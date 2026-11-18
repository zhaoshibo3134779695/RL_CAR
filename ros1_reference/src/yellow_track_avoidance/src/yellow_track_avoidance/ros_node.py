"""ROS1 wrapper for the ROS-independent avoidance pipeline."""

import json
import threading
from typing import Dict, Optional

import numpy as np
import rosgraph
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

from .config import ConfigError, load_config
from .model import ControlCommand, MotionState, PipelineResult
from .perception_latch import LOCKED, MissionPerceptionLatch
from .pipeline import AvoidancePipeline
from .passage_navigation import infer_parallel_boundary


class YellowTrackAvoidanceNode:
    def __init__(self):
        config_file = rospy.get_param("~config_file")
        self.config = load_config(config_file)
        self._apply_ros_overrides()
        self.offline_planning_enabled = bool(
            rospy.get_param("~enable_planning", False)
        )
        self.pipeline = AvoidancePipeline(self.config)
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.last_image_received_sec: Optional[float] = None
        self.last_result: Optional[PipelineResult] = None
        self.camera_info: Optional[CameraInfo] = None
        self.output_conflict_reported = False
        self.last_publisher_audit_sec = -1.0
        self.cached_output_conflicts = []
        self.debug_frame_index = 0
        self.require_photometry_ready = bool(
            rospy.get_param("~require_photometry_ready", False)
        )
        self.photometry_ready = not self.require_photometry_ready
        self.photometry_preview = False
        self.photometry_wait_frames = 0
        self.photometry_ready_topic = str(
            rospy.get_param(
                "~photometry_ready_topic",
                "/yellow_track_avoidance/camera_photometry_ready",
            )
        )
        self.photometry_preview_topic = str(
            rospy.get_param(
                "~photometry_preview_topic",
                "/yellow_track_avoidance/camera_photometry_preview",
            )
        )
        self.mission_latch_enabled = bool(
            rospy.get_param("~enable_mission_perception_latch", False)
        )
        self.corridor_status_topic = str(
            rospy.get_param(
                "~competition_corridor_status_topic",
                "/yellow_track_competition_corridor_map/status",
            )
        )
        self.disc_bridge_status_topic = str(
            rospy.get_param(
                "~disc_bridge_status_topic",
                "/yellow_track_disc_costmap_bridge/status",
            )
        )
        self.scene_reset_topic = str(
            rospy.get_param(
                "~scene_reset_topic", "/yellow_track_avoidance/reset_scene"
            )
        )
        self.locked_status_period = max(
            0.2, float(rospy.get_param("~locked_status_period_sec", 1.0))
        )
        self.perception_latch = MissionPerceptionLatch(
            expected_disc_count=int(
                rospy.get_param("~expected_frozen_disc_count", 2)
            ),
            stable_sec=float(
                rospy.get_param("~perception_lock_stable_sec", 0.5)
            ),
        )
        self.last_locked_status_publish_sec = -1.0

        ros_cfg = self.config["ros"]
        queue_size = int(ros_cfg["queue_size"])
        self.cmd_topic = str(ros_cfg["output_cmd_topic"])
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        prefix = str(ros_cfg["debug_topic_prefix"]).rstrip("/")
        self.debug_publishers = {
            name: rospy.Publisher(prefix + "/" + topic, Image, queue_size=1)
            for name, topic in {
                "raw_overlay": "raw_overlay",
                "yellow_mask": "yellow_mask",
                "yellow_score": "yellow_score",
                "red_mask": "red_mask",
                "disc_overlay": "disc_overlay",
                "disc_red_mask": "disc_red_mask",
                "disc_white_mask": "disc_white_mask",
                "disc_candidates": "disc_candidates",
                "bev": "bev",
                "plan": "plan",
            }.items()
        }
        self.status_pub = rospy.Publisher(prefix + "/status", String, queue_size=1, latch=True)
        self.detections_pub = rospy.Publisher(prefix + "/disks", String, queue_size=1)
        self.boundaries_pub = rospy.Publisher(
            prefix + "/boundaries", String, queue_size=1
        )
        self.disc_status_pub = rospy.Publisher(
            prefix + "/disc_status", String, queue_size=1
        )
        self.candidates_pub = rospy.Publisher(prefix + "/candidates", String, queue_size=1)
        self.path_pub = rospy.Publisher(prefix + "/selected_path", Path, queue_size=1)

        self.image_sub = rospy.Subscriber(
            str(ros_cfg["image_topic"]), Image, self._image_callback,
            queue_size=queue_size, buff_size=2 ** 24,
        )
        self.camera_info_sub = rospy.Subscriber(
            str(ros_cfg["camera_info_topic"]), CameraInfo, self._camera_info_callback,
            queue_size=1,
        )
        if self.require_photometry_ready:
            self.photometry_ready_sub = rospy.Subscriber(
                self.photometry_ready_topic,
                Bool,
                self._photometry_ready_callback,
                queue_size=1,
            )
            self.photometry_preview_sub = rospy.Subscriber(
                self.photometry_preview_topic,
                Bool,
                self._photometry_preview_callback,
                queue_size=1,
            )
        else:
            self.photometry_ready_sub = None
            self.photometry_preview_sub = None
        if self.mission_latch_enabled:
            self.corridor_status_sub = rospy.Subscriber(
                self.corridor_status_topic,
                String,
                self._corridor_status_callback,
                queue_size=1,
            )
            self.disc_bridge_status_sub = rospy.Subscriber(
                self.disc_bridge_status_topic,
                String,
                self._disc_bridge_status_callback,
                queue_size=1,
            )
            self.scene_reset_sub = rospy.Subscriber(
                self.scene_reset_topic,
                String,
                self._scene_reset_callback,
                queue_size=1,
            )
        else:
            self.corridor_status_sub = None
            self.disc_bridge_status_sub = None
            self.scene_reset_sub = None
        self.reset_service = rospy.Service("~reset_stop", Trigger, self._reset_stop)
        self.watchdog = rospy.Timer(rospy.Duration(0.05), self._watchdog_callback)
        rospy.on_shutdown(self._shutdown)

        rospy.loginfo("yellow_track_avoidance ready: ROS1 image=%s output=%s motion=%s planning=%s calibrated=%s",
                      ros_cfg["image_topic"], self.cmd_topic,
                      self.config["control"]["enable_motion"],
                      self.offline_planning_enabled,
                      self.pipeline.calibrated)
        if not self.pipeline.calibrated:
            rospy.logwarn("BEV/physical calibration incomplete: node is locked in STOP")
        if not bool(self.config["control"]["enable_motion"]):
            rospy.logwarn("enable_motion=false: only zero velocity commands can be published")
        if self.mission_latch_enabled:
            rospy.loginfo(
                "competition perception latch enabled: collect for %.3fs after "
                "at least %d confirmed discs, then freeze the complete set",
                self.perception_latch.stable_sec,
                self.perception_latch.expected_disc_count,
            )
        if self.require_photometry_ready:
            rospy.loginfo(
                "perception waits for fixed camera photometry on %s "
                "(safe preview: %s)",
                self.photometry_ready_topic,
                self.photometry_preview_topic,
            )

    def _apply_ros_overrides(self) -> None:
        mappings = {
            "image_topic": ("ros", "image_topic"),
            "camera_info_topic": ("ros", "camera_info_topic"),
            "output_cmd_topic": ("ros", "output_cmd_topic"),
            "enable_motion": ("control", "enable_motion"),
            "robot_width_m": ("physical", "robot_width_m"),
            "robot_length_m": ("physical", "robot_length_m"),
            "local_forward_origin_offset_m": (
                "physical",
                "local_forward_origin_offset_m",
            ),
        }
        for parameter, path in mappings.items():
            private = "~" + parameter
            if rospy.has_param(private):
                self.config[path[0]][path[1]] = rospy.get_param(private)

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _photometry_ready_callback(self, message: Bool) -> None:
        with self.lock:
            was_ready = self.photometry_ready
            self.photometry_ready = bool(message.data)
            if self.photometry_ready and not was_ready:
                # Start mission tracking from clean IDs, but keep the bounded
                # red-mask evidence collected at the detector-approved final
                # exposure.  Clearing it here created a second five-frame
                # blind acquisition window in which an intermittent far disc
                # could disappear after photometry had already accepted two.
                self.pipeline.reset_perception_history(
                    preserve_far_red_evidence=True
                )
        if self.photometry_ready and not was_ready:
            rospy.loginfo(
                "camera photometry is stable; starting visual perception"
            )

    def _photometry_preview_callback(self, message: Bool) -> None:
        with self.lock:
            self.photometry_preview = bool(message.data)

    @staticmethod
    def _status_payload(message, label):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid %s status JSON", label)
            return None
        if not isinstance(payload, dict):
            rospy.logwarn_throttle(1.0, "invalid %s status payload", label)
            return None
        return payload

    def _corridor_status_callback(self, message) -> None:
        payload = self._status_payload(message, "competition corridor")
        if payload is None:
            return
        now_sec = rospy.Time.now().to_sec()
        with self.lock:
            newly_locked = self.perception_latch.update_corridor(
                payload, now_sec
            )
            if newly_locked:
                self._on_perception_locked(now_sec)

    def _disc_bridge_status_callback(self, message) -> None:
        payload = self._status_payload(message, "disc bridge")
        if payload is None:
            return
        now_sec = rospy.Time.now().to_sec()
        with self.lock:
            newly_locked = self.perception_latch.update_bridge(
                payload, now_sec
            )
            if newly_locked:
                self._on_perception_locked(now_sec)

    def _on_perception_locked(self, now_sec) -> None:
        rospy.loginfo(
            "competition perception LOCKED: corridor cloud ready and %d "
            "odom discs frozen in both costmaps",
            self.perception_latch.frozen_disc_count,
        )
        self._publish_locked_status(now_sec)

    def _scene_reset_callback(self, message) -> None:
        payload = self._status_payload(message, "scene reset")
        if payload is None:
            return
        try:
            generation = int(payload["scene_generation"])
        except (KeyError, TypeError, ValueError, OverflowError):
            rospy.logwarn_throttle(1.0, "scene reset generation is invalid")
            return
        with self.lock:
            if generation <= self.perception_latch.scene_generation:
                return
            self.perception_latch.reset(scene_generation=generation)
            # A new mission must not inherit any detector or temporal BEV
            # evidence from the previous physical scene.
            self.pipeline.reset_perception_history(
                preserve_far_red_evidence=False
            )
            self.pipeline.reset_stop()
            self.last_result = None
            self.last_locked_status_publish_sec = -1.0
        rospy.logwarn(
            "competition perception reset to ACQUIRE at scene generation=%d",
            generation,
        )

    def _resolution_valid(self, message: Image) -> bool:
        camera = self.config["camera"]
        expected_width = int(camera["expected_width"])
        expected_height = int(camera["expected_height"])
        configured_encodings = camera.get("expected_encoding", "")
        if isinstance(configured_encodings, (list, tuple)):
            expected_encodings = {
                str(value).strip()
                for value in configured_encodings
                if str(value).strip()
            }
        else:
            value = str(configured_encodings).strip()
            expected_encodings = {value} if value else set()
        if expected_width and int(message.width) != expected_width:
            return False
        if expected_height and int(message.height) != expected_height:
            return False
        if expected_encodings and message.encoding not in expected_encodings:
            return False
        return True

    def _image_callback(self, message: Image) -> None:
        receipt_sec = rospy.Time.now().to_sec()
        with self.lock:
            self.last_image_received_sec = receipt_sec
            if (
                self.mission_latch_enabled
                and self.perception_latch.mode == LOCKED
            ):
                self.perception_latch.note_received_frame()
                # Mission world geometry is already frozen downstream.
                # Return before resolution checks and, critically, before
                # cv_bridge copies or any OpenCV work.
                self.perception_latch.note_skipped_frame()
                return
            waiting_for_photometry = bool(
                getattr(self, "require_photometry_ready", False)
                and not getattr(self, "photometry_ready", False)
            )
            preview_only = bool(
                waiting_for_photometry
                and getattr(self, "photometry_preview", False)
            )
            if waiting_for_photometry and not preview_only:
                self.photometry_wait_frames += 1
                rospy.loginfo_throttle(
                    2.0,
                    "waiting for fixed camera photometry; visual frame skipped",
                )
                return
            if self.mission_latch_enabled and not preview_only:
                self.perception_latch.note_received_frame()
            if not self._resolution_valid(message):
                result = self.pipeline.camera_timeout(receipt_sec)
                result.stop_reason = "camera_format_mismatch"
                result.diagnostics["received_encoding"] = message.encoding
                result.diagnostics["received_width"] = int(message.width)
                result.diagnostics["received_height"] = int(message.height)
                self.pipeline.state_machine.force_stop("camera_format_mismatch", receipt_sec)
                self._publish_result(
                    result, message.header, preview_only=preview_only
                )
                return
            try:
                image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                stamp_sec = message.header.stamp.to_sec() if message.header.stamp else receipt_sec
                if stamp_sec <= 0.0:
                    stamp_sec = receipt_sec
                requested_debug = set()
                debug_stride = max(
                    1, int(self.config.get("debug", {}).get("image_publish_stride", 1))
                )
                publish_debug_frame = self.debug_frame_index % debug_stride == 0
                self.debug_frame_index += 1
                if (
                    bool(self.config["ros"]["publish_debug_images"])
                    and publish_debug_frame
                ):
                    requested_debug = {
                        name
                        for name, publisher in self.debug_publishers.items()
                        if publisher.get_num_connections() > 0
                    }
                run_planning = (not preview_only) and (
                    self.offline_planning_enabled
                    or bool(self.config["control"]["enable_motion"])
                    or (
                        "plan" in requested_debug
                        or self.candidates_pub.get_num_connections() > 0
                        or self.path_pub.get_num_connections() > 0
                    )
                )
                result = self.pipeline.process(
                    image,
                    stamp_sec,
                    debug_image_names=requested_debug,
                    run_planning=run_planning,
                )
                if self.mission_latch_enabled and not preview_only:
                    self.perception_latch.note_processed_frame()
            except (CvBridgeError, ConfigError, ValueError, RuntimeError) as exc:
                rospy.logerr_throttle(1.0, "image processing failed: %s", exc)
                result = self.pipeline.camera_timeout(receipt_sec)
                result.stop_reason = "image_processing_error"
                result.diagnostics["exception"] = str(exc)
                self.pipeline.state_machine.force_stop("image_processing_error", receipt_sec)
            except Exception as exc:
                rospy.logerr_throttle(1.0, "unexpected image processing failure: %s", exc)
                result = self.pipeline.camera_timeout(receipt_sec)
                result.stop_reason = "image_processing_error"
                result.diagnostics["exception"] = str(exc)
                self.pipeline.state_machine.force_stop("image_processing_error", receipt_sec)
            self._publish_result(
                result, message.header, preview_only=preview_only
            )

    def _other_output_publishers(self):
        if not bool(self.config["ros"]["require_exclusive_output"]):
            return []
        now_sec = rospy.Time.now().to_sec()
        interval = float(self.config["ros"]["publisher_audit_interval_sec"])
        if self.last_publisher_audit_sec >= 0.0 and now_sec - self.last_publisher_audit_sec < interval:
            return list(self.cached_output_conflicts)
        try:
            publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
            resolved = rospy.resolve_name(self.cmd_topic)
            own_name = rospy.get_name()
            for topic, nodes in publishers:
                if rospy.resolve_name(topic) == resolved:
                    self.cached_output_conflicts = [node for node in nodes if node != own_name]
                    self.last_publisher_audit_sec = now_sec
                    return list(self.cached_output_conflicts)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "could not audit output publishers: %s", exc)
            return ["master_audit_failed"]
        self.cached_output_conflicts = []
        self.last_publisher_audit_sec = now_sec
        return []

    def _authorized_command(self, result: PipelineResult) -> ControlCommand:
        if not bool(self.config["control"]["enable_motion"]):
            return ControlCommand()
        conflicts = self._other_output_publishers()
        if conflicts:
            if not self.output_conflict_reported:
                rospy.logerr("refusing motion: output topic %s also has publishers %s", self.cmd_topic, conflicts)
                self.output_conflict_reported = True
            self.pipeline.state_machine.force_stop("output_topic_has_other_publishers", rospy.Time.now().to_sec())
            result.state = MotionState.STOP
            result.stop_reason = "output_topic_has_other_publishers"
            result.diagnostics["conflicting_publishers"] = conflicts
            return ControlCommand()
        self.output_conflict_reported = False
        return result.command

    @staticmethod
    def _twist(command: ControlCommand) -> Twist:
        message = Twist()
        message.linear.x = float(command.linear_x)
        message.linear.y = float(command.linear_y)
        message.angular.z = float(command.angular_z)
        return message

    def _publish_result(
        self, result: PipelineResult, header, preview_only: bool = False
    ) -> None:
        result.diagnostics["photometry_preview_only"] = bool(preview_only)
        command = (
            ControlCommand()
            if preview_only
            else self._authorized_command(result)
        )
        self.cmd_pub.publish(self._twist(command))
        result.command = command
        self.last_result = result

        if bool(self.config["ros"]["publish_debug_images"]):
            for name, image in result.debug_images.items():
                if name not in self.debug_publishers:
                    continue
                if self.debug_publishers[name].get_num_connections() <= 0:
                    continue
                debug_message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
                debug_message.header = header
                self.debug_publishers[name].publish(debug_message)

        self.status_pub.publish(String(data=json.dumps(self._status_dict(result), ensure_ascii=False, allow_nan=False)))
        # Preview frames are real detector measurements for selecting the
        # startup exposure.  They must never enter the frozen odom world model.
        if not preview_only and self.detections_pub.get_num_connections() > 0:
            self.detections_pub.publish(
                String(
                    data=json.dumps(
                        self._disk_dict(result), ensure_ascii=False, allow_nan=False
                    )
                )
            )
        if not preview_only and self.boundaries_pub.get_num_connections() > 0:
            self.boundaries_pub.publish(
                String(
                    data=json.dumps(
                        self._boundary_dict(result),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            )
        if self.disc_status_pub.get_num_connections() > 0:
            self.disc_status_pub.publish(
                String(
                    data=json.dumps(
                        self._disc_status_dict(result, header),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            )
        if (
            not preview_only
            and
            bool(self.config["ros"]["publish_candidates"])
            and self.candidates_pub.get_num_connections() > 0
        ):
            self.candidates_pub.publish(
                String(
                    data=json.dumps(
                        self._candidate_dict(result),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            )
        if not preview_only and self.path_pub.get_num_connections() > 0:
            self._publish_selected_path(result, header)

    @staticmethod
    def _finite(value):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    def _boundary_dict(self, result: PipelineResult) -> Dict:
        boundary = result.boundary
        boundary_mode = str(
            self.config.get("boundary", {}).get("detection_mode", "both")
        ).strip().lower()
        left_only = boundary_mode == "left_only"
        if (
            boundary is None
            or self.pipeline.transformer is None
            or boundary.rows_px.size == 0
        ):
            return {
                "stamp": result.stamp_sec,
                "frame_id": "base_link",
                "left_detected": False,
                "right_detected": False,
                "boundary_detection_mode": boundary_mode,
                "boundary_pair_valid": False,
                "boundary_inference": "none",
                "quality": 0.0,
                "left_base_m": [],
                "right_base_m": [],
                "right_wall_base_m": [],
            }
        observed_range = (
            (boundary.rows_px >= int(boundary.observed_min_row_px))
            & (boundary.rows_px <= int(boundary.observed_max_row_px))
        )
        forward_offset = float(
            self.config["physical"].get(
                "local_forward_origin_offset_m", 0.0
            )
        )

        def metric_side(values, detected):
            observed = observed_range & np.isfinite(values)
            if not detected or not np.any(observed):
                return np.empty((0, 2), dtype=np.float64)
            pixels = np.column_stack(
                [values[observed], boundary.rows_px[observed]]
            )
            metric = self.pipeline.transformer.pixel_to_metric(pixels)
            metric[:, 0] += forward_offset
            return metric

        left_base = metric_side(
            boundary.left_px, boundary.left_detected
        )
        right_base = metric_side(
            boundary.right_px, boundary.right_detected
        )
        if left_only:
            right_base = np.empty((0, 2), dtype=np.float64)
        inference = "none"
        expected_separation = float(
            self.config["physical"]["track_gray_width_m"]
        ) + float(self.config["physical"]["yellow_line_width_m"])
        if len(left_base) >= 2 and len(right_base) < 2:
            right_base = np.asarray(
                infer_parallel_boundary(
                    left_base.tolist(), -expected_separation
                ),
                dtype=np.float64,
            )
            inference = "right_from_left_metric_width"
        elif (
            not left_only
            and len(right_base) >= 2
            and len(left_base) < 2
        ):
            left_base = np.asarray(
                infer_parallel_boundary(
                    right_base.tolist(), expected_separation
                ),
                dtype=np.float64,
            )
            inference = "left_from_right_metric_width"
        right_wall = right_base.copy()
        if len(right_wall):
            # Vehicle coordinates use positive Y to the left. The fixed outer
            # wall is one measured yellow-line width beyond the right line.
            right_wall[:, 1] -= float(
                self.config["physical"]["yellow_line_width_m"]
            )

        def sampled(points):
            if len(points) == 0:
                return []
            count = min(20, len(points))
            indices = np.linspace(
                0, len(points) - 1, count
            ).astype(np.int32)
            return points[indices].tolist()

        return {
            "stamp": result.stamp_sec,
            "frame_id": "base_link",
            "left_detected": bool(boundary.left_detected),
            "right_detected": bool(
                boundary.right_detected and not left_only
            ),
            "boundary_detection_mode": boundary_mode,
            "quality": self._finite(boundary.quality),
            "boundary_pair_valid": bool(
                len(left_base) >= 2 and len(right_base) >= 2
            ),
            "boundary_inference": inference,
            "expected_center_separation_m": expected_separation,
            "expected_inner_clear_width_m": float(
                self.config["physical"]["track_gray_width_m"]
            ),
            "yellow_line_width_m": float(
                self.config["physical"]["yellow_line_width_m"]
            ),
            "left_base_m": sampled(left_base),
            "right_base_m": sampled(right_base),
            "right_wall_base_m": sampled(right_wall),
        }

    def _status_dict(self, result: PipelineResult) -> Dict:
        diagnostics = dict(result.diagnostics)
        if self.mission_latch_enabled:
            diagnostics.update(
                self.perception_latch.snapshot(result.stamp_sec)
            )
        diagnostics["photometry_ready"] = bool(
            getattr(self, "photometry_ready", True)
        )
        diagnostics["photometry_preview"] = bool(
            getattr(self, "photometry_preview", False)
        )
        diagnostics["photometry_wait_frames"] = int(
            getattr(self, "photometry_wait_frames", 0)
        )
        return {
            "stamp": result.stamp_sec,
            "state": result.state.value,
            "stop_reason": result.stop_reason,
            "command": {
                "linear_x": result.command.linear_x,
                "linear_y": result.command.linear_y,
                "angular_z": result.command.angular_z,
            },
            "camera_info_received": self.camera_info is not None,
            "diagnostics": diagnostics,
        }

    def _locked_status_dict(self, now_sec) -> Dict:
        return {
            "stamp": float(now_sec),
            "state": MotionState.STOP.value,
            "stop_reason": "perception_world_model_locked",
            "command": {
                "linear_x": 0.0,
                "linear_y": 0.0,
                "angular_z": 0.0,
            },
            "camera_info_received": self.camera_info is not None,
            "diagnostics": self.perception_latch.snapshot(now_sec),
        }

    def _publish_locked_status(self, now_sec) -> None:
        self.status_pub.publish(
            String(
                data=json.dumps(
                    self._locked_status_dict(now_sec),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        )
        self.last_locked_status_publish_sec = float(now_sec)

    def _disk_dict(self, result: PipelineResult) -> Dict:
        forward_offset = float(
            self.config["physical"].get(
                "local_forward_origin_offset_m", 0.0
            )
        )
        raw = []
        for disk in result.raw_disks:
            contour = disk.contour.reshape(-1, 2)
            stride = max(1, len(contour) // 48)
            raw.append({
                "track_id": disk.track_id,
                "confirmed": disk.confirmed,
                "center_px": [disk.center_px[0], disk.center_px[1]],
                "center_m": [self._finite(disk.center_m[0]), self._finite(disk.center_m[1])],
                "center_base_m": [
                    self._finite(disk.center_m[0] + forward_offset),
                    self._finite(disk.center_m[1]),
                ],
                "ellipse": [list(disk.ellipse[0]), list(disk.ellipse[1]), disk.ellipse[2]],
                "contour": contour[::stride].astype(int).tolist(),
                "estimated_diameter_m": disk.estimated_diameter_m,
                "confidence": disk.confidence,
                "red_annulus_ratio": disk.red_annulus_ratio,
                "white_center_ratio": disk.white_center_ratio,
                "red_sector_coverage": disk.red_sector_coverage,
                "white_center_contrast": disk.white_center_contrast,
                "proposal_source": disk.proposal_source,
                "candidate_id": disk.candidate_id,
                "contour_area_px": disk.contour_area_px,
                "red_area_px": disk.red_area_px,
                "white_area_px": disk.white_area_px,
                "red_white_center_distance_px": disk.red_white_center_distance_px,
                "inside_lane": disk.inside_lane,
                "distance_to_left_line_px": self._finite(
                    disk.distance_to_left_line_px
                ),
                "distance_to_right_line_px": self._finite(
                    disk.distance_to_right_line_px
                ),
                "used_history": disk.used_history,
                "metric_projection": disk.projection_diagnostics,
                "stamp": disk.stamp_sec,
            })
        tracked = [
            {
                "track_id": disk.track_id,
                "center_px": list(disk.center_px),
                "center_m": list(disk.center_m),
                "center_base_m": [
                    disk.center_m[0] + forward_offset,
                    disk.center_m[1],
                ],
                "estimated_diameter_m": disk.estimated_diameter_m,
                "confidence": disk.confidence,
                "hits": disk.hits,
                "misses": disk.misses,
                "stamp": disk.stamp_sec,
            }
            for disk in result.tracked_disks
        ]
        return {
            "stamp": result.stamp_sec,
            "raw": raw,
            "confirmed_tracks": tracked,
        }

    def _disc_status_dict(self, result: PipelineResult, header) -> Dict:
        detections = sorted(
            result.raw_disks, key=lambda item: item.confidence, reverse=True
        )
        primary = (
            min(detections, key=lambda item: item.center_px[0])
            if detections
            else None
        )
        candidates = []
        detector = self.pipeline.disk_detector
        for item in detector.last_candidates:
            ellipse = item.get("ellipse")
            candidates.append(
                {
                    "candidate_id": int(item.get("candidate_id", -1)),
                    "source": str(item.get("source", "")),
                    "accepted": bool(item.get("accepted", False)),
                    "reject_reason": str(item.get("reject_reason", "")),
                    "ellipse": None
                    if ellipse is None
                    else [
                        [self._finite(ellipse[0][0]), self._finite(ellipse[0][1])],
                        [self._finite(ellipse[1][0]), self._finite(ellipse[1][1])],
                        self._finite(ellipse[2]),
                    ],
                    "contour_area_px": self._finite(
                        item.get("contour_area_px", 0.0)
                    ),
                    "red_area_px": int(item.get("red_area_px", 0)),
                    "white_area_px": int(item.get("white_area_px", 0)),
                    "red_white_center_distance_px": (
                        None
                        if item.get("red_white_center_distance_px") is None
                        else self._finite(item["red_white_center_distance_px"])
                    ),
                    "red_annulus_ratio": self._finite(
                        item.get("red_annulus_ratio", 0.0)
                    ),
                    "white_center_ratio": self._finite(
                        item.get("white_center_ratio", 0.0)
                    ),
                    "neutral_center_ratio": self._finite(
                        item.get("neutral_center_ratio", 0.0)
                    ),
                    "red_center_ratio": self._finite(
                        item.get("red_center_ratio", 0.0)
                    ),
                    "ring_sector_coverage": self._finite(
                        item.get("ring_sector_coverage", 0.0)
                    ),
                    "missing_sector_fraction": self._finite(
                        item.get("missing_sector_fraction", 0.0)
                    ),
                    "center_brightness_contrast": self._finite(
                        item.get("center_brightness_contrast", 0.0)
                    ),
                    "nearby_red_area_px": int(
                        item.get("nearby_red_area_px", 0)
                    ),
                    "strong_ring_recovery": bool(
                        item.get("strong_ring_recovery", False)
                    ),
                    "far_white_center_recovery": bool(
                        item.get("far_white_center_recovery", False)
                    ),
                    "far_red_ring_recovery": bool(
                        item.get("far_red_ring_recovery", False)
                    ),
                    "confidence": self._finite(item.get("confidence", 0.0)),
                    "inside_lane": item.get("inside_lane"),
                    "distance_to_left_line_px": (
                        None
                        if item.get("distance_to_left_line_px") is None
                        else self._finite(item["distance_to_left_line_px"])
                    ),
                    "distance_to_right_line_px": (
                        None
                        if item.get("distance_to_right_line_px") is None
                        else self._finite(item["distance_to_right_line_px"])
                    ),
                    "metric_projection": item.get("metric_projection"),
                }
            )

        serialized_detections = []
        for disk in detections:
            major = float(max(disk.ellipse[1]))
            minor = float(min(disk.ellipse[1]))
            serialized_detections.append(
                {
                    "candidate_id": disk.candidate_id,
                    "center_x_px": disk.center_px[0],
                    "center_y_px": disk.center_px[1],
                    "major_axis_px": major,
                    "minor_axis_px": minor,
                    "angle_deg": float(disk.ellipse[2]),
                    "contour_area_px": disk.contour_area_px,
                    "confidence": disk.confidence,
                    "red_area_px": disk.red_area_px,
                    "white_area_px": disk.white_area_px,
                    "red_white_center_distance_px": disk.red_white_center_distance_px,
                    "inside_lane": disk.inside_lane,
                    "distance_to_left_line_px": self._finite(
                        disk.distance_to_left_line_px
                    ),
                    "distance_to_right_line_px": self._finite(
                        disk.distance_to_right_line_px
                    ),
                    "used_history": False,
                    "metric_projection": disk.projection_diagnostics,
                }
            )

        diagnostics = result.diagnostics
        return {
            "timestamp": result.stamp_sec,
            "frame_id": int(getattr(header, "seq", 0)),
            "left_line_detected": bool(diagnostics.get("left_detected", False)),
            "right_line_detected": bool(diagnostics.get("right_detected", False)),
            "left_line_quality": self._finite(
                diagnostics.get(
                    "left_quality", diagnostics.get("boundary_quality", 0.0)
                )
            ),
            "disc_detected": primary is not None,
            "disc_count": len(detections),
            "disc_center_x_px": None if primary is None else primary.center_px[0],
            "disc_center_y_px": None if primary is None else primary.center_px[1],
            "disc_major_axis_px": None if primary is None else float(max(primary.ellipse[1])),
            "disc_minor_axis_px": None if primary is None else float(min(primary.ellipse[1])),
            "disc_angle_deg": None if primary is None else float(primary.ellipse[2]),
            "disc_area_px": None if primary is None else primary.contour_area_px,
            "disc_confidence": None if primary is None else primary.confidence,
            "red_area_px": (
                int(diagnostics.get("disc_red_mask_area_px", 0))
                if primary is None
                else primary.red_area_px
            ),
            "white_area_px": (
                int(diagnostics.get("disc_white_mask_area_px", 0))
                if primary is None
                else primary.white_area_px
            ),
            "red_mask_area_px": int(diagnostics.get("disc_red_mask_area_px", 0)),
            "white_mask_area_px": int(diagnostics.get("disc_white_mask_area_px", 0)),
            "red_white_center_distance_px": (
                None if primary is None else primary.red_white_center_distance_px
            ),
            "candidate_count": int(diagnostics.get("disc_candidate_count", 0)),
            "rejected_candidate_count": int(
                diagnostics.get("disc_rejected_candidate_count", 0)
            ),
            "disc_inside_lane": None if primary is None else primary.inside_lane,
            "distance_to_left_line_px": (
                None
                if primary is None
                else self._finite(primary.distance_to_left_line_px)
            ),
            "distance_to_right_line_px": (
                None
                if primary is None
                else self._finite(primary.distance_to_right_line_px)
            ),
            "processing_time_ms": self._finite(
                diagnostics.get("processing_time_ms", 0.0)
            ),
            "used_history": False,
            "motion_enabled": bool(diagnostics.get("motion_enabled", False)),
            "detections": serialized_detections,
            "candidates": candidates,
        }

    @staticmethod
    def _candidate_dict(result: PipelineResult) -> Dict:
        if result.planning is None:
            return {
                "stamp": result.stamp_sec,
                "safe": False,
                "reason": result.stop_reason,
                "candidates": [],
            }
        return {
            "stamp": result.stamp_sec,
            "safe": result.planning.safe,
            "reason": result.planning.reason,
            "selected": None if result.planning.selected is None else result.planning.selected.name,
            "left_space_m": result.planning.left_space_m,
            "right_space_m": result.planning.right_space_m,
            "candidates": [
                {
                    "name": item.name,
                    "offset_m": item.offset_m,
                    "direction": item.direction,
                    "collision_free": item.collision_free,
                    "score": None if not np.isfinite(item.score) else item.score,
                    "reject_reason": item.reject_reason,
                    "minimum_boundary_clearance_m": item.minimum_boundary_clearance_m,
                    "minimum_obstacle_clearance_m": None if not np.isfinite(item.minimum_obstacle_clearance_m) else item.minimum_obstacle_clearance_m,
                    "max_curvature": item.max_curvature,
                    "costs": item.costs,
                    "points_m": item.points_m.tolist(),
                }
                for item in result.planning.candidates
            ],
        }

    def _publish_selected_path(self, result: PipelineResult, header) -> None:
        path = Path()
        path.header = header
        path.header.frame_id = "yellow_track_bev"
        if result.planning is not None and result.planning.selected is not None:
            for forward, left in result.planning.selected.points_m:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = float(forward)
                pose.pose.position.y = float(left)
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
        self.path_pub.publish(path)

    def _watchdog_callback(self, _event) -> None:
        now_sec = rospy.Time.now().to_sec()
        with self.lock:
            if (
                self.mission_latch_enabled
                and self.perception_latch.mode == LOCKED
            ):
                if (
                    self.last_locked_status_publish_sec < 0.0
                    or now_sec - self.last_locked_status_publish_sec
                    >= self.locked_status_period
                ):
                    self._publish_locked_status(now_sec)
                return
            timeout = float(self.config["ros"]["camera_timeout_sec"])
            if self.last_image_received_sec is None or now_sec - self.last_image_received_sec > timeout:
                result = self.pipeline.camera_timeout(now_sec)
                command = self._authorized_command(result)
                self.cmd_pub.publish(self._twist(command))
                self.last_result = result
                self.status_pub.publish(String(data=json.dumps(self._status_dict(result), ensure_ascii=False, allow_nan=False)))

    def _reset_stop(self, _request) -> TriggerResponse:
        with self.lock:
            self.pipeline.reset_stop()
        return TriggerResponse(success=True, message="reset requested; transition requires consecutive safe frames")

    def _shutdown(self) -> None:
        stop = Twist()
        repeats = int(self.config["control"]["shutdown_stop_repeats"])
        interval = float(self.config["control"]["shutdown_stop_interval_sec"])
        for _ in range(max(1, repeats)):
            try:
                self.cmd_pub.publish(stop)
                rospy.sleep(interval)
            except Exception:
                break


def main() -> None:
    rospy.init_node("yellow_track_avoidance")
    try:
        YellowTrackAvoidanceNode()
        rospy.spin()
    except (ConfigError, KeyError, ValueError) as exc:
        rospy.logfatal("yellow_track_avoidance configuration error: %s", exc)
        raise
