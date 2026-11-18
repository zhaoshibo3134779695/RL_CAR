#!/usr/bin/env python3
"""Expose camera-detected red/white discs as a clearing LaserScan for costmap_2d."""

import json
import math
import threading

import numpy as np
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

from yellow_track_avoidance.disc_passage import (
    PHASE_PASSED_CANDIDATE,
    PHASE_TOMBSTONE,
    PassageConfig,
    cumulative_lengths,
    reset_candidate,
    tombstone_should_reactivate,
    update_passage,
)
from yellow_track_avoidance.disc_memory import StableDiscMemory


class DiscLaserScanBridge:
    def __init__(self):
        self.lock = threading.RLock()
        self.input_topic = str(
            rospy.get_param(
                "~disk_topic", "/yellow_track_avoidance/debug/disks"
            )
        )
        self.output_topic = str(
            rospy.get_param(
                "~scan_topic", "/yellow_track_avoidance/disc_scan"
            )
        )
        self.marker_topic = str(
            rospy.get_param(
                "~marker_topic", "/yellow_track_avoidance/disc_markers"
            )
        )
        self.require_localization_ready = bool(
            rospy.get_param("~require_localization_ready", False)
        )
        self.enable_costmap_writes = bool(
            rospy.get_param("~enable_costmap_writes", True)
        )
        self.competition_two_disc_mode = bool(
            rospy.get_param("~competition_two_disc_mode", False)
        )
        self.lock_confirmed_discs_for_mission = bool(
            rospy.get_param(
                "~lock_confirmed_discs_for_mission", False
            )
        )
        self.localization_status_topic = str(
            rospy.get_param(
                "~localization_status_topic",
                "/yellow_track_field_localization/status",
            )
        )
        self.global_path_topic = str(
            rospy.get_param(
                "~global_path_topic",
                "/move_base/SBPLLatticePlanner/plan",
            )
        )
        self.move_base_status_topic = str(
            rospy.get_param("~move_base_status_topic", "/move_base/status")
        )
        self.navigation_gate_status_topic = str(
            rospy.get_param(
                "~navigation_gate_status_topic",
                "/yellow_track_navigation_gate/status",
            )
        )
        self.scene_reset_topic = str(
            rospy.get_param(
                "~scene_reset_topic", "/yellow_track_avoidance/reset_scene"
            )
        )
        self.require_navigation_gate_for_scene_reset = bool(
            rospy.get_param(
                "~require_navigation_gate_for_scene_reset", True
            )
        )
        self.output_command_topic = str(
            rospy.get_param("~output_command_topic", "/cmd_vel")
        )
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.reset_state_timeout = float(
            rospy.get_param("~reset_state_timeout_sec", 0.75)
        )
        self.reset_linear_speed_limit = float(
            rospy.get_param("~reset_linear_speed_max_mps", 0.005)
        )
        self.reset_angular_speed_limit = float(
            rospy.get_param("~reset_angular_speed_max_radps", 0.01)
        )
        self.frame_id = str(
            rospy.get_param("~frame_id", "base_link")
        ).lstrip("/")
        self.fixed_frame = str(
            rospy.get_param("~fixed_frame", "odom")
        ).lstrip("/")
        self.forward_offset = float(
            rospy.get_param("~local_forward_origin_offset_m", 0.0)
        )
        # Memory is world-fixed, so an active disc must remain visible after
        # it moves beside or behind the robot. A forward-only virtual scan
        # would make the obstacle disappear before passage confirmation.
        self.angle_min = float(
            rospy.get_param("~angle_min_rad", -math.pi)
        )
        self.angle_max = float(
            rospy.get_param("~angle_max_rad", math.pi)
        )
        self.angle_increment = math.radians(
            float(rospy.get_param("~angle_increment_deg", 1.0))
        )
        self.range_min = float(rospy.get_param("~range_min_m", 0.08))
        self.range_max = float(rospy.get_param("~range_max_m", 1.70))
        # Keep the measured/physical disc size separate from the collision
        # envelope written to costmap.  Stage 1 can therefore reserve a hard
        # positioning allowance without corrupting the reported 0.10 m disc
        # diameter or the passage bookkeeping that uses the physical object.
        self.disk_radius = float(rospy.get_param("~disk_radius_m", 0.05))
        self.costmap_disk_radius = float(
            rospy.get_param("~costmap_disk_radius_m", self.disk_radius)
        )
        if (
            not math.isfinite(self.disk_radius)
            or not math.isfinite(self.costmap_disk_radius)
            or self.disk_radius <= 0.0
            or self.costmap_disk_radius < self.disk_radius
        ):
            raise ValueError(
                "costmap disc radius must be finite and no smaller than the "
                "physical disc radius"
            )
        self.input_timeout = float(
            rospy.get_param("~input_timeout_sec", 0.35)
        )
        self.publish_rate = float(
            rospy.get_param("~publish_rate_hz", 10.0)
        )
        self.marker_persistence = max(
            0.1,
            float(rospy.get_param("~marker_persistence_sec", 0.3)),
        )
        self.confirmed_memory_timeout = float(
            rospy.get_param("~confirmed_memory_timeout_sec", 8.0)
        )
        self.unconfirmed_memory_timeout = float(
            rospy.get_param("~unconfirmed_memory_timeout_sec", 0.8)
        )
        self.tombstone_timeout = max(
            0.0, float(rospy.get_param("~passed_tombstone_sec", 20.0))
        )
        self.path_timeout = max(
            0.0, float(rospy.get_param("~global_path_timeout_sec", 5.0))
        )
        self.passage_config = PassageConfig(
            vehicle_half_length_m=float(
                rospy.get_param("~vehicle_half_length_m", 0.165)
            ),
            vehicle_half_width_m=float(
                rospy.get_param("~vehicle_half_width_m", 0.150)
            ),
            disc_radius_m=self.disk_radius,
            progress_margin_m=float(
                rospy.get_param("~pass_progress_margin_m", 0.05)
            ),
            clearance_m=float(
                rospy.get_param("~pass_clearance_m", 0.03)
            ),
            confirm_duration_sec=float(
                rospy.get_param("~pass_confirm_duration_sec", 1.0)
            ),
            confirm_frames=max(
                1, int(rospy.get_param("~pass_confirm_frames", 8))
            ),
            minimum_advance_m=float(
                rospy.get_param("~pass_min_advance_m", 0.10)
            ),
            projection_max_distance_m=float(
                rospy.get_param(
                    "~pass_projection_max_distance_m", 0.35
                )
            ),
            projection_back_segments=int(
                rospy.get_param("~pass_projection_back_segments", 5)
            ),
            projection_forward_segments=int(
                rospy.get_param("~pass_projection_forward_segments", 20)
            ),
            maximum_progress_jump_m=float(
                rospy.get_param("~pass_max_progress_jump_m", 0.10)
            ),
            path_back_length_m=float(
                rospy.get_param("~pass_path_back_length_m", 0.60)
            ),
            path_forward_length_m=float(
                rospy.get_param("~pass_path_forward_length_m", 1.20)
            ),
            path_resample_m=float(
                rospy.get_param("~pass_path_resample_m", 0.03)
            ),
            candidate_backtrack_reset_m=float(
                rospy.get_param(
                    "~pass_candidate_backtrack_reset_m", 0.03
                )
            ),
            tombstone_reactivate_backtrack_m=float(
                rospy.get_param(
                    "~tombstone_reactivate_backtrack_m", 0.20
                )
            ),
        )
        self.memory = StableDiscMemory(
            confirmed_timeout_sec=self.confirmed_memory_timeout,
            unconfirmed_timeout_sec=self.unconfirmed_memory_timeout,
            merge_distance_m=float(
                rospy.get_param("~memory_merge_distance_m", 0.18)
            ),
            smoothing_alpha=float(
                rospy.get_param("~memory_smoothing_alpha", 0.35)
            ),
            tombstone_match_distance_m=float(
                rospy.get_param(
                    "~passed_reacquire_suppression_m", 0.15
                )
            ),
            freeze_confirmed_position=(
                self.lock_confirmed_discs_for_mission
            ),
        )
        if (
            self.angle_increment <= 0.0
            or self.angle_max <= self.angle_min
            or self.range_max <= self.range_min
            or self.reset_state_timeout <= 0.0
            or self.reset_linear_speed_limit <= 0.0
            or self.reset_angular_speed_limit <= 0.0
        ):
            raise ValueError("invalid virtual LaserScan geometry")

        self.bin_count = int(
            math.floor((self.angle_max - self.angle_min) / self.angle_increment)
        ) + 1
        self.latest_receipt_sec = None
        self.latest_raw_count = 0
        self.latest_live_tracks = ()
        self.timestamp_transform_drop_count = 0
        self.previous_marker_ids = set()
        self.latest_global_path = ()
        self.latest_global_path_cumulative = ()
        self.latest_global_path_receipt_sec = None
        self.latest_goal_id = None
        self.navigation_active = False
        self.navigation_gate_receipt_sec = None
        self.navigation_gate_enabled = True
        self.navigation_gate_allowed = True
        self.output_command_receipt_sec = None
        self.output_command = None
        self.odom_receipt_sec = None
        self.odom_linear_speed = math.inf
        self.odom_angular_speed = math.inf
        self.scene_generation = 0
        self.scene_reset_stamp_sec = None
        self.scene_reset_drop_count = 0
        self.scene_reset_count = 0
        self.passed_removal_count = 0
        self.timeout_removal_count = 0
        self.tombstone_reactivation_count = 0
        self.tombstone_suppression_count = 0
        self.tombstone_gc_count = 0
        self.last_removal_reason = None
        self.localization_ready = not self.require_localization_ready
        self.localization_state = (
            "not_required"
            if self.localization_ready
            else "waiting_for_localization"
        )
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.scan_pub = rospy.Publisher(
            self.output_topic, LaserScan, queue_size=1
        )
        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1
        )
        self.status_pub = rospy.Publisher(
            "~status", String, queue_size=1, latch=True
        )
        self.scene_reset_pub = rospy.Publisher(
            self.scene_reset_topic, String, queue_size=1
        )
        self.disk_sub = rospy.Subscriber(
            self.input_topic, String, self._disk_callback, queue_size=1
        )
        self.global_path_sub = rospy.Subscriber(
            self.global_path_topic,
            Path,
            self._global_path_callback,
            queue_size=1,
        )
        self.move_base_status_sub = rospy.Subscriber(
            self.move_base_status_topic,
            GoalStatusArray,
            self._move_base_status_callback,
            queue_size=1,
        )
        self.navigation_gate_status_sub = rospy.Subscriber(
            self.navigation_gate_status_topic,
            String,
            self._navigation_gate_status_callback,
            queue_size=1,
        )
        self.output_command_sub = rospy.Subscriber(
            self.output_command_topic,
            Twist,
            self._output_command_callback,
            queue_size=1,
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self._odom_callback,
            queue_size=1,
        )
        if self.require_localization_ready:
            self.localization_status_sub = rospy.Subscriber(
                self.localization_status_topic,
                String,
                self._localization_status_callback,
                queue_size=1,
            )
        else:
            self.localization_status_sub = None
        self.reset_scene_service = rospy.Service(
            "~reset_scene", Trigger, self._reset_scene_callback
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.publish_rate)),
            self._timer_callback,
        )
        rospy.loginfo(
            "disc costmap bridge ready: %s -> %s and %s frame=%s range=%.2fm "
            "fixed_frame=%s rate=%.1fHz memory=%.1fs path=%s "
            "localization_gate=%s",
            self.input_topic,
            self.output_topic,
            self.marker_topic,
            self.frame_id,
            self.range_max,
            self.fixed_frame,
            self.publish_rate,
            self.confirmed_memory_timeout,
            self.global_path_topic,
            self.require_localization_ready,
        )

    @staticmethod
    def _transform_xy(transform, point):
        rotation = transform.transform.rotation
        translation = transform.transform.translation
        sin_yaw = 2.0 * (
            rotation.w * rotation.z + rotation.x * rotation.y
        )
        cos_yaw = 1.0 - 2.0 * (
            rotation.y * rotation.y + rotation.z * rotation.z
        )
        yaw = math.atan2(sin_yaw, cos_yaw)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return (
            float(translation.x) + cosine * point[0] - sine * point[1],
            float(translation.y) + sine * point[0] + cosine * point[1],
        )

    @staticmethod
    def _transform_yaw(transform):
        rotation = transform.transform.rotation
        return math.atan2(
            2.0 * (
                rotation.w * rotation.z
                + rotation.x * rotation.y
            ),
            1.0
            - 2.0
            * (
                rotation.y * rotation.y
                + rotation.z * rotation.z
            ),
        )

    @classmethod
    def _inverse_transform_xy(cls, transform, point):
        """Apply the inverse of a TransformStamped to a planar point."""
        yaw = cls._transform_yaw(transform)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        translation = transform.transform.translation
        delta_x = float(point[0]) - float(translation.x)
        delta_y = float(point[1]) - float(translation.y)
        return (
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
        )

    def _lookup_transform(
        self, target_frame, source_frame, stamp, allow_latest=True
    ):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                rospy.Duration(0.05),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            if stamp == rospy.Time(0) or not allow_latest:
                raise
            rospy.logwarn_throttle(
                2.0,
                "disc timestamp transform unavailable; using latest %s -> %s",
                source_frame,
                target_frame,
            )
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0),
                rospy.Duration(0.05),
            )

    def _invalidate_global_path(self):
        with self.lock:
            self.latest_global_path = ()
            self.latest_global_path_cumulative = ()
            self.latest_global_path_receipt_sec = None

    def _global_path_callback(self, message):
        source_frame = str(message.header.frame_id).lstrip("/")
        if not source_frame and message.poses:
            source_frame = str(
                message.poses[0].header.frame_id
            ).lstrip("/")
        if not source_frame:
            rospy.logwarn_throttle(
                1.0, "global path has no coordinate frame"
            )
            self._invalidate_global_path()
            return

        pose_frames = {
            str(pose.header.frame_id).lstrip("/")
            for pose in message.poses
            if pose.header.frame_id
        }
        if any(frame != source_frame for frame in pose_frames):
            rospy.logwarn_throttle(
                1.0, "global path contains mixed coordinate frames"
            )
            self._invalidate_global_path()
            return

        transform = None
        if source_frame != self.fixed_frame:
            try:
                transform = self._lookup_transform(
                    self.fixed_frame,
                    source_frame,
                    message.header.stamp,
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                rospy.logwarn_throttle(
                    1.0,
                    "cannot transform global path %s -> %s: %s",
                    source_frame,
                    self.fixed_frame,
                    exc,
                )
                self._invalidate_global_path()
                return

        points = []
        for pose in message.poses:
            point = (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
            )
            if transform is not None:
                point = self._transform_xy(transform, point)
            if not all(math.isfinite(value) for value in point):
                continue
            if points and math.hypot(
                point[0] - points[-1][0],
                point[1] - points[-1][1],
            ) <= 1e-5:
                continue
            points.append(point)

        if len(points) < 2:
            rospy.logwarn_throttle(
                1.0, "global path has fewer than two distinct points"
            )
            self._invalidate_global_path()
            return
        path = tuple(points)
        receipt_sec = rospy.Time.now().to_sec()
        with self.lock:
            self.latest_global_path = path
            self.latest_global_path_cumulative = cumulative_lengths(path)
            self.latest_global_path_receipt_sec = receipt_sec

    def _move_base_status_callback(self, message):
        active_statuses = {
            GoalStatus.PENDING,
            GoalStatus.ACTIVE,
            GoalStatus.PREEMPTING,
            GoalStatus.RECALLING,
        }
        candidates = [
            (status.goal_id.stamp.to_sec(), index, status)
            for index, status in enumerate(message.status_list)
            if (
                status.status in active_statuses
                and status.goal_id.id
            )
        ]
        selected = max(candidates, default=None, key=lambda item: item[:2])
        goal_id = None if selected is None else selected[2].goal_id.id
        new_epoch = None
        with self.lock:
            if goal_id is not None and goal_id != self.latest_goal_id:
                self.latest_goal_id = goal_id
                new_epoch = self.memory.begin_navigation_epoch()
                # Do not let a new goal inherit the tail of the preceding
                # goal's route while waiting for SBPL's first fresh plan.
                self.latest_global_path = ()
                self.latest_global_path_cumulative = ()
                self.latest_global_path_receipt_sec = None
            self.navigation_active = goal_id is not None
        if new_epoch is not None:
            rospy.loginfo(
                "new navigation epoch=%d goal_id=%s",
                new_epoch,
                goal_id,
            )

    def _navigation_gate_status_callback(self, message):
        try:
            payload = json.loads(message.data)
            enabled = bool(payload["enabled"])
            allowed = bool(payload["allowed"])
        except (KeyError, TypeError, ValueError):
            rospy.logwarn_throttle(
                1.0, "invalid navigation-gate status for scene reset"
            )
            return
        with self.lock:
            self.navigation_gate_receipt_sec = rospy.Time.now().to_sec()
            self.navigation_gate_enabled = enabled
            self.navigation_gate_allowed = allowed

    def _output_command_callback(self, message):
        with self.lock:
            self.output_command_receipt_sec = rospy.Time.now().to_sec()
            self.output_command = (
                float(message.linear.x),
                float(message.linear.y),
                float(message.angular.z),
            )

    def _odom_callback(self, message):
        linear = message.twist.twist.linear
        angular = message.twist.twist.angular
        with self.lock:
            self.odom_receipt_sec = rospy.Time.now().to_sec()
            self.odom_linear_speed = math.sqrt(
                float(linear.x) ** 2
                + float(linear.y) ** 2
                + float(linear.z) ** 2
            )
            self.odom_angular_speed = math.sqrt(
                float(angular.x) ** 2
                + float(angular.y) ** 2
                + float(angular.z) ** 2
            )

    def _scene_reset_block_reason(self, now_sec):
        state_stamps = {
            "output_command_stale": self.output_command_receipt_sec,
            "odom_stale": self.odom_receipt_sec,
        }
        if self.require_navigation_gate_for_scene_reset:
            state_stamps["navigation_gate_status_stale"] = (
                self.navigation_gate_receipt_sec
            )
        for reason, stamp_sec in state_stamps.items():
            if (
                stamp_sec is None
                or float(now_sec) - float(stamp_sec)
                > self.reset_state_timeout
            ):
                return reason
        if (
            self.require_navigation_gate_for_scene_reset
            and (self.navigation_gate_enabled or self.navigation_gate_allowed)
        ):
            return "navigation_gate_not_locked"
        if self.navigation_active:
            return "navigation_goal_active"
        if self.output_command is None:
            return "output_command_unavailable"
        command_linear = math.hypot(
            self.output_command[0], self.output_command[1]
        )
        if command_linear > self.reset_linear_speed_limit:
            return "output_command_linear_nonzero"
        if abs(self.output_command[2]) > self.reset_angular_speed_limit:
            return "output_command_angular_nonzero"
        if self.odom_linear_speed > self.reset_linear_speed_limit:
            return "odom_linear_speed_nonzero"
        if self.odom_angular_speed > self.reset_angular_speed_limit:
            return "odom_angular_speed_nonzero"
        return ""

    def _reset_scene_callback(self, _request):
        now_sec = rospy.Time.now().to_sec()
        with self.lock:
            block_reason = self._scene_reset_block_reason(now_sec)
            if block_reason:
                return TriggerResponse(
                    False, "scene reset rejected: " + block_reason
                )
            self.scene_generation += 1
            self.scene_reset_count += 1
            self.scene_reset_stamp_sec = now_sec
            self.memory.clear(reset_identity=True)
            self.latest_live_tracks = ()
            self.latest_raw_count = 0
            self.latest_global_path = ()
            self.latest_global_path_cumulative = ()
            self.latest_global_path_receipt_sec = None
            self.latest_goal_id = None
            self.last_removal_reason = "manual_scene_reset"
            generation = self.scene_generation
        reset_event = {
            "stamp": now_sec,
            "scene_generation": generation,
            "source": rospy.get_name(),
            "reason": "manual_scene_reset",
        }
        self.scene_reset_pub.publish(
            String(data=json.dumps(reset_event, allow_nan=False))
        )
        rospy.logwarn(
            "cleared disc scene memory at generation=%d while locked/stationary",
            generation,
        )
        return TriggerResponse(
            True, "disc scene reset; generation={}".format(generation)
        )

    def _localization_status_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(
                1.0, "invalid field-localization status JSON"
            )
            return
        state = str(payload.get("state", "unknown"))
        ready = state == "localized"
        with self.lock:
            changed = ready != self.localization_ready
            self.localization_ready = ready
            self.localization_state = state
            if changed:
                # Never carry observations across a localization reset. Fresh
                # detections repopulate the fixed-frame memory immediately.
                self.memory.clear()
                self.latest_global_path = ()
                self.latest_global_path_cumulative = ()
                self.latest_global_path_receipt_sec = None
        if changed:
            if ready:
                rospy.loginfo(
                    "field localization is ready; enabling disc costmap writes"
                )
            else:
                rospy.logwarn(
                    "field localization state is %s; suppressing disc "
                    "costmap writes",
                    state,
                )

    def _disk_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid disc JSON for costmap bridge")
            return

        raw_items = list(payload.get("raw", []))
        confirmed_ids = {
            int(item["track_id"])
            for item in payload.get("confirmed_tracks", [])
            if item.get("track_id") is not None
        }
        try:
            payload_stamp_sec = float(
                payload.get("stamp", rospy.Time.now().to_sec())
            )
            with self.lock:
                if (
                    self.scene_reset_stamp_sec is not None
                    and payload_stamp_sec <= self.scene_reset_stamp_sec
                ):
                    self.scene_reset_drop_count += 1
                    return
            fixed_from_base = self._lookup_transform(
                self.fixed_frame,
                self.frame_id,
                rospy.Time.from_sec(payload_stamp_sec),
                allow_latest=False,
            )
        except (
            TypeError,
            ValueError,
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                1.0,
                "cannot stabilize disc observations in %s: %s",
                self.fixed_frame,
                exc,
            )
            with self.lock:
                self.latest_receipt_sec = rospy.Time.now().to_sec()
                self.latest_raw_count = 0
                self.latest_live_tracks = ()
                self.timestamp_transform_drop_count += 1
            return

        observations = []
        seen_tracks = set()
        # Only current-frame detections update the fixed location. Reusing a
        # smoothed track while the vehicle moves would drag the obstacle along
        # with the vehicle and create the exact costmap trail we want to avoid.
        for item in raw_items:
            track_id = item.get("track_id")
            if track_id is not None and track_id in seen_tracks:
                continue
            center_base = item.get("center_base_m")
            center = (
                center_base
                if isinstance(center_base, list) and len(center_base) >= 2
                else item.get("center_m")
            )
            if not isinstance(center, list) or len(center) < 2:
                continue
            try:
                forward = float(center[0])
                left = float(center[1])
                if center is not center_base:
                    forward += self.forward_offset
            except (TypeError, ValueError):
                continue
            if np.isfinite(forward) and np.isfinite(left):
                observations.append(
                    (
                        track_id,
                        self._transform_xy(
                            fixed_from_base, (forward, left)
                        ),
                        (forward, left),
                        float(item.get("stamp", payload_stamp_sec)),
                        bool(item.get("confirmed", False))
                        or (
                            track_id is not None
                            and int(track_id) in confirmed_ids
                        ),
                        (
                            None
                            if item.get("estimated_diameter_m") is None
                            else float(item["estimated_diameter_m"])
                        ),
                        (
                            None
                            if item.get("confidence") is None
                            else float(item["confidence"])
                        ),
                    )
                )
                if track_id is not None:
                    seen_tracks.add(track_id)
        with self.lock:
            if (
                self.scene_reset_stamp_sec is not None
                and payload_stamp_sec <= self.scene_reset_stamp_sec
            ):
                self.scene_reset_drop_count += 1
                return
            self.latest_receipt_sec = rospy.Time.now().to_sec()
            self.latest_raw_count = len(observations)
            live_tracks = []
            for (
                track_id,
                fixed_xy,
                base_xy,
                stamp_sec,
                confirmed,
                estimated_diameter_m,
                confidence,
            ) in observations:
                entry = self.memory.observe(
                    track_id,
                    fixed_xy,
                    stamp_sec,
                    confirmed,
                    estimated_diameter_m=estimated_diameter_m,
                    confidence=confidence,
                )
                live_tracks.append(
                    {
                        "memory_id": int(entry.memory_id),
                        "fixed_xy": (
                            float(fixed_xy[0]), float(fixed_xy[1])
                        ),
                        "base_xy": (
                            float(base_xy[0]), float(base_xy[1])
                        ),
                        "stamp_sec": float(stamp_sec),
                        "confirmed": bool(entry.confirmed),
                        "estimated_diameter_m": (
                            entry.estimated_diameter_m
                        ),
                        "confidence": entry.confidence,
                        "phase": str(entry.passage.phase),
                    }
                )
                if entry.passage.phase != PHASE_TOMBSTONE:
                    continue
                robot_fixed_xy = self._transform_xy(
                    fixed_from_base, (0.0, 0.0)
                )
                if tombstone_should_reactivate(
                    entry.passage,
                    robot_fixed_xy,
                    self.passage_config,
                ):
                    if self.memory.reactivate_tombstone(entry.memory_id):
                        self.tombstone_reactivation_count += 1
                        rospy.loginfo(
                            "reactivated disc memory id=%d after upstream "
                            "return",
                            entry.memory_id,
                        )
                else:
                    self.tombstone_suppression_count += 1
            self.latest_live_tracks = tuple(live_tracks)

    def _current_centers(self, now):
        try:
            fixed_from_base = self._lookup_transform(
                self.fixed_frame, self.frame_id, rospy.Time(0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn_throttle(
                1.0,
                "cannot read fixed disc memory in %s: %s",
                self.fixed_frame,
                exc,
            )
            for entry in self.memory.entries():
                if entry.passage.phase == PHASE_PASSED_CANDIDATE:
                    reset_candidate(entry.passage)
            # Do not publish an all-clear scan when the fixed memory cannot
            # be transformed. Skipping this cycle lets costmap_2d retain its
            # last obstacle marks through a transient TF outage.
            return None
        now_sec = now.to_sec()
        robot_fixed_xy = self._transform_xy(
            fixed_from_base, (0.0, 0.0)
        )
        robot_yaw = self._transform_yaw(fixed_from_base)
        path_fresh = (
            self.navigation_active
            and self.latest_global_path_receipt_sec is not None
            and now_sec - self.latest_global_path_receipt_sec
            <= self.path_timeout
            and len(self.latest_global_path) >= 2
        )
        centers = []
        removals = []
        for entry in self.memory.entries():
            forward, left = self._inverse_transform_xy(
                fixed_from_base, entry.fixed_xy
            )

            if self.lock_confirmed_discs_for_mission:
                if not entry.confirmed:
                    if self.memory.is_stale(entry, now_sec):
                        removals.append(
                            (entry.memory_id, "unconfirmed_timeout")
                        )
                    # A single-frame candidate remains diagnostic-only.  It
                    # cannot mark either costmap until the existing tracker
                    # has confirmed it over multiple frames.
                    continue
                centers.append(
                    {
                        "memory_id": entry.memory_id,
                        "forward": forward,
                        "left": left,
                        "fixed_x": float(entry.fixed_xy[0]),
                        "fixed_y": float(entry.fixed_xy[1]),
                        "confirmed": True,
                        "last_seen_age_sec": max(
                            0.0, now_sec - float(entry.last_seen_sec)
                        ),
                        "estimated_diameter_m": (
                            entry.estimated_diameter_m
                        ),
                        "confidence": entry.confidence,
                        "phase": entry.passage.phase,
                    }
                )
                # Formal competition uses a one-pass, reset-scoped world
                # model.  Do not age, pass-confirm or tombstone a locked disc;
                # visual loss therefore cannot remove it from DWA.
                continue

            if self.competition_two_disc_mode:
                if self.memory.is_stale(entry, now_sec):
                    removals.append((entry.memory_id, "competition_timeout"))
                    continue
                centers.append(
                    {
                        "memory_id": entry.memory_id,
                        "forward": forward,
                        "left": left,
                        "fixed_x": float(entry.fixed_xy[0]),
                        "fixed_y": float(entry.fixed_xy[1]),
                        "confirmed": bool(entry.confirmed),
                        "last_seen_age_sec": max(
                            0.0, now_sec - float(entry.last_seen_sec)
                        ),
                        "estimated_diameter_m": (
                            entry.estimated_diameter_m
                        ),
                        "confidence": entry.confidence,
                        "phase": entry.passage.phase,
                    }
                )
                continue

            if entry.passage.phase == PHASE_TOMBSTONE:
                can_reactivate = (
                    self.navigation_active
                    and entry.passage.navigation_epoch
                    == self.memory.navigation_epoch
                    and tombstone_should_reactivate(
                        entry.passage,
                        robot_fixed_xy,
                        self.passage_config,
                    )
                )
                if not can_reactivate:
                    continue
                if self.memory.reactivate_tombstone(entry.memory_id):
                    self.tombstone_reactivation_count += 1
                    rospy.loginfo(
                        "reactivated disc memory id=%d after upstream return",
                        entry.memory_id,
                    )

            reason = self.memory.stale_removal_reason(
                entry=entry,
                now_sec=now_sec,
                distance_m=math.hypot(forward, left),
                active_range_m=self.range_max + self.costmap_disk_radius,
            )
            if reason is not None:
                removals.append((entry.memory_id, reason))
                continue

            if (
                not path_fresh
                and entry.passage.phase == PHASE_PASSED_CANDIDATE
            ):
                # Candidate confirmation must be continuous. Losing the
                # active goal or a fresh SBPL path invalidates the hold period
                # rather than merely pausing its wall-clock timer.
                reset_candidate(entry.passage)

            if path_fresh and entry.confirmed:
                passage_update = update_passage(
                    state=entry.passage,
                    now_sec=now_sec,
                    disc_fixed_xy=entry.fixed_xy,
                    disc_base_xy=(forward, left),
                    robot_fixed_xy=robot_fixed_xy,
                    robot_yaw=robot_yaw,
                    latest_path=self.latest_global_path,
                    latest_cumulative_s=(
                        self.latest_global_path_cumulative
                    ),
                    config=self.passage_config,
                )
                if passage_update.entered_tombstone:
                    self.passed_removal_count += 1
                    self.last_removal_reason = "passed_tombstone"
                    rospy.loginfo(
                        "disc memory id=%d entered tombstone: %s",
                        entry.memory_id,
                        json.dumps(entry.passage.debug, sort_keys=True),
                    )
                    continue

            centers.append(
                {
                    "memory_id": entry.memory_id,
                    "forward": forward,
                    "left": left,
                    "fixed_x": float(entry.fixed_xy[0]),
                    "fixed_y": float(entry.fixed_xy[1]),
                    "confirmed": bool(entry.confirmed),
                    "last_seen_age_sec": max(
                        0.0, now_sec - float(entry.last_seen_sec)
                    ),
                    "estimated_diameter_m": entry.estimated_diameter_m,
                    "confidence": entry.confidence,
                    "phase": entry.passage.phase,
                }
            )
        for memory_id, reason in removals:
            self.memory.remove(memory_id)
            self.last_removal_reason = reason
            self.timeout_removal_count += 1
            rospy.loginfo(
                "removed disc memory id=%d reason=%s",
                memory_id,
                reason,
            )
        garbage_collected = (
            []
            if self.competition_two_disc_mode
            else self.memory.garbage_collect_tombstones(
                now_sec=now_sec,
                navigation_active=self.navigation_active,
                timeout_sec=self.tombstone_timeout,
            )
        )
        if garbage_collected:
            self.tombstone_gc_count += len(garbage_collected)
            self.last_removal_reason = "tombstone_gc"
            rospy.loginfo(
                "garbage-collected disc tombstones ids=%s",
                garbage_collected,
            )
        return centers

    def _scan(self, now, centers):
        message = LaserScan()
        message.header.stamp = now
        message.header.frame_id = self.frame_id
        message.angle_min = self.angle_min
        message.angle_max = self.angle_min + (
            self.bin_count - 1
        ) * self.angle_increment
        message.angle_increment = self.angle_increment
        message.time_increment = 0.0
        message.scan_time = 1.0 / max(1.0, self.publish_rate)
        message.range_min = self.range_min
        message.range_max = self.range_max
        ranges = np.full(self.bin_count, np.inf, dtype=np.float64)
        ray_angles = (
            self.angle_min
            + np.arange(self.bin_count, dtype=np.float64)
            * self.angle_increment
        )
        accepted = 0
        for center in centers:
            forward = center["forward"]
            left = center["left"]
            distance = math.hypot(forward, left)
            angle = math.atan2(left, forward)
            if (
                not math.isfinite(distance)
                or distance + self.costmap_disk_radius < self.range_min
                or distance - self.costmap_disk_radius > self.range_max
            ):
                continue
            # Wrap angular differences so a disc centered at +pi also marks
            # rays near -pi instead of being cut in half at the scan seam.
            delta = np.arctan2(
                np.sin(ray_angles - angle),
                np.cos(ray_angles - angle),
            )
            perpendicular = distance * np.sin(delta)
            along = distance * np.cos(delta)
            intersects = (
                (np.abs(perpendicular) <= self.costmap_disk_radius)
                & (along >= 0.0)
            )
            intersection_ranges = np.full(
                self.bin_count, np.inf, dtype=np.float64
            )
            intersection_ranges[intersects] = (
                along[intersects]
                - np.sqrt(
                    np.maximum(
                        self.costmap_disk_radius * self.costmap_disk_radius
                        - perpendicular[intersects] ** 2,
                        0.0,
                    )
                )
            )
            intersection_ranges = np.maximum(
                intersection_ranges, self.range_min
            )
            valid = intersection_ranges <= self.range_max
            if not np.any(valid):
                continue
            ranges[valid] = np.minimum(
                ranges[valid], intersection_ranges[valid]
            )
            accepted += 1
        message.ranges = ranges.tolist()
        message.intensities = []
        return message, accepted

    def _markers(self, now, centers):
        message = MarkerArray()
        visible_centers = [
            center
            for center in centers
            if (
                np.isfinite(center["forward"])
                and np.isfinite(center["left"])
            )
        ]
        lifetime = rospy.Duration(
            max(
                self.marker_persistence,
                2.0 / max(1.0, self.publish_rate),
            )
        )
        current_ids = set()
        for disc in visible_centers:
            memory_id = int(disc["memory_id"])
            forward = disc["forward"]
            left = disc["left"]
            candidate = disc["phase"] == PHASE_PASSED_CANDIDATE
            current_ids.add(memory_id)
            outer = Marker()
            outer.header.stamp = now
            outer.header.frame_id = self.frame_id
            outer.ns = "yellow_track_disc_outer"
            outer.id = memory_id
            outer.type = Marker.CYLINDER
            outer.action = Marker.ADD
            outer.pose.position.x = forward
            outer.pose.position.y = left
            outer.pose.position.z = 0.012
            outer.pose.orientation.w = 1.0
            outer.scale.x = 2.0 * self.disk_radius
            outer.scale.y = 2.0 * self.disk_radius
            outer.scale.z = 0.024
            outer.color.r = 1.0 if candidate else 0.90
            outer.color.g = 0.55 if candidate else 0.05
            outer.color.b = 0.0 if candidate else 0.03
            outer.color.a = 0.95
            outer.lifetime = lifetime
            message.markers.append(outer)

            center = Marker()
            center.header.stamp = now
            center.header.frame_id = self.frame_id
            center.ns = "yellow_track_disc_center"
            center.id = memory_id
            center.type = Marker.CYLINDER
            center.action = Marker.ADD
            center.pose.position.x = forward
            center.pose.position.y = left
            center.pose.position.z = 0.026
            center.pose.orientation.w = 1.0
            center.scale.x = 0.45 * 2.0 * self.disk_radius
            center.scale.y = 0.45 * 2.0 * self.disk_radius
            center.scale.z = 0.006
            center.color.r = 1.0
            center.color.g = 1.0
            center.color.b = 1.0
            center.color.a = 1.0
            center.lifetime = lifetime
            message.markers.append(center)
        # MarkerArray does not implicitly remove IDs absent from a later
        # message. Explicitly delete surplus IDs so a short-lived false
        # detection cannot remain in RViz as an apparent extra disc.
        for memory_id in self.previous_marker_ids - current_ids:
            for namespace in (
                "yellow_track_disc_outer",
                "yellow_track_disc_center",
            ):
                stale = Marker()
                stale.header.stamp = now
                stale.header.frame_id = self.frame_id
                stale.ns = namespace
                stale.id = int(memory_id)
                stale.action = Marker.DELETE
                message.markers.append(stale)
        self.previous_marker_ids = current_ids
        return message

    def _timer_callback(self, _event):
        with self.lock:
            now = rospy.Time.now()
            centers = self._current_centers(now)
            input_fresh = (
                self.latest_receipt_sec is not None
                and now.to_sec() - self.latest_receipt_sec
                <= self.input_timeout
            )
            costmap_writes_enabled = (
                self.enable_costmap_writes and self.localization_ready
            )
            transform_available = centers is not None
            count = 0
            if transform_available:
                scan, count = self._scan(now, centers)
                if not costmap_writes_enabled:
                    # Continue publishing an all-clear scan so costmap_2d has
                    # a fresh observation, but never mark a disc using AMCL's
                    # provisional startup transform. Detection markers remain
                    # visible for commissioning.
                    scan.ranges = [math.inf] * self.bin_count
                self.scan_pub.publish(scan)
                self.marker_pub.publish(self._markers(now, centers))
            path_age_sec = (
                None
                if self.latest_global_path_receipt_sec is None
                else max(
                    0.0,
                    now.to_sec() - self.latest_global_path_receipt_sec,
                )
            )
            path_fresh = (
                self.navigation_active
                and path_age_sec is not None
                and path_age_sec <= self.path_timeout
                and len(self.latest_global_path) >= 2
            )
            memory_entries = self.memory.entries()
            passage_states = []
            for entry in memory_entries:
                passage = {
                    "memory_id": entry.memory_id,
                    "phase": entry.passage.phase,
                    "confirmed": entry.confirmed,
                    "navigation_epoch": entry.passage.navigation_epoch,
                }
                passage.update(entry.passage.debug)
                passage_states.append(passage)
            status = {
                "stamp": now.to_sec(),
                "input_fresh": input_fresh,
                "latest_raw_count": self.latest_raw_count,
                "timestamp_transform_drop_count": (
                    self.timestamp_transform_drop_count
                ),
                "disc_count": count,
                "costmap_disc_count": (
                    count if costmap_writes_enabled else 0
                ),
                "costmap_writes_enabled": costmap_writes_enabled,
                "costmap_writes_configured": self.enable_costmap_writes,
                "fixed_transform_available": transform_available,
                "localization_required": self.require_localization_ready,
                "localization_state": self.localization_state,
                "frame_id": self.frame_id,
                "fixed_frame": self.fixed_frame,
                "memory_mode": (
                    "mission_locked_confirmed_until_reset"
                    if self.lock_confirmed_discs_for_mission
                    else (
                        "competition_two_disc_{:g}s".format(
                            self.confirmed_memory_timeout
                        )
                        if self.competition_two_disc_mode
                        else (
                            "path_progress_clearance_confirmation_tombstone"
                        )
                    )
                ),
                "lock_confirmed_discs_for_mission": (
                    self.lock_confirmed_discs_for_mission
                ),
                "confirmed_positions_frozen": (
                    self.lock_confirmed_discs_for_mission
                ),
                "costmap_requires_confirmed": (
                    self.lock_confirmed_discs_for_mission
                ),
                "memory_count": len(self.memory),
                "active_memory_count": len(
                    self.memory.active_entries()
                ),
                "tombstone_count": len(
                    self.memory.tombstone_entries()
                ),
                "passed_candidate_count": sum(
                    entry.passage.phase == PHASE_PASSED_CANDIDATE
                    for entry in memory_entries
                ),
                "passed_removal_count": self.passed_removal_count,
                "timeout_removal_count": self.timeout_removal_count,
                "tombstone_gc_count": self.tombstone_gc_count,
                "tombstone_reactivation_count": (
                    self.tombstone_reactivation_count
                ),
                "tombstone_suppression_count": (
                    self.tombstone_suppression_count
                ),
                "last_removal_reason": self.last_removal_reason,
                "confirmed_memory_timeout_sec": (
                    self.confirmed_memory_timeout
                ),
                "confirmed_timeout_scope": (
                    "disabled_until_scene_reset"
                    if self.lock_confirmed_discs_for_mission
                    else (
                        "all_tracks_competition_mode"
                        if self.competition_two_disc_mode
                        else "outside_active_scan_range_only"
                    )
                ),
                "unconfirmed_memory_timeout_sec": (
                    self.unconfirmed_memory_timeout
                ),
                "navigation_active": self.navigation_active,
                "navigation_epoch": self.memory.navigation_epoch,
                "navigation_goal_id": self.latest_goal_id,
                "scene_generation": self.scene_generation,
                "scene_reset_count": self.scene_reset_count,
                "scene_reset_stamp_sec": self.scene_reset_stamp_sec,
                "scene_reset_drop_count": self.scene_reset_drop_count,
                "scene_reset_safety": {
                    "require_navigation_gate": (
                        self.require_navigation_gate_for_scene_reset
                    ),
                    "navigation_gate_enabled": (
                        self.navigation_gate_enabled
                    ),
                    "navigation_gate_allowed": (
                        self.navigation_gate_allowed
                    ),
                    "navigation_active": self.navigation_active,
                    "output_command": self.output_command,
                    "odom_linear_speed_mps": self.odom_linear_speed,
                    "odom_angular_speed_radps": (
                        self.odom_angular_speed
                    ),
                },
                "global_path_topic": self.global_path_topic,
                "global_path_points": len(self.latest_global_path),
                "global_path_age_sec": path_age_sec,
                "global_path_fresh": path_fresh,
                "global_path_timeout_sec": self.path_timeout,
                # The passage supervisor consumes the same world-fixed tracks
                # used to draw the costmap scan. Publishing these coordinates
                # avoids a second, inconsistent perception memory or another
                # application of the camera-origin offset.
                "active_tracks": [
                    {
                        "memory_id": int(center["memory_id"]),
                        "fixed_xy": [
                            float(center["fixed_x"]),
                            float(center["fixed_y"]),
                        ],
                        "base_xy": [
                            float(center["forward"]),
                            float(center["left"]),
                        ],
                        "confirmed": bool(center["confirmed"]),
                        "estimated_diameter_m": center.get(
                            "estimated_diameter_m"
                        ),
                        "confidence": center.get("confidence"),
                        "phase": str(center["phase"]),
                        "last_seen_age_sec": float(
                            center["last_seen_age_sec"]
                        ),
                    }
                    for center in (centers or [])
                ],
                "live_tracks": [
                    {
                        "memory_id": int(track["memory_id"]),
                        "fixed_xy": [
                            float(track["fixed_xy"][0]),
                            float(track["fixed_xy"][1]),
                        ],
                        "base_xy": [
                            float(track["base_xy"][0]),
                            float(track["base_xy"][1]),
                        ],
                        "confirmed": bool(track["confirmed"]),
                        "estimated_diameter_m": track.get(
                            "estimated_diameter_m"
                        ),
                        "confidence": track.get("confidence"),
                        "phase": str(track["phase"]),
                        "last_seen_age_sec": max(
                            0.0,
                            now.to_sec() - float(track["stamp_sec"]),
                        ),
                    }
                    for track in self.latest_live_tracks
                ],
                "pass_progress_margin_m": (
                    self.passage_config.progress_margin_m
                ),
                "pass_clearance_m": self.passage_config.clearance_m,
                "physical_disc_radius_m": self.disk_radius,
                "costmap_disc_radius_m": self.costmap_disk_radius,
                "costmap_disc_hard_clearance_m": (
                    self.costmap_disk_radius - self.disk_radius
                ),
                "pass_confirm_duration_sec": (
                    self.passage_config.confirm_duration_sec
                ),
                "pass_confirm_frames": (
                    self.passage_config.confirm_frames
                ),
                "pass_min_advance_m": (
                    self.passage_config.minimum_advance_m
                ),
                "passed_tombstone_sec": self.tombstone_timeout,
                "passage_states": passage_states,
                "publish_rate_hz": self.publish_rate,
                "scan_angle_min_rad": self.angle_min,
                "scan_angle_max_rad": self.angle_max,
                "scan_topic": self.output_topic,
                "marker_topic": self.marker_topic,
                "scene_reset_topic": self.scene_reset_topic,
                "marker_persistence_sec": self.marker_persistence,
            }
            self.status_pub.publish(
                String(data=json.dumps(status, ensure_ascii=False))
            )


if __name__ == "__main__":
    rospy.init_node("yellow_track_disc_laserscan_bridge")
    DiscLaserScanBridge()
    rospy.spin()
