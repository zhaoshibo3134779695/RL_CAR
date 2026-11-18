#!/usr/bin/env python3
"""Capture, freeze, preview and explicitly send the Stage-1 lidar goal."""

import json
import math
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

from yellow_track_avoidance.stage1_lidar_goal import (
    build_stage1_geometry,
    fit_right_wall,
    voxel_downsample,
)


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _transform_xy(transform, point):
    angle = _yaw(transform.transform.rotation)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        (
            transform.transform.translation.x
            + cosine * point[0]
            - sine * point[1],
            transform.transform.translation.y
            + sine * point[0]
            + cosine * point[1],
        ),
        dtype=np.float64,
    )


class Stage1LidarGoalNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.scan_topic = str(rospy.get_param("~scan_topic", "/scan"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link")).lstrip("/")
        self.target_frame = str(rospy.get_param("~target_frame", "map")).lstrip("/")
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.goal_topic = str(rospy.get_param("~goal_topic", "/move_base_simple/goal"))
        self.marker_topic = str(
            rospy.get_param(
                "~marker_topic", "/yellow_track_avoidance/stage1_lidar_goal_markers"
            )
        )
        self.goal_pose_topic = str(
            rospy.get_param(
                "~goal_pose_topic", "/yellow_track_avoidance/stage1_lidar_goal"
            )
        )
        self.preview_topic = str(
            rospy.get_param(
                "~preview_plan_topic", "/yellow_track_avoidance/stage1_lidar_goal_plan"
            )
        )
        self.corridor_status_topic = str(
            rospy.get_param(
                "~corridor_status_topic", "/yellow_track_competition_corridor_map/status"
            )
        )
        self.disc_status_topic = str(
            rospy.get_param(
                "~disc_status_topic", "/yellow_track_disc_costmap_bridge/status"
            )
        )
        self.perception_status_topic = str(
            rospy.get_param(
                "~perception_status_topic", "/yellow_track_avoidance/debug/status"
            )
        )
        self.compact_status_topic = str(
            rospy.get_param(
                "~compact_status_topic",
                "/yellow_track_teb_compact_obstacles/status",
            )
        )
        self.require_passage_closures = bool(
            rospy.get_param("~require_passage_closures", False)
        )
        self.scene_reset_topic = str(
            rospy.get_param("~scene_reset_topic", "/yellow_track_avoidance/reset_scene")
        )
        self.make_plan_service = str(
            rospy.get_param("~make_plan_service", "/move_base/make_plan")
        )
        self.scan_frame_count = max(1, int(rospy.get_param("~scan_frame_count", 60)))
        self.capture_retry_overlap_frames = max(
            0,
            min(
                self.scan_frame_count - 1,
                int(rospy.get_param("~capture_retry_overlap_frames", 0)),
            ),
        )
        self.heading_scan_frame_count = max(
            1,
            int(
                rospy.get_param(
                    "~heading_scan_frame_count", self.scan_frame_count
                )
            ),
        )
        self.heading_topic = str(
            rospy.get_param(
                "~heading_topic",
                "/yellow_track_avoidance/stage1_lidar_heading",
            )
        )
        self.heading_maximum_rms_m = float(
            rospy.get_param("~heading_maximum_rms_m", 0.020)
        )
        self.heading_minimum_inliers = int(
            rospy.get_param("~heading_minimum_inliers", 30)
        )
        self.heading_minimum_span_m = float(
            rospy.get_param("~heading_minimum_span_m", 1.0)
        )
        legacy_disc_count = int(rospy.get_param("~expected_disc_count", 2))
        self.minimum_disc_count = max(
            1, int(rospy.get_param("~minimum_disc_count", legacy_disc_count))
        )
        self.expected_width = float(rospy.get_param("~expected_width_m", 0.485))
        self.width_tolerance = float(rospy.get_param("~width_tolerance_m", 0.08))
        self.vehicle_front_offset = float(
            rospy.get_param("~vehicle_front_offset_m", 0.17)
        )
        self.front_clearance = float(
            rospy.get_param("~desired_front_clearance_m", 0.10)
        )
        self.voxel_resolution = float(
            rospy.get_param("~voxel_resolution_m", 0.015)
        )
        self.world_stable_sec = float(rospy.get_param("~world_stable_sec", 0.5))
        self.status_timeout = float(rospy.get_param("~status_timeout_sec", 2.5))
        self.auto_capture = bool(rospy.get_param("~auto_capture_when_ready", True))
        # Keep false during staged commissioning: sending an active move_base
        # goal before the operator unlocks would trip its 6 s oscillation timer.
        self.auto_send = bool(rospy.get_param("~auto_send_goal_when_ready", False))
        self.maximum_linear_speed = float(
            rospy.get_param("~maximum_capture_linear_speed_mps", 0.01)
        )
        self.maximum_angular_speed = float(
            rospy.get_param("~maximum_capture_angular_speed_radps", 0.03)
        )
        self.fit_options = {
            "minimum_x_m": float(rospy.get_param("~right_wall_minimum_x_m", 0.2)),
            "maximum_x_m": float(rospy.get_param("~right_wall_maximum_x_m", 5.8)),
            "maximum_y_m": float(rospy.get_param("~right_wall_maximum_y_m", -0.12)),
            "ransac_threshold_m": float(
                rospy.get_param("~right_wall_ransac_threshold_m", 0.025)
            ),
            "minimum_inliers": int(
                rospy.get_param("~right_wall_minimum_inliers", 16)
            ),
            "straight_length_m": float(
                rospy.get_param("~narrow_straight_length_m", 1.40)
            ),
            "expected_turn_depth_m": float(
                rospy.get_param("~terminal_expected_turn_depth_m", 0.485)
            ),
            "turn_depth_tolerance_m": float(
                rospy.get_param("~terminal_turn_depth_tolerance_m", 0.18)
            ),
            "inner_expected_forward_m": float(
                rospy.get_param("~terminal_inner_expected_forward_m", 3.95)
            ),
            "inner_forward_tolerance_m": float(
                rospy.get_param("~terminal_inner_forward_tolerance_m", 0.55)
            ),
            "inner_maximum_lateral_m": float(
                rospy.get_param("~terminal_inner_maximum_lateral_m", 1.90)
            ),
            "outer_minimum_lateral_m": float(
                rospy.get_param("~terminal_outer_minimum_lateral_m", 0.05)
            ),
            "outer_lateral_margin_m": float(
                rospy.get_param("~terminal_outer_lateral_margin_m", 0.16)
            ),
            "face_ransac_threshold_m": float(
                rospy.get_param("~terminal_face_ransac_threshold_m", 0.035)
            ),
            "face_minimum_support": int(
                rospy.get_param("~terminal_face_minimum_support", 10)
            ),
            "face_minimum_lateral_span_m": float(
                rospy.get_param("~terminal_face_minimum_lateral_span_m", 0.22)
            ),
            "face_maximum_abs_slope_ds_dd": float(
                rospy.get_param("~terminal_face_maximum_abs_slope_ds_dd", 0.55)
            ),
        }

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.corridor_status = None
        self.corridor_receipt = None
        self.disc_status = None
        self.disc_receipt = None
        self.perception_status = None
        self.perception_receipt = None
        self.compact_status = None
        self.compact_receipt = None
        self.ready_since = None
        self.latest_odom = None
        self.frames = []
        self.heading_frames = []
        self.heading_processing = False
        self.heading_pose = None
        self.heading_rms_m = None
        self.heading_inliers = 0
        self.heading_span_m = None
        self.heading_last_error = None
        self.capturing = False
        self.processing = False
        self.capture_stamp = None
        self.geometry_base = None
        self.fixed_geometry = None
        self.goal_pose = None
        self.goal_sent = False
        self.preview_pose_count = 0
        self.last_error = None
        self.scene_generation = 0

        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1, latch=True
        )
        self.goal_pose_pub = rospy.Publisher(
            self.goal_pose_topic, PoseStamped, queue_size=1, latch=True
        )
        self.heading_pub = rospy.Publisher(
            self.heading_topic, PoseStamped, queue_size=1, latch=True
        )
        self.preview_pub = rospy.Publisher(
            self.preview_topic, Path, queue_size=1, latch=True
        )
        self.navigation_goal_pub = rospy.Publisher(
            self.goal_topic, PoseStamped, queue_size=1
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.make_plan = rospy.ServiceProxy(self.make_plan_service, GetPlan)
        self.capture_service = rospy.Service("~capture", Trigger, self._capture_service)
        self.send_service = rospy.Service("~send_goal", Trigger, self._send_goal_service)
        self.reset_service = rospy.Service("~reset", Trigger, self._reset_service)
        self.scan_sub = rospy.Subscriber(
            self.scan_topic, LaserScan, self._scan_callback, queue_size=30
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_callback, queue_size=1
        )
        self.corridor_sub = rospy.Subscriber(
            self.corridor_status_topic, String, self._corridor_callback, queue_size=1
        )
        self.disc_sub = rospy.Subscriber(
            self.disc_status_topic, String, self._disc_callback, queue_size=1
        )
        self.perception_sub = rospy.Subscriber(
            self.perception_status_topic,
            String,
            self._perception_callback,
            queue_size=1,
        )
        self.compact_sub = rospy.Subscriber(
            self.compact_status_topic,
            String,
            self._compact_callback,
            queue_size=1,
        )
        self.reset_sub = rospy.Subscriber(
            self.scene_reset_topic, String, self._scene_reset_callback, queue_size=1
        )
        self.timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)
        rospy.loginfo(
            "Stage-1 lidar goal ready: %d scans, width %.3f m, front clearance %.3f m; no cmd_vel publisher",
            self.scan_frame_count,
            self.expected_width,
            self.front_clearance,
        )

    @staticmethod
    def _json(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError):
            return None

    def _corridor_callback(self, message):
        payload = self._json(message)
        if payload is not None:
            with self.lock:
                self.corridor_status = payload
                self.corridor_receipt = rospy.Time.now()

    def _disc_callback(self, message):
        payload = self._json(message)
        if payload is not None:
            with self.lock:
                self.disc_status = payload
                self.disc_receipt = rospy.Time.now()

    def _perception_callback(self, message):
        payload = self._json(message)
        if payload is not None:
            with self.lock:
                self.perception_status = payload
                self.perception_receipt = rospy.Time.now()

    def _compact_callback(self, message):
        payload = self._json(message)
        if payload is not None:
            with self.lock:
                self.compact_status = payload
                self.compact_receipt = rospy.Time.now()

    def _odom_callback(self, message):
        with self.lock:
            self.latest_odom = message

    def _scene_reset_callback(self, message):
        payload = self._json(message)
        if payload is None:
            return
        try:
            generation = int(payload["scene_generation"])
        except (KeyError, TypeError, ValueError):
            return
        with self.lock:
            if generation <= self.scene_generation:
                return
            self.scene_generation = generation
            self._clear_locked_state()
        rospy.logwarn("Stage-1 lidar goal reset for scene generation %d", generation)

    def _clear_locked_state(self):
        self.frames = []
        self.heading_frames = []
        self.heading_processing = False
        self.heading_pose = None
        self.heading_rms_m = None
        self.heading_inliers = 0
        self.heading_span_m = None
        self.heading_last_error = None
        self.capturing = False
        self.processing = False
        self.capture_stamp = None
        self.geometry_base = None
        self.fixed_geometry = None
        self.goal_pose = None
        self.goal_sent = False
        self.preview_pose_count = 0
        self.ready_since = None
        self.last_error = None

    def _world_ready(self, now):
        corridor = self.corridor_status or {}
        discs = self.disc_status or {}
        perception = self.perception_status or {}
        if (
            self.corridor_receipt is None
            or self.disc_receipt is None
            or self.perception_receipt is None
        ):
            return False, "waiting_for_status"
        if (now - self.corridor_receipt).to_sec() > self.status_timeout:
            return False, "corridor_status_stale"
        if (now - self.disc_receipt).to_sec() > self.status_timeout:
            return False, "disc_status_stale"
        if (now - self.perception_receipt).to_sec() > self.status_timeout:
            return False, "perception_status_stale"
        diagnostics = perception.get("diagnostics", {})
        if not isinstance(diagnostics, dict) or diagnostics.get(
            "perception_mode"
        ) != "LOCKED":
            return False, "disc_collection_not_complete"
        if corridor.get("state") != "locked" or not corridor.get(
            "corridor_cloud_ready", False
        ):
            return False, "corridor_not_locked"
        if not discs.get("confirmed_positions_frozen", False):
            return False, "disc_positions_not_frozen"
        if not discs.get("costmap_writes_enabled", False):
            return False, "costmap_writes_disabled"
        tracks = discs.get("active_tracks", [])
        if not isinstance(tracks, list) or len(tracks) < self.minimum_disc_count:
            return False, "confirmed_disc_count_below_minimum"
        if not all(
            bool(track.get("confirmed", False)) for track in tracks
        ):
            return False, "disc_tracks_not_confirmed"
        disc_count = len(tracks)
        if int(discs.get("costmap_disc_count", 0)) != disc_count:
            return False, "costmap_disc_count_mismatch"
        if int(discs.get("memory_count", 0)) < disc_count:
            return False, "disc_memory_count_mismatch"
        if int(corridor.get("scene_generation", 0)) != int(
            discs.get("scene_generation", 0)
        ):
            return False, "scene_generation_mismatch"
        if int(diagnostics.get("scene_generation", -1)) != int(
            discs.get("scene_generation", 0)
        ):
            return False, "perception_scene_generation_mismatch"
        if self.require_passage_closures:
            compact = self.compact_status or {}
            if self.compact_receipt is None:
                return False, "waiting_for_passage_status"
            if (now - self.compact_receipt).to_sec() > self.status_timeout:
                return False, "passage_status_stale"
            if not compact.get("ready", False) or not compact.get(
                "passage_closures_enabled", False
            ):
                return False, "passage_closures_not_ready"
            if int(compact.get("scene_generation", -1)) != int(
                corridor.get("scene_generation", 0)
            ):
                return False, "passage_scene_generation_mismatch"
            if len(compact.get("passage_decisions", [])) != disc_count:
                return False, "passage_decision_count_mismatch"
        return True, "ready"

    def _stationary(self):
        if self.latest_odom is None:
            return False
        twist = self.latest_odom.twist.twist
        linear = math.hypot(twist.linear.x, twist.linear.y)
        return (
            linear <= self.maximum_linear_speed
            and abs(twist.angular.z) <= self.maximum_angular_speed
        )

    def _start_capture(self):
        # A failed short window may simply have fallen between the very sparse
        # returns from the entrance wall's thin end face.  Keep the configured
        # tail so automatic retries form an overlapping rolling window instead
        # of discarding valid recent lidar evidence.
        if len(self.frames) > self.capture_retry_overlap_frames:
            self.frames = self.frames[-self.capture_retry_overlap_frames :]
        self.capturing = True
        self.capture_stamp = None
        self.last_error = None
        rospy.loginfo(
            "Stage-1 lidar capture started (%d frames, %d retained)",
            self.scan_frame_count,
            len(self.frames),
        )

    def _capture_service(self, _request):
        now = rospy.Time.now()
        with self.lock:
            ready, reason = self._world_ready(now)
            if self.goal_pose is not None:
                return TriggerResponse(True, "Stage-1 goal is already frozen")
            if self.capturing or self.processing:
                return TriggerResponse(True, "Stage-1 lidar capture is already running")
            if not ready:
                return TriggerResponse(False, "world model is not ready: " + reason)
            if not self._stationary():
                return TriggerResponse(False, "robot odometry is not stationary")
            self._start_capture()
        return TriggerResponse(True, "Stage-1 lidar capture started")

    def _reset_service(self, _request):
        with self.lock:
            self._clear_locked_state()
        return TriggerResponse(True, "Stage-1 lidar goal cleared")

    def _scan_points(self, message):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            message.header.frame_id,
            message.header.stamp,
            rospy.Duration(0.2),
        )
        ranges = np.asarray(message.ranges, dtype=np.float64)
        angles = message.angle_min + np.arange(len(ranges)) * message.angle_increment
        valid = (
            np.isfinite(ranges)
            & (ranges >= message.range_min)
            & (ranges <= message.range_max)
        )
        laser_points = np.column_stack(
            (ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid]))
        )
        angle = _yaw(transform.transform.rotation)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
            dtype=np.float64,
        )
        translation = np.asarray(
            (transform.transform.translation.x, transform.transform.translation.y),
            dtype=np.float64,
        )
        return laser_points @ rotation.T + translation

    def _scan_callback(self, message):
        with self.lock:
            stationary = self._stationary()
            collect_heading = (
                self.heading_pose is None
                and not self.heading_processing
                and stationary
            )
            collect_goal = self.capturing and self.goal_pose is None
            if not collect_heading and not collect_goal:
                return
            if collect_goal and not stationary:
                self.capturing = False
                self.frames = []
                self.last_error = "robot moved during lidar capture"
                collect_goal = False
        try:
            points = self._scan_points(message)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(1.0, "waiting for lidar TF: %s", error)
            return
        with self.lock:
            heading_job = None
            goal_job = None
            if collect_heading and self.heading_pose is None:
                self.heading_frames.append(points)
                if len(self.heading_frames) >= self.heading_scan_frame_count:
                    heading_job = (self.heading_frames, message.header.stamp)
                    self.heading_frames = []
                    self.heading_processing = True
            if collect_goal and self.capturing:
                self.frames.append(points)
                self.capture_stamp = message.header.stamp
                if len(self.frames) >= self.scan_frame_count:
                    goal_job = (self.frames, self.capture_stamp)
                    self.frames = []
                    self.capturing = False
                    self.processing = True
        if heading_job is not None:
            self._finish_heading_capture(*heading_job)
        if goal_job is not None:
            self._finish_capture(*goal_job)

    def _finish_heading_capture(self, frames, stamp):
        try:
            points = voxel_downsample(
                np.vstack(frames), resolution_m=self.voxel_resolution
            )
            right_keys = {
                "minimum_x_m",
                "maximum_x_m",
                "maximum_y_m",
                "ransac_threshold_m",
                "minimum_inliers",
            }
            wall = fit_right_wall(
                points,
                **{
                    key: value
                    for key, value in self.fit_options.items()
                    if key in right_keys
                }
            )
            span_m = float(wall.maximum_s_m - wall.minimum_s_m)
            if wall.rms_m > self.heading_maximum_rms_m:
                raise ValueError(
                    "right wall RMS %.4f m exceeds %.4f m"
                    % (wall.rms_m, self.heading_maximum_rms_m)
                )
            if len(wall.inliers) < self.heading_minimum_inliers:
                raise ValueError("right wall has too few inliers")
            if span_m < self.heading_minimum_span_m:
                raise ValueError(
                    "right wall span %.3f m is shorter than %.3f m"
                    % (span_m, self.heading_minimum_span_m)
                )
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.base_frame,
                stamp,
                rospy.Duration(0.3),
            )
            heading = _normalize(
                _yaw(transform.transform.rotation)
                + math.atan2(wall.tangent[1], wall.tangent[0])
            )
            pose = self._goal_message(
                stamp,
                (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                ),
                heading,
            )
        except Exception as error:
            with self.lock:
                self.heading_processing = False
                self.heading_last_error = str(error)
            rospy.logwarn_throttle(
                1.0, "Stage-1 right-wall heading capture failed: %s", error
            )
            return
        with self.lock:
            self.heading_processing = False
            self.heading_pose = pose
            self.heading_rms_m = wall.rms_m
            self.heading_inliers = len(wall.inliers)
            self.heading_span_m = span_m
            self.heading_last_error = None
        self.heading_pub.publish(pose)
        self.marker_pub.publish(
            self._heading_marker(
                rospy.Time.now(),
                transform,
                wall,
            )
        )
        rospy.loginfo(
            "Stage-1 lidar heading frozen from right wall: %.3f deg, "
            "RMS %.4f m, %d inliers",
            math.degrees(heading),
            wall.rms_m,
            len(wall.inliers),
        )

    def _heading_marker(self, stamp, transform, wall):
        start_base = (
            wall.anchor + wall.minimum_s_m * wall.tangent
        )
        end_base = wall.anchor + wall.maximum_s_m * wall.tangent
        start = _transform_xy(transform, start_base)
        end = _transform_xy(transform, end_base)
        green = self._color(0.1, 1.0, 0.2)
        white = self._color(1.0, 1.0, 1.0)
        return MarkerArray(
            markers=[
                self._line(
                    stamp,
                    101,
                    "right_wall_heading_reference",
                    start,
                    end,
                    green,
                    0.045,
                ),
                self._text(
                    stamp,
                    102,
                    0.5 * (start + end),
                    "雷达右墙方向参考",
                    white,
                    0.22,
                ),
            ]
        )

    def _finish_capture(self, frames, stamp):
        try:
            points = voxel_downsample(
                np.vstack(frames), resolution_m=self.voxel_resolution
            )
            geometry = build_stage1_geometry(
                points,
                expected_width_m=self.expected_width,
                width_tolerance_m=self.width_tolerance,
                vehicle_front_offset_m=self.vehicle_front_offset,
                desired_front_clearance_m=self.front_clearance,
                **self.fit_options
            )
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.base_frame,
                stamp,
                rospy.Duration(0.3),
            )
            fixed = self._fixed_geometry(transform, geometry)
            goal = self._goal_message(stamp, fixed["goal"], fixed["heading"])
        except Exception as error:  # report the geometric/TF reason verbatim
            with self.lock:
                if self.capture_retry_overlap_frames:
                    self.frames = list(frames[-self.capture_retry_overlap_frames :])
                else:
                    self.frames = []
                self.processing = False
                self.last_error = str(error)
            rospy.logerr("Stage-1 lidar goal capture failed: %s", error)
            return
        with self.lock:
            self.frames = []
            self.capturing = False
            self.processing = False
            self.geometry_base = geometry
            self.fixed_geometry = fixed
            self.goal_pose = goal
            self.last_error = None
        self.goal_pose_pub.publish(goal)
        self.marker_pub.publish(self._markers(rospy.Time.now(), geometry, fixed))
        self._publish_preview_plan(goal)
        rospy.loginfo(
            "Stage-1 lidar goal frozen in %s: (%.3f, %.3f), yaw %.2f deg; "
            "width %.3f m, terminal pair %.3f m",
            self.target_frame,
            goal.pose.position.x,
            goal.pose.position.y,
            math.degrees(fixed["heading"]),
            geometry.measured_width_m,
            geometry.terminal_pair_separation_m,
        )

    def _fixed_geometry(self, transform, geometry):
        transform_yaw = _yaw(transform.transform.rotation)
        output = {
            "left": _transform_xy(transform, geometry.left_entrance),
            "right": _transform_xy(transform, geometry.right_projection),
            "center": _transform_xy(transform, geometry.entrance_center),
            "terminal_inner": _transform_xy(
                transform, geometry.terminal_inner_corner
            ),
            "terminal_outer": _transform_xy(
                transform, geometry.terminal_outer_corner
            ),
            "goal": _transform_xy(transform, geometry.goal_base_center),
            "front": _transform_xy(transform, geometry.goal_front_center),
            "wall_start": _transform_xy(
                transform,
                geometry.right_wall.anchor
                + geometry.right_wall.minimum_s_m * geometry.right_wall.tangent,
            ),
            "wall_end": _transform_xy(
                transform,
                geometry.right_wall.anchor
                + geometry.right_wall.maximum_s_m * geometry.right_wall.tangent,
            ),
            "heading": _normalize(transform_yaw + geometry.heading_rad),
        }
        for name, face in (
            ("terminal_inner_face", geometry.terminal_inner_face),
            ("terminal_outer_face", geometry.terminal_outer_face),
        ):
            first_base = (
                geometry.right_wall.anchor
                + face.longitudinal_at(face.minimum_d_m)
                * geometry.right_wall.tangent
                + face.minimum_d_m * geometry.right_wall.left_normal
            )
            second_base = (
                geometry.right_wall.anchor
                + face.longitudinal_at(face.maximum_d_m)
                * geometry.right_wall.tangent
                + face.maximum_d_m * geometry.right_wall.left_normal
            )
            output[name] = (
                _transform_xy(transform, first_base),
                _transform_xy(transform, second_base),
            )
        output["tangent"] = np.asarray(
            (math.cos(output["heading"]), math.sin(output["heading"])),
            dtype=np.float64,
        )
        output["left_normal"] = np.asarray(
            (-output["tangent"][1], output["tangent"][0]), dtype=np.float64
        )
        return output

    def _goal_message(self, stamp, xy, heading):
        message = PoseStamped()
        message.header.frame_id = self.target_frame
        message.header.stamp = stamp
        message.pose.position.x = float(xy[0])
        message.pose.position.y = float(xy[1])
        message.pose.orientation.z = math.sin(0.5 * heading)
        message.pose.orientation.w = math.cos(0.5 * heading)
        return message

    @staticmethod
    def _point(xy, z=0.08):
        message = Point()
        message.x, message.y, message.z = float(xy[0]), float(xy[1]), float(z)
        return message

    @staticmethod
    def _color(red, green, blue, alpha=1.0):
        return ColorRGBA(red, green, blue, alpha)

    def _line(self, stamp, marker_id, namespace, first, second, color, width=0.05):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns, marker.id = namespace, marker_id
        marker.type, marker.action = Marker.LINE_LIST, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.color = width, color
        marker.points = [self._point(first), self._point(second)]
        return marker

    def _sphere(self, stamp, marker_id, namespace, xy, color, size=0.11):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns, marker.id = namespace, marker_id
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position = self._point(xy, 0.12)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color = color
        return marker

    def _text(self, stamp, marker_id, xy, text, color, z=0.28):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns, marker.id = "stage1_labels", marker_id
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position = self._point(xy, z)
        marker.pose.orientation.w = 1.0
        marker.scale.z, marker.color, marker.text = 0.11, color, text
        return marker

    def _markers(self, stamp, geometry, fixed):
        green = self._color(0.1, 1.0, 0.2)
        orange = self._color(1.0, 0.55, 0.0)
        cyan = self._color(0.0, 0.9, 1.0)
        yellow = self._color(1.0, 0.9, 0.0)
        magenta = self._color(1.0, 0.05, 0.75)
        white = self._color(1.0, 1.0, 1.0)
        markers = [
            self._line(stamp, 1, "right_wall", fixed["wall_start"], fixed["wall_end"], green),
            self._line(stamp, 2, "entrance", fixed["right"], fixed["left"], white, 0.06),
            self._sphere(stamp, 3, "inferred_entrance", fixed["left"], orange, 0.13),
            self._sphere(stamp, 4, "entrance_points", fixed["right"], cyan),
            self._sphere(stamp, 5, "entrance_center", fixed["center"], yellow),
            self._line(stamp, 6, "front_clearance", fixed["front"], fixed["center"], orange, 0.04),
            self._line(
                stamp,
                15,
                "terminal_inner_face",
                fixed["terminal_inner_face"][0],
                fixed["terminal_inner_face"][1],
                cyan,
                0.055,
            ),
            self._line(
                stamp,
                16,
                "terminal_outer_face",
                fixed["terminal_outer_face"][0],
                fixed["terminal_outer_face"][1],
                orange,
                0.055,
            ),
            self._sphere(
                stamp, 17, "terminal_corners", fixed["terminal_inner"], cyan, 0.13
            ),
            self._sphere(
                stamp, 18, "terminal_corners", fixed["terminal_outer"], orange, 0.13
            ),
            self._line(
                stamp,
                21,
                "terminal_corner_pair",
                fixed["terminal_inner"],
                fixed["terminal_outer"],
                white,
                0.035,
            ),
        ]
        tangent, normal = fixed["tangent"], fixed["left_normal"]
        goal = fixed["goal"]
        corners = [
            goal + self.vehicle_front_offset * tangent + 0.165 * normal,
            goal + self.vehicle_front_offset * tangent - 0.165 * normal,
            goal - self.vehicle_front_offset * tangent - 0.165 * normal,
            goal - self.vehicle_front_offset * tangent + 0.165 * normal,
        ]
        footprint = Marker()
        footprint.header.frame_id, footprint.header.stamp = self.target_frame, stamp
        footprint.ns, footprint.id = "stage1_goal_vehicle", 7
        footprint.type, footprint.action = Marker.LINE_STRIP, Marker.ADD
        footprint.pose.orientation.w = 1.0
        footprint.scale.x, footprint.color = 0.055, magenta
        footprint.points = [self._point(value, 0.13) for value in corners + [corners[0]]]
        markers.append(footprint)
        arrow = Marker()
        arrow.header.frame_id, arrow.header.stamp = self.target_frame, stamp
        arrow.ns, arrow.id = "stage1_goal_arrow", 8
        arrow.type, arrow.action = Marker.ARROW, Marker.ADD
        arrow.pose.orientation.w = 1.0
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.05, 0.10, 0.14
        arrow.color = magenta
        arrow.points = [self._point(goal, 0.15), self._point(goal + 0.4 * tangent, 0.15)]
        markers.append(arrow)
        markers.extend(
            (
                self._text(stamp, 10, fixed["left"], "反推左入口点", orange),
                self._text(stamp, 11, fixed["right"], "P_R 右墙垂足", cyan),
                self._text(stamp, 12, fixed["center"], "入口中心 E", yellow, 0.36),
                self._text(stamp, 13, goal, "Stage 1 雷达终点", magenta, 0.34),
                self._text(
                    stamp,
                    19,
                    fixed["terminal_inner"],
                    "远端内侧面",
                    cyan,
                    0.34,
                ),
                self._text(
                    stamp,
                    20,
                    fixed["terminal_outer"],
                    "远端外侧面上端点",
                    orange,
                    0.34,
                ),
                self._text(
                    stamp,
                    14,
                    goal - 0.25 * normal,
                    "入口宽 %.3fm  直道 %.2fm  车头预留 %.2fm\n"
                    "yaw %.2f°  远端间距 %.3fm  入口闪点(仅校验) %d"
                    % (
                        geometry.measured_width_m,
                        geometry.straight_length_m,
                        self.front_clearance,
                        math.degrees(fixed["heading"]),
                        geometry.terminal_pair_separation_m,
                        geometry.entrance_support_voxels,
                    ),
                    white,
                    0.28,
                ),
            )
        )
        return MarkerArray(markers=markers)

    def _publish_preview_plan(self, goal):
        try:
            robot = self.tf_buffer.lookup_transform(
                self.target_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.2)
            )
            start = self._goal_message(
                rospy.Time.now(),
                (robot.transform.translation.x, robot.transform.translation.y),
                _yaw(robot.transform.rotation),
            )
            request_goal = PoseStamped()
            request_goal.header.frame_id = goal.header.frame_id
            request_goal.header.stamp = rospy.Time.now()
            request_goal.pose = goal.pose
            response = self.make_plan(start, request_goal, 0.08)
        except Exception as error:
            rospy.logwarn("Stage-1 make_plan preview unavailable: %s", error)
            return
        if not response.plan.poses:
            rospy.logwarn("Stage-1 make_plan returned an empty path")
            return
        response.plan.header.stamp = rospy.Time.now()
        self.preview_pub.publish(response.plan)
        with self.lock:
            self.preview_pose_count = len(response.plan.poses)

    def _send_goal_service(self, _request):
        with self.lock:
            goal = self.goal_pose
        if goal is None:
            return TriggerResponse(False, "Stage-1 lidar goal is not frozen")
        if self.navigation_goal_pub.get_num_connections() <= 0:
            return TriggerResponse(False, "move_base goal subscriber is missing")
        message = PoseStamped()
        message.header.frame_id = goal.header.frame_id
        message.header.stamp = rospy.Time.now()
        message.pose = goal.pose
        self.navigation_goal_pub.publish(message)
        with self.lock:
            self.goal_sent = True
        return TriggerResponse(
            True,
            "Stage-1 goal sent: (%.3f, %.3f), yaw %.3f rad"
            % (
                message.pose.position.x,
                message.pose.position.y,
                2.0 * math.atan2(message.pose.orientation.z, message.pose.orientation.w),
            ),
        )

    def _timer_callback(self, _event):
        now = rospy.Time.now()
        with self.lock:
            ready, reason = self._world_ready(now)
            if ready:
                if self.ready_since is None:
                    self.ready_since = now
            else:
                self.ready_since = None
            stable = (
                ready
                and self.ready_since is not None
                and (now - self.ready_since).to_sec() >= self.world_stable_sec
            )
            if (
                self.auto_capture
                and stable
                and self.goal_pose is None
                and not self.capturing
                and not self.processing
                and self._stationary()
            ):
                self._start_capture()
            goal = self.goal_pose
            geometry = self.geometry_base
            fixed = self.fixed_geometry
            if self.processing:
                state = "processing_scan"
            elif self.capturing:
                state = "collecting_scan"
            elif goal is not None:
                state = "goal_sent" if self.goal_sent else "goal_ready"
            elif self.last_error:
                state = "capture_failed"
            else:
                state = "waiting_world_model"
            status = {
                "stamp": now.to_sec(),
                "state": state,
                "world_ready": ready,
                "world_ready_reason": reason,
                "world_stable": stable,
                "passage_closures_required": self.require_passage_closures,
                "passage_closures_ready": bool(
                    (self.compact_status or {}).get("passage_ready", False)
                ),
                "passage_decisions": (self.compact_status or {}).get(
                    "passage_decisions", []
                ),
                "scan_frames_collected": len(self.frames),
                "scan_frames_required": self.scan_frame_count,
                "target_frame": self.target_frame,
                "goal_frozen": goal is not None,
                "goal_sent": self.goal_sent,
                "publishes_cmd_vel": False,
                "auto_capture_when_ready": self.auto_capture,
                "auto_send_goal_when_ready": self.auto_send,
                "preview_plan_pose_count": self.preview_pose_count,
                "capture_service": rospy.resolve_name("~capture"),
                "send_goal_service": rospy.resolve_name("~send_goal"),
                "last_error": self.last_error,
                "heading_reference": {
                    "ready": self.heading_pose is not None,
                    "topic": self.heading_topic,
                    "scan_frames_required": self.heading_scan_frame_count,
                    "scan_frames_collected": len(self.heading_frames),
                    "yaw_rad": None
                    if self.heading_pose is None
                    else _yaw(self.heading_pose.pose.orientation),
                    "right_wall_rms_m": self.heading_rms_m,
                    "right_wall_inliers": self.heading_inliers,
                    "right_wall_span_m": self.heading_span_m,
                    "last_error": self.heading_last_error,
                },
                "goal": None
                if goal is None
                else {
                    "frame_id": goal.header.frame_id,
                    "xy_m": [goal.pose.position.x, goal.pose.position.y],
                    "yaw_rad": fixed["heading"],
                    "front_clearance_m": self.front_clearance,
                    "measured_entrance_width_m": geometry.measured_width_m,
                    "right_wall_rms_m": geometry.right_wall.rms_m,
                    "right_wall_inliers": len(geometry.right_wall.inliers),
                    "geometry_source": "far_terminal_pair_back_projection",
                    "narrow_straight_length_m": geometry.straight_length_m,
                    "terminal_pair_separation_m": geometry.terminal_pair_separation_m,
                    "terminal_inner_support_voxels": len(
                        geometry.terminal_inner_face.inliers
                    ),
                    "terminal_outer_support_voxels": len(
                        geometry.terminal_outer_face.inliers
                    ),
                    "terminal_inner_rms_m": geometry.terminal_inner_face.rms_m,
                    "terminal_outer_rms_m": geometry.terminal_outer_face.rms_m,
                    "optional_entrance_validation_voxels": geometry.entrance_support_voxels,
                },
            }
        if goal is not None and geometry is not None and fixed is not None:
            self.goal_pose_pub.publish(goal)
            self.marker_pub.publish(self._markers(now, geometry, fixed))
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
        if self.auto_send and goal is not None and not self.goal_sent:
            self._send_goal_service(None)


def main():
    rospy.init_node("yellow_track_stage1_lidar_goal")
    Stage1LidarGoalNode()
    rospy.spin()


if __name__ == "__main__":
    main()
