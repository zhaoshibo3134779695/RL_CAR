#!/usr/bin/env python3
"""Conditionally take over from move_base to cross a close two-disc gate."""

import copy
import json
import math
import threading

import numpy as np
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

from yellow_track_avoidance.narrow_gate import (
    DiscTrack,
    NarrowGateConfig,
    NarrowGateStateMachine,
    RobotPose,
    STATE_FAILED_STOP,
    wrap_angle,
)


class NarrowGateControllerNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.enabled = bool(rospy.get_param("~enabled", True))
        self.fixed_frame = str(
            rospy.get_param("~fixed_frame", "odom")
        ).lstrip("/")
        self.global_path_topic = str(
            rospy.get_param(
                "~global_path_topic",
                "/move_base/SBPLLatticePlanner/plan",
            )
        )
        self.disc_status_topic = str(
            rospy.get_param(
                "~disc_status_topic",
                "/yellow_track_disc_costmap_bridge/status",
            )
        )
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.move_base_status_topic = str(
            rospy.get_param("~move_base_status_topic", "/move_base/status")
        )
        self.current_goal_topic = str(
            rospy.get_param(
                "~current_goal_topic", "/move_base/current_goal"
            )
        )
        self.cancel_topic = str(
            rospy.get_param("~cancel_topic", "/move_base/cancel")
        )
        self.resume_goal_topic = str(
            rospy.get_param(
                "~resume_goal_topic", "/move_base_simple/goal"
            )
        )
        self.command_topic = str(
            rospy.get_param(
                "~command_topic",
                "/yellow_track_avoidance/narrow_gate_cmd_vel",
            )
        )
        self.status_topic = str(
            rospy.get_param(
                "~status_topic",
                "/yellow_track_avoidance/narrow_gate/status",
            )
        )
        self.marker_topic = str(
            rospy.get_param(
                "~marker_topic",
                "/yellow_track_avoidance/narrow_gate/markers",
            )
        )
        self.control_rate_hz = float(
            rospy.get_param("~control_rate_hz", 20.0)
        )
        self.robot_timeout_sec = float(
            rospy.get_param("~robot_pose_timeout_sec", 0.30)
        )
        self.disc_timeout_sec = float(
            rospy.get_param("~disc_status_timeout_sec", 0.40)
        )
        self.path_timeout_sec = float(
            rospy.get_param("~global_path_timeout_sec", 5.0)
        )
        self.stationary_linear_limit = float(
            rospy.get_param("~reset_linear_speed_max_mps", 0.01)
        )
        self.stationary_angular_limit = float(
            rospy.get_param("~reset_angular_speed_max_radps", 0.03)
        )
        if (
            self.control_rate_hz <= 0.0
            or self.robot_timeout_sec <= 0.0
            or self.disc_timeout_sec <= 0.0
            or self.path_timeout_sec <= 0.0
        ):
            raise ValueError("invalid narrow gate timing configuration")

        self.config = NarrowGateConfig(
            disc_radius_m=float(
                rospy.get_param("~disc_radius_m", 0.050)
            ),
            vehicle_half_length_m=float(
                rospy.get_param("~vehicle_half_length_m", 0.165)
            ),
            vehicle_half_width_m=float(
                rospy.get_param("~vehicle_half_width_m", 0.150)
            ),
            trigger_edge_gap_m=float(
                rospy.get_param("~trigger_edge_gap_m", 0.450)
            ),
            release_edge_gap_m=float(
                rospy.get_param("~release_edge_gap_m", 0.480)
            ),
            pair_confirm_duration_sec=float(
                rospy.get_param("~pair_confirm_duration_sec", 0.60)
            ),
            pair_midpoint_variation_max_m=float(
                rospy.get_param(
                    "~pair_midpoint_variation_max_m", 0.015
                )
            ),
            pair_distance_variation_max_m=float(
                rospy.get_param(
                    "~pair_distance_variation_max_m", 0.015
                )
            ),
            pair_angle_variation_max_rad=math.radians(
                float(
                    rospy.get_param(
                        "~pair_angle_variation_max_deg", 1.5
                    )
                )
            ),
            maximum_candidate_track_age_sec=float(
                rospy.get_param(
                    "~maximum_candidate_track_age_sec", 0.40
                )
            ),
            midpoint_path_distance_max_m=float(
                rospy.get_param(
                    "~midpoint_path_distance_max_m", 0.15
                )
            ),
            pair_along_path_mismatch_max_m=float(
                rospy.get_param(
                    "~pair_along_path_mismatch_max_m", 0.10
                )
            ),
            crossing_angle_deviation_max_rad=math.radians(
                float(
                    rospy.get_param(
                        "~crossing_angle_deviation_max_deg", 20.0
                    )
                )
            ),
            minimum_opposite_side_offset_m=float(
                rospy.get_param(
                    "~minimum_opposite_side_offset_m", 0.02
                )
            ),
            gate_ahead_max_m=float(
                rospy.get_param("~gate_ahead_max_m", 1.20)
            ),
            gate_ahead_min_m=float(
                rospy.get_param("~gate_ahead_min_m", 0.10)
            ),
            prealign_distance_m=float(
                rospy.get_param("~prealign_distance_m", 0.50)
            ),
            latest_takeover_progress_m=float(
                rospy.get_param(
                    "~latest_takeover_progress_m", -0.10
                )
            ),
            align_lateral_tolerance_m=float(
                rospy.get_param(
                    "~align_lateral_tolerance_m", 0.008
                )
            ),
            align_yaw_tolerance_rad=math.radians(
                float(
                    rospy.get_param(
                        "~align_yaw_tolerance_deg", 1.5
                    )
                )
            ),
            align_hold_duration_sec=float(
                rospy.get_param("~align_hold_duration_sec", 0.30)
            ),
            alignment_lateral_abort_m=float(
                rospy.get_param("~alignment_lateral_abort_m", 0.15)
            ),
            alignment_yaw_abort_rad=math.radians(
                float(
                    rospy.get_param(
                        "~alignment_yaw_abort_deg", 45.0
                    )
                )
            ),
            pass_speed_mps=float(
                rospy.get_param("~pass_speed_mps", 0.040)
            ),
            exit_speed_mps=float(
                rospy.get_param("~exit_speed_mps", 0.030)
            ),
            lateral_gain=float(
                rospy.get_param("~lateral_gain", 0.80)
            ),
            yaw_gain=float(rospy.get_param("~yaw_gain", 1.50)),
            prealign_longitudinal_gain=float(
                rospy.get_param(
                    "~prealign_longitudinal_gain", 0.80
                )
            ),
            maximum_lateral_speed_mps=float(
                rospy.get_param(
                    "~maximum_lateral_speed_mps", 0.015
                )
            ),
            maximum_align_longitudinal_speed_mps=float(
                rospy.get_param(
                    "~maximum_align_longitudinal_speed_mps", 0.015
                )
            ),
            maximum_angular_speed_radps=float(
                rospy.get_param(
                    "~maximum_angular_speed_radps", 0.080
                )
            ),
            slow_lateral_error_m=float(
                rospy.get_param("~slow_lateral_error_m", 0.012)
            ),
            slow_yaw_error_rad=math.radians(
                float(
                    rospy.get_param("~slow_yaw_error_deg", 2.0)
                )
            ),
            stop_lateral_error_m=float(
                rospy.get_param("~stop_lateral_error_m", 0.030)
            ),
            stop_yaw_error_rad=math.radians(
                float(
                    rospy.get_param("~stop_yaw_error_deg", 6.0)
                )
            ),
            minimum_physical_clearance_m=float(
                rospy.get_param(
                    "~minimum_physical_clearance_m", 0.008
                )
            ),
            geometry_uncertainty_m=float(
                rospy.get_param("~geometry_uncertainty_m", 0.005)
            ),
            emergency_direct_clearance_m=float(
                rospy.get_param(
                    "~emergency_direct_clearance_m", 0.005
                )
            ),
            exit_clearance_margin_m=float(
                rospy.get_param(
                    "~exit_clearance_margin_m", 0.050
                )
            ),
            exit_distance_m=float(
                rospy.get_param("~exit_distance_m", 0.350)
            ),
            stop_hold_duration_sec=float(
                rospy.get_param("~stop_hold_duration_sec", 0.30)
            ),
            acceleration_x_mps2=float(
                rospy.get_param("~acceleration_x_mps2", 0.10)
            ),
            acceleration_y_mps2=float(
                rospy.get_param("~acceleration_y_mps2", 0.08)
            ),
            acceleration_yaw_radps2=float(
                rospy.get_param(
                    "~acceleration_yaw_radps2", 0.25
                )
            ),
        )
        self.machine = NarrowGateStateMachine(self.config)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.robot_pose = None
        self.robot_receipt_sec = None
        self.robot_linear_speed = math.inf
        self.robot_angular_speed = math.inf
        self.tracks = []
        self.disc_receipt_sec = None
        self.disc_input_fresh = False
        self.disc_fixed_transform_available = False
        self.path = ()
        self.path_receipt_sec = None
        self.navigation_active = False
        self.current_goal = None
        self.resume_goal = None
        self.last_decision = None
        self.cancel_count = 0
        self.resume_count = 0

        self.command_pub = rospy.Publisher(
            self.command_topic, Twist, queue_size=1
        )
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1
        )
        self.cancel_pub = rospy.Publisher(
            self.cancel_topic, GoalID, queue_size=1
        )
        self.resume_pub = rospy.Publisher(
            self.resume_goal_topic, PoseStamped, queue_size=1
        )
        self.path_sub = rospy.Subscriber(
            self.global_path_topic,
            Path,
            self._path_callback,
            queue_size=1,
        )
        self.disc_sub = rospy.Subscriber(
            self.disc_status_topic,
            String,
            self._disc_callback,
            queue_size=1,
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self._odom_callback,
            queue_size=1,
        )
        self.move_base_status_sub = rospy.Subscriber(
            self.move_base_status_topic,
            GoalStatusArray,
            self._move_base_status_callback,
            queue_size=1,
        )
        self.current_goal_sub = rospy.Subscriber(
            self.current_goal_topic,
            PoseStamped,
            self._current_goal_callback,
            queue_size=1,
        )
        self.reset_service = rospy.Service(
            "~reset", Trigger, self._reset_callback
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate_hz),
            self._timer_callback,
        )
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "narrow gate controller ready: edge_gap<%.3fm path=%s "
            "command=%s fixed_frame=%s enabled=%s",
            self.config.trigger_edge_gap_m,
            self.global_path_topic,
            self.command_topic,
            self.fixed_frame,
            self.enabled,
        )

    @staticmethod
    def _yaw(quaternion):
        return math.atan2(
            2.0 * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0
            - 2.0
            * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )

    @staticmethod
    def _transform_xy(transform, point):
        yaw = NarrowGateControllerNode._yaw(
            transform.transform.rotation
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        translation = transform.transform.translation
        return (
            float(translation.x)
            + cosine * float(point[0])
            - sine * float(point[1]),
            float(translation.y)
            + sine * float(point[0])
            + cosine * float(point[1]),
        )

    def _lookup_transform(self, target, source, stamp):
        if target == source:
            return None
        try:
            return self.tf_buffer.lookup_transform(
                target, source, stamp, rospy.Duration(0.05)
            )
        except tf2_ros.ExtrapolationException:
            return self.tf_buffer.lookup_transform(
                target, source, rospy.Time(0), rospy.Duration(0.05)
            )

    def _odom_callback(self, message):
        source = str(message.header.frame_id).lstrip("/")
        if not source:
            return
        point = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
        )
        yaw = self._yaw(message.pose.pose.orientation)
        try:
            transform = self._lookup_transform(
                self.fixed_frame, source, message.header.stamp
            )
            if transform is not None:
                point = self._transform_xy(transform, point)
                yaw = wrap_angle(
                    self._yaw(transform.transform.rotation) + yaw
                )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return
        values = (point[0], point[1], yaw)
        if not all(np.isfinite(value) for value in values):
            return
        linear_speed = math.hypot(
            float(message.twist.twist.linear.x),
            float(message.twist.twist.linear.y),
        )
        angular_speed = abs(float(message.twist.twist.angular.z))
        with self.lock:
            self.robot_pose = RobotPose(*values)
            self.robot_receipt_sec = rospy.Time.now().to_sec()
            self.robot_linear_speed = linear_speed
            self.robot_angular_speed = angular_speed

    def _path_callback(self, message):
        source = str(message.header.frame_id).lstrip("/")
        if not source and message.poses:
            source = str(message.poses[0].header.frame_id).lstrip("/")
        if not source:
            return
        try:
            transform = self._lookup_transform(
                self.fixed_frame, source, message.header.stamp
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return
        points = []
        for pose in message.poses:
            point = (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
            )
            if transform is not None:
                point = self._transform_xy(transform, point)
            if not all(np.isfinite(value) for value in point):
                continue
            if points and math.hypot(
                point[0] - points[-1][0],
                point[1] - points[-1][1],
            ) <= 1e-4:
                continue
            points.append(point)
        if len(points) < 2:
            return
        with self.lock:
            self.path = tuple(points)
            self.path_receipt_sec = rospy.Time.now().to_sec()

    def _disc_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if str(payload.get("fixed_frame", "")).lstrip("/") != self.fixed_frame:
            return
        tracks = []
        for item in payload.get("active_tracks", []):
            fixed_xy = item.get("fixed_xy")
            if not isinstance(fixed_xy, list) or len(fixed_xy) < 2:
                continue
            try:
                values = float(fixed_xy[0]), float(fixed_xy[1])
                memory_id = int(item["memory_id"])
                last_seen_age_sec = float(
                    item.get("last_seen_age_sec", math.inf)
                )
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not all(np.isfinite(value) for value in values)
                or not np.isfinite(last_seen_age_sec)
                or last_seen_age_sec < 0.0
            ):
                continue
            tracks.append(
                DiscTrack(
                    memory_id=memory_id,
                    fixed_xy=values,
                    confirmed=bool(item.get("confirmed", False)),
                    phase=str(item.get("phase", "unconfirmed")),
                    last_seen_age_sec=last_seen_age_sec,
                )
            )
        with self.lock:
            self.tracks = tracks
            self.disc_receipt_sec = rospy.Time.now().to_sec()
            self.disc_input_fresh = bool(
                payload.get("input_fresh", False)
            )
            self.disc_fixed_transform_available = bool(
                payload.get("fixed_transform_available", False)
            )

    def _move_base_status_callback(self, message):
        active_states = {
            GoalStatus.PENDING,
            GoalStatus.ACTIVE,
            GoalStatus.PREEMPTING,
            GoalStatus.RECALLING,
        }
        active = any(
            status.status in active_states and status.goal_id.id
            for status in message.status_list
        )
        with self.lock:
            self.navigation_active = active

    def _current_goal_callback(self, message):
        with self.lock:
            self.current_goal = copy.deepcopy(message)
            if not self.machine.authority_requested:
                self.resume_goal = copy.deepcopy(message)

    def _reset_callback(self, _request):
        with self.lock:
            if (
                self.robot_linear_speed > self.stationary_linear_limit
                or self.robot_angular_speed > self.stationary_angular_limit
            ):
                return TriggerResponse(
                    False, "robot must be stationary before gate reset"
                )
            self.machine.reset()
            self.command_pub.publish(Twist())
            return TriggerResponse(True, "narrow gate state reset")

    @staticmethod
    def _twist(command):
        message = Twist()
        message.linear.x = float(command[0])
        message.linear.y = float(command[1])
        message.angular.z = float(command[2])
        return message

    @staticmethod
    def _frame_status(frame):
        if frame is None:
            return None
        return {
            "pair_ids": list(frame.pair_ids),
            "disc_first_xy": list(frame.disc_first_xy),
            "disc_second_xy": list(frame.disc_second_xy),
            "midpoint_xy": list(frame.midpoint_xy),
            "normal_xy": list(frame.normal_xy),
            "travel_xy": list(frame.travel_xy),
            "yaw_ref": frame.yaw_ref,
            "center_distance_m": frame.center_distance_m,
            "edge_gap_m": frame.edge_gap_m,
            "midpoint_path_distance_m":
                frame.midpoint_path_distance_m,
            "along_path_mismatch_m": frame.along_path_mismatch_m,
            "crossing_angle_deviation_rad":
                frame.crossing_angle_deviation_rad,
        }

    @staticmethod
    def _error_status(errors):
        if errors is None:
            return None
        return {
            "lateral_m": errors.lateral_m,
            "yaw_rad": errors.yaw_rad,
            "progress_m": errors.progress_m,
            "projected_width_m": errors.projected_width_m,
            "predicted_clearance_m": errors.predicted_clearance_m,
            "direct_clearance_first_m":
                errors.direct_clearance_first_m,
            "direct_clearance_second_m":
                errors.direct_clearance_second_m,
            "minimum_direct_clearance_m":
                errors.minimum_direct_clearance_m,
        }

    @staticmethod
    def _point(x_value, y_value, z_value=0.04):
        point = Point()
        point.x = float(x_value)
        point.y = float(y_value)
        point.z = float(z_value)
        return point

    def _markers(self, now, decision):
        message = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        message.markers.append(clear)
        frame = decision.frame
        if frame is None:
            return message

        color = {
            "normal_teb": (0.3, 0.7, 0.3),
            "gate_armed": (1.0, 0.65, 0.0),
            "align": (0.2, 0.6, 1.0),
            "crossing": (0.0, 1.0, 0.3),
            "exit_clear": (0.0, 0.8, 0.8),
            "stop_and_release": (0.7, 0.7, 1.0),
            "failed_stop": (1.0, 0.0, 0.0),
        }.get(decision.state, (0.8, 0.8, 0.8))

        pair = Marker()
        pair.header.frame_id = self.fixed_frame
        pair.header.stamp = now
        pair.ns = "narrow_gate_pair"
        pair.id = 1
        pair.type = Marker.LINE_STRIP
        pair.action = Marker.ADD
        pair.scale.x = 0.018
        pair.color.r, pair.color.g, pair.color.b = color
        pair.color.a = 0.95
        pair.points = [
            self._point(*frame.disc_first_xy),
            self._point(*frame.disc_second_xy),
        ]
        message.markers.append(pair)

        approach = Marker()
        approach.header.frame_id = self.fixed_frame
        approach.header.stamp = now
        approach.ns = "narrow_gate_travel"
        approach.id = 2
        approach.type = Marker.ARROW
        approach.action = Marker.ADD
        approach.scale.x = 0.018
        approach.scale.y = 0.040
        approach.scale.z = 0.060
        approach.color.r, approach.color.g, approach.color.b = color
        approach.color.a = 0.95
        approach.points = [
            self._point(
                frame.midpoint_xy[0]
                - self.config.prealign_distance_m
                * frame.travel_xy[0],
                frame.midpoint_xy[1]
                - self.config.prealign_distance_m
                * frame.travel_xy[1],
                0.06,
            ),
            self._point(
                frame.midpoint_xy[0]
                + self.config.exit_distance_m * frame.travel_xy[0],
                frame.midpoint_xy[1]
                + self.config.exit_distance_m * frame.travel_xy[1],
                0.06,
            ),
        ]
        message.markers.append(approach)

        midpoint = Marker()
        midpoint.header.frame_id = self.fixed_frame
        midpoint.header.stamp = now
        midpoint.ns = "narrow_gate_midpoint"
        midpoint.id = 3
        midpoint.type = Marker.SPHERE
        midpoint.action = Marker.ADD
        midpoint.pose.position = self._point(
            *frame.midpoint_xy, z_value=0.04
        )
        midpoint.pose.orientation.w = 1.0
        midpoint.scale.x = midpoint.scale.y = midpoint.scale.z = 0.05
        midpoint.color.r, midpoint.color.g, midpoint.color.b = color
        midpoint.color.a = 0.95
        message.markers.append(midpoint)

        label = Marker()
        label.header.frame_id = self.fixed_frame
        label.header.stamp = now
        label.ns = "narrow_gate_state"
        label.id = 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = self._point(
            frame.midpoint_xy[0],
            frame.midpoint_xy[1],
            z_value=0.18,
        )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.08
        label.color.r, label.color.g, label.color.b = color
        label.color.a = 1.0
        label.text = "{} gap={:.3f}m".format(
            decision.state, frame.edge_gap_m
        )
        message.markers.append(label)
        return message

    def _publish_status(
        self,
        now_sec,
        decision,
        robot_fresh,
        disc_fresh,
        path_fresh,
        pair_live,
    ):
        status = {
            "stamp": now_sec,
            "enabled": self.enabled,
            "state": decision.state,
            "authority_requested": decision.authority_requested,
            "safe_to_move": (
                decision.state != STATE_FAILED_STOP
                and robot_fresh
                and disc_fresh
                and (
                    pair_live
                    or not decision.authority_requested
                )
            ),
            "reason": decision.reason,
            "command_fresh": True,
            "robot_pose_fresh": robot_fresh,
            "disc_status_fresh": disc_fresh,
            "global_path_fresh": path_fresh,
            "locked_pair_live": pair_live,
            "navigation_active": self.navigation_active,
            "candidate_stable_sec": decision.candidate_stable_sec,
            "trigger_edge_gap_m": self.config.trigger_edge_gap_m,
            "release_edge_gap_m": self.config.release_edge_gap_m,
            "required_predicted_clearance_m": (
                self.config.minimum_physical_clearance_m
                + self.config.geometry_uncertainty_m
            ),
            "cancel_count": self.cancel_count,
            "resume_count": self.resume_count,
            "resume_goal_available": self.resume_goal is not None,
            "marker_topic": self.marker_topic,
            "frame": self._frame_status(decision.frame),
            "errors": self._error_status(decision.errors),
            "command": {
                "linear_x": decision.command_body[0],
                "linear_y": decision.command_body[1],
                "angular_z": decision.command_body[2],
            },
        }
        self.status_pub.publish(
            String(
                data=json.dumps(
                    status, ensure_ascii=False, allow_nan=False
                )
            )
        )

    def _timer_callback(self, _event):
        with self.lock:
            now_sec = rospy.Time.now().to_sec()
            robot_fresh = (
                self.robot_pose is not None
                and self.robot_receipt_sec is not None
                and now_sec - self.robot_receipt_sec
                <= self.robot_timeout_sec
            )
            disc_fresh = (
                self.disc_receipt_sec is not None
                and now_sec - self.disc_receipt_sec
                <= self.disc_timeout_sec
                and self.disc_input_fresh
                and self.disc_fixed_transform_available
            )
            path_fresh = (
                self.path_receipt_sec is not None
                and now_sec - self.path_receipt_sec
                <= self.path_timeout_sec
                and len(self.path) >= 2
            )
            pair_live = True
            if (
                self.machine.authority_requested
                and self.machine.locked_frame is not None
            ):
                live_ids = {track.memory_id for track in self.tracks}
                pair_live = all(
                    memory_id in live_ids
                    for memory_id in self.machine.locked_frame.pair_ids
                )
            inputs_fresh = (
                robot_fresh
                and disc_fresh
                and pair_live
                and (
                    path_fresh
                    or self.machine.authority_requested
                )
            )
            decision = self.machine.update(
                now_sec=now_sec,
                robot=self.robot_pose if robot_fresh else None,
                tracks=self.tracks,
                path=self.path if path_fresh else (),
                navigation_active=(
                    self.enabled and self.navigation_active
                ),
                inputs_fresh=inputs_fresh,
            )

            if decision.cancel_move_base:
                if self.resume_goal is None and self.current_goal is not None:
                    self.resume_goal = copy.deepcopy(self.current_goal)
                if self.resume_goal is None:
                    self.machine.fail("resume_goal_missing")
                    decision = self.machine.update(
                        now_sec,
                        self.robot_pose if robot_fresh else None,
                        self.tracks,
                        (),
                        False,
                        inputs_fresh,
                    )
                self.cancel_pub.publish(GoalID())
                self.cancel_count += 1

            command = (
                decision.command_body
                if self.enabled and decision.authority_requested
                else (0.0, 0.0, 0.0)
            )
            self.command_pub.publish(self._twist(command))

            if decision.resume_move_base and self.resume_goal is not None:
                goal = copy.deepcopy(self.resume_goal)
                goal.header.stamp = rospy.Time.now()
                self.resume_pub.publish(goal)
                self.resume_count += 1

            self.last_decision = decision
            now = rospy.Time.from_sec(now_sec)
            self.marker_pub.publish(self._markers(now, decision))
            self._publish_status(
                now_sec,
                decision,
                robot_fresh,
                disc_fresh,
                path_fresh,
                pair_live,
            )

    def _shutdown(self):
        for _ in range(5):
            try:
                self.command_pub.publish(Twist())
                rospy.sleep(0.03)
            except Exception:
                break
def main():
    rospy.init_node("yellow_track_narrow_gate_controller")
    NarrowGateControllerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
