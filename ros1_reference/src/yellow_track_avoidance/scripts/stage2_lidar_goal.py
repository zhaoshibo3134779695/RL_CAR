#!/usr/bin/env python3
"""Freeze, preview and explicitly send the Stage-2 lidar terminal goal."""

import json
import math
import threading
from types import SimpleNamespace

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

from yellow_track_avoidance.stage2_lidar_goal import (
    build_stage2_geometry,
    voxel_downsample,
)
from yellow_track_avoidance.stage2_frozen_scene import goal_from_payload


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _normalize(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _transform_xy(transform, value):
    angle = _yaw(transform.transform.rotation)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        (
            transform.transform.translation.x + cosine * value[0] - sine * value[1],
            transform.transform.translation.y + sine * value[0] + cosine * value[1],
        ),
        dtype=np.float64,
    )


class Stage2LidarGoalNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.scan_topic = str(rospy.get_param("~scan_topic", "/scan"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link")).lstrip("/")
        self.target_frame = str(rospy.get_param("~target_frame", "map")).lstrip("/")
        self.goal_topic = str(rospy.get_param("~goal_topic", "/move_base_simple/goal"))
        self.frozen_geometry_topic = str(
            rospy.get_param("~frozen_geometry_topic", "")
        )
        self.marker_topic = str(
            rospy.get_param(
                "~marker_topic", "/yellow_track_avoidance/stage2_lidar_goal_markers"
            )
        )
        self.goal_pose_topic = str(
            rospy.get_param(
                "~goal_pose_topic", "/yellow_track_avoidance/stage2_lidar_goal"
            )
        )
        self.preview_topic = str(
            rospy.get_param(
                "~preview_plan_topic", "/yellow_track_avoidance/stage2_lidar_goal_plan"
            )
        )
        self.make_plan_service = str(
            rospy.get_param("~make_plan_service", "/move_base/make_plan")
        )
        self.scan_frame_count = max(1, int(rospy.get_param("~scan_frame_count", 40)))
        self.voxel_resolution = float(rospy.get_param("~voxel_resolution_m", 0.015))
        self.expected_width = float(rospy.get_param("~expected_width_m", 0.485))
        self.width_tolerance = float(rospy.get_param("~width_tolerance_m", 0.10))
        self.vehicle_front_offset = float(
            rospy.get_param("~vehicle_front_offset_m", 0.17)
        )
        self.vehicle_half_width = float(
            rospy.get_param("~vehicle_half_width_m", 0.165)
        )
        self.front_clearance = float(
            rospy.get_param("~desired_front_clearance_m", 0.10)
        )
        self.maximum_linear_speed = float(
            rospy.get_param("~maximum_capture_linear_speed_mps", 0.01)
        )
        self.maximum_angular_speed = float(
            rospy.get_param("~maximum_capture_angular_speed_radps", 0.03)
        )
        self.stationary_stable_sec = float(
            rospy.get_param("~stationary_stable_sec", 0.5)
        )
        self.auto_capture = bool(rospy.get_param("~auto_capture_when_ready", True))
        self.auto_send = bool(rospy.get_param("~auto_send_goal_when_ready", False))
        self.preview_retry_sec = max(
            0.2, float(rospy.get_param("~preview_plan_retry_sec", 1.0))
        )
        self.fit_options = {
            "minimum_x_m": float(rospy.get_param("~right_wall_minimum_x_m", 0.20)),
            "maximum_x_m": float(rospy.get_param("~right_wall_maximum_x_m", 3.50)),
            "maximum_y_m": float(rospy.get_param("~right_wall_maximum_y_m", -0.10)),
            "ransac_threshold_m": float(
                rospy.get_param("~right_wall_ransac_threshold_m", 0.025)
            ),
            "minimum_inliers": int(
                rospy.get_param("~right_wall_minimum_inliers", 18)
            ),
            "terminal_minimum_forward_m": float(
                rospy.get_param("~terminal_minimum_forward_m", 1.0)
            ),
            "terminal_cluster_gap_m": float(
                rospy.get_param("~terminal_cluster_gap_m", 0.08)
            ),
            "terminal_minimum_support": int(
                rospy.get_param("~terminal_minimum_support", 8)
            ),
            "terminal_minimum_span_m": float(
                rospy.get_param("~terminal_minimum_span_m", 0.22)
            ),
        }

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.latest_odom = None
        self.stationary_since = None
        self.frames = []
        self.capturing = False
        self.processing = False
        self.capture_stamp = None
        self.geometry_base = None
        self.fixed_geometry = None
        self.goal_pose = None
        self.goal_sent = False
        self.preview_pose_count = 0
        self.last_preview_attempt = None
        self.last_error = None

        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1, latch=True
        )
        self.goal_pose_pub = rospy.Publisher(
            self.goal_pose_topic, PoseStamped, queue_size=1, latch=True
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
        self.scan_sub = None
        self.frozen_geometry_sub = None
        if self.frozen_geometry_topic:
            # Phase 2 must reuse the map captured in Phase 1.  Re-fitting a new
            # scene here could produce a goal inconsistent with the static map.
            self.auto_capture = False
            self.frozen_geometry_sub = rospy.Subscriber(
                self.frozen_geometry_topic,
                String,
                self._frozen_geometry_callback,
                queue_size=1,
            )
        else:
            self.scan_sub = rospy.Subscriber(
                self.scan_topic, LaserScan, self._scan_callback, queue_size=30
            )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_callback, queue_size=1
        )
        self.timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)
        rospy.loginfo(
            "Stage-2 lidar goal node ready: %s, front clearance %.3f m; no cmd_vel publisher",
            "frozen geometry"
            if self.frozen_geometry_topic
            else "%d scans" % self.scan_frame_count,
            self.front_clearance,
        )

    def _frozen_geometry_callback(self, message):
        with self.lock:
            if self.goal_pose is not None:
                return
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("frozen geometry is not an object")
            frame_id = str(payload.get("frame_id", "")).lstrip("/")
            if frame_id != self.target_frame:
                raise ValueError(
                    "frozen geometry frame %r does not match %r"
                    % (frame_id, self.target_frame)
                )
            goal_xy, heading = goal_from_payload(
                payload,
                vehicle_front_offset_m=self.vehicle_front_offset,
                front_clearance_m=self.front_clearance,
            )
            fixed = {}
            for key in (
                "left_wall_start",
                "left_wall_end",
                "right_wall_start",
                "right_wall_end",
                "terminal_right",
                "terminal_left",
                "terminal_center",
                "outer_corner",
            ):
                fixed[key] = np.asarray(payload[key], dtype=np.float64)
            tangent = np.asarray(
                (math.cos(heading), math.sin(heading)), dtype=np.float64
            )
            fixed["heading"] = heading
            fixed["tangent"] = tangent
            fixed["normal"] = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
            fixed["goal"] = goal_xy
            fixed["goal_base_center"] = goal_xy
            fixed["front"] = goal_xy + self.vehicle_front_offset * tangent
            fixed["goal_front_center"] = fixed["front"]
            geometry = SimpleNamespace(
                measured_width_m=float(payload["measured_width_m"]),
                terminal_support_voxels=int(payload["terminal_support_voxels"]),
                terminal_rms_m=float(payload["terminal_rms_m"]),
            )
            goal = self._goal_message(rospy.Time.now(), goal_xy, heading)
        except (KeyError, TypeError, ValueError) as error:
            with self.lock:
                self.last_error = "invalid frozen geometry: %s" % error
            rospy.logerr_throttle(1.0, "%s", self.last_error)
            return
        with self.lock:
            self.geometry_base = geometry
            self.fixed_geometry = fixed
            self.goal_pose = goal
            self.last_error = None
        self.goal_pose_pub.publish(goal)
        self.marker_pub.publish(self._markers(rospy.Time.now(), geometry, fixed))
        self._publish_preview_plan(goal)
        rospy.loginfo(
            "Stage-2 goal derived from frozen map: (%.3f, %.3f), yaw %.2f deg",
            goal.pose.position.x,
            goal.pose.position.y,
            math.degrees(heading),
        )

    def _odom_callback(self, message):
        with self.lock:
            self.latest_odom = message

    def _stationary(self):
        if self.latest_odom is None:
            return False
        twist = self.latest_odom.twist.twist
        return (
            math.hypot(twist.linear.x, twist.linear.y) <= self.maximum_linear_speed
            and abs(twist.angular.z) <= self.maximum_angular_speed
        )

    def _start_capture(self):
        self.frames = []
        self.capturing = True
        self.processing = False
        self.capture_stamp = None
        self.last_error = None
        rospy.loginfo("Stage-2 lidar capture started (%d frames)", self.scan_frame_count)

    def _capture_service(self, _request):
        with self.lock:
            if self.frozen_geometry_topic:
                return TriggerResponse(
                    False, "goal mode reuses Phase-1 frozen geometry; recapture disabled"
                )
            if self.goal_pose is not None:
                return TriggerResponse(True, "Stage-2 goal is already frozen")
            if self.capturing or self.processing:
                return TriggerResponse(True, "Stage-2 capture is already running")
            if not self._stationary():
                return TriggerResponse(False, "robot odometry is not stationary")
            self._start_capture()
        return TriggerResponse(True, "Stage-2 lidar capture started")

    def _reset_service(self, _request):
        with self.lock:
            self.frames = []
            self.capturing = False
            self.processing = False
            self.capture_stamp = None
            self.geometry_base = None
            self.fixed_geometry = None
            self.goal_pose = None
            self.goal_sent = False
            self.preview_pose_count = 0
            self.last_error = None
        return TriggerResponse(True, "Stage-2 lidar goal cleared")

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
        laser = np.column_stack(
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
        return laser @ rotation.T + translation

    def _scan_callback(self, message):
        with self.lock:
            if not self.capturing or self.goal_pose is not None:
                return
            if not self._stationary():
                self.frames = []
                self.capturing = False
                self.last_error = "robot moved during Stage-2 lidar capture"
                return
        try:
            points = self._scan_points(message)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(1.0, "waiting for Stage-2 lidar TF: %s", error)
            return
        with self.lock:
            if not self.capturing:
                return
            self.frames.append(points)
            self.capture_stamp = message.header.stamp
            if len(self.frames) < self.scan_frame_count:
                return
            frames = self.frames
            stamp = self.capture_stamp
            self.capturing = False
            self.processing = True
        self._finish_capture(frames, stamp)

    def _finish_capture(self, frames, stamp):
        try:
            points = voxel_downsample(np.vstack(frames), self.voxel_resolution)
            geometry = build_stage2_geometry(
                points,
                expected_width_m=self.expected_width,
                width_tolerance_m=self.width_tolerance,
                vehicle_front_offset_m=self.vehicle_front_offset,
                desired_front_clearance_m=self.front_clearance,
                **self.fit_options
            )
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self.base_frame, stamp, rospy.Duration(0.3)
            )
            fixed = self._fixed_geometry(transform, geometry)
            goal = self._goal_message(stamp, fixed["goal"], fixed["heading"])
        except Exception as error:
            with self.lock:
                self.frames = []
                self.processing = False
                self.last_error = str(error)
            rospy.logerr("Stage-2 lidar goal capture failed: %s", error)
            return
        with self.lock:
            self.frames = []
            self.processing = False
            self.geometry_base = geometry
            self.fixed_geometry = fixed
            self.goal_pose = goal
            self.last_error = None
        self.goal_pose_pub.publish(goal)
        self.marker_pub.publish(self._markers(rospy.Time.now(), geometry, fixed))
        self._publish_preview_plan(goal)
        rospy.loginfo(
            "Stage-2 goal frozen in %s: (%.3f, %.3f), yaw %.2f deg; width %.3f m, terminal support %d",
            self.target_frame,
            goal.pose.position.x,
            goal.pose.position.y,
            math.degrees(fixed["heading"]),
            geometry.measured_width_m,
            geometry.terminal_support_voxels,
        )

    def _fixed_geometry(self, transform, geometry):
        transform_yaw = _yaw(transform.transform.rotation)
        keys = (
            "left_wall_start",
            "left_wall_end",
            "right_wall_start",
            "right_wall_end",
            "terminal_right",
            "terminal_left",
            "terminal_center",
            "outer_corner",
            "goal_base_center",
            "goal_front_center",
        )
        output = {key: _transform_xy(transform, getattr(geometry, key)) for key in keys}
        output["goal"] = output["goal_base_center"]
        output["front"] = output["goal_front_center"]
        output["heading"] = _normalize(transform_yaw + geometry.heading_rad)
        output["tangent"] = np.asarray(
            (math.cos(output["heading"]), math.sin(output["heading"])), dtype=np.float64
        )
        output["normal"] = np.asarray(
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
        value = Point()
        value.x, value.y, value.z = float(xy[0]), float(xy[1]), float(z)
        return value

    @staticmethod
    def _color(red, green, blue, alpha=1.0):
        return ColorRGBA(red, green, blue, alpha)

    def _line(self, stamp, marker_id, namespace, first, second, color, width=0.04):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = self.target_frame, stamp
        marker.ns, marker.id = namespace, marker_id
        marker.type, marker.action = Marker.LINE_LIST, Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.color = width, color
        marker.points = [self._point(first), self._point(second)]
        return marker

    def _sphere(self, stamp, marker_id, namespace, xy, color, size=0.10):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = self.target_frame, stamp
        marker.ns, marker.id = namespace, marker_id
        marker.type, marker.action = Marker.SPHERE, Marker.ADD
        marker.pose.position = self._point(xy, 0.12)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = size
        marker.color = color
        return marker

    def _text(self, stamp, marker_id, xy, text, color):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = self.target_frame, stamp
        marker.ns, marker.id = "stage2_labels", marker_id
        marker.type, marker.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        marker.pose.position = self._point(xy, 0.28)
        marker.pose.orientation.w = 1.0
        marker.scale.z, marker.color, marker.text = 0.09, color, text
        return marker

    def _markers(self, stamp, geometry, fixed):
        green = self._color(0.1, 1.0, 0.2)
        cyan = self._color(0.0, 0.9, 1.0)
        purple = self._color(0.75, 0.2, 1.0)
        yellow = self._color(1.0, 0.9, 0.0)
        orange = self._color(1.0, 0.35, 0.0)
        magenta = self._color(1.0, 0.05, 0.75)
        blue = self._color(0.05, 0.25, 1.0)
        red = self._color(1.0, 0.1, 0.1)
        dark = self._color(0.05, 0.05, 0.05)
        markers = [
            self._line(stamp, 1, "stage2_right_wall", fixed["right_wall_start"], fixed["right_wall_end"], green),
            self._line(stamp, 2, "stage2_left_wall", fixed["left_wall_start"], fixed["left_wall_end"], cyan),
            self._line(stamp, 3, "stage2_terminal_wall", fixed["terminal_right"], fixed["terminal_left"], purple, 0.055),
            self._sphere(stamp, 4, "stage2_terminal", fixed["terminal_center"], yellow, 0.09),
            self._sphere(stamp, 5, "stage2_outer_corner", fixed["outer_corner"], orange, 0.12),
            self._line(stamp, 6, "stage2_front_clearance", fixed["front"], fixed["terminal_center"], red, 0.025),
        ]
        tangent, normal, goal = fixed["tangent"], fixed["normal"], fixed["goal"]
        corners = [
            goal + self.vehicle_front_offset * tangent + self.vehicle_half_width * normal,
            goal + self.vehicle_front_offset * tangent - self.vehicle_half_width * normal,
            goal - self.vehicle_front_offset * tangent - self.vehicle_half_width * normal,
            goal - self.vehicle_front_offset * tangent + self.vehicle_half_width * normal,
        ]
        footprint = Marker()
        footprint.header.frame_id, footprint.header.stamp = self.target_frame, stamp
        footprint.ns, footprint.id = "stage2_goal_vehicle", 7
        footprint.type, footprint.action = Marker.LINE_STRIP, Marker.ADD
        footprint.pose.orientation.w = 1.0
        footprint.scale.x, footprint.color = 0.035, blue
        footprint.points = [self._point(value, 0.12) for value in corners + [corners[0]]]
        markers.append(footprint)
        arrow = Marker()
        arrow.header.frame_id, arrow.header.stamp = self.target_frame, stamp
        arrow.ns, arrow.id = "stage2_goal_arrow", 8
        arrow.type, arrow.action = Marker.ARROW, Marker.ADD
        arrow.pose.orientation.w = 1.0
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.055, 0.10, 0.12
        arrow.color = magenta
        arrow.points = [self._point(goal, 0.15), self._point(goal + 0.16 * tangent, 0.15)]
        markers.append(arrow)
        markers.extend(
            (
                self._text(stamp, 10, fixed["terminal_center"] + 0.25 * normal, "窄道末端截面", yellow),
                self._text(stamp, 11, fixed["outer_corner"] - 0.18 * normal, "外侧拐角", orange),
                self._text(
                    stamp,
                    12,
                    goal + 0.42 * normal,
                    "Stage 2 雷达终点  车头预留 %.2fm\n宽 %.3fm  yaw %.2f°  末端支持 %d"
                    % (
                        self.front_clearance,
                        geometry.measured_width_m,
                        math.degrees(fixed["heading"]),
                        geometry.terminal_support_voxels,
                    ),
                    dark,
                ),
            )
        )
        return MarkerArray(markers=markers)

    def _publish_preview_plan(self, goal):
        with self.lock:
            self.last_preview_attempt = rospy.Time.now()
        try:
            robot = self.tf_buffer.lookup_transform(
                self.target_frame, self.base_frame, rospy.Time(0), rospy.Duration(0.3)
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
            rospy.logwarn("Stage-2 make_plan preview unavailable: %s", error)
            return
        if not response.plan.poses:
            rospy.logwarn("Stage-2 make_plan returned an empty path")
            return
        response.plan.header.stamp = rospy.Time.now()
        self.preview_pub.publish(response.plan)
        with self.lock:
            self.preview_pose_count = len(response.plan.poses)

    def _send_goal_service(self, _request):
        with self.lock:
            goal = self.goal_pose
        if goal is None:
            return TriggerResponse(False, "Stage-2 lidar goal is not frozen")
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
            "Stage-2 goal sent: (%.3f, %.3f), yaw %.3f rad"
            % (
                message.pose.position.x,
                message.pose.position.y,
                2.0 * math.atan2(message.pose.orientation.z, message.pose.orientation.w),
            ),
        )

    def _timer_callback(self, _event):
        now = rospy.Time.now()
        with self.lock:
            stationary = self._stationary()
            if stationary:
                if self.stationary_since is None:
                    self.stationary_since = now
            else:
                self.stationary_since = None
            stable = (
                stationary
                and self.stationary_since is not None
                and (now - self.stationary_since).to_sec() >= self.stationary_stable_sec
            )
            if (
                self.auto_capture
                and stable
                and self.goal_pose is None
                and not self.capturing
                and not self.processing
            ):
                self._start_capture()
            if self.processing:
                state = "processing_scan"
            elif self.capturing:
                state = "collecting_scan"
            elif self.goal_pose is not None:
                state = "goal_sent" if self.goal_sent else "goal_ready"
            elif self.last_error:
                state = "capture_failed"
            else:
                state = "waiting_stationary"
            goal = self.goal_pose
            geometry = self.geometry_base
            fixed = self.fixed_geometry
            retry_preview = (
                goal is not None
                and self.preview_pose_count <= 1
                and (
                    self.last_preview_attempt is None
                    or (now - self.last_preview_attempt).to_sec()
                    >= self.preview_retry_sec
                )
            )
            status = {
                "stamp": now.to_sec(),
                "state": state,
                "stationary": stationary,
                "stationary_stable": stable,
                "scan_frames_collected": len(self.frames),
                "scan_frames_required": self.scan_frame_count,
                "goal_frozen": goal is not None,
                "goal_sent": self.goal_sent,
                "publishes_cmd_vel": False,
                "preview_plan_pose_count": self.preview_pose_count,
                "capture_service": rospy.resolve_name("~capture"),
                "send_goal_service": rospy.resolve_name("~send_goal"),
                "last_error": self.last_error,
                "goal": None
                if goal is None
                else {
                    "frame_id": goal.header.frame_id,
                    "xy_m": [goal.pose.position.x, goal.pose.position.y],
                    "yaw_rad": fixed["heading"],
                    "front_clearance_m": self.front_clearance,
                    "measured_width_m": geometry.measured_width_m,
                    "terminal_support_voxels": geometry.terminal_support_voxels,
                    "terminal_rms_m": geometry.terminal_rms_m,
                },
            }
        if goal is not None and geometry is not None and fixed is not None:
            self.goal_pose_pub.publish(goal)
            self.marker_pub.publish(self._markers(now, geometry, fixed))
        if retry_preview:
            self._publish_preview_plan(goal)
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))
        if self.auto_send and goal is not None and not self.goal_sent:
            self._send_goal_service(None)


def main():
    rospy.init_node("yellow_track_stage2_lidar_goal")
    Stage2LidarGoalNode()
    rospy.spin()


if __name__ == "__main__":
    main()
