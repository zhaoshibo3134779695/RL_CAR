#!/usr/bin/env python3
"""Follow Sebastian's frozen Stage-1 global path without TEB velocity control."""

import bisect
import copy
import json
import math
import threading
import time

import rosgraph
import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import MarkerArray


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def move_toward(current, target, maximum_change):
    return current + clamp(target - current, -maximum_change, maximum_change)


def path_geometry(message, minimum_spacing=0.003):
    points = []
    for pose in message.poses:
        point = (float(pose.pose.position.x), float(pose.pose.position.y))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("path contains a non-finite point")
        if not points or math.hypot(
                point[0] - points[-1][0], point[1] - points[-1][1]
        ) >= minimum_spacing:
            points.append(point)
    if len(points) < 2:
        raise ValueError("path must contain at least two distinct points")

    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(second[0] - first[0], second[1] - first[1])
        )
    if cumulative[-1] < 0.20:
        raise ValueError("path is shorter than 0.20 m")

    quaternion = message.poses[-1].pose.orientation
    norm = math.hypot(quaternion.z, quaternion.w)
    if norm > 1e-6:
        final_yaw = math.atan2(
            2.0 * quaternion.w * quaternion.z,
            1.0 - 2.0 * quaternion.z * quaternion.z,
        )
    else:
        final_yaw = math.atan2(
            points[-1][1] - points[-2][1],
            points[-1][0] - points[-2][0],
        )
    return points, cumulative, final_yaw


def path_frame_id(message):
    frames = []
    if message.header.frame_id:
        frames.append(message.header.frame_id.lstrip("/"))
    frames.extend(
        pose.header.frame_id.lstrip("/")
        for pose in message.poses
        if pose.header.frame_id
    )
    unique_frames = set(frames)
    if not unique_frames:
        raise ValueError("path does not declare a frame_id")
    if len(unique_frames) != 1:
        raise ValueError("path contains mixed frame_ids: %s" % sorted(unique_frames))
    return unique_frames.pop()


def offset_path_tail(message, final_yaw, lateral_offset, blend_distance):
    adjusted = copy.deepcopy(message)
    if abs(lateral_offset) < 1e-9:
        return adjusted

    raw_points = [
        (float(pose.pose.position.x), float(pose.pose.position.y))
        for pose in adjusted.poses
    ]
    cumulative = [0.0]
    for first, second in zip(raw_points, raw_points[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(second[0] - first[0], second[1] - first[1])
        )
    total_length = cumulative[-1]
    effective_blend = min(max(blend_distance, 0.01), total_length)
    blend_start = total_length - effective_blend
    left_x = -math.sin(final_yaw)
    left_y = math.cos(final_yaw)

    for pose, distance in zip(adjusted.poses, cumulative):
        ratio = clamp((distance - blend_start) / effective_blend, 0.0, 1.0)
        weight = ratio * ratio * (3.0 - 2.0 * ratio)
        pose.pose.position.x += left_x * lateral_offset * weight
        pose.pose.position.y += left_y * lateral_offset * weight
    return adjusted


def extend_path_to_terminal(
        message,
        final_yaw,
        terminal_xy,
        vehicle_front_offset,
        terminal_forward_offset=0.0,
        spacing=0.01,
):
    adjusted = copy.deepcopy(message)
    if not adjusted.poses:
        raise ValueError("cannot extend an empty path")
    tangent_x = math.cos(final_yaw)
    tangent_y = math.sin(final_yaw)
    endpoint = adjusted.poses[-1].pose.position
    center_offset = terminal_forward_offset - vehicle_front_offset
    center_target_x = float(terminal_xy[0]) + center_offset * tangent_x
    center_target_y = float(terminal_xy[1]) + center_offset * tangent_y
    extension = (
        (center_target_x - endpoint.x) * tangent_x
        + (center_target_y - endpoint.y) * tangent_y
    )
    if extension <= 0.02:
        raise ValueError(
            "terminal marker is not ahead of the path endpoint (%.3f m)" % extension
        )
    if extension > 2.50:
        raise ValueError("terminal extension is implausibly long (%.3f m)" % extension)

    count = max(2, int(math.ceil(extension / max(spacing, 0.005))))
    start_x, start_y = float(endpoint.x), float(endpoint.y)
    template = adjusted.poses[-1]
    for index in range(1, count + 1):
        distance = extension * index / count
        pose = copy.deepcopy(template)
        pose.pose.position.x = start_x + distance * tangent_x
        pose.pose.position.y = start_y + distance * tangent_y
        adjusted.poses.append(pose)
    return adjusted, extension


def rectangle_disc_clearance(forward, left, half_length, half_width, radius):
    outside_x = max(abs(forward) - half_length, 0.0)
    outside_y = max(abs(left) - half_width, 0.0)
    return math.hypot(outside_x, outside_y) - radius


class Stage1DirectPathFollower:
    def __init__(self):
        self.lock = threading.RLock()
        self.path_topic = str(rospy.get_param(
            "~path_topic", "/yellow_track_avoidance/stage1_lidar_goal_plan"
        ))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.disc_status_topic = str(rospy.get_param(
            "~disc_status_topic", "/yellow_track_disc_costmap_bridge/status"
        ))
        self.terminal_marker_topic = str(rospy.get_param(
            "~terminal_marker_topic",
            "/yellow_track_avoidance/stage1_lidar_goal_markers",
        ))
        self.command_topic = str(rospy.get_param(
            "~command_topic", "/cmd_vel_stage1_direct"
        ))
        self.base_frame = str(rospy.get_param(
            "~base_frame", "base_footprint"
        )).lstrip("/")
        self.required_command_subscriber = str(rospy.get_param(
            "~required_command_subscriber", "/direct_cmd_vel_listener"
        ))
        self.enable_motion = bool(rospy.get_param("~enable_motion", False))
        self.auto_start = bool(rospy.get_param("~auto_start", False))
        legacy_require_two_discs = bool(rospy.get_param(
            "~require_two_confirmed_discs", True
        ))
        default_minimum_disc_count = 2 if legacy_require_two_discs else 0
        self.minimum_disc_count = max(
            0,
            int(rospy.get_param(
                "~minimum_confirmed_disc_count", default_minimum_disc_count
            )),
        )
        self.extend_to_terminal = bool(rospy.get_param(
            "~extend_to_terminal_marker", False
        ))

        self.control_rate = float(rospy.get_param("~control_rate_hz", 30.0))
        self.lookahead = float(rospy.get_param("~lookahead_m", 0.18))
        self.max_linear = float(rospy.get_param("~max_linear_mps", 0.12))
        self.terminal_max_linear = float(rospy.get_param(
            "~terminal_max_linear_mps", self.max_linear
        ))
        self.min_linear = float(rospy.get_param("~min_linear_mps", 0.035))
        self.max_angular = float(rospy.get_param("~max_angular_radps", 0.24))
        self.max_linear_accel = float(rospy.get_param(
            "~max_linear_accel_mps2", 0.18
        ))
        self.max_angular_accel = float(rospy.get_param(
            "~max_angular_accel_radps2", 0.35
        ))
        self.rotate_only_angle = math.radians(float(rospy.get_param(
            "~rotate_only_angle_deg", 32.0
        )))
        self.position_tolerance = float(rospy.get_param(
            "~position_tolerance_m", 0.045
        ))
        self.yaw_tolerance = math.radians(float(rospy.get_param(
            "~yaw_tolerance_deg", 3.0
        )))
        self.max_cross_track = float(rospy.get_param(
            "~maximum_cross_track_error_m", 0.08
        ))
        self.minimum_disc_clearance = float(rospy.get_param(
            "~minimum_disc_clearance_m", 0.02
        ))
        self.stop_on_disc_clearance_violation = bool(rospy.get_param(
            "~stop_on_disc_clearance_violation", True
        ))
        self.final_lateral_offset = float(rospy.get_param(
            "~final_lateral_offset_m", 0.0
        ))
        self.final_offset_blend_distance = float(rospy.get_param(
            "~final_offset_blend_distance_m", 0.30
        ))
        self.vehicle_half_length = float(rospy.get_param(
            "~vehicle_half_length_m", 0.17
        ))
        self.vehicle_half_width = float(rospy.get_param(
            "~vehicle_half_width_m", 0.165
        ))
        self.vehicle_front_offset = float(rospy.get_param(
            "~vehicle_front_offset_m", 0.17
        ))
        self.terminal_forward_offset = float(rospy.get_param(
            "~terminal_forward_offset_m", 0.0
        ))
        self.default_disc_radius = float(rospy.get_param(
            "~physical_disc_radius_m", 0.05
        ))
        self.odom_timeout = float(rospy.get_param("~odom_timeout_sec", 0.40))
        self.disc_timeout = float(rospy.get_param(
            "~disc_status_timeout_sec", 0.50
        ))
        self.stall_timeout = float(rospy.get_param("~stall_timeout_sec", 3.0))
        self.mission_timeout = float(rospy.get_param(
            "~mission_timeout_sec", 120.0
        ))

        if self.control_rate <= 0.0 or self.lookahead <= 0.05:
            raise ValueError("control rate and lookahead must be positive")
        if not 0.0 < self.min_linear <= self.max_linear:
            raise ValueError("linear speed limits are invalid")
        if self.terminal_max_linear < self.min_linear:
            raise ValueError("terminal linear speed limit is invalid")
        if self.max_angular <= 0.0:
            raise ValueError("maximum angular speed must be positive")
        if abs(self.final_lateral_offset) > 0.10:
            raise ValueError("final lateral offset must not exceed 0.10 m")
        if not 0.0 <= self.terminal_forward_offset <= 1.0:
            raise ValueError("terminal forward offset must be between 0 and 1.0 m")

        self.path_message = None
        self.path_frame = None
        self.points = ()
        self.cumulative = ()
        self.final_yaw = 0.0
        self.progress_index = 0
        self.progress_s = 0.0
        self.cross_track = None
        self.remaining = None
        self.latest_odom_receipt = None
        self.odom_linear_speed = None
        self.odom_angular_speed = None
        self.latest_disc_receipt = None
        self.disc_tracks = ()
        self.disc_radius = self.default_disc_radius
        self.disc_positions_frozen = False
        self.costmap_disc_count = 0
        self.costmap_writes_enabled = False
        self.locked_disc_ids = None
        self.minimum_observed_disc_clearance = None
        self.terminal_target_xy = None
        self.terminal_target_frame = None
        self.terminal_extension = None
        self.terminal_speed_active = False
        self.pending_path_message = None
        self.state = "WAITING_PATH"
        self.reason = "waiting_for_preview_path"
        self.started_at = None
        self.last_progress_at = None
        self.last_progress_s = 0.0
        self.last_tick = time.monotonic()
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.stable_goal_cycles = 0
        self.auto_start_attempted = False
        self.last_status_publish = 0.0
        self.last_topology_check = 0.0
        self.topology_reason = "not_checked"

        self.tf_listener = tf.TransformListener()
        self.command_pub = rospy.Publisher(
            self.command_topic, Twist, queue_size=1
        )
        self.path_pub = rospy.Publisher("~path", Path, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(
            "~status", String, queue_size=1, latch=True
        )
        rospy.Subscriber(self.path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(
            self.disc_status_topic, String, self.disc_status_callback, queue_size=1
        )
        rospy.Subscriber(
            self.terminal_marker_topic,
            MarkerArray,
            self.terminal_marker_callback,
            queue_size=1,
        )
        rospy.Service("~start", Trigger, self.start_service)
        rospy.Service("~check", Trigger, self.check_service)
        rospy.Service("~stop", Trigger, self.stop_service)
        rospy.Service("~reset", Trigger, self.reset_service)
        rospy.on_shutdown(self.shutdown)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self.timer_callback
        )
        rospy.loginfo(
            "Stage-1 direct follower ready: path=%s output=%s motion=%s; "
            "linear.y is always zero",
            self.path_topic,
            self.command_topic,
            self.enable_motion,
        )

    def path_callback(self, message):
        try:
            frame_id = path_frame_id(message)
            points, cumulative, final_yaw = path_geometry(message)
            if self.extend_to_terminal:
                with self.lock:
                    terminal_xy = self.terminal_target_xy
                    terminal_frame = self.terminal_target_frame
                if terminal_xy is None:
                    with self.lock:
                        self.pending_path_message = copy.deepcopy(message)
                        self.reason = "waiting_for_terminal_marker"
                    rospy.logwarn_throttle(
                        2.0, "Stage-1 direct follower waiting for terminal marker"
                    )
                    return
                if terminal_frame != frame_id:
                    raise ValueError(
                        "terminal marker frame %s does not match path frame %s"
                        % (terminal_frame, frame_id)
                    )
            message = offset_path_tail(
                message,
                final_yaw,
                self.final_lateral_offset,
                self.final_offset_blend_distance,
            )
            terminal_extension = None
            if self.extend_to_terminal:
                message, terminal_extension = extend_path_to_terminal(
                    message,
                    final_yaw,
                    terminal_xy,
                    self.vehicle_front_offset,
                    self.terminal_forward_offset,
                )
            points, cumulative, final_yaw = path_geometry(message)
        except ValueError as error:
            rospy.logwarn_throttle(2.0, "Ignoring invalid Stage-1 path: %s", error)
            return
        with self.lock:
            if self.state == "RUNNING":
                return
            if self.points:
                return
            message.header.frame_id = frame_id
            self.path_message = message
            self.path_frame = frame_id
            self.points = tuple(points)
            self.cumulative = tuple(cumulative)
            self.final_yaw = final_yaw
            self.terminal_extension = terminal_extension
            self.terminal_speed_active = False
            self.pending_path_message = None
            self.progress_index = 0
            self.progress_s = 0.0
            self.remaining = cumulative[-1]
            self.state = "READY"
            self.reason = "preview_path_ready"
            self.path_pub.publish(message)
            rospy.loginfo(
                "Locked Stage-1 preview path: %d points, %.3f m, frame=%s, "
                "final left offset=%+.3f m, terminal extension=%s",
                len(points), cumulative[-1], self.path_frame,
                self.final_lateral_offset,
                "disabled" if terminal_extension is None else "%.3f m" % terminal_extension,
            )

    def terminal_marker_callback(self, message):
        target = None
        frame_id = None
        for marker in message.markers:
            if marker.ns == "terminal_corners" and marker.id == 17:
                target = (float(marker.pose.position.x), float(marker.pose.position.y))
                frame_id = marker.header.frame_id.lstrip("/")
                break
        if target is None or not frame_id:
            return
        with self.lock:
            self.terminal_target_xy = target
            self.terminal_target_frame = frame_id
            pending = self.pending_path_message
            self.pending_path_message = None
        if pending is not None:
            self.path_callback(pending)

    def odom_callback(self, message):
        with self.lock:
            self.latest_odom_receipt = rospy.Time.now()
            twist = message.twist.twist
            self.odom_linear_speed = math.hypot(twist.linear.x, twist.linear.y)
            self.odom_angular_speed = abs(twist.angular.z)

    def disc_status_callback(self, message):
        try:
            payload = json.loads(message.data)
            tracks = []
            for track in payload.get("active_tracks", []):
                base_xy = track.get("base_xy", ())
                if len(base_xy) != 2 or not bool(track.get("confirmed", False)):
                    continue
                forward, left = float(base_xy[0]), float(base_xy[1])
                if math.isfinite(forward) and math.isfinite(left):
                    tracks.append((forward, left, int(track.get("memory_id", -1))))
            radius = float(payload.get(
                "physical_disc_radius_m", self.default_disc_radius
            ))
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError("invalid physical disc radius")
            positions_frozen = bool(
                payload.get("confirmed_positions_frozen", False)
            )
            costmap_disc_count = int(payload.get("costmap_disc_count", -1))
            costmap_writes_enabled = bool(
                payload.get("costmap_writes_enabled", False)
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        with self.lock:
            self.disc_tracks = tuple(tracks)
            self.disc_radius = radius
            self.disc_positions_frozen = positions_frozen
            self.costmap_disc_count = costmap_disc_count
            self.costmap_writes_enabled = costmap_writes_enabled
            self.latest_disc_receipt = rospy.Time.now()

    def current_pose(self):
        translation, rotation = self.tf_listener.lookupTransform(
            self.path_frame, self.base_frame, rospy.Time(0)
        )
        yaw = tf.transformations.euler_from_quaternion(rotation)[2]
        return float(translation[0]), float(translation[1]), float(yaw)

    def freshness_reason(self, now):
        if self.latest_odom_receipt is None:
            return "odometry_missing"
        if (now - self.latest_odom_receipt).to_sec() > self.odom_timeout:
            return "odometry_stale"
        if self.minimum_disc_count > 0:
            if self.latest_disc_receipt is None:
                return "disc_status_missing"
            if (now - self.latest_disc_receipt).to_sec() > self.disc_timeout:
                return "disc_status_stale"
            if not self.disc_positions_frozen:
                return "disc_positions_not_frozen"
            if len(self.disc_tracks) < self.minimum_disc_count:
                return "confirmed_disc_count_below_minimum"
            if not self.costmap_writes_enabled:
                return "costmap_disc_writes_disabled"
            if (
                self.costmap_disc_count < 0
                or self.costmap_disc_count > len(self.disc_tracks)
            ):
                return "costmap_disc_count_mismatch"
            # Before motion, every frozen disc must be represented in the
            # virtual scan. During motion, passed discs naturally leave the
            # scan's finite range while remaining in the frozen mission set.
            if (
                self.state != "RUNNING"
                and self.costmap_disc_count != len(self.disc_tracks)
            ):
                return "costmap_disc_count_mismatch"
            disc_ids = tuple(sorted(track[2] for track in self.disc_tracks))
            if len(set(disc_ids)) != len(disc_ids) or any(
                memory_id < 0 for memory_id in disc_ids
            ):
                return "confirmed_disc_ids_invalid"
            if self.locked_disc_ids is not None and disc_ids != self.locked_disc_ids:
                return "frozen_disc_set_changed"
        return "ready"

    def command_topology_reason(self):
        now = time.monotonic()
        if now - self.last_topology_check < 0.5:
            return self.topology_reason
        try:
            publishers, subscribers, _ = rosgraph.Master(
                rospy.get_name()
            ).getSystemState()
            publisher_map = dict(publishers)
            subscriber_map = dict(subscribers)
            direct_publishers = set(publisher_map.get(self.command_topic, ()))
            direct_subscribers = set(subscriber_map.get(self.command_topic, ()))
            raw_subscribers = set(subscriber_map.get("/cmd_vel", ()))
            if direct_publishers != {rospy.get_name()}:
                reason = "direct_command_has_other_publishers"
            elif self.required_command_subscriber not in direct_subscribers:
                reason = "direct_chassis_subscriber_missing"
            elif "/cmd_vel_listener" in raw_subscribers:
                reason = "standard_cmd_vel_chassis_still_connected"
            else:
                reason = "ready"
        except Exception as error:
            reason = "master_query_failed:" + str(error)
        self.last_topology_check = now
        self.topology_reason = reason
        return reason

    def readiness_reason(self):
        if not self.enable_motion:
            return "motion_disabled"
        if not self.points:
            return "preview_path_missing"
        now = rospy.Time.now()
        freshness = self.freshness_reason(now)
        if freshness != "ready":
            return freshness
        if (
            self.odom_linear_speed is None
            or self.odom_angular_speed is None
            or self.odom_linear_speed > 0.01
            or self.odom_angular_speed > 0.02
        ):
            return "robot_not_stationary"
        topology = self.command_topology_reason()
        if topology != "ready":
            return topology
        try:
            x, y, _ = self.current_pose()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return "path_to_robot_tf_missing"
        start_distance = min(
            math.hypot(x - point[0], y - point[1])
            for point in self.points[: min(len(self.points), 200)]
        )
        if start_distance > self.max_cross_track:
            return "robot_too_far_from_path_start"
        return "ready"

    def start_service(self, _request):
        with self.lock:
            if self.state == "RUNNING":
                return TriggerResponse(True, "direct follower is already running")
            reason = self.readiness_reason()
            if reason != "ready":
                self.stop_commands()
                return TriggerResponse(False, reason)
            self.progress_index = 0
            self.progress_s = 0.0
            self.last_progress_s = 0.0
            self.started_at = time.monotonic()
            self.last_progress_at = self.started_at
            self.last_tick = self.started_at
            self.last_linear = 0.0
            self.last_angular = 0.0
            self.locked_disc_ids = tuple(
                sorted(track[2] for track in self.disc_tracks)
            )
            self.stable_goal_cycles = 0
            self.state = "RUNNING"
            self.reason = "explicit_start"
            rospy.logwarn("Stage-1 direct path motion started")
            return TriggerResponse(True, "direct path motion started")

    def check_service(self, _request):
        with self.lock:
            reason = self.readiness_reason()
            return TriggerResponse(reason == "ready", reason)

    def stop_service(self, _request):
        with self.lock:
            self.state = "STOPPED"
            self.reason = "operator_stop"
            self.stop_commands()
        return TriggerResponse(True, "direct path motion stopped")

    def reset_service(self, _request):
        with self.lock:
            if self.state == "RUNNING":
                return TriggerResponse(False, "stop the controller before reset")
            self.stop_commands()
            self.path_message = None
            self.path_frame = None
            self.points = ()
            self.cumulative = ()
            self.progress_index = 0
            self.progress_s = 0.0
            self.locked_disc_ids = None
            self.terminal_speed_active = False
            self.state = "WAITING_PATH"
            self.reason = "operator_reset"
            self.auto_start_attempted = False
        return TriggerResponse(True, "path latch reset")

    def fault(self, reason):
        self.state = "FAULT"
        self.reason = reason
        self.stop_commands()
        rospy.logerr("Stage-1 direct follower stopped: %s", reason)

    def stop_commands(self):
        self.last_linear = 0.0
        self.last_angular = 0.0
        zero = Twist()
        for _ in range(8):
            self.command_pub.publish(zero)

    def publish_command(self, linear_target, angular_target, dt):
        linear_limit = (
            self.terminal_max_linear
            if self.terminal_speed_active
            else self.max_linear
        )
        self.last_linear = move_toward(
            self.last_linear,
            clamp(linear_target, 0.0, linear_limit),
            self.max_linear_accel * dt,
        )
        self.last_angular = move_toward(
            self.last_angular,
            clamp(angular_target, -self.max_angular, self.max_angular),
            self.max_angular_accel * dt,
        )
        command = Twist()
        command.linear.x = self.last_linear
        command.linear.y = 0.0
        command.angular.z = self.last_angular
        self.command_pub.publish(command)

    def nearest_path_index(self, x, y):
        start = max(0, self.progress_index - 5)
        stop = min(len(self.points), max(start + 1, self.progress_index + 250))
        nearest = min(
            range(start, stop),
            key=lambda index: math.hypot(
                x - self.points[index][0], y - self.points[index][1]
            ),
        )
        return max(self.progress_index, nearest)

    def minimum_disc_gap(self):
        if not self.disc_tracks:
            return None
        return min(
            rectangle_disc_clearance(
                forward,
                left,
                self.vehicle_half_length,
                self.vehicle_half_width,
                self.disc_radius,
            )
            for forward, left, _memory_id in self.disc_tracks
        )

    def control_step(self, now, dt):
        freshness = self.freshness_reason(now)
        if freshness != "ready":
            self.fault(freshness)
            return
        topology = self.command_topology_reason()
        if topology != "ready":
            self.fault(topology)
            return
        if time.monotonic() - self.started_at > self.mission_timeout:
            self.fault("mission_timeout")
            return
        try:
            x, y, yaw = self.current_pose()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            self.fault("path_to_robot_tf_lost")
            return

        index = self.nearest_path_index(x, y)
        nearest = self.points[index]
        cross_track = math.hypot(x - nearest[0], y - nearest[1])
        self.progress_index = index
        self.progress_s = self.cumulative[index]
        self.cross_track = cross_track
        self.remaining = max(0.0, self.cumulative[-1] - self.progress_s)
        if cross_track > self.max_cross_track:
            self.fault("cross_track_error_exceeded")
            return

        disc_gap = self.minimum_disc_gap()
        self.minimum_observed_disc_clearance = disc_gap
        if disc_gap is not None and disc_gap < self.minimum_disc_clearance:
            if self.stop_on_disc_clearance_violation:
                self.fault("disc_clearance_below_limit")
                return
            rospy.logwarn_throttle(
                1.0,
                "Disc clearance %.3f m is below %.3f m; continuing because "
                "the competition override is enabled",
                disc_gap,
                self.minimum_disc_clearance,
            )

        goal = self.points[-1]
        goal_distance = math.hypot(goal[0] - x, goal[1] - y)
        if goal_distance <= self.position_tolerance:
            yaw_error = normalize_angle(self.final_yaw - yaw)
            if abs(yaw_error) <= self.yaw_tolerance:
                self.stable_goal_cycles += 1
                self.publish_command(0.0, 0.0, dt)
                if self.stable_goal_cycles >= 6:
                    self.state = "FINISHED"
                    self.reason = "goal_reached"
                    self.stop_commands()
                    rospy.loginfo("Stage-1 direct path completed")
                return
            self.stable_goal_cycles = 0
            self.publish_command(0.0, 1.4 * yaw_error, dt)
            return

        self.stable_goal_cycles = 0
        target_s = min(self.cumulative[-1], self.progress_s + self.lookahead)
        target_index = min(
            len(self.points) - 1,
            bisect.bisect_left(self.cumulative, target_s),
        )
        target = self.points[target_index]
        target_dx, target_dy = target[0] - x, target[1] - y
        target_distance = max(0.05, math.hypot(target_dx, target_dy))
        heading_error = normalize_angle(math.atan2(target_dy, target_dx) - yaw)

        if abs(heading_error) >= self.rotate_only_angle:
            linear_target = 0.0
            angular_target = 1.4 * heading_error
        else:
            curvature = 2.0 * math.sin(heading_error) / target_distance
            curvature_scale = 1.0 / (1.0 + 1.7 * abs(curvature))
            finish_scale = clamp(self.remaining / 0.40, 0.30, 1.0)
            self.terminal_speed_active = (
                self.terminal_extension is not None
                and self.remaining <= self.terminal_extension + 0.02
            )
            linear_limit = (
                self.terminal_max_linear
                if self.terminal_speed_active
                else self.max_linear
            )
            linear_target = max(
                self.min_linear,
                linear_limit * curvature_scale * finish_scale,
            )
            angular_target = linear_target * curvature + 0.25 * heading_error
        self.publish_command(linear_target, angular_target, dt)

        if self.progress_s >= self.last_progress_s + 0.01:
            self.last_progress_s = self.progress_s
            self.last_progress_at = time.monotonic()
        elif (
            self.last_linear > 0.03
            and time.monotonic() - self.last_progress_at > self.stall_timeout
        ):
            self.fault("path_progress_stalled")

    def publish_status(self):
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "state": self.state,
            "reason": self.reason,
            "enable_motion": self.enable_motion,
            "auto_start": self.auto_start,
            "path_topic": self.path_topic,
            "path_frame": self.path_frame,
            "path_points": len(self.points),
            "path_length_m": self.cumulative[-1] if self.cumulative else None,
            "progress_m": self.progress_s,
            "remaining_m": self.remaining,
            "cross_track_error_m": self.cross_track,
            "confirmed_disc_count": len(self.disc_tracks),
            "minimum_confirmed_disc_count": self.minimum_disc_count,
            "costmap_disc_count": self.costmap_disc_count,
            "costmap_disc_writes_enabled": self.costmap_writes_enabled,
            "locked_disc_ids": self.locked_disc_ids,
            "minimum_disc_clearance_m": self.minimum_observed_disc_clearance,
            "stop_on_disc_clearance_violation": (
                self.stop_on_disc_clearance_violation
            ),
            "terminal_target_xy_m": self.terminal_target_xy,
            "terminal_extension_m": self.terminal_extension,
            "terminal_speed_active": self.terminal_speed_active,
            "terminal_max_linear_mps": self.terminal_max_linear,
            "terminal_forward_offset_m": self.terminal_forward_offset,
            "odom_linear_speed_mps": self.odom_linear_speed,
            "odom_angular_speed_radps": self.odom_angular_speed,
            "command_topic": self.command_topic,
            "command_subscribers": self.command_pub.get_num_connections(),
            "linear_x": self.last_linear,
            "linear_y": 0.0,
            "angular_z": self.last_angular,
            "topology": self.topology_reason,
        }
        self.status_pub.publish(String(data=json.dumps(payload, allow_nan=False)))

    def timer_callback(self, _event):
        with self.lock:
            now_monotonic = time.monotonic()
            dt = clamp(now_monotonic - self.last_tick, 0.001, 0.10)
            self.last_tick = now_monotonic
            if (
                self.auto_start
                and not self.auto_start_attempted
                and self.state == "READY"
            ):
                reason = self.readiness_reason()
                if reason == "ready":
                    self.auto_start_attempted = True
                    response = self.start_service(None)
                    if not response.success:
                        self.reason = response.message
                else:
                    self.reason = reason
                    rospy.logwarn_throttle(
                        2.0, "Stage-1 direct follower waiting: %s", reason
                    )
            elif self.auto_start and self.state == "WAITING_PATH":
                rospy.logwarn_throttle(
                    2.0, "Stage-1 direct follower waiting: %s", self.reason
                )
            if self.state == "RUNNING":
                self.control_step(rospy.Time.now(), dt)
            if now_monotonic - self.last_status_publish >= 0.20:
                self.last_status_publish = now_monotonic
                self.publish_status()

    def shutdown(self):
        with self.lock:
            self.stop_commands()


def main():
    rospy.init_node("stage1_direct_path_follower")
    Stage1DirectPathFollower()
    rospy.spin()


if __name__ == "__main__":
    main()
