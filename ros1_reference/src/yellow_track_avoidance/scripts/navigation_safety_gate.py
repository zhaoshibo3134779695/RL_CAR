#!/usr/bin/env python3
"""Gate move_base velocity on live yellow-track perception safety."""

import json
import math
import threading

import numpy as np
import rosgraph
import rospy
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import (
    SetBool,
    SetBoolResponse,
    Trigger,
    TriggerResponse,
)
from yellow_track_avoidance.terminal_control import (
    STATE_DWA_FALLBACK,
    TerminalConfig,
    TerminalLineObservation,
    TerminalVisualController,
    fit_left_boundary_line,
    goal_frame_errors,
)


def terminal_takeover_allowed(
    auto_localization_active,
    special_authority_latched,
    special_status,
):
    """Allow terminal control only after obstacle passage has finished."""
    if auto_localization_active:
        return False
    if not special_authority_latched:
        return True
    if not isinstance(special_status, dict):
        return False
    state = str(special_status.get("state", ""))
    return bool(special_status.get("competition_pair_consumed", False)) or (
        state.startswith("finish_recovery_")
    )


def zone_navigation_safety(
    passage_status,
    status_fresh,
    command_fresh,
    special_authority_latched,
):
    """Validate an ordinary move_base command inside the visual corridor."""
    if special_authority_latched:
        return True, "passage_authority"
    if not command_fresh:
        return True, "zone_waiting_for_command"
    if passage_status is None or not status_fresh:
        return False, "passage_status_timeout"
    zone = passage_status.get("activation_zone", {})
    if not bool(zone.get("enabled", False)):
        return True, "activation_zone_disabled"
    if not bool(zone.get("pose_available", False)):
        return False, "activation_zone_pose_unavailable"
    if not bool(zone.get("active", False)):
        return True, "outside_activation_zone"
    if not bool(passage_status.get("boundary_fresh", False)):
        return False, "zone_boundary_stale"
    safety = passage_status.get("normal_command_safety")
    if not isinstance(safety, dict):
        return False, "zone_swept_guard:missing"
    if not bool(safety.get("safe", False)):
        return False, "zone_swept_guard:" + str(
            safety.get("reason", "unsafe")
        )
    return True, "zone_swept_safe"


class NavigationSafetyGate:
    def __init__(self):
        self.lock = threading.RLock()
        self.input_topic = str(
            rospy.get_param(
                "~input_cmd_topic",
                "/yellow_track_avoidance/move_base_cmd_vel",
            )
        )
        self.special_input_topic = str(
            rospy.get_param(
                "~special_input_cmd_topic",
                "/yellow_track_avoidance/passage_cmd_vel",
            )
        )
        self.special_status_topic = str(
            rospy.get_param(
                "~special_status_topic",
                "/yellow_track_avoidance/passage/status",
            )
        )
        self.output_topic = str(rospy.get_param("~output_cmd_topic", "/cmd_vel"))
        self.status_topic = str(
            rospy.get_param(
                "~perception_status_topic",
                "/yellow_track_avoidance/debug/status",
            )
        )
        self.disk_topic = str(
            rospy.get_param(
                "~disk_topic", "/yellow_track_avoidance/debug/disks"
            )
        )
        self.motion_capable = bool(rospy.get_param("~enable_motion", False))
        self.enabled = self.motion_capable and bool(
            rospy.get_param("~start_enabled", False)
        )
        self.status_timeout = float(
            rospy.get_param("~perception_status_timeout_sec", 0.50)
        )
        self.command_timeout = float(
            rospy.get_param("~command_timeout_sec", 0.30)
        )
        self.special_timeout = float(
            rospy.get_param("~special_timeout_sec", 0.25)
        )
        self.disk_timeout = float(
            rospy.get_param("~disk_status_timeout_sec", 0.50)
        )
        self.emergency_clearance = float(
            rospy.get_param("~emergency_clearance_m", 0.03)
        )
        self.enforce_disc_emergency_clearance = bool(
            rospy.get_param(
                "~enforce_disc_emergency_clearance", True
            )
        )
        self.vehicle_half_length = float(
            rospy.get_param("~vehicle_half_length_m", 0.165)
        )
        self.vehicle_half_width = float(
            rospy.get_param("~vehicle_half_width_m", 0.150)
        )
        self.disk_radius = float(
            rospy.get_param("~disk_radius_m", 0.05)
        )
        self.forward_offset = float(
            rospy.get_param("~local_forward_origin_offset_m", 0.0)
        )
        self.minimum_boundary_quality = float(
            rospy.get_param("~minimum_boundary_quality", 0.50)
        )
        self.enforce_perception_safety = bool(
            rospy.get_param("~enforce_perception_safety", True)
        )
        self.enforce_boundary_in_activation_zone = bool(
            rospy.get_param(
                "~enforce_boundary_in_activation_zone", False
            )
        )
        self.allow_move_base_fallback = bool(
            rospy.get_param(
                "~allow_move_base_when_local_planner_blocked", False
            )
        )
        self.maximum_linear = float(
            rospy.get_param("~maximum_linear_speed_mps", 0.10)
        )
        self.maximum_angular = float(
            rospy.get_param("~maximum_angular_speed_radps", 0.30)
        )
        self.require_exclusive_output = bool(
            rospy.get_param("~require_exclusive_output", True)
        )
        self.field_localization_service_name = str(
            rospy.get_param(
                "~field_localization_service",
                "/yellow_track_field_localization/localize",
            )
        )
        self.move_base_cancel_topic = str(
            rospy.get_param("~move_base_cancel_topic", "/move_base/cancel")
        )
        self.terminal_goal_stop_enabled = bool(
            rospy.get_param("~terminal_goal_stop_enabled", False)
        )
        self.terminal_goal_topic = str(
            rospy.get_param(
                "~terminal_goal_topic",
                "/yellow_track_avoidance/corridor_goal",
            )
        )
        self.terminal_goal_odom_topic = str(
            rospy.get_param("~terminal_goal_odom_topic", "/odom")
        )
        self.terminal_boundary_topic = str(
            rospy.get_param(
                "~terminal_boundary_topic",
                "/yellow_track_avoidance/debug/boundaries",
            )
        )
        self.terminal_line_minimum_quality = float(
            rospy.get_param("~terminal_line_minimum_quality", 0.50)
        )
        self.terminal_goal_odom_timeout = float(
            rospy.get_param("~terminal_goal_odom_timeout_sec", 0.30)
        )
        terminal_defaults = TerminalConfig(
            vehicle_length_m=2.0 * self.vehicle_half_length,
            vehicle_width_m=2.0 * self.vehicle_half_width,
        )
        self.terminal_config = TerminalConfig(
            takeover_remaining_m=float(
                rospy.get_param(
                    "~terminal_takeover_remaining_m",
                    terminal_defaults.takeover_remaining_m,
                )
            ),
            required_line_frames=int(
                rospy.get_param(
                    "~terminal_required_line_frames",
                    terminal_defaults.required_line_frames,
                )
            ),
            line_propagation_sec=float(
                rospy.get_param(
                    "~terminal_line_propagation_sec",
                    terminal_defaults.line_propagation_sec,
                )
            ),
            line_wait_sec=float(
                rospy.get_param(
                    "~terminal_line_wait_sec",
                    terminal_defaults.line_wait_sec,
                )
            ),
            left_reference_m=float(
                rospy.get_param(
                    "~terminal_left_line_reference_m",
                    terminal_defaults.left_reference_m,
                )
            ),
            corridor_width_m=float(
                rospy.get_param(
                    "~terminal_corridor_width_m",
                    terminal_defaults.corridor_width_m,
                )
            ),
            vehicle_length_m=2.0 * self.vehicle_half_length,
            vehicle_width_m=2.0 * self.vehicle_half_width,
            boundary_margin_m=float(
                rospy.get_param(
                    "~terminal_boundary_margin_m",
                    terminal_defaults.boundary_margin_m,
                )
            ),
            minimum_sweep_clearance_m=float(
                rospy.get_param(
                    "~terminal_minimum_sweep_clearance_m",
                    terminal_defaults.minimum_sweep_clearance_m,
                )
            ),
            aligned_stable_sec=float(
                rospy.get_param(
                    "~terminal_aligned_stable_sec",
                    terminal_defaults.aligned_stable_sec,
                )
            ),
            stop_min_remaining_m=float(
                rospy.get_param(
                    "~terminal_stop_min_remaining_m",
                    terminal_defaults.stop_min_remaining_m,
                )
            ),
            stop_max_remaining_m=float(
                rospy.get_param(
                    "~terminal_stop_max_remaining_m",
                    terminal_defaults.stop_max_remaining_m,
                )
            ),
            stop_lateral_m=float(
                rospy.get_param(
                    "~terminal_stop_lateral_m",
                    terminal_defaults.stop_lateral_m,
                )
            ),
            stop_yaw_rad=float(
                rospy.get_param(
                    "~terminal_stop_yaw_rad",
                    terminal_defaults.stop_yaw_rad,
                )
            ),
            stop_linear_speed_mps=float(
                rospy.get_param(
                    "~terminal_stop_linear_speed_mps",
                    terminal_defaults.stop_linear_speed_mps,
                )
            ),
            stop_angular_speed_radps=float(
                rospy.get_param(
                    "~terminal_stop_angular_speed_radps",
                    terminal_defaults.stop_angular_speed_radps,
                )
            ),
            stop_stable_sec=float(
                rospy.get_param(
                    "~terminal_stop_stable_sec",
                    terminal_defaults.stop_stable_sec,
                )
            ),
            overrun_m=float(
                rospy.get_param(
                    "~terminal_overrun_m", terminal_defaults.overrun_m
                )
            ),
        )
        self.terminal_dwa_fallback_linear = float(
            rospy.get_param("~terminal_dwa_fallback_linear_mps", 0.03)
        )
        self.terminal_dwa_fallback_angular = float(
            rospy.get_param("~terminal_dwa_fallback_angular_radps", 0.10)
        )
        self.terminal_controller = TerminalVisualController(
            self.terminal_config
        )
        self.perception_status = None
        self.status_receipt_sec = None
        self.latest_command = None
        self.command_receipt_sec = None
        self.latest_special_command = None
        self.special_command_receipt_sec = None
        self.special_status = None
        self.special_status_receipt_sec = None
        # If passage status disappears after requesting authority, fail closed
        # instead of falling back to a still-fresh move_base command.
        self.special_authority_latched = False
        self.minimum_disk_distance = None
        self.minimum_disk_clearance = None
        self.disk_receipt_sec = None
        self.last_conflict_audit_sec = -1.0
        self.cached_conflicts = []
        self.auto_localization_active = False
        self.auto_localization_state = "idle"
        self.auto_localization_message = ""
        self.auto_localization_start_sec = None
        self.auto_localization_elapsed_sec = 0.0
        self.terminal_goal = None
        self.terminal_goal_signature = None
        self.terminal_goal_armed = False
        self.terminal_stop_latched = False
        self.terminal_stop_reason = ""
        self.terminal_robot = None
        self.terminal_robot_receipt_sec = None
        self.terminal_line = None
        self.terminal_decision = None
        self.terminal_goal_metrics = {
            "available": False,
            "frame_match": False,
            "distance_m": None,
            "remaining_m": None,
            "lateral_error_m": None,
        }

        self.output_pub = rospy.Publisher(
            self.output_topic, Twist, queue_size=1
        )
        self.goal_cancel_pub = rospy.Publisher(
            self.move_base_cancel_topic, GoalID, queue_size=1
        )
        self.gate_status_pub = rospy.Publisher(
            "~status", String, queue_size=1, latch=True
        )
        self.command_sub = rospy.Subscriber(
            self.input_topic, Twist, self._command_callback, queue_size=1
        )
        self.special_command_sub = rospy.Subscriber(
            self.special_input_topic,
            Twist,
            self._special_command_callback,
            queue_size=1,
        )
        self.special_status_sub = rospy.Subscriber(
            self.special_status_topic,
            String,
            self._special_status_callback,
            queue_size=1,
        )
        self.status_sub = rospy.Subscriber(
            self.status_topic, String, self._status_callback, queue_size=1
        )
        self.disk_sub = rospy.Subscriber(
            self.disk_topic, String, self._disk_callback, queue_size=1
        )
        self.terminal_goal_sub = None
        self.terminal_odom_sub = None
        self.terminal_boundary_sub = None
        if self.terminal_goal_stop_enabled:
            self.terminal_goal_sub = rospy.Subscriber(
                self.terminal_goal_topic,
                PoseStamped,
                self._terminal_goal_callback,
                queue_size=1,
            )
            self.terminal_odom_sub = rospy.Subscriber(
                self.terminal_goal_odom_topic,
                Odometry,
                self._terminal_odom_callback,
                queue_size=1,
            )
            self.terminal_boundary_sub = rospy.Subscriber(
                self.terminal_boundary_topic,
                String,
                self._terminal_boundary_callback,
                queue_size=1,
            )
        self.enable_service = rospy.Service(
            "~set_enabled", SetBool, self._set_enabled
        )
        self.start_auto_localization_service = rospy.Service(
            "~start_auto_localization",
            Trigger,
            self._start_auto_localization,
        )
        self.cancel_auto_localization_service = rospy.Service(
            "~cancel_auto_localization",
            Trigger,
            self._cancel_auto_localization,
        )
        self.timer = rospy.Timer(rospy.Duration(0.05), self._timer_callback)
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "navigation safety gate ready: %s -> %s capable=%s enabled=%s",
            self.input_topic,
            self.output_topic,
            self.motion_capable,
            self.enabled,
        )

    def _command_callback(self, message):
        with self.lock:
            self.latest_command = message
            self.command_receipt_sec = rospy.Time.now().to_sec()

    def _special_command_callback(self, message):
        with self.lock:
            self.latest_special_command = message
            self.special_command_receipt_sec = (
                rospy.Time.now().to_sec()
            )

    def _special_status_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid passage status JSON")
            return
        authority = bool(payload.get("authority_requested", False))
        with self.lock:
            self.special_status = payload
            self.special_status_receipt_sec = (
                rospy.Time.now().to_sec()
            )
            self.special_authority_latched = authority

    def _status_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid perception status JSON")
            return
        with self.lock:
            self.perception_status = payload
            self.status_receipt_sec = rospy.Time.now().to_sec()

    def _disk_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid disc status JSON")
            return
        distances = []
        clearances = []
        seen_tracks = set()
        items = list(payload.get("confirmed_tracks", [])) + list(
            payload.get("raw", [])
        )
        for item in items:
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
                forward, left = float(center[0]), float(center[1])
                if center is not center_base:
                    forward += self.forward_offset
            except (TypeError, ValueError):
                continue
            if np.isfinite(forward) and np.isfinite(left):
                distances.append(float(np.hypot(forward, left)))
                longitudinal_gap = max(
                    abs(forward) - self.vehicle_half_length, 0.0
                )
                lateral_gap = max(
                    abs(left) - self.vehicle_half_width, 0.0
                )
                clearances.append(
                    float(np.hypot(longitudinal_gap, lateral_gap))
                    - self.disk_radius
                )
                if track_id is not None:
                    seen_tracks.add(track_id)
        with self.lock:
            self.minimum_disk_distance = (
                min(distances) if distances else None
            )
            self.minimum_disk_clearance = (
                min(clearances) if clearances else None
            )
            self.disk_receipt_sec = rospy.Time.now().to_sec()

    @staticmethod
    def _normalized_frame(frame_id):
        return str(frame_id).lstrip("/")

    def _terminal_goal_callback(self, message):
        orientation = message.pose.orientation
        yaw = math.atan2(
            2.0
            * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        goal = (
            self._normalized_frame(message.header.frame_id),
            float(message.pose.position.x),
            float(message.pose.position.y),
            float(yaw),
        )
        signature = (
            goal[0],
            round(goal[1], 4),
            round(goal[2], 4),
            round(goal[3], 4),
        )
        with self.lock:
            changed = signature != self.terminal_goal_signature
            self.terminal_goal = goal
            self.terminal_goal_signature = signature
            if changed:
                self.terminal_stop_latched = False
                self.terminal_stop_reason = ""
                self.terminal_goal_armed = self.enabled
                self.terminal_controller.reset()
                self.terminal_decision = None

    def _terminal_odom_callback(self, message):
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0
            * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0
            - 2.0
            * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )
        twist = message.twist.twist
        robot = (
            self._normalized_frame(message.header.frame_id),
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            float(yaw),
            float(math.hypot(twist.linear.x, twist.linear.y)),
            float(twist.angular.z),
        )
        with self.lock:
            self.terminal_robot = robot
            self.terminal_robot_receipt_sec = rospy.Time.now().to_sec()

    def _terminal_boundary_callback(self, message):
        try:
            payload = json.loads(message.data)
            quality = float(payload.get("quality", 0.0))
            source_sec = float(payload["stamp"])
            frame_id = self._normalized_frame(
                payload.get("frame_id", "")
            )
            points = payload.get("left_base_m", [])
            valid = (
                bool(payload.get("left_detected", False))
                and quality >= self.terminal_line_minimum_quality
                and math.isfinite(source_sec)
                and frame_id in ("base_link", "base_footprint")
                and isinstance(points, list)
                and len(points) >= 2
            )
            if not valid:
                raise ValueError("invalid terminal left-line observation")
            left_distance, yaw_error = fit_left_boundary_line(points)
        except (KeyError, TypeError, ValueError, OverflowError):
            with self.lock:
                self.terminal_controller.note_invalid_line_frame()
            return
        with self.lock:
            odom_yaw = (
                0.0
                if self.terminal_robot is None
                else float(self.terminal_robot[3])
            )
            self.terminal_line = TerminalLineObservation(
                stamp_sec=source_sec,
                left_distance_m=left_distance,
                yaw_error_rad=yaw_error,
                odom_yaw_rad=odom_yaw,
            )

    def _terminal_control_locked(self, now_sec, takeover_allowed):
        metrics = {
            "available": False,
            "frame_match": False,
            "distance_m": None,
            "remaining_m": None,
            "lateral_error_m": None,
        }
        if (
            not self.terminal_goal_stop_enabled
            or not self.enabled
            or not self.terminal_goal_armed
            or self.terminal_goal is None
        ):
            self.terminal_goal_metrics = metrics
            self.terminal_decision = self.terminal_controller.update(
                now_sec=now_sec,
                armed=False,
                takeover_allowed=False,
                remaining_m=None,
                goal_lateral_m=None,
                odom_yaw_rad=0.0,
                linear_speed_mps=0.0,
                angular_speed_radps=0.0,
                line=None,
            )
            return self.terminal_decision
        if (
            self.terminal_robot is None
            or self.terminal_robot_receipt_sec is None
            or now_sec - self.terminal_robot_receipt_sec
            > self.terminal_goal_odom_timeout
        ):
            previous_remaining = self.terminal_goal_metrics.get(
                "remaining_m"
            )
            previous_lateral = self.terminal_goal_metrics.get(
                "lateral_error_m"
            )
            self.terminal_goal_metrics = metrics
            self.terminal_decision = self.terminal_controller.hold(
                "odom_stale_hold",
                previous_remaining,
                previous_lateral,
            )
            return self.terminal_decision
        metrics["available"] = True
        metrics["frame_match"] = (
            self.terminal_goal[0] == self.terminal_robot[0]
        )
        if not metrics["frame_match"]:
            self.terminal_goal_metrics = metrics
            self.terminal_decision = self.terminal_controller.hold(
                "frame_mismatch_hold", None, None
            )
            return self.terminal_decision
        distance, remaining, lateral = goal_frame_errors(
            self.terminal_robot[1:3],
            self.terminal_goal[1:3],
            self.terminal_goal[3],
        )
        metrics.update(
            {
                "distance_m": distance,
                "remaining_m": remaining,
                "lateral_error_m": lateral,
                "robot_yaw_rad": self.terminal_robot[3],
                "linear_speed_mps": self.terminal_robot[4],
                "angular_speed_radps": self.terminal_robot[5],
            }
        )
        self.terminal_goal_metrics = metrics
        self.terminal_decision = self.terminal_controller.update(
            now_sec=now_sec,
            armed=True,
            takeover_allowed=takeover_allowed,
            remaining_m=remaining,
            goal_lateral_m=lateral,
            odom_yaw_rad=self.terminal_robot[3],
            linear_speed_mps=self.terminal_robot[4],
            angular_speed_radps=self.terminal_robot[5],
            line=self.terminal_line,
        )
        return self.terminal_decision

    def _latch_terminal_stop_locked(self, reason):
        if self.terminal_stop_latched:
            return
        self.terminal_stop_latched = True
        self.terminal_goal_armed = False
        self.terminal_stop_reason = str(reason)
        self.enabled = False
        self.output_pub.publish(Twist())
        self.goal_cancel_pub.publish(GoalID())
        rospy.logwarn(
            "terminal stop latched (%s): chassis stopped and navigation cancelled",
            self.terminal_stop_reason,
        )

    def _output_conflicts(self, force=False):
        if not self.require_exclusive_output:
            return []
        now_sec = rospy.Time.now().to_sec()
        if (
            not force
            and self.last_conflict_audit_sec >= 0.0
            and now_sec - self.last_conflict_audit_sec < 0.5
        ):
            return list(self.cached_conflicts)
        try:
            publishers, _, _ = rosgraph.Master(
                rospy.get_name()
            ).getSystemState()
            resolved = rospy.resolve_name(self.output_topic)
            own_name = rospy.get_name()
            conflicts = []
            for topic, nodes in publishers:
                if rospy.resolve_name(topic) == resolved:
                    conflicts = [
                        node for node in nodes if node != own_name
                    ]
                    break
        except Exception:
            conflicts = ["master_audit_failed"]
        self.cached_conflicts = conflicts
        self.last_conflict_audit_sec = now_sec
        return list(conflicts)

    def _ready(self, now_sec):
        conflicts = self._output_conflicts()
        if conflicts:
            return False, "output_publisher_conflict:" + ",".join(conflicts)
        # In finish-first mode discs are a scoring penalty, not a hard stop.
        # Requiring their camera stream while hard enforcement is disabled
        # would turn a camera stall into a permanent chassis lock.
        if (
            self.enforce_disc_emergency_clearance
            and (
                self.disk_receipt_sec is None
                or now_sec - self.disk_receipt_sec > self.disk_timeout
            )
        ):
            return False, "disc_status_timeout"
        if (
            self.enforce_disc_emergency_clearance
            and
            self.minimum_disk_clearance is not None
            and self.minimum_disk_clearance < self.emergency_clearance
        ):
            return False, "disc_inside_emergency_clearance"
        if self.enforce_boundary_in_activation_zone:
            special_status_fresh = (
                self.special_status is not None
                and self.special_status_receipt_sec is not None
                and now_sec - self.special_status_receipt_sec
                <= self.special_timeout
            )
            command_fresh = (
                self.latest_command is not None
                and self.command_receipt_sec is not None
                and now_sec - self.command_receipt_sec
                <= self.command_timeout
            )
            zone_safe, zone_reason = zone_navigation_safety(
                self.special_status,
                special_status_fresh,
                command_fresh,
                self.special_authority_latched,
            )
            if not zone_safe:
                return False, zone_reason
        if not self.enforce_perception_safety:
            return True, (
                zone_reason
                if self.enforce_boundary_in_activation_zone
                else "ready_competition_mode"
            )
        if self.perception_status is None or self.status_receipt_sec is None:
            return False, "perception_status_missing"
        if now_sec - self.status_receipt_sec > self.status_timeout:
            return False, "perception_status_timeout"

        state = str(self.perception_status.get("state", "STOP"))
        stop_reason = str(
            self.perception_status.get("stop_reason", "unknown")
        )
        ready_reason = "ready"
        if state == "STOP":
            diagnostics = self.perception_status.get("diagnostics", {})
            live_track_safe = (
                bool(diagnostics.get("calibrated", False))
                and bool(diagnostics.get("left_detected", False))
                and bool(diagnostics.get("right_detected", False))
                and float(diagnostics.get("boundary_quality", 0.0))
                >= self.minimum_boundary_quality
            )
            fallback_safe = (
                self.allow_move_base_fallback
                and stop_reason in (
                    "no_collision_free_candidate",
                    "yellow_boundaries_lost",
                )
                and live_track_safe
            )
            if not fallback_safe:
                return False, "perception_stop:" + stop_reason
            # In click-navigation mode TEB is the only local planner. A
            # latched STOP from the perception-only path planner must not hold
            # the chassis forever after current-frame yellow tracking has
            # recovered. Actual current boundary loss and emergency disc
            # clearance still fail closed above.
            ready_reason = "ready_move_base_authority"
        return True, ready_reason

    def _set_enabled(self, request):
        with self.lock:
            if not request.data:
                self.enabled = False
                self.terminal_goal_armed = False
                self.terminal_controller.reset()
                self.terminal_decision = None
                self.output_pub.publish(Twist())
                return SetBoolResponse(True, "navigation safety gate disabled")
            if self.auto_localization_active:
                return SetBoolResponse(
                    False, "stationary field localization is running"
                )
            if self.terminal_stop_latched:
                return SetBoolResponse(
                    False,
                    "terminal goal stop is latched; restart or set a new goal",
                )
            if not self.motion_capable:
                return SetBoolResponse(
                    False, "launch with enable_motion:=true first"
                )
            ready, reason = self._ready(rospy.Time.now().to_sec())
            if not ready:
                return SetBoolResponse(False, reason)
            if self.output_pub.get_num_connections() <= 0:
                return SetBoolResponse(
                    False, "chassis /cmd_vel subscriber is missing"
                )
            self.enabled = True
            self.terminal_goal_armed = (
                self.terminal_goal_stop_enabled
                and self.terminal_goal is not None
            )
            self.terminal_controller.reset()
            self.terminal_decision = None
            return SetBoolResponse(True, "navigation safety gate enabled")

    def _stop_auto_localization_locked(self, state, message):
        if self.auto_localization_start_sec is not None:
            self.auto_localization_elapsed_sec = max(
                0.0,
                rospy.Time.now().to_sec()
                - self.auto_localization_start_sec,
            )
        self.auto_localization_active = False
        self.auto_localization_state = str(state)
        self.auto_localization_message = str(message)
        self.auto_localization_start_sec = None
        self.output_pub.publish(Twist())

    def _start_auto_localization(self, _request):
        with self.lock:
            if self.auto_localization_active:
                return TriggerResponse(
                    False, "auto-localization is already running"
                )
            self.enabled = False
            self.latest_command = None
            self.command_receipt_sec = None
            self.auto_localization_active = True
            self.auto_localization_state = "matching"
            self.auto_localization_message = (
                "stationary camera-yellow and lidar-map fusion"
            )
            self.auto_localization_elapsed_sec = 0.0
            self.auto_localization_start_sec = rospy.Time.now().to_sec()
            self.output_pub.publish(Twist())

        self.goal_cancel_pub.publish(GoalID())
        try:
            rospy.wait_for_service(
                self.field_localization_service_name, timeout=3.0
            )
            result = rospy.ServiceProxy(
                self.field_localization_service_name, Trigger
            )()
        except (rospy.ROSException, rospy.ServiceException) as error:
            with self.lock:
                self._stop_auto_localization_locked(
                    "failed",
                    "stationary field localization failed: {}".format(error),
                )
            return TriggerResponse(
                False,
                "stationary field localization failed: {}".format(error),
            )

        with self.lock:
            if not self.auto_localization_active:
                self._stop_auto_localization_locked(
                    "cancelled",
                    "stationary field localization was cancelled",
                )
                return TriggerResponse(
                    False, "stationary field localization was cancelled"
                )
            state = "localized" if result.success else "rejected"
            self._stop_auto_localization_locked(
                state, str(result.message)
            )
        return TriggerResponse(
            bool(result.success), str(result.message)
        )

    def _cancel_auto_localization(self, _request):
        with self.lock:
            if (
                not self.auto_localization_active
            ):
                return TriggerResponse(
                    False, "auto-localization is not running"
                )
            self._stop_auto_localization_locked(
                "cancelled", "cancelled by service request"
            )
        return TriggerResponse(True, "auto-localization cancelled")

    def _limited(self, command):
        return self._limited_to(
            command, self.maximum_linear, self.maximum_angular
        )

    @staticmethod
    def _limited_to(command, maximum_linear, maximum_angular):
        values = np.asarray(
            [command.linear.x, command.linear.y], dtype=np.float64
        )
        norm = float(np.linalg.norm(values))
        if norm > maximum_linear and norm > 1e-9:
            values *= maximum_linear / norm
        output = Twist()
        output.linear.x = float(values[0])
        output.linear.y = float(values[1])
        output.angular.z = float(
            np.clip(
                command.angular.z,
                -maximum_angular,
                maximum_angular,
            )
        )
        return output

    @staticmethod
    def _terminal_twist(decision):
        output = Twist()
        output.linear.x = decision.command_body[0]
        output.linear.y = decision.command_body[1]
        output.angular.z = decision.command_body[2]
        return output

    def _timer_callback(self, _event):
        if rospy.is_shutdown():
            return
        with self.lock:
            now_sec = rospy.Time.now().to_sec()
            ready, reason = self._ready(now_sec)
            navigation_command_fresh = (
                self.latest_command is not None
                and self.command_receipt_sec is not None
                and now_sec - self.command_receipt_sec
                <= self.command_timeout
            )
            special_command_fresh = (
                self.latest_special_command is not None
                and self.special_command_receipt_sec is not None
                and now_sec - self.special_command_receipt_sec
                <= self.special_timeout
            )
            special_status_fresh = (
                self.special_status is not None
                and self.special_status_receipt_sec is not None
                and now_sec - self.special_status_receipt_sec
                <= self.special_timeout
            )
            special_safe = (
                special_status_fresh
                and bool(
                    self.special_status.get("safe_to_move", False)
                )
            )
            special_owns_output = self.special_authority_latched
            auto_owns_output = self.auto_localization_active
            terminal_decision = self.terminal_decision
            if not self.terminal_stop_latched:
                terminal_decision = self._terminal_control_locked(
                    now_sec,
                    takeover_allowed=terminal_takeover_allowed(
                        auto_owns_output,
                        special_owns_output,
                        self.special_status,
                    ),
                )
                if terminal_decision.latch_reason:
                    self._latch_terminal_stop_locked(
                        terminal_decision.latch_reason
                    )
            if self.terminal_stop_latched:
                allowed = False
                output = Twist()
                command_fresh = True
                output_reason = self.terminal_stop_reason
                selected_source = "terminal_goal_guard"
            elif auto_owns_output:
                allowed = False
                output = Twist()
                command_fresh = True
                output_reason = "stationary_field_localizing"
                selected_source = "stationary_field_localization"
            elif terminal_decision is not None and (
                terminal_decision.owns_output
            ):
                command_fresh = True
                allowed = (
                    self.motion_capable
                    and self.enabled
                    and ready
                )
                output = (
                    self._limited(
                        self._terminal_twist(terminal_decision)
                    )
                    if allowed
                    else Twist()
                )
                output_reason = "{}:{}".format(
                    terminal_decision.state,
                    terminal_decision.line_mode,
                )
                selected_source = "terminal_visual_controller"
            elif special_owns_output:
                command_fresh = (
                    special_command_fresh
                    and special_status_fresh
                )
                allowed = (
                    self.motion_capable
                    and self.enabled
                    and ready
                    and command_fresh
                    and special_safe
                )
                output = (
                    self._limited(self.latest_special_command)
                    if allowed
                    else Twist()
                )
                passage_state = (
                    "missing"
                    if self.special_status is None
                    else str(
                        self.special_status.get("state", "unknown")
                    )
                )
                output_reason = (
                    "passage:" + passage_state
                    if allowed
                    else "passage_stop:" + passage_state
                )
                selected_source = "passage"
            else:
                command_fresh = navigation_command_fresh
                allowed = (
                    self.motion_capable
                    and self.enabled
                    and ready
                    and command_fresh
                )
                terminal_fallback = (
                    terminal_decision is not None
                    and terminal_decision.state == STATE_DWA_FALLBACK
                )
                if allowed and terminal_fallback:
                    output = self._limited_to(
                        self.latest_command,
                        self.terminal_dwa_fallback_linear,
                        self.terminal_dwa_fallback_angular,
                    )
                    output_reason = "terminal_dwa_fallback_speed_limited"
                    selected_source = "move_base_terminal_fallback"
                else:
                    output = (
                        self._limited(self.latest_command)
                        if allowed
                        else Twist()
                    )
                    output_reason = reason
                    selected_source = "move_base"
            self.output_pub.publish(output)
            elapsed = self.auto_localization_elapsed_sec
            if self.auto_localization_start_sec is not None:
                elapsed = max(
                    0.0,
                    now_sec - self.auto_localization_start_sec,
                )
            status = {
                "stamp": now_sec,
                "motion_capable": self.motion_capable,
                "enabled": self.enabled,
                "ready": ready,
                "command_fresh": command_fresh,
                "allowed": allowed,
                "reason": output_reason,
                "selected_command_source": selected_source,
                "minimum_disk_distance_m": self.minimum_disk_distance,
                "minimum_disk_clearance_m": self.minimum_disk_clearance,
                "emergency_clearance_m": self.emergency_clearance,
                "enforce_disc_emergency_clearance": (
                    self.enforce_disc_emergency_clearance
                ),
                "enforce_perception_safety": self.enforce_perception_safety,
                "enforce_boundary_in_activation_zone":
                    self.enforce_boundary_in_activation_zone,
                "activation_zone": (
                    None
                    if self.special_status is None
                    else self.special_status.get("activation_zone")
                ),
                "normal_command_safety": (
                    None
                    if self.special_status is None
                    else self.special_status.get("normal_command_safety")
                ),
                "local_forward_origin_offset_m": self.forward_offset,
                "input_cmd_topic": self.input_topic,
                "special_input_cmd_topic": self.special_input_topic,
                "special_status_topic": self.special_status_topic,
                "special_authority_latched":
                    self.special_authority_latched,
                "special_status_fresh": special_status_fresh,
                "special_command_fresh": special_command_fresh,
                "special_safe": special_safe,
                "output_cmd_topic": self.output_topic,
                "terminal_goal_guard": {
                    "enabled": self.terminal_goal_stop_enabled,
                    "goal_topic": self.terminal_goal_topic,
                    "odom_topic": self.terminal_goal_odom_topic,
                    "boundary_topic": self.terminal_boundary_topic,
                    "armed": self.terminal_goal_armed,
                    "stop_latched": self.terminal_stop_latched,
                    "stop_reason": self.terminal_stop_reason,
                    "state": (
                        None
                        if terminal_decision is None
                        else terminal_decision.state
                    ),
                    "owns_output": (
                        False
                        if terminal_decision is None
                        else terminal_decision.owns_output
                    ),
                    "line_mode": (
                        "unavailable"
                        if terminal_decision is None
                        else terminal_decision.line_mode
                    ),
                    "valid_line_frames": (
                        0
                        if terminal_decision is None
                        else terminal_decision.valid_line_frames
                    ),
                    "visual_lateral_error_m": (
                        None
                        if terminal_decision is None
                        else terminal_decision.visual_lateral_error_m
                    ),
                    "visual_yaw_error_rad": (
                        None
                        if terminal_decision is None
                        else terminal_decision.visual_yaw_error_rad
                    ),
                    "left_sweep_clearance_m": (
                        None
                        if terminal_decision is None
                        else terminal_decision.left_clearance_m
                    ),
                    "right_sweep_clearance_m": (
                        None
                        if terminal_decision is None
                        else terminal_decision.right_clearance_m
                    ),
                    "stable_sec": (
                        0.0
                        if terminal_decision is None
                        else terminal_decision.stable_sec
                    ),
                    "takeover_remaining_m": (
                        self.terminal_config.takeover_remaining_m
                    ),
                    "left_line_reference_m": (
                        self.terminal_config.left_reference_m
                    ),
                    "stop_remaining_window_m": [
                        self.terminal_config.stop_min_remaining_m,
                        self.terminal_config.stop_max_remaining_m,
                    ],
                    "stop_lateral_m": self.terminal_config.stop_lateral_m,
                    "stop_yaw_rad": self.terminal_config.stop_yaw_rad,
                    "stop_stable_sec": self.terminal_config.stop_stable_sec,
                    "overrun_m": self.terminal_config.overrun_m,
                    **self.terminal_goal_metrics,
                },
                "auto_localization": {
                    "mode": "stationary_camera_lidar_fusion",
                    "active": self.auto_localization_active,
                    "state": self.auto_localization_state,
                    "message": self.auto_localization_message,
                    "elapsed_sec": elapsed,
                },
                "command": {
                    "linear_x": output.linear.x,
                    "linear_y": output.linear.y,
                    "angular_z": output.angular.z,
                },
            }
            self.gate_status_pub.publish(
                String(
                    data=json.dumps(
                        status, ensure_ascii=False, allow_nan=False
                    )
                )
            )

    def _shutdown(self):
        try:
            self.timer.shutdown()
        except Exception:
            pass
        for _ in range(5):
            try:
                self.output_pub.publish(Twist())
                rospy.sleep(0.04)
            except Exception:
                break


if __name__ == "__main__":
    rospy.init_node("yellow_track_navigation_safety_gate")
    NavigationSafetyGate()
    rospy.spin()
