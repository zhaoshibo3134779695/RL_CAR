#!/usr/bin/env python3
"""Safely track the live map-frame fused path using localization TF."""

import json
import math
import threading

import numpy as np
import rosgraph
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, SetBoolResponse

from yellow_track_avoidance.navigation_control import compute_navigation_command


def _quaternion_yaw(quaternion):
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


class MapPathController:
    def __init__(self):
        self.lock = threading.RLock()
        self.map_frame = str(rospy.get_param("~map_frame", "map")).lstrip("/")
        self.base_frame = str(rospy.get_param("~base_frame", "base_link")).lstrip("/")
        self.path_topic = str(
            rospy.get_param(
                "~path_topic", "/yellow_track_avoidance/map_fused_path"
            )
        )
        self.fusion_status_topic = str(
            rospy.get_param(
                "~fusion_status_topic",
                "/yellow_track_avoidance/map_fusion_status",
            )
        )
        self.perception_status_topic = str(
            rospy.get_param(
                "~perception_status_topic",
                "/yellow_track_avoidance/debug/status",
            )
        )
        self.output_topic = str(
            rospy.get_param(
                "~output_cmd_topic",
                "/yellow_track_avoidance/navigation_cmd_vel",
            )
        )
        self.chassis_topic = str(rospy.get_param("~chassis_cmd_topic", "/cmd_vel"))
        self.expected_chassis_publishers = set(
            str(value) for value in rospy.get_param(
                "~expected_chassis_publishers", ["/vel_ctrl"]
            )
        )
        self.motion_capable = bool(rospy.get_param("~enable_motion", False))
        self.armed = self.motion_capable and bool(
            rospy.get_param("~start_armed", False)
        )
        self.require_exclusive_output = bool(
            rospy.get_param("~require_exclusive_output", True)
        )

        self.rate_hz = float(rospy.get_param("~control_rate_hz", 20.0))
        self.path_timeout = float(rospy.get_param("~path_timeout_sec", 0.50))
        self.status_timeout = float(rospy.get_param("~status_timeout_sec", 0.50))
        self.tf_timeout = float(rospy.get_param("~tf_lookup_timeout_sec", 0.05))
        self.maximum_tf_age = float(rospy.get_param("~maximum_tf_age_sec", 0.30))
        self.lookahead = float(rospy.get_param("~lookahead_m", 0.30))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance_m", 0.06))
        self.goal_slowdown_distance = float(
            rospy.get_param("~goal_slowdown_distance_m", 0.45)
        )
        self.position_gain = float(rospy.get_param("~position_gain", 0.8))
        self.heading_gain = float(rospy.get_param("~heading_gain", 1.2))
        self.maximum_linear = float(
            rospy.get_param("~maximum_linear_speed_mps", 0.08)
        )
        self.maximum_angular = float(
            rospy.get_param("~maximum_angular_speed_radps", 0.30)
        )
        self.maximum_linear_accel = float(
            rospy.get_param("~maximum_linear_accel_mps2", 0.20)
        )
        self.maximum_angular_accel = float(
            rospy.get_param("~maximum_angular_accel_radps2", 0.60)
        )
        self.yaw_slow_threshold = math.radians(
            float(rospy.get_param("~yaw_slow_threshold_deg", 10.0))
        )
        self.yaw_stop_threshold = math.radians(
            float(rospy.get_param("~yaw_stop_threshold_deg", 25.0))
        )

        self.path = None
        self.path_stamp = None
        self.path_receipt = None
        self.fusion_status = None
        self.fusion_status_receipt = None
        self.perception_status = None
        self.perception_status_receipt = None
        self.last_command = np.zeros(3, dtype=np.float64)
        self.last_step_time = rospy.Time.now()
        self.last_conflict_audit = rospy.Time(0)
        self.cached_conflicts = []
        self.goal_reported = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.command_pub = rospy.Publisher(self.output_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.finished_pub = rospy.Publisher("~finished", Bool, queue_size=1, latch=True)
        self.path_sub = rospy.Subscriber(
            self.path_topic, Path, self._path_callback, queue_size=1
        )
        self.fusion_status_sub = rospy.Subscriber(
            self.fusion_status_topic,
            String,
            self._fusion_status_callback,
            queue_size=1,
        )
        self.perception_status_sub = rospy.Subscriber(
            self.perception_status_topic,
            String,
            self._perception_status_callback,
            queue_size=1,
        )
        self.enable_service = rospy.Service(
            "~set_enabled", SetBool, self._set_enabled
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.rate_hz, 1.0)), self._timer_callback
        )
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "map path controller ready: path=%s output=%s motion_capable=%s armed=%s",
            self.path_topic,
            self.output_topic,
            self.motion_capable,
            self.armed,
        )
        if not self.motion_capable:
            rospy.logwarn("enable_motion=false: navigation controller is dry-run only")

    def _path_callback(self, message):
        with self.lock:
            self.path = np.asarray(
                [
                    [pose.pose.position.x, pose.pose.position.y]
                    for pose in message.poses
                ],
                dtype=np.float64,
            ).reshape(-1, 2)
            self.path_stamp = message.header.stamp.to_sec()
            self.path_receipt = rospy.Time.now().to_sec()

    def _fusion_status_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid map fusion status JSON")
            return
        with self.lock:
            self.fusion_status = payload
            self.fusion_status_receipt = rospy.Time.now().to_sec()

    def _perception_status_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid perception status JSON")
            return
        with self.lock:
            self.perception_status = payload
            self.perception_status_receipt = rospy.Time.now().to_sec()

    def _lookup_pose(self):
        transform = self.tf_buffer.lookup_transform(
            self.map_frame,
            self.base_frame,
            rospy.Time(0),
            rospy.Duration(self.tf_timeout),
        )
        translation = transform.transform.translation
        transform_stamp = transform.header.stamp.to_sec()
        if transform_stamp <= 0.0 or (
            rospy.Time.now().to_sec() - transform_stamp > self.maximum_tf_age
        ):
            raise RuntimeError("localization TF is stale")
        return (
            float(translation.x),
            float(translation.y),
            _quaternion_yaw(transform.transform.rotation),
        )

    def _topic_publishers(self, topic):
        try:
            publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
            resolved = rospy.resolve_name(topic)
            for published_topic, nodes in publishers:
                if rospy.resolve_name(published_topic) == resolved:
                    return set(nodes)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "publisher audit failed: %s", exc)
            return {"master_audit_failed"}
        return set()

    def _conflicts(self, force=False):
        if not self.require_exclusive_output:
            return []
        now = rospy.Time.now()
        if not force and (now - self.last_conflict_audit).to_sec() < 0.5:
            return list(self.cached_conflicts)
        own_name = rospy.get_name()
        output_publishers = self._topic_publishers(self.output_topic) - {own_name}
        conflicts = ["output:" + name for name in sorted(output_publishers)]
        if rospy.resolve_name(self.chassis_topic) != rospy.resolve_name(self.output_topic):
            chassis_publishers = self._topic_publishers(self.chassis_topic)
            unexpected = chassis_publishers - self.expected_chassis_publishers
            conflicts.extend("chassis:" + name for name in sorted(unexpected))
        self.cached_conflicts = conflicts
        self.last_conflict_audit = now
        return list(conflicts)

    def _inputs_ready(self, now_sec):
        if self.path is None or len(self.path) < 2:
            return False, "empty_fused_path"
        if self.path_receipt is None or now_sec - self.path_receipt > self.path_timeout:
            return False, "fused_path_timeout"
        if self.fusion_status is None or self.fusion_status_receipt is None:
            return False, "fusion_status_missing"
        if now_sec - self.fusion_status_receipt > self.status_timeout:
            return False, "fusion_status_timeout"
        if not bool(self.fusion_status.get("safe", False)):
            return False, "fusion_unsafe:" + str(self.fusion_status.get("mode", "unknown"))
        fusion_stamp = self.fusion_status.get("stamp")
        if fusion_stamp is None or self.path_stamp is None or abs(
            float(fusion_stamp) - float(self.path_stamp)
        ) > 1e-3:
            return False, "path_status_stamp_mismatch"
        if self.perception_status is None or self.perception_status_receipt is None:
            return False, "perception_status_missing"
        if now_sec - self.perception_status_receipt > self.status_timeout:
            return False, "perception_status_timeout"
        if str(self.perception_status.get("state", "STOP")) == "STOP":
            return False, "perception_stop:" + str(
                self.perception_status.get("stop_reason", "unknown")
            )
        conflicts = self._conflicts()
        if conflicts:
            return False, "publisher_conflict:" + ",".join(conflicts)
        return True, "ready"

    def _set_enabled(self, request):
        with self.lock:
            if not request.data:
                self.armed = False
                self._publish_zero()
                return SetBoolResponse(success=True, message="controller disarmed")
            if not self.motion_capable:
                return SetBoolResponse(
                    success=False,
                    message="launch with enable_motion:=true before arming",
                )
            ready, reason = self._inputs_ready(rospy.Time.now().to_sec())
            if not ready:
                return SetBoolResponse(success=False, message=reason)
            if self.command_pub.get_num_connections() <= 0:
                return SetBoolResponse(
                    success=False,
                    message="output command topic has no subscriber",
                )
            self.armed = True
            return SetBoolResponse(success=True, message="controller armed")

    @staticmethod
    def _twist(values):
        message = Twist()
        message.linear.x = float(values[0])
        message.linear.y = float(values[1])
        message.angular.z = float(values[2])
        return message

    def _publish_zero(self):
        self.last_command[:] = 0.0
        self.command_pub.publish(Twist())

    def _limited_command(self, desired, dt):
        values = np.asarray(
            [desired["linear_x"], desired["linear_y"], desired["angular_z"]],
            dtype=np.float64,
        )
        linear_delta = self.maximum_linear_accel * max(dt, 0.0)
        angular_delta = self.maximum_angular_accel * max(dt, 0.0)
        values[:2] = np.clip(
            values[:2],
            self.last_command[:2] - linear_delta,
            self.last_command[:2] + linear_delta,
        )
        values[2] = np.clip(
            values[2],
            self.last_command[2] - angular_delta,
            self.last_command[2] + angular_delta,
        )
        norm = float(np.linalg.norm(values[:2]))
        if norm > self.maximum_linear and norm > 1e-9:
            values[:2] *= self.maximum_linear / norm
        self.last_command = values
        return values

    def _publish_status(self, ready, reason, pose, desired=None):
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "motion_capable": self.motion_capable,
            "armed": self.armed,
            "ready": bool(ready),
            "reason": reason,
            "output_cmd_topic": self.output_topic,
            "robot_pose_map": None if pose is None else list(pose),
            "command": {
                "linear_x": float(self.last_command[0]),
                "linear_y": float(self.last_command[1]),
                "angular_z": float(self.last_command[2]),
            },
            "goal_distance_m": None if desired is None else desired["goal_distance_m"],
            "yaw_error_rad": None if desired is None else desired["yaw_error_rad"],
        }
        self.status_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, allow_nan=False))
        )

    def _timer_callback(self, _event):
        with self.lock:
            now = rospy.Time.now()
            now_sec = now.to_sec()
            dt = max(0.0, min(0.2, (now - self.last_step_time).to_sec()))
            self.last_step_time = now
            ready, reason = self._inputs_ready(now_sec)
            pose = None
            desired = None
            if ready:
                try:
                    pose = self._lookup_pose()
                except Exception as exc:
                    ready = False
                    reason = "tf_unavailable:" + str(exc)
            if ready:
                try:
                    desired = compute_navigation_command(
                        self.path,
                        pose,
                        self.lookahead,
                        self.goal_tolerance,
                        self.goal_slowdown_distance,
                        self.position_gain,
                        self.heading_gain,
                        self.maximum_linear,
                        self.maximum_angular,
                        self.yaw_slow_threshold,
                        self.yaw_stop_threshold,
                    )
                except ValueError as exc:
                    ready = False
                    reason = "invalid_path:" + str(exc)
            if not ready or not self.motion_capable or not self.armed:
                self._publish_zero()
            elif desired["goal_reached"]:
                self._publish_zero()
                if not self.goal_reported:
                    self.finished_pub.publish(Bool(data=True))
                    rospy.loginfo("yellow-track navigation goal reached")
                    self.goal_reported = True
            else:
                self.goal_reported = False
                command = self._limited_command(desired, dt)
                self.command_pub.publish(self._twist(command))
            self._publish_status(ready, reason, pose, desired)

    def _shutdown(self):
        for _ in range(5):
            try:
                self.command_pub.publish(Twist())
                rospy.sleep(0.04)
            except Exception:
                break


if __name__ == "__main__":
    rospy.init_node("yellow_track_map_path_controller")
    MapPathController()
    rospy.spin()
