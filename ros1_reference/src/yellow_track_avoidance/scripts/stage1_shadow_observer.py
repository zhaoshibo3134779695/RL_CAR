#!/usr/bin/env python3
"""Measure Stage-1 shadow trajectory geometry without publishing control.

This node is deliberately diagnostic-only.  It observes the frozen semantic
scene, simulated odometry and TEB local path, then reports exact oriented-
rectangle clearances.  It has no publishers consumed by navigation.
"""

import json
import math
import threading

from nav_msgs.msg import Odometry, Path
import rospy
from std_msgs.msg import String


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _unit(vector):
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        raise ValueError("zero-length corridor vector")
    return vector[0] / length, vector[1] / length


def _inner_edges(boundary):
    left = boundary["left_base_m"]
    right = boundary["right_base_m"]
    half_width = 0.5 * float(boundary["yellow_line_width_m"])
    left_inner = []
    right_inner = []
    for left_point, right_point in zip(left, right):
        across = _unit(
            (right_point[0] - left_point[0], right_point[1] - left_point[1])
        )
        left_inner.append(
            (left_point[0] + across[0] * half_width,
             left_point[1] + across[1] * half_width)
        )
        right_inner.append(
            (right_point[0] - across[0] * half_width,
             right_point[1] - across[1] * half_width)
        )
    return tuple(left_inner), tuple(right_inner)


def _rectangle_vertices(x_value, y_value, yaw_value, half_length, half_width):
    cosine = math.cos(yaw_value)
    sine = math.sin(yaw_value)
    output = []
    for local_x, local_y in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        output.append(
            (
                x_value + cosine * local_x - sine * local_y,
                y_value + sine * local_x + cosine * local_y,
            )
        )
    return tuple(output)


def _circle_clearance(x_value, y_value, yaw_value, center, radius,
                      half_length, half_width):
    dx = float(center[0]) - x_value
    dy = float(center[1]) - y_value
    cosine = math.cos(yaw_value)
    sine = math.sin(yaw_value)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    closest_x = min(half_length, max(-half_length, local_x))
    closest_y = min(half_width, max(-half_width, local_y))
    return math.hypot(local_x - closest_x, local_y - closest_y) - radius


def _corridor_clearance(x_value, y_value, yaw_value, boundary,
                        half_length, half_width):
    left, right = _inner_edges(boundary)
    tangent = _unit((left[1][0] - left[0][0], left[1][1] - left[0][1]))
    across = _unit((right[0][0] - left[0][0], right[0][1] - left[0][1]))
    width = (
        (right[0][0] - left[0][0]) * across[0]
        + (right[0][1] - left[0][1]) * across[1]
    )
    lateral_values = []
    for vertex in _rectangle_vertices(
        x_value, y_value, yaw_value, half_length, half_width
    ):
        relative = vertex[0] - left[0][0], vertex[1] - left[0][1]
        lateral_values.append(relative[0] * across[0] + relative[1] * across[1])
    return min(min(lateral_values), width - max(lateral_values)), math.atan2(
        tangent[1], tangent[0]
    )


def _angle_difference(first, second):
    return math.atan2(math.sin(first - second), math.cos(first - second))


class Stage1ShadowObserver:
    def __init__(self):
        self.lock = threading.RLock()
        self.half_length = float(rospy.get_param("~vehicle_half_length_m", 0.17))
        self.half_width = float(rospy.get_param("~vehicle_half_width_m", 0.165))
        self.disc_radius = float(rospy.get_param("~disc_collision_radius_m", 0.09))
        self.boundary = None
        self.discs = ()
        self.current = None
        self.local_plan = None
        self.max_abs_yaw_error = 0.0
        self.minimum_current_disc_clearance = float("inf")
        self.minimum_current_boundary_clearance = float("inf")
        self.minimum_predicted_disc_clearance = float("inf")
        self.minimum_predicted_boundary_clearance = float("inf")
        self.last_payload = None

        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        rospy.Subscriber(
            "/yellow_track_avoidance/corridor_boundaries",
            String,
            self._boundary_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yellow_track_teb_compact_obstacles/status",
            String,
            self._obstacle_callback,
            queue_size=1,
        )
        rospy.Subscriber("/odom", Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber(
            "/move_base/TebLocalPlannerROS/local_plan",
            Path,
            self._path_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(0.5), self._timer_callback)
        rospy.on_shutdown(self._shutdown)

    @staticmethod
    def _decode(message):
        try:
            value = json.loads(message.data)
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError):
            return None

    def _boundary_callback(self, message):
        value = self._decode(message)
        if value is not None and value.get("boundary_pair_valid"):
            with self.lock:
                self.boundary = value

    def _obstacle_callback(self, message):
        value = self._decode(message)
        if value is None or not value.get("ready"):
            return
        discs = []
        for decision in value.get("passage_decisions", []):
            center = decision.get("center_m")
            if isinstance(center, list) and len(center) >= 2:
                discs.append((float(center[0]), float(center[1])))
        if discs:
            with self.lock:
                self.discs = tuple(discs)

    def _odom_callback(self, message):
        pose = message.pose.pose
        value = (float(pose.position.x), float(pose.position.y), _yaw(pose.orientation))
        with self.lock:
            self.current = value

    def _path_callback(self, message):
        values = []
        for stamped in message.poses:
            pose = stamped.pose
            values.append(
                (float(pose.position.x), float(pose.position.y), _yaw(pose.orientation))
            )
        with self.lock:
            self.local_plan = tuple(values)

    @staticmethod
    def _finite_or_none(value):
        if value is None:
            return None
        return float(value) if math.isfinite(value) else None

    def _evaluate_pose(self, pose, boundary, discs):
        x_value, y_value, yaw_value = pose
        disc_clearance = min(
            (
                _circle_clearance(
                    x_value,
                    y_value,
                    yaw_value,
                    center,
                    self.disc_radius,
                    self.half_length,
                    self.half_width,
                )
                for center in discs
            ),
            default=float("inf"),
        )
        if boundary is None:
            return disc_clearance, float("inf"), None
        boundary_clearance, corridor_yaw = _corridor_clearance(
            x_value,
            y_value,
            yaw_value,
            boundary,
            self.half_length,
            self.half_width,
        )
        yaw_error = abs(_angle_difference(yaw_value, corridor_yaw))
        return disc_clearance, boundary_clearance, yaw_error

    def _timer_callback(self, _event):
        with self.lock:
            boundary = self.boundary
            discs = self.discs
            current = self.current
            local_plan = self.local_plan
        current_disc = current_boundary = yaw_error = None
        if current is not None:
            current_disc, current_boundary, yaw_error = self._evaluate_pose(
                current, boundary, discs
            )
        predicted_disc = predicted_boundary = float("inf")
        predicted_max_yaw_error = 0.0
        for pose in local_plan or ():
            disc_value, boundary_value, pose_yaw_error = self._evaluate_pose(
                pose, boundary, discs
            )
            predicted_disc = min(predicted_disc, disc_value)
            predicted_boundary = min(predicted_boundary, boundary_value)
            if pose_yaw_error is not None:
                predicted_max_yaw_error = max(predicted_max_yaw_error, pose_yaw_error)

        with self.lock:
            if yaw_error is not None:
                self.max_abs_yaw_error = max(self.max_abs_yaw_error, yaw_error)
            if current_disc is not None:
                self.minimum_current_disc_clearance = min(
                    self.minimum_current_disc_clearance, current_disc
                )
                self.minimum_current_boundary_clearance = min(
                    self.minimum_current_boundary_clearance, current_boundary
                )
            self.minimum_predicted_disc_clearance = min(
                self.minimum_predicted_disc_clearance, predicted_disc
            )
            self.minimum_predicted_boundary_clearance = min(
                self.minimum_predicted_boundary_clearance, predicted_boundary
            )
            self.max_abs_yaw_error = max(
                self.max_abs_yaw_error, predicted_max_yaw_error
            )
            payload = {
                "stamp": rospy.Time.now().to_sec(),
                "physical_chassis_enabled": False,
                "scene_ready": boundary is not None and len(discs) == 2,
                "disc_count": len(discs),
                "disc_centers_m": [list(value) for value in discs],
                "corridor_left_centerline_m": (
                    boundary.get("left_base_m") if boundary is not None else None
                ),
                "corridor_right_centerline_m": (
                    boundary.get("right_base_m") if boundary is not None else None
                ),
                "current_pose": list(current) if current is not None else None,
                "local_plan_pose_count": len(local_plan or ()),
                "current_disc_clearance_m": self._finite_or_none(current_disc),
                "current_boundary_clearance_m": self._finite_or_none(current_boundary),
                "predicted_disc_clearance_m": self._finite_or_none(predicted_disc),
                "predicted_boundary_clearance_m": self._finite_or_none(predicted_boundary),
                "maximum_abs_yaw_error_deg": math.degrees(self.max_abs_yaw_error),
                "minimum_current_disc_clearance_m": self._finite_or_none(
                    self.minimum_current_disc_clearance
                ),
                "minimum_current_boundary_clearance_m": self._finite_or_none(
                    self.minimum_current_boundary_clearance
                ),
                "minimum_predicted_disc_clearance_m": self._finite_or_none(
                    self.minimum_predicted_disc_clearance
                ),
                "minimum_predicted_boundary_clearance_m": self._finite_or_none(
                    self.minimum_predicted_boundary_clearance
                ),
            }
            self.last_payload = payload
        message = String(data=json.dumps(payload, allow_nan=False, sort_keys=True))
        self.status_pub.publish(message)
        rospy.loginfo_throttle(1.0, "SHADOW_GEOMETRY %s", message.data)

    def _shutdown(self):
        with self.lock:
            payload = self.last_payload
        if payload is not None:
            rospy.logwarn("SHADOW_GEOMETRY_FINAL %s", json.dumps(payload, sort_keys=True))


def main():
    rospy.init_node("stage1_shadow_observer")
    Stage1ShadowObserver()
    rospy.spin()


if __name__ == "__main__":
    main()
