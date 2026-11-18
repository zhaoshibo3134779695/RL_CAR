#!/usr/bin/env python3
"""Drive a constant-radius arc and stop at an odometry yaw change."""

import math
import time

import rospy
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class OdomArcCalibration:
    def __init__(self):
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.command_topic = rospy.get_param(
            "~command_topic", "/cmd_vel_nav_raw")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.publisher = rospy.Publisher(
            self.command_topic, Twist, queue_size=1)
        self.cancel_publisher = rospy.Publisher(
            "/move_base/cancel", GoalID, queue_size=1)
        rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.on_shutdown(self.stop)

    def odom_callback(self, message):
        self.current_x = message.pose.pose.position.x
        self.current_y = message.pose.pose.position.y
        q = message.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def stop(self):
        if not hasattr(self, "publisher"):
            return
        zero = Twist()
        for _ in range(15):
            self.publisher.publish(zero)
            rospy.sleep(0.02)

    def wait_for_odom(self, timeout):
        deadline = time.monotonic() + timeout
        while self.current_yaw is None and not rospy.is_shutdown():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "No odometry received from {}".format(self.odom_topic))
            rospy.sleep(0.02)

    @staticmethod
    def estimate_radius(body_dx, body_dy, direction, angle):
        coefficient_x = math.sin(angle)
        coefficient_y = direction * (1.0 - math.cos(angle))
        denominator = (
            coefficient_x * coefficient_x +
            coefficient_y * coefficient_y
        )
        if denominator <= 1e-9:
            return float("nan")
        return (
            body_dx * coefficient_x +
            body_dy * coefficient_y
        ) / denominator

    def run(self):
        radius = float(rospy.get_param("~radius", 0.75))
        angle_deg = float(rospy.get_param("~angle_deg", -45.0))
        max_linear = abs(float(rospy.get_param("~speed", 0.10)))
        min_linear = abs(float(rospy.get_param("~min_speed", 0.04)))
        tolerance = math.radians(abs(float(
            rospy.get_param("~tolerance_deg", 0.8))))
        slowdown_angle = math.radians(abs(float(
            rospy.get_param("~slowdown_angle_deg", 15.0))))
        timeout = float(rospy.get_param("~timeout", 30.0))
        stall_timeout = float(rospy.get_param("~stall_timeout", 3.0))
        start_delay = float(rospy.get_param("~start_delay", 2.0))
        rate_hz = float(rospy.get_param("~rate", 30.0))

        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if abs(angle_deg) < 1.0:
            raise ValueError("angle_deg magnitude must be at least 1 degree")
        if max_linear <= 0.0:
            raise ValueError("speed must be positive")
        if rate_hz <= 0.0:
            raise ValueError("rate must be positive")

        min_linear = min(min_linear, max_linear)
        direction = 1.0 if angle_deg > 0.0 else -1.0
        target_angle = math.radians(abs(angle_deg))

        self.wait_for_odom(5.0)
        self.cancel_publisher.publish(GoalID())
        self.stop()

        rospy.logwarn(
            "Arc calibration starts in %.1f s: radius=%.3f m, "
            "angle=%+.1f deg, speed<=%.3f m/s, topic=%s",
            start_delay, radius, angle_deg, max_linear,
            self.command_topic)
        if start_delay > 0.0:
            rospy.sleep(start_delay)

        start_x = self.current_x
        start_y = self.current_y
        start_yaw = self.current_yaw
        previous_yaw = self.current_yaw
        accumulated = 0.0
        last_progress = 0.0
        started = time.monotonic()
        last_motion = started
        rate = rospy.Rate(rate_hz)

        while not rospy.is_shutdown():
            accumulated += normalize_angle(self.current_yaw - previous_yaw)
            previous_yaw = self.current_yaw
            progress = direction * accumulated
            remaining = target_angle - progress

            if progress > last_progress + math.radians(0.1):
                last_progress = progress
                last_motion = time.monotonic()

            if remaining <= tolerance:
                break
            if time.monotonic() - last_motion > stall_timeout:
                raise RuntimeError(
                    "No odometry yaw progress for {:.1f} seconds".format(
                        stall_timeout))
            if time.monotonic() - started > timeout:
                raise RuntimeError(
                    "Arc timed out at {:.2f} degrees".format(
                        math.degrees(progress)))

            if slowdown_angle > 0.0:
                speed_scale = min(1.0, max(0.0, remaining / slowdown_angle))
            else:
                speed_scale = 1.0
            linear = max(min_linear, max_linear * speed_scale)
            angular = direction * linear / radius

            command = Twist()
            command.linear.x = linear
            command.angular.z = angular
            self.publisher.publish(command)
            rate.sleep()

        self.stop()
        rospy.sleep(0.30)

        signed_angle = accumulated
        map_dx = self.current_x - start_x
        map_dy = self.current_y - start_y
        body_dx = (
            math.cos(start_yaw) * map_dx +
            math.sin(start_yaw) * map_dy
        )
        body_dy = (
            -math.sin(start_yaw) * map_dx +
            math.cos(start_yaw) * map_dy
        )
        measured_angle = abs(accumulated)
        odom_radius = self.estimate_radius(
            body_dx, body_dy, direction, measured_angle)
        ideal_dx = radius * math.sin(measured_angle)
        ideal_dy = direction * radius * (1.0 - math.cos(measured_angle))

        rospy.loginfo(
            "Arc stopped at odom angle %+.2f deg; start-body odom "
            "displacement dx=%+.4f m dy=%+.4f m; ideal dx=%+.4f m "
            "dy=%+.4f m; odom-fit radius=%.4f m",
            math.degrees(signed_angle), body_dx, body_dy,
            ideal_dx, ideal_dy, odom_radius)
        rospy.loginfo(
            "Measure the REAL vehicle displacement or circle radius; "
            "odom-fit radius alone cannot reveal wheel slip or scale error.")


def main():
    rospy.init_node("drive_arc_by_odom")
    calibration = OdomArcCalibration()
    try:
        calibration.run()
    except (RuntimeError, ValueError, rospy.ROSException) as error:
        calibration.stop()
        rospy.logerr("Arc calibration failed: %s", error)


if __name__ == "__main__":
    main()
