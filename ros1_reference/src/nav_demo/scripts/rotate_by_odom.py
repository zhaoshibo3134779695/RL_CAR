#!/usr/bin/env python3
import math
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class OdomRotation:
    def __init__(self):
        self.current_yaw = None
        self.current_x = None
        self.current_y = None
        command_topic = rospy.get_param("~command_topic", "/cmd_vel")
        self.publisher = rospy.Publisher(command_topic, Twist, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        rospy.on_shutdown(self.stop)
        rospy.loginfo("Rotation test command topic: %s", command_topic)

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def stop(self):
        if not hasattr(self, "publisher"):
            return
        zero = Twist()
        for _ in range(10):
            self.publisher.publish(zero)
            rospy.sleep(0.02)

    def run(self):
        angle_deg = rospy.get_param("~angle_deg", 90.0)
        max_speed = abs(rospy.get_param("~speed", 0.10))
        min_speed = abs(rospy.get_param("~min_speed", 0.08))
        center_offset_x = rospy.get_param("~center_offset_x", 0.0)
        center_offset_y = rospy.get_param("~center_offset_y", 0.0)
        tolerance = math.radians(abs(rospy.get_param("~tolerance_deg", 0.5)))
        timeout = rospy.get_param("~timeout", 60.0)
        stall_timeout = rospy.get_param("~stall_timeout", 3.0)

        if abs(angle_deg) < 0.1 or max_speed <= 0.0:
            rospy.logerr("angle_deg and speed must be non-zero")
            return

        while self.current_yaw is None and not rospy.is_shutdown():
            rospy.sleep(0.02)

        direction = 1.0 if angle_deg > 0.0 else -1.0
        target = math.radians(abs(angle_deg))
        min_speed = min(min_speed, max_speed)
        start_x = self.current_x
        start_y = self.current_y
        start_yaw = self.current_yaw
        previous_yaw = self.current_yaw
        accumulated = 0.0
        started = time.monotonic()
        last_progress = 0.0
        last_motion = started
        rate = rospy.Rate(30)

        rospy.loginfo(
            "Rotating %.1f deg at up to %.3f rad/s around body offset "
            "(%.4f, %.4f) m",
            angle_deg, max_speed, center_offset_x, center_offset_y)
        while not rospy.is_shutdown():
            accumulated += normalize_angle(self.current_yaw - previous_yaw)
            previous_yaw = self.current_yaw
            progress = direction * accumulated
            remaining = target - progress

            if progress > last_progress + math.radians(0.1):
                last_progress = progress
                last_motion = time.monotonic()

            if remaining <= tolerance:
                break
            if time.monotonic() - last_motion > stall_timeout:
                rospy.logerr("No odom rotation detected for %.1f seconds; stopping", stall_timeout)
                self.stop()
                return
            if time.monotonic() - started > timeout:
                rospy.logerr("Rotation timed out at %.2f deg", math.degrees(progress))
                self.stop()
                return

            command = Twist()
            command.angular.z = direction * min(
                max_speed, max(min_speed, remaining * 0.8)
            )
            # Shift the commanded instantaneous center of rotation from the
            # base origin to the requested body-center offset.
            command.linear.x = command.angular.z * center_offset_y
            command.linear.y = -command.angular.z * center_offset_x
            self.publisher.publish(command)
            rate.sleep()

        self.stop()
        rospy.sleep(0.30)
        delta_x = self.current_x - start_x
        delta_y = self.current_y - start_y
        body_delta_x = (
            math.cos(start_yaw) * delta_x + math.sin(start_yaw) * delta_y
        )
        body_delta_y = (
            -math.sin(start_yaw) * delta_x + math.cos(start_yaw) * delta_y
        )
        rospy.loginfo(
            "Stopped at odom rotation %.2f deg; odom displacement "
            "dx=%+.4f m dy=%+.4f m distance=%.4f m; "
            "start-body displacement dx=%+.4f m dy=%+.4f m",
            math.degrees(direction * accumulated),
            delta_x,
            delta_y,
            math.hypot(delta_x, delta_y),
            body_delta_x,
            body_delta_y,
        )


if __name__ == "__main__":
    rospy.init_node("rotate_by_odom")
    OdomRotation().run()
