#!/usr/bin/env python3
"""Express chassis velocity feedback at the compensated rotation center."""

import math

import rospy
from geometry_msgs.msg import Twist


class VelocityCenterCorrector:
    def __init__(self):
        input_topic = rospy.get_param("~input_topic", "/vel_raw")
        output_topic = rospy.get_param("~output_topic", "/vel_raw_centered")
        command_topic = rospy.get_param("~command_topic", "/cmd_vel_nav_raw")

        self.max_command_translation = float(
            rospy.get_param("~max_command_translation", 0.03)
        )
        self.min_command_angular = float(
            rospy.get_param("~min_command_angular", 0.05)
        )
        self.min_feedback_angular = float(
            rospy.get_param("~min_feedback_angular", 0.02)
        )
        self.hold_time = float(rospy.get_param("~hold_time", 0.40))
        self.apply_during_translation = bool(
            rospy.get_param("~apply_during_translation", True)
        )

        # Coefficients use the reported /vel_raw angular velocity. The CCW
        # values were fitted from rviz_rotation_drift.bag on 2026-07-18.
        self.ccw_vx_per_w = float(rospy.get_param("~ccw_vx_per_w", 0.108895))
        self.ccw_vy_per_w = float(rospy.get_param("~ccw_vy_per_w", -0.116189))
        self.cw_vx_per_w = float(rospy.get_param("~cw_vx_per_w", 0.143333))
        self.cw_vy_per_w = float(rospy.get_param("~cw_vy_per_w", -0.106667))

        self.active_until = rospy.Time(0)
        self.shutting_down = False
        self.publisher = rospy.Publisher(output_topic, Twist, queue_size=20)
        self.command_subscriber = rospy.Subscriber(
            command_topic, Twist, self.command_callback, queue_size=10
        )
        self.feedback_subscriber = rospy.Subscriber(
            input_topic, Twist, self.feedback_callback, queue_size=50,
            tcp_nodelay=True
        )
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo(
            "Velocity-center odometry correction enabled: input=%s output=%s "
            "command=%s CCW=(%.6f, %.6f) CW=(%.6f, %.6f)",
            input_topic,
            output_topic,
            command_topic,
            self.ccw_vx_per_w,
            self.ccw_vy_per_w,
            self.cw_vx_per_w,
            self.cw_vy_per_w,
        )

    def shutdown(self):
        self.shutting_down = True

    def command_callback(self, msg):
        if self.shutting_down:
            return
        translation = math.hypot(msg.linear.x, msg.linear.y)
        now = rospy.Time.now()
        correction_allowed = (
            self.apply_during_translation
            or translation <= self.max_command_translation
        )
        if correction_allowed and abs(msg.angular.z) >= self.min_command_angular:
            self.active_until = now + rospy.Duration(self.hold_time)
        elif not correction_allowed:
            self.active_until = now

    def feedback_callback(self, msg):
        if self.shutting_down:
            return
        corrected = Twist()
        corrected.linear.x = msg.linear.x
        corrected.linear.y = msg.linear.y
        corrected.linear.z = msg.linear.z
        corrected.angular.x = msg.angular.x
        corrected.angular.y = msg.angular.y
        corrected.angular.z = msg.angular.z

        if (
            rospy.Time.now() <= self.active_until
            and abs(msg.angular.z) >= self.min_feedback_angular
        ):
            if msg.angular.z > 0.0:
                vx_per_w = self.ccw_vx_per_w
                vy_per_w = self.ccw_vy_per_w
            else:
                vx_per_w = self.cw_vx_per_w
                vy_per_w = self.cw_vy_per_w
            corrected.linear.x -= vx_per_w * msg.angular.z
            corrected.linear.y -= vy_per_w * msg.angular.z

        try:
            self.publisher.publish(corrected)
        except rospy.ROSException:
            if not self.shutting_down and not rospy.is_shutdown():
                raise


if __name__ == "__main__":
    rospy.init_node("vel_center_corrector")
    VelocityCenterCorrector()
    rospy.spin()
