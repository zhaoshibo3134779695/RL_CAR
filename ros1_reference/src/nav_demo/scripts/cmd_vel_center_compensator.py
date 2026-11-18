#!/usr/bin/env python3

import math

import rospy
from geometry_msgs.msg import Twist


class CmdVelCenterCompensator:
    def __init__(self):
        input_topic = rospy.get_param("~input_topic", "/cmd_vel_nav_raw")
        output_topic = rospy.get_param("~output_topic", "/cmd_vel")
        default_x = rospy.get_param("~center_offset_x", 0.0)
        default_y = rospy.get_param("~center_offset_y", 0.0)
        self.ccw_offset_x = rospy.get_param("~ccw_center_offset_x", default_x)
        self.ccw_offset_y = rospy.get_param("~ccw_center_offset_y", default_y)
        self.cw_offset_x = rospy.get_param("~cw_center_offset_x", default_x)
        self.cw_offset_y = rospy.get_param("~cw_center_offset_y", default_y)
        self.max_translation_speed = rospy.get_param(
            "~max_translation_speed", 0.03
        )
        self.apply_during_translation = rospy.get_param(
            "~apply_during_translation", True
        )

        self.publisher = rospy.Publisher(output_topic, Twist, queue_size=1)
        self.subscriber = rospy.Subscriber(
            input_topic, Twist, self.cmd_callback, queue_size=1, tcp_nodelay=True
        )
        rospy.loginfo(
            "Navigation rotation-center compensation enabled: "
            "input=%s output=%s ccw_offset=(%.4f, %.4f) m "
            "cw_offset=(%.4f, %.4f) m max_translation_speed=%.3f m/s",
            input_topic,
            output_topic,
            self.ccw_offset_x,
            self.ccw_offset_y,
            self.cw_offset_x,
            self.cw_offset_y,
            self.max_translation_speed,
        )

    def cmd_callback(self, msg):
        if msg.angular.z < 0.0:
            offset_x = self.cw_offset_x
            offset_y = self.cw_offset_y
        else:
            offset_x = self.ccw_offset_x
            offset_y = self.ccw_offset_y

        command = Twist()
        translation_speed = math.hypot(msg.linear.x, msg.linear.y)
        if (
            self.apply_during_translation
            or translation_speed <= self.max_translation_speed
        ):
            command.linear.x = msg.linear.x + msg.angular.z * offset_y
            command.linear.y = msg.linear.y - msg.angular.z * offset_x
        else:
            command.linear.x = msg.linear.x
            command.linear.y = msg.linear.y
        command.linear.z = msg.linear.z
        command.angular.x = msg.angular.x
        command.angular.y = msg.angular.y
        command.angular.z = msg.angular.z
        self.publisher.publish(command)


if __name__ == "__main__":
    rospy.init_node("cmd_vel_center_compensator")
    CmdVelCenterCompensator()
    rospy.spin()
