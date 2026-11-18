#!/usr/bin/env python3
import math

import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped


class ManualMapTf:
    def __init__(self):
        self.global_frame = rospy.get_param("~global_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.x = rospy.get_param("~x", 0.0)
        self.y = rospy.get_param("~y", 0.0)
        self.z = rospy.get_param("~z", 0.0)
        self.yaw = rospy.get_param("~yaw", 0.0)
        self.br = tf.TransformBroadcaster()
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initialpose_cb, queue_size=1)

    def initialpose_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, self.yaw = tf.transformations.euler_from_quaternion(
            [quat.x, quat.y, quat.z, quat.w]
        )
        rospy.loginfo(
            "Manual map TF set to x=%.3f y=%.3f yaw=%.1f deg",
            self.x,
            self.y,
            math.degrees(self.yaw),
        )

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            quat = tf.transformations.quaternion_from_euler(0.0, 0.0, self.yaw)
            self.br.sendTransform(
                (self.x, self.y, self.z),
                quat,
                rospy.Time.now(),
                self.base_frame,
                self.global_frame,
            )
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("manual_map_tf")
    ManualMapTf().spin()
