#!/usr/bin/env python3
import math
import threading

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped


class OdomMapTf:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.transform_tolerance = rospy.get_param("~transform_tolerance", 0.5)
        self.desired_pose = (
            rospy.get_param("~initial_x", 0.0),
            rospy.get_param("~initial_y", 0.0),
            rospy.get_param("~initial_yaw", 0.0),
        )
        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()
        self.lock = threading.Lock()
        self.pending_update = True
        self.map_to_odom = None
        rospy.Subscriber(
            "/initialpose", PoseWithCovarianceStamped, self.initialpose_cb, queue_size=1
        )

    def initialpose_cb(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        with self.lock:
            self.desired_pose = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                yaw,
            )
            self.pending_update = True
        rospy.loginfo(
            "Odom-only initial pose requested: x=%.3f y=%.3f yaw=%.1f deg",
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            math.degrees(yaw),
        )

    @staticmethod
    def pose_matrix(translation, quaternion):
        return np.dot(
            tf.transformations.translation_matrix(translation),
            tf.transformations.quaternion_matrix(quaternion),
        )

    def update_map_to_odom(self):
        try:
            odom_translation, odom_rotation = self.listener.lookupTransform(
                self.odom_frame, self.base_frame, rospy.Time(0)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        with self.lock:
            x, y, yaw = self.desired_pose
            desired = self.pose_matrix(
                (x, y, 0.0), tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
            )
            odom_to_base = self.pose_matrix(odom_translation, odom_rotation)
            map_to_odom = np.dot(desired, tf.transformations.inverse_matrix(odom_to_base))
            translation = tf.transformations.translation_from_matrix(map_to_odom)
            rotation = tf.transformations.quaternion_from_matrix(map_to_odom)
            self.map_to_odom = (translation, rotation)
            self.pending_update = False
        rospy.loginfo("map->odom fixed from the current odometry pose")

    def spin(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                pending = self.pending_update
                transform = self.map_to_odom
            if pending:
                self.update_map_to_odom()
                with self.lock:
                    transform = self.map_to_odom
            if transform is not None:
                translation, rotation = transform
                self.broadcaster.sendTransform(
                    translation,
                    rotation,
                    rospy.Time.now() + rospy.Duration(self.transform_tolerance),
                    self.odom_frame,
                    self.map_frame,
                )
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("odom_map_tf")
    OdomMapTf().spin()
