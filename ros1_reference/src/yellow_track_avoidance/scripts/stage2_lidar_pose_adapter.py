#!/usr/bin/env python3
"""Add honest covariance and jump rejection to RF2O for Stage-2 EKF fusion."""

import json
import math
import threading

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class Stage2LidarPoseAdapter:
    def __init__(self):
        self.lock = threading.Lock()
        self.input_topic = str(rospy.get_param("~input_topic", "/odom_rf2o"))
        self.output_topic = str(
            rospy.get_param(
                "~output_topic", "/yellow_track_avoidance/stage2_lidar_pose"
            )
        )
        self.frame_id = str(rospy.get_param("~frame_id", "odom")).lstrip("/")
        self.xy_sigma = float(rospy.get_param("~xy_sigma_m", 0.025))
        self.maximum_step = float(rospy.get_param("~maximum_step_m", 0.15))
        self.maximum_gap = float(rospy.get_param("~maximum_gap_sec", 0.5))
        self.last_stamp = None
        self.last_xy = None
        self.published = 0
        self.rejected = 0
        self.pub = rospy.Publisher(
            self.output_topic, PoseWithCovarianceStamped, queue_size=5
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(
            self.input_topic, Odometry, self._callback, queue_size=10
        )
        self.timer = rospy.Timer(rospy.Duration(1.0), self._status)
        rospy.loginfo(
            "Stage-2 RF2O adapter: %s -> %s, xy sigma %.3f m",
            self.input_topic,
            self.output_topic,
            self.xy_sigma,
        )

    def _callback(self, message):
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        if not math.isfinite(x) or not math.isfinite(y):
            with self.lock:
                self.rejected += 1
            return
        stamp = message.header.stamp
        with self.lock:
            if self.last_xy is not None and self.last_stamp is not None:
                dt = (stamp - self.last_stamp).to_sec()
                step = math.hypot(x - self.last_xy[0], y - self.last_xy[1])
                if 0.0 < dt <= self.maximum_gap and step > self.maximum_step:
                    self.rejected += 1
                    rospy.logwarn_throttle(
                        1.0,
                        "rejecting RF2O jump %.3f m in %.3f s",
                        step,
                        dt,
                    )
                    return
            self.last_stamp = stamp
            self.last_xy = (x, y)
            self.published += 1

        output = PoseWithCovarianceStamped()
        output.header.stamp = stamp
        output.header.frame_id = self.frame_id
        output.pose.pose = message.pose.pose
        variance = self.xy_sigma * self.xy_sigma
        output.pose.covariance = [0.0] * 36
        output.pose.covariance[0] = variance
        output.pose.covariance[7] = variance
        output.pose.covariance[14] = 1.0e6
        output.pose.covariance[21] = 1.0e6
        output.pose.covariance[28] = 1.0e6
        # RF2O yaw is deliberately not fused; Yesense remains the yaw owner.
        output.pose.covariance[35] = 1.0e3
        self.pub.publish(output)

    def _status(self, _event):
        with self.lock:
            payload = {
                "input_topic": self.input_topic,
                "output_topic": self.output_topic,
                "xy_sigma_m": self.xy_sigma,
                "published": self.published,
                "rejected": self.rejected,
                "last_xy_m": self.last_xy,
            }
        self.status_pub.publish(String(data=json.dumps(payload, allow_nan=False)))


def main():
    rospy.init_node("yellow_track_stage2_lidar_pose_adapter")
    Stage2LidarPoseAdapter()
    rospy.spin()


if __name__ == "__main__":
    main()
