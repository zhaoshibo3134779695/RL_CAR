#!/usr/bin/env python3
"""Publish simulated odometry for a no-chassis Stage-1 planning run."""

import json
import math
import threading

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
import rospy
from std_msgs.msg import String
import tf2_ros

from yellow_track_avoidance.shadow_odom import PlanarState, integrate_body_twist


class Stage1ShadowOdom:
    def __init__(self):
        self.lock = threading.RLock()
        self.odom_frame = str(rospy.get_param("~odom_frame", "odom"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_footprint"))
        self.cmd_topic = str(rospy.get_param("~cmd_topic", "/cmd_vel"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.command_timeout_sec = float(
            rospy.get_param("~command_timeout_sec", 0.35)
        )
        if self.rate_hz <= 0.0 or self.command_timeout_sec <= 0.0:
            raise ValueError("shadow odometry timing parameters must be positive")

        self.state = PlanarState()
        self.command = Twist()
        self.last_command_stamp = None
        self.last_update_stamp = rospy.Time.now()
        self.distance_m = 0.0
        self.command_count = 0
        self.maximum_abs_yaw_rad = 0.0
        self.maximum_abs_wz_radps = 0.0
        self.last_status = None

        self.odom_pub = rospy.Publisher(
            self.odom_topic, Odometry, queue_size=5
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.cmd_sub = rospy.Subscriber(
            self.cmd_topic, Twist, self._command_callback, queue_size=1
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.rate_hz), self._timer_callback
        )
        rospy.logwarn(
            "STAGE-1 SHADOW MODE: /cmd_vel is integrated in software; "
            "no physical chassis driver is present"
        )
        rospy.on_shutdown(self._shutdown)

    def _command_callback(self, message):
        with self.lock:
            self.command = message
            self.last_command_stamp = rospy.Time.now()
            self.command_count += 1

    @staticmethod
    def _quaternion(yaw):
        half = 0.5 * float(yaw)
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def _active_twist(self, now):
        if self.last_command_stamp is None:
            return 0.0, 0.0, 0.0
        if (now - self.last_command_stamp).to_sec() > self.command_timeout_sec:
            return 0.0, 0.0, 0.0
        return (
            float(self.command.linear.x),
            float(self.command.linear.y),
            float(self.command.angular.z),
        )

    def _timer_callback(self, _event):
        now = rospy.Time.now()
        with self.lock:
            dt = min(0.1, max(0.0, (now - self.last_update_stamp).to_sec()))
            self.last_update_stamp = now
            vx, vy, wz = self._active_twist(now)
            previous = self.state
            self.state = integrate_body_twist(previous, vx, vy, wz, dt)
            self.distance_m += math.hypot(
                self.state.x_m - previous.x_m,
                self.state.y_m - previous.y_m,
            )
            self.maximum_abs_yaw_rad = max(
                self.maximum_abs_yaw_rad, abs(self.state.yaw_rad)
            )
            self.maximum_abs_wz_radps = max(self.maximum_abs_wz_radps, abs(wz))
            qx, qy, qz, qw = self._quaternion(self.state.yaw_rad)

            odom = Odometry()
            odom.header.stamp = now
            odom.header.frame_id = self.odom_frame
            odom.child_frame_id = self.base_frame
            odom.pose.pose.position.x = self.state.x_m
            odom.pose.pose.position.y = self.state.y_m
            odom.pose.pose.orientation.x = qx
            odom.pose.pose.orientation.y = qy
            odom.pose.pose.orientation.z = qz
            odom.pose.pose.orientation.w = qw
            odom.twist.twist.linear.x = vx
            odom.twist.twist.linear.y = vy
            odom.twist.twist.angular.z = wz
            self.odom_pub.publish(odom)

            transform = TransformStamped()
            transform.header.stamp = now
            transform.header.frame_id = self.odom_frame
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = self.state.x_m
            transform.transform.translation.y = self.state.y_m
            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(transform)

            status = {
                            "mode": "shadow_kinematic_simulation",
                            "physical_chassis_enabled": False,
                            "x_m": self.state.x_m,
                            "y_m": self.state.y_m,
                            "yaw_rad": self.state.yaw_rad,
                            "vx_mps": vx,
                            "vy_mps": vy,
                            "wz_radps": wz,
                            "integrated_distance_m": self.distance_m,
                            "command_count": self.command_count,
                            "maximum_abs_yaw_rad": self.maximum_abs_yaw_rad,
                            "maximum_abs_wz_radps": self.maximum_abs_wz_radps,
                        }
            self.last_status = status
            encoded = json.dumps(status, sort_keys=True)
            self.status_pub.publish(String(data=encoded))
            rospy.loginfo_throttle(1.0, "SHADOW_STATE %s", encoded)

    def _shutdown(self):
        with self.lock:
            status = self.last_status
        if status is not None:
            rospy.logwarn("SHADOW_STATE_FINAL %s", json.dumps(status, sort_keys=True))


def main():
    rospy.init_node("stage1_shadow_odom")
    Stage1ShadowOdom()
    rospy.spin()


if __name__ == "__main__":
    main()
