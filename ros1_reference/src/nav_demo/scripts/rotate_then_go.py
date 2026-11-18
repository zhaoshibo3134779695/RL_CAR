#!/usr/bin/env python3
"""Execute RViz goals as rotate, straight drive, then final rotate."""

import math
import threading

import actionlib
import rospy
import tf
from actionlib_msgs.msg import GoalID, GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class RotateThenGo:
    def __init__(self):
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_footprint")
        self.max_linear = rospy.get_param("~max_linear", 0.22)
        self.min_linear = rospy.get_param("~min_linear", 0.08)
        self.max_angular = rospy.get_param("~max_angular", 0.40)
        self.min_angular = rospy.get_param("~min_angular", 0.08)
        self.rotate_accel = rospy.get_param("~rotate_accel", 0.35)
        self.position_tolerance = rospy.get_param("~position_tolerance", 0.05)
        self.heading_tolerance = rospy.get_param(
            "~heading_tolerance", math.radians(2.0)
        )
        self.rotate_kp = rospy.get_param("~rotate_kp", 1.5)
        self.heading_kp = rospy.get_param("~heading_kp", 2.0)
        self.use_dwa_straight = rospy.get_param("~use_dwa_straight", True)

        self.listener = tf.TransformListener()
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.lock = threading.Lock()
        self.goal_event = threading.Event()
        self.latest_goal = None
        self.goal_sequence = 0
        rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=1
        )
        rospy.on_shutdown(self.stop)
        threading.Thread(target=self.worker, daemon=True).start()
        rospy.loginfo("Direct rotate-straight-rotate controller is ready")

    def goal_cb(self, msg):
        with self.lock:
            self.latest_goal = msg
            self.goal_sequence += 1
        self.cancel_pub.publish(GoalID())
        self.goal_event.set()

    def current_pose(self):
        translation, rotation = self.listener.lookupTransform(
            self.map_frame, self.base_frame, rospy.Time(0)
        )
        yaw = tf.transformations.euler_from_quaternion(rotation)[2]
        return translation[0], translation[1], yaw

    def publish_cmd(self, linear=0.0, angular=0.0):
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

    def stop(self):
        self.publish_cmd()

    def is_current(self, sequence):
        with self.lock:
            return sequence == self.goal_sequence

    def rotate_to(self, target_yaw, sequence, timeout=20.0):
        started = rospy.Time.now()
        stable_cycles = 0
        angular_command = 0.0
        previous_command_time = rospy.Time.now()
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and self.is_current(sequence):
            if (rospy.Time.now() - started).to_sec() > timeout:
                self.stop()
                return False
            try:
                _, _, yaw = self.current_pose()
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rate.sleep()
                continue
            error = normalize_angle(target_yaw - yaw)
            if abs(error) <= self.heading_tolerance:
                stable_cycles += 1
                self.stop()
                angular_command = 0.0
                previous_command_time = rospy.Time.now()
                if stable_cycles >= 5:
                    return True
            else:
                stable_cycles = 0
                desired = max(
                    -self.max_angular,
                    min(self.max_angular, self.rotate_kp * error),
                )
                if abs(desired) < self.min_angular:
                    desired = math.copysign(self.min_angular, desired)

                now = rospy.Time.now()
                dt = max(0.0, min(0.1, (now - previous_command_time).to_sec()))
                previous_command_time = now
                max_change = self.rotate_accel * dt
                angular_command += max(
                    -max_change,
                    min(max_change, desired - angular_command),
                )
                self.publish_cmd(0.0, angular_command)
            rate.sleep()
        self.stop()
        return False

    def drive_straight_to(self, target_x, target_y, sequence, timeout=60.0):
        started = rospy.Time.now()
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and self.is_current(sequence):
            if (rospy.Time.now() - started).to_sec() > timeout:
                self.stop()
                return False
            try:
                x, y, yaw = self.current_pose()
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rate.sleep()
                continue

            dx = target_x - x
            dy = target_y - y
            distance = math.hypot(dx, dy)
            if distance <= self.position_tolerance:
                self.stop()
                return True

            desired_yaw = math.atan2(dy, dx)
            heading_error = normalize_angle(desired_yaw - yaw)
            if abs(heading_error) > math.radians(8.0):
                linear = 0.0
            else:
                linear = max(self.min_linear, min(self.max_linear, 0.8 * distance))
            angular = max(
                -0.20, min(0.20, self.heading_kp * heading_error)
            )
            self.publish_cmd(linear, angular)
            rate.sleep()
        self.stop()
        return False

    def drive_with_dwa(self, target_x, target_y, travel_yaw, sequence,
                       timeout=60.0):
        if not self.move_base.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("move_base action server is unavailable")
            return False

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = target_x
        goal.target_pose.pose.position.y = target_y
        quaternion = tf.transformations.quaternion_from_euler(
            0.0, 0.0, travel_yaw
        )
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]
        self.move_base.send_goal(goal)

        started = rospy.Time.now()
        rate = rospy.Rate(20)
        failed_states = {
            GoalStatus.PREEMPTED,
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.RECALLED,
            GoalStatus.LOST,
        }
        while not rospy.is_shutdown() and self.is_current(sequence):
            if (rospy.Time.now() - started).to_sec() > timeout:
                rospy.logerr("DWA straight drive timed out")
                self.move_base.cancel_goal()
                self.stop()
                return False
            try:
                x, y, _ = self.current_pose()
            except (tf.LookupException, tf.ConnectivityException,
                    tf.ExtrapolationException):
                rate.sleep()
                continue

            if math.hypot(target_x - x, target_y - y) <= self.position_tolerance:
                self.move_base.cancel_goal()
                self.stop()
                rospy.sleep(0.15)
                return True

            state = self.move_base.get_state()
            if state == GoalStatus.SUCCEEDED:
                self.stop()
                return True
            if state in failed_states:
                rospy.logerr("DWA straight drive failed with state %d", state)
                self.stop()
                return False
            rate.sleep()

        self.move_base.cancel_goal()
        self.stop()
        return False

    def execute_goal(self, goal, sequence):
        try:
            x, y, _ = self.current_pose()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logerr("Cannot start direct goal: %s", exc)
            return

        target_x = goal.pose.position.x
        target_y = goal.pose.position.y
        q = goal.pose.orientation
        target_yaw = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )[2]
        travel_yaw = math.atan2(target_y - y, target_x - x)

        rospy.loginfo("Phase 1/3: rotate in place to the travel heading")
        if not self.rotate_to(travel_yaw, sequence):
            rospy.logerr("Initial in-place rotation failed")
            return
        if self.use_dwa_straight:
            rospy.loginfo("Phase 2/3: DWA drive with continuous path correction")
            drove_to_goal = self.drive_with_dwa(
                target_x, target_y, travel_yaw, sequence
            )
        else:
            rospy.loginfo("Phase 2/3: direct straight drive with IMU heading hold")
            drove_to_goal = self.drive_straight_to(
                target_x, target_y, sequence
            )
        if not drove_to_goal:
            rospy.logerr("Straight drive failed")
            return
        rospy.loginfo("Phase 3/3: rotate in place to the requested final heading")
        if not self.rotate_to(target_yaw, sequence):
            rospy.logerr("Final in-place rotation failed")
            return
        rospy.loginfo("Direct goal reached")

    def worker(self):
        while not rospy.is_shutdown():
            self.goal_event.wait(0.2)
            if not self.goal_event.is_set():
                continue
            self.goal_event.clear()
            with self.lock:
                goal = self.latest_goal
                sequence = self.goal_sequence
            if goal is not None:
                self.execute_goal(goal, sequence)


if __name__ == "__main__":
    rospy.init_node("rotate_then_go")
    RotateThenGo()
    rospy.spin()
