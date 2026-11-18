#!/usr/bin/env python3
"""State-driven one-command runner for the complete Stage-1 mission."""

import json
import math
import threading

import rosgraph
import rospy
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger

from yellow_track_avoidance.stage1_auto import (
    command_topology_reason,
    goal_is_ready,
)


class Stage1AutoRunner:
    def __init__(self):
        self.lock = threading.RLock()
        self.enable_motion = bool(rospy.get_param("~enable_motion", False))
        self.goal_status_topic = str(
            rospy.get_param(
                "~goal_status_topic", "/yellow_track_stage1_lidar_goal/status"
            )
        )
        self.move_base_status_topic = str(
            rospy.get_param("~move_base_status_topic", "/move_base/status")
        )
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.send_goal_service = str(
            rospy.get_param(
                "~send_goal_service", "/yellow_track_stage1_lidar_goal/send_goal"
            )
        )
        self.cmd_topic = str(rospy.get_param("~cmd_topic", "/cmd_vel"))
        self.goal_ready_stable_sec = float(
            rospy.get_param("~goal_ready_stable_sec", 0.30)
        )
        self.startup_timeout_sec = float(
            rospy.get_param("~startup_timeout_sec", 35.0)
        )
        self.navigation_timeout_sec = float(
            rospy.get_param("~navigation_timeout_sec", 120.0)
        )
        self.status_timeout_sec = float(rospy.get_param("~status_timeout_sec", 2.0))
        self.maximum_linear_speed = float(
            rospy.get_param("~maximum_start_linear_speed_mps", 0.01)
        )
        self.maximum_angular_speed = float(
            rospy.get_param("~maximum_start_angular_speed_radps", 0.03)
        )

        self.started = rospy.Time.now()
        self.goal_status = None
        self.goal_status_receipt = None
        self.goal_ready_since = None
        self.goal_ready_at_elapsed_sec = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.latest_move_base_status = None
        self.goal_sent_stamp = None
        self.send_in_progress = False
        self.terminal_state = None
        self.terminal_elapsed_sec = None
        self.last_reason = "starting"
        self.topology_reason = "unchecked"
        self.last_topology_check = None
        self.send_response = None

        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.cancel_pub = rospy.Publisher(
            "/move_base/cancel", GoalID, queue_size=1, latch=False
        )
        self.send_goal = rospy.ServiceProxy(self.send_goal_service, Trigger)
        self.goal_status_sub = rospy.Subscriber(
            self.goal_status_topic, String, self._goal_status_callback, queue_size=1
        )
        self.move_base_status_sub = rospy.Subscriber(
            self.move_base_status_topic,
            GoalStatusArray,
            self._move_base_status_callback,
            queue_size=1,
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_callback, queue_size=1
        )
        self.timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)
        rospy.loginfo(
            "Stage-1 auto runner ready: motion=%s, state-driven waits, no fixed launch sleep",
            self.enable_motion,
        )

    @staticmethod
    def _json(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError):
            return None

    def _goal_status_callback(self, message):
        payload = self._json(message)
        if payload is None:
            return
        with self.lock:
            self.goal_status = payload
            self.goal_status_receipt = rospy.Time.now()

    def _odom_callback(self, message):
        with self.lock:
            self.latest_odom = message
            self.latest_odom_receipt = rospy.Time.now()

    def _odom_snapshot(self):
        if self.latest_odom is None:
            return None
        pose = self.latest_odom.pose.pose
        twist = self.latest_odom.twist.twist
        quaternion = pose.orientation
        yaw = math.atan2(
            2.0 * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0 - 2.0 * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )
        return {
            "x_m": float(pose.position.x),
            "y_m": float(pose.position.y),
            "yaw_rad": float(yaw),
            "vx_mps": float(twist.linear.x),
            "vy_mps": float(twist.linear.y),
            "wz_radps": float(twist.angular.z),
        }

    def _move_base_status_callback(self, message):
        if not message.status_list:
            return
        latest = max(
            message.status_list,
            key=lambda status: (status.goal_id.stamp.secs, status.goal_id.stamp.nsecs),
        )
        with self.lock:
            self.latest_move_base_status = latest

    def _stationary(self, now):
        if self.latest_odom is None or self.latest_odom_receipt is None:
            return False
        if (now - self.latest_odom_receipt).to_sec() > self.status_timeout_sec:
            return False
        twist = self.latest_odom.twist.twist
        return (
            math.hypot(twist.linear.x, twist.linear.y) <= self.maximum_linear_speed
            and abs(twist.angular.z) <= self.maximum_angular_speed
        )

    def _topology(self, now):
        if (
            self.last_topology_check is not None
            and (now - self.last_topology_check).to_sec() < 0.5
        ):
            return self.topology_reason
        try:
            publishers, subscribers, _ = rosgraph.Master(
                rospy.get_name()
            ).getSystemState()
        except Exception as error:  # ROS master failure is reported, never ignored.
            reason = "master_query_failed:" + str(error)
            self.last_topology_check = now
            self.topology_reason = reason
            return reason
        publisher_map = dict(publishers)
        subscriber_map = dict(subscribers)
        reason = command_topology_reason(
            publisher_map.get(self.cmd_topic, ()),
            subscriber_map.get(self.cmd_topic, ()),
            self.enable_motion,
        )
        self.last_topology_check = now
        self.topology_reason = reason
        return reason

    def _fresh_goal_ready(self, now):
        if self.goal_status_receipt is None:
            return False
        if (now - self.goal_status_receipt).to_sec() > self.status_timeout_sec:
            return False
        return goal_is_ready(self.goal_status)

    def _call_send_goal(self):
        try:
            rospy.wait_for_service(self.send_goal_service, timeout=2.0)
            response = self.send_goal()
            success = bool(response.success)
            message = str(response.message)
        except Exception as error:
            success = False
            message = str(error)
        now = rospy.Time.now()
        with self.lock:
            self.send_in_progress = False
            self.send_response = message
            if success:
                self.goal_sent_stamp = now
                self.last_reason = "goal_sent"
                rospy.loginfo("Stage-1 automatic goal dispatch succeeded: %s", message)
            else:
                self.terminal_state = "FAILED"
                self.last_reason = "send_goal_failed:" + message
                rospy.logerr("Stage-1 automatic goal dispatch failed: %s", message)

    def _navigation_state(self, now):
        if self.goal_sent_stamp is None:
            return None
        status = self.latest_move_base_status
        if status is None:
            return "WAITING_ACCEPTANCE"
        # Ignore statuses belonging to goals that predate this dispatch.
        if status.goal_id.stamp + rospy.Duration(0.5) < self.goal_sent_stamp:
            return "WAITING_ACCEPTANCE"
        if status.status == GoalStatus.SUCCEEDED:
            return "SUCCEEDED"
        if status.status in (
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.RECALLED,
            GoalStatus.LOST,
        ):
            return "FAILED"
        if status.status in (
            GoalStatus.PENDING,
            GoalStatus.ACTIVE,
            GoalStatus.PREEMPTING,
            GoalStatus.RECALLING,
        ):
            return "ACTIVE"
        return "WAITING_ACCEPTANCE"

    def _timer_callback(self, _event):
        if rospy.is_shutdown():
            return
        now = rospy.Time.now()
        elapsed = (now - self.started).to_sec()
        with self.lock:
            topology_reason = self._topology(now)
            self.topology_reason = topology_reason
            ready = self._fresh_goal_ready(now)
            if ready:
                if self.goal_ready_since is None:
                    self.goal_ready_since = now
                    self.goal_ready_at_elapsed_sec = elapsed
            else:
                self.goal_ready_since = None
            ready_stable = (
                ready
                and self.goal_ready_since is not None
                and (now - self.goal_ready_since).to_sec()
                >= self.goal_ready_stable_sec
            )
            stationary = self._stationary(now)

            navigation_state = self._navigation_state(now)
            if self.terminal_state is None and navigation_state == "SUCCEEDED":
                self.terminal_state = "SUCCEEDED"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "stage1_goal_reached"
                rospy.loginfo("Stage-1 automatic mission reached its frozen goal")
            elif self.terminal_state is None and navigation_state == "FAILED":
                self.terminal_state = "FAILED"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "move_base_failed"
                rospy.logerr(
                    "Stage-1 automatic mission failed in move_base; odom=%s",
                    json.dumps(self._odom_snapshot(), sort_keys=True),
                )

            if (
                self.terminal_state is None
                and self.goal_sent_stamp is not None
                and (now - self.goal_sent_stamp).to_sec() > self.navigation_timeout_sec
            ):
                self.cancel_pub.publish(GoalID())
                self.terminal_state = "FAILED"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "navigation_timeout"
                rospy.logerr("Stage-1 navigation timed out; goal cancelled")

            if ready_stable and not self.enable_motion and self.terminal_state is None:
                self.terminal_state = "DRY_RUN_READY"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "goal_ready_motion_disabled"

            if self.terminal_state is not None:
                state = self.terminal_state
            elif self.goal_sent_stamp is not None:
                state = navigation_state or "GOAL_SENT"
            elif ready_stable and topology_reason == "ready" and stationary:
                state = "SENDING_GOAL"
                if not self.send_in_progress:
                    self.send_in_progress = True
                    threading.Thread(target=self._call_send_goal, daemon=True).start()
            elif ready_stable and topology_reason != "ready":
                state = "WAITING_COMMAND_TOPOLOGY"
                self.last_reason = topology_reason
            elif ready_stable and not stationary:
                state = "WAITING_STATIONARY"
                self.last_reason = "odom_not_stationary"
            else:
                state = "WAITING_WORLD_MODEL"
                self.last_reason = (
                    (self.goal_status or {}).get("world_ready_reason", "goal_not_ready")
                )

            if (
                self.terminal_state is None
                and self.goal_sent_stamp is None
                and elapsed > self.startup_timeout_sec
            ):
                self.terminal_state = "FAILED"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "startup_timeout"
                state = self.terminal_state
                rospy.logerr("Stage-1 startup timed out without a valid frozen goal")

            ready_elapsed = (
                None
                if self.goal_ready_since is None
                else (now - self.goal_ready_since).to_sec()
            )
            navigation_elapsed = (
                None
                if self.goal_sent_stamp is None
                else (now - self.goal_sent_stamp).to_sec()
            )
            status_payload = {
                "stamp": now.to_sec(),
                "state": state,
                "enable_motion": self.enable_motion,
                "elapsed_sec": elapsed,
                "goal_ready": ready,
                "goal_ready_stable": ready_stable,
                "goal_ready_elapsed_sec": ready_elapsed,
                "goal_ready_at_elapsed_sec": self.goal_ready_at_elapsed_sec,
                "stationary": stationary,
                "command_topology": topology_reason,
                "goal_sent": self.goal_sent_stamp is not None,
                "navigation_elapsed_sec": navigation_elapsed,
                "reason": self.last_reason,
                "terminal_elapsed_sec": self.terminal_elapsed_sec,
                "send_response": self.send_response,
                "odom": self._odom_snapshot(),
                "scan_frames_required": (self.goal_status or {}).get(
                    "scan_frames_required"
                ),
                "preview_plan_pose_count": (self.goal_status or {}).get(
                    "preview_plan_pose_count", 0
                ),
            }
        if not rospy.is_shutdown():
            try:
                self.status_pub.publish(
                    String(data=json.dumps(status_payload, allow_nan=False))
                )
            except rospy.ROSException:
                # A timer callback can cross the shutdown boundary after the
                # publisher has already been unregistered. This is harmless
                # and must not leave a misleading traceback in trial logs.
                if not rospy.is_shutdown():
                    raise


def main():
    rospy.init_node("yellow_track_stage1_auto_runner")
    Stage1AutoRunner()
    rospy.spin()


if __name__ == "__main__":
    main()
