#!/usr/bin/env python3
"""State-driven one-command runner for the complete Stage-2 mission."""

import json
import math
import threading

import rosgraph
import rospy
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger

from yellow_track_avoidance.stage2_auto import (
    command_topology_reason,
    goal_is_ready,
    scene_is_ready,
)


class Stage2AutoRunner:
    def __init__(self):
        self.lock = threading.RLock()
        self.enable_motion = bool(rospy.get_param("~enable_motion", False))
        self.scene_status_topic = str(
            rospy.get_param(
                "~scene_status_topic", "/yellow_track_stage2_frozen_scene/status"
            )
        )
        self.goal_status_topic = str(
            rospy.get_param(
                "~goal_status_topic", "/yellow_track_stage2_lidar_goal/status"
            )
        )
        self.move_base_status_topic = str(
            rospy.get_param("~move_base_status_topic", "/move_base/status")
        )
        self.amcl_pose_topic = str(rospy.get_param("~amcl_pose_topic", "/amcl_pose"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.send_goal_service = str(
            rospy.get_param(
                "~send_goal_service", "/yellow_track_stage2_lidar_goal/send_goal"
            )
        )
        self.cmd_topic = str(rospy.get_param("~cmd_topic", "/cmd_vel"))
        self.ready_stable_sec = float(rospy.get_param("~ready_stable_sec", 0.50))
        self.startup_timeout_sec = float(
            rospy.get_param("~startup_timeout_sec", 35.0)
        )
        self.navigation_timeout_sec = float(
            rospy.get_param("~navigation_timeout_sec", 90.0)
        )
        self.status_timeout_sec = float(rospy.get_param("~status_timeout_sec", 2.0))
        self.maximum_linear_speed = float(
            rospy.get_param("~maximum_start_linear_speed_mps", 0.01)
        )
        self.maximum_angular_speed = float(
            rospy.get_param("~maximum_start_angular_speed_radps", 0.03)
        )

        self.started = rospy.Time.now()
        self.scene_status = None
        self.scene_status_receipt = None
        self.goal_status = None
        self.goal_status_receipt = None
        self.latest_odom = None
        self.latest_odom_receipt = None
        self.latest_amcl_pose_receipt = None
        self.latest_move_base_status = None
        self.ready_since = None
        self.ready_at_elapsed_sec = None
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
        self.scene_status_sub = rospy.Subscriber(
            self.scene_status_topic, String, self._scene_status_callback, queue_size=1
        )
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
        self.amcl_pose_sub = rospy.Subscriber(
            self.amcl_pose_topic,
            PoseWithCovarianceStamped,
            self._amcl_pose_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)
        rospy.loginfo(
            "Stage-2 auto runner ready: motion=%s, frozen-map readiness is state-driven",
            self.enable_motion,
        )

    @staticmethod
    def _json(message):
        try:
            return json.loads(message.data)
        except (TypeError, ValueError):
            return None

    def _scene_status_callback(self, message):
        payload = self._json(message)
        if payload is None:
            return
        with self.lock:
            self.scene_status = payload
            self.scene_status_receipt = rospy.Time.now()

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

    def _amcl_pose_callback(self, _message):
        with self.lock:
            self.latest_amcl_pose_receipt = rospy.Time.now()

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

    def _amcl_ready(self, now):
        # A stationary AMCL instance may publish only its initial posterior.
        # Requiring a high-rate pose stream here would deadlock a valid frozen
        # scene.  A non-empty make_plan preview separately proves that the
        # current map->odom->base transform chain is usable.
        return self.latest_amcl_pose_receipt is not None

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
        except Exception as error:
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

    def _fresh_ready(self, now):
        if self.scene_status_receipt is None or self.goal_status_receipt is None:
            return False
        if (now - self.scene_status_receipt).to_sec() > self.status_timeout_sec:
            return False
        if (now - self.goal_status_receipt).to_sec() > self.status_timeout_sec:
            return False
        return scene_is_ready(self.scene_status) and goal_is_ready(self.goal_status)

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
                rospy.loginfo("Stage-2 automatic goal dispatch succeeded: %s", message)
            else:
                self.terminal_state = "FAILED"
                self.last_reason = "send_goal_failed:" + message
                rospy.logerr("Stage-2 automatic goal dispatch failed: %s", message)

    def _navigation_state(self, now):
        if self.goal_sent_stamp is None:
            return None
        status = self.latest_move_base_status
        if status is None:
            return "WAITING_ACCEPTANCE"
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
            ready = self._fresh_ready(now)
            amcl_ready = self._amcl_ready(now)
            if ready and amcl_ready:
                if self.ready_since is None:
                    self.ready_since = now
                    self.ready_at_elapsed_sec = elapsed
            else:
                self.ready_since = None
            ready_stable = (
                ready
                and amcl_ready
                and self.ready_since is not None
                and (now - self.ready_since).to_sec() >= self.ready_stable_sec
            )
            stationary = self._stationary(now)

            navigation_state = self._navigation_state(now)
            if self.terminal_state is None and navigation_state == "SUCCEEDED":
                self.terminal_state = "SUCCEEDED"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "stage2_goal_reached"
                rospy.loginfo("Stage-2 automatic mission reached its frozen goal")
            elif self.terminal_state is None and navigation_state == "FAILED":
                self.terminal_state = "FAILED"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "move_base_failed"
                rospy.logerr("Stage-2 automatic mission failed in move_base")

            if (
                self.terminal_state is None
                and self.goal_sent_stamp is not None
                and (now - self.goal_sent_stamp).to_sec() > self.navigation_timeout_sec
            ):
                self.cancel_pub.publish(GoalID())
                self.terminal_state = "FAILED"
                self.terminal_elapsed_sec = elapsed
                self.last_reason = "navigation_timeout"
                rospy.logerr("Stage-2 navigation timed out; goal cancelled")

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
            elif ready and not amcl_ready:
                state = "WAITING_AMCL"
                self.last_reason = "amcl_pose_not_fresh"
            else:
                state = "WAITING_FROZEN_SCENE"
                self.last_reason = (
                    (self.scene_status or {}).get("last_error")
                    or (self.goal_status or {}).get("last_error")
                    or "scene_or_goal_not_ready"
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
                rospy.logerr("Stage-2 startup timed out without a valid frozen scene")

            ready_elapsed = (
                None
                if self.ready_since is None
                else (now - self.ready_since).to_sec()
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
                "scene_ready": scene_is_ready(self.scene_status),
                "goal_ready": goal_is_ready(self.goal_status),
                "amcl_ready": amcl_ready,
                "ready_stable": ready_stable,
                "ready_elapsed_sec": ready_elapsed,
                "ready_at_elapsed_sec": self.ready_at_elapsed_sec,
                "stationary": stationary,
                "command_topology": topology_reason,
                "goal_sent": self.goal_sent_stamp is not None,
                "navigation_elapsed_sec": navigation_elapsed,
                "reason": self.last_reason,
                "terminal_elapsed_sec": self.terminal_elapsed_sec,
                "send_response": self.send_response,
                "captured_frames": (self.scene_status or {}).get("captured_frames"),
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
                if not rospy.is_shutdown():
                    raise


def main():
    rospy.init_node("yellow_track_stage2_auto_runner")
    Stage2AutoRunner()
    rospy.spin()


if __name__ == "__main__":
    main()
