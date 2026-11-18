#!/usr/bin/env python3
"""Replay DWA Trial 3 without a chassis and audit one planner profile.

The probe refuses to send a goal unless move_base is the sole /cmd_vel
publisher and no hardware command consumer is connected. It publishes the
world geometry recovered from the 2026-08-09 left-staggered Trial 3, waits for
the corridor and both discs to lock, records planner outputs, and cancels.
"""

import argparse
import json
import math
import os
import statistics
import time

import rosgraph
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rosgraph_msgs.msg import Log
from std_msgs.msg import String
from std_srvs.srv import Trigger
from teb_local_planner.msg import FeedbackMsg


TRIAL3_DISCS = (
    (0.6553191071523864, -0.1500485535500179),
    (1.1359505038459874, 0.21536528214195472),
)
TRIAL3_GOAL = (2.408, -0.283)
TRIAL3_YAW = -0.057
VEHICLE_HALF_LENGTH = 0.170
VEHICLE_HALF_WIDTH = 0.165
DISC_RADIUS = 0.050
YELLOW_HALF_WIDTH = 0.015
CORRIDOR_WIDTH = 0.910


def wrapped(value):
    return math.atan2(math.sin(value), math.cos(value))


def yaw_of(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round(fraction * (len(ordered) - 1)))
    return float(ordered[index])


def process_sample():
    """Return move_base and whole-machine counters from /proc."""
    move_base_pid = None
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/cmdline".format(name), "rb") as stream:
                command = stream.read().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
        except (OSError, IOError):
            continue
        if "/move_base/move_base" in command and "__name:=move_base" in command:
            move_base_pid = int(name)
            break
    if move_base_pid is None:
        return None
    with open("/proc/stat", "r", encoding="utf-8") as stream:
        cpu_fields = [int(item) for item in stream.readline().split()[1:]]
    with open(
        "/proc/{}/stat".format(move_base_pid), "r", encoding="utf-8"
    ) as stream:
        process_fields = stream.read().split()
    with open(
        "/proc/{}/statm".format(move_base_pid), "r", encoding="utf-8"
    ) as stream:
        resident_pages = int(stream.read().split()[1])
    return {
        "pid": move_base_pid,
        "system_total_ticks": sum(cpu_fields),
        "system_busy_ticks": sum(cpu_fields[:3]) + sum(cpu_fields[5:8]),
        "process_ticks": int(process_fields[13]) + int(process_fields[14]),
        "rss_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE"),
        "cpu_count": os.cpu_count() or 1,
    }


def cpu_delta(before, after):
    if before is None or after is None or before["pid"] != after["pid"]:
        return {}
    total = after["system_total_ticks"] - before["system_total_ticks"]
    process = after["process_ticks"] - before["process_ticks"]
    busy = after["system_busy_ticks"] - before["system_busy_ticks"]
    if total <= 0:
        return {}
    return {
        "move_base_cpu_percent_one_core_scale": (
            100.0 * process * before["cpu_count"] / total
        ),
        "move_base_rss_mib_end": after["rss_bytes"] / (1024.0 * 1024.0),
        "whole_machine_busy_percent": 100.0 * busy / total,
    }


def graph_state():
    publishers, subscribers, _ = rosgraph.Master(
        "/locked_planner_ab_probe"
    ).getSystemState()
    return dict(publishers), dict(subscribers)


def assert_motion_disconnected():
    publishers, subscribers = graph_state()
    command_publishers = sorted(publishers.get("/cmd_vel", []))
    command_subscribers = sorted(subscribers.get("/cmd_vel", []))
    if command_publishers != ["/move_base"]:
        raise RuntimeError(
            "/cmd_vel publishers are not the one-authority baseline: {}".format(
                command_publishers
            )
        )
    permitted = {"/yellow_track_disc_costmap_bridge"}
    unsafe = [item for item in command_subscribers if item not in permitted]
    if unsafe or "/cmd_vel_listener" in command_subscribers:
        raise RuntimeError(
            "refusing locked replay; hardware/unknown /cmd_vel subscribers: {}"
            .format(unsafe)
        )
    return {
        "publishers": command_publishers,
        "subscribers_before_probe": command_subscribers,
        "hardware_command_consumer_present": False,
    }


def derive_left_line():
    """Choose the left line that regenerates the recorded Trial 3 goal."""
    tangent = (math.cos(TRIAL3_YAW), math.sin(TRIAL3_YAW))
    normal = (-tangent[1], tangent[0])
    residual = (
        TRIAL3_GOAL[0] - tangent[0] * 2.42,
        TRIAL3_GOAL[1] - tangent[1] * 2.42,
    )
    offset = 0.67 + residual[0] * normal[0] + residual[1] * normal[1]
    points = []
    for along in (0.25, 0.45, 0.65, 0.85, 1.05, 1.25, 1.45, 1.65):
        points.append(
            [
                tangent[0] * along + normal[0] * offset,
                tangent[1] * along + normal[1] * offset,
            ]
        )
    return tangent, normal, offset, points


def pose_clearances(x, y, yaw, normal, left_offset):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    disc_clearance = float("inf")
    disc_detail = None
    for disc_index, (disc_x, disc_y) in enumerate(TRIAL3_DISCS):
        delta_x = disc_x - x
        delta_y = disc_y - y
        body_x = cosine * delta_x + sine * delta_y
        body_y = -sine * delta_x + cosine * delta_y
        outside_x = max(abs(body_x) - VEHICLE_HALF_LENGTH, 0.0)
        outside_y = max(abs(body_y) - VEHICLE_HALF_WIDTH, 0.0)
        clearance = math.hypot(outside_x, outside_y) - DISC_RADIUS
        if clearance < disc_clearance:
            disc_clearance = clearance
            disc_detail = {
                "disc_index": disc_index,
                "disc_body_m": [body_x, body_y],
            }

    corner_lateral = []
    for forward in (-VEHICLE_HALF_LENGTH, VEHICLE_HALF_LENGTH):
        for left in (-VEHICLE_HALF_WIDTH, VEHICLE_HALF_WIDTH):
            corner_x = x + cosine * forward - sine * left
            corner_y = y + sine * forward + cosine * left
            corner_lateral.append(
                corner_x * normal[0] + corner_y * normal[1]
            )
    left_inner = left_offset - YELLOW_HALF_WIDTH
    right_inner = left_offset - CORRIDOR_WIDTH + YELLOW_HALF_WIDTH
    yellow_clearance = min(
        left_inner - max(corner_lateral),
        min(corner_lateral) - right_inner,
    )
    return disc_clearance, disc_detail, yellow_clearance, corner_lateral


def path_metrics(path, normal, left_offset):
    if path is None or not path.poses:
        return {"points": 0}
    values = []
    yaws = []
    for stamped in path.poses:
        pose = stamped.pose
        values.append((pose.position.x, pose.position.y))
        yaws.append(yaw_of(pose.orientation))
    segments = [
        math.hypot(
            values[i][0] - values[i - 1][0],
            values[i][1] - values[i - 1][1],
        )
        for i in range(1, len(values))
    ]
    direction_angles = []
    for i in range(1, len(values)):
        dx = values[i][0] - values[i - 1][0]
        dy = values[i][1] - values[i - 1][1]
        if math.hypot(dx, dy) > 1e-8:
            direction_angles.append(math.atan2(dy, dx))
    direction_changes = [
        abs(wrapped(direction_angles[i] - direction_angles[i - 1]))
        for i in range(1, len(direction_angles))
    ]
    yaw_changes = [
        abs(wrapped(yaws[i] - yaws[i - 1])) for i in range(1, len(yaws))
    ]

    sampled_poses = []
    for index, ((x, y), yaw) in enumerate(zip(values, yaws)):
        if index == 0:
            sampled_poses.append((x, y, yaw))
            continue
        previous_x, previous_y = values[index - 1]
        previous_yaw = yaws[index - 1]
        distance = math.hypot(x - previous_x, y - previous_y)
        yaw_delta = wrapped(yaw - previous_yaw)
        sample_count = max(
            1,
            int(math.ceil(distance / 0.005)),
            int(math.ceil(abs(yaw_delta) / math.radians(1.0))),
        )
        for sample_index in range(1, sample_count + 1):
            ratio = float(sample_index) / sample_count
            sampled_poses.append(
                (
                    previous_x + ratio * (x - previous_x),
                    previous_y + ratio * (y - previous_y),
                    wrapped(previous_yaw + ratio * yaw_delta),
                )
            )

    min_disc_clearance = float("inf")
    min_disc_detail = None
    min_yellow_clearance = float("inf")
    min_yellow_detail = None
    for sample_index, (x, y, yaw) in enumerate(sampled_poses):
        disc_clearance, disc_detail, yellow_clearance, corner_lateral = (
            pose_clearances(x, y, yaw, normal, left_offset)
        )
        if disc_clearance < min_disc_clearance:
            min_disc_clearance = disc_clearance
            min_disc_detail = dict(disc_detail)
            min_disc_detail.update(
                {"sample_index": sample_index, "pose_m_rad": [x, y, yaw]}
            )
        if yellow_clearance < min_yellow_clearance:
            min_yellow_clearance = yellow_clearance
            min_yellow_detail = {
                "sample_index": sample_index,
                "pose_m_rad": [x, y, yaw],
                "corner_lateral_range_m": [
                    min(corner_lateral),
                    max(corner_lateral),
                ],
            }

    node_clearances = [
        pose_clearances(x, y, yaw, normal, left_offset)
        for (x, y), yaw in zip(values, yaws)
    ]

    return {
        "points": len(values),
        "length_m": sum(segments),
        "endpoint_m": list(values[-1]),
        "min_disc_clearance_m": min_disc_clearance,
        "min_disc_clearance_detail": min_disc_detail,
        "min_yellow_clearance_m": min_yellow_clearance,
        "min_yellow_clearance_detail": min_yellow_detail,
        "node_min_disc_clearance_m": min(item[0] for item in node_clearances),
        "node_min_yellow_clearance_m": min(item[2] for item in node_clearances),
        "swept_pose_samples": len(sampled_poses),
        "max_adjacent_translation_m": max(segments) if segments else 0.0,
        "max_adjacent_yaw_change_rad": max(yaw_changes) if yaw_changes else 0.0,
        "max_adjacent_direction_change_rad": (
            max(direction_changes) if direction_changes else 0.0
        ),
    }


class Capture:
    def __init__(self, global_topic):
        self.global_path = None
        self.first_global_path = None
        self.local_path = None
        self.first_local_path = None
        self.feedback = None
        self.first_feedback = None
        self.feedback_history = []
        self.preview_path = None
        self.commands = []
        self.disc_status = None
        self.corridor_status = None
        self.compact_obstacle_status = None
        self.perception_status = None
        self.measurement_logs = []
        self.measurement_start_sec = None
        self.global_costmap = None
        self.local_costmap = None
        rospy.Subscriber(global_topic, Path, self._global, queue_size=1)
        rospy.Subscriber(
            "/move_base/TebLocalPlannerROS/local_plan",
            Path,
            self._local,
            queue_size=1,
        )
        rospy.Subscriber(
            "/move_base/TebLocalPlannerROS/teb_feedback",
            FeedbackMsg,
            self._feedback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yellow_track_avoidance/corridor_goal_plan",
            Path,
            self._preview,
            queue_size=1,
        )
        rospy.Subscriber("/cmd_vel", Twist, self._command, queue_size=40)
        rospy.Subscriber(
            "/yellow_track_disc_costmap_bridge/status",
            String,
            self._disc_status,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yellow_track_competition_corridor_map/status",
            String,
            self._corridor_status,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yellow_track_teb_compact_obstacles/status",
            String,
            self._compact_obstacle_status,
            queue_size=1,
        )
        rospy.Subscriber(
            "/yellow_track_avoidance/debug/status",
            String,
            self._perception_status,
            queue_size=1,
        )
        rospy.Subscriber("/rosout_agg", Log, self._rosout, queue_size=80)
        rospy.Subscriber(
            "/move_base/global_costmap/costmap",
            OccupancyGrid,
            self._global_costmap,
            queue_size=1,
        )
        rospy.Subscriber(
            "/move_base/local_costmap/costmap",
            OccupancyGrid,
            self._local_costmap,
            queue_size=1,
        )
        rospy.Subscriber(
            "/move_base/global_costmap/costmap_updates",
            OccupancyGridUpdate,
            self._global_costmap_update,
            queue_size=4,
        )
        rospy.Subscriber(
            "/move_base/local_costmap/costmap_updates",
            OccupancyGridUpdate,
            self._local_costmap_update,
            queue_size=8,
        )

    def _global(self, message):
        if self.first_global_path is None and message.poses:
            self.first_global_path = message
        self.global_path = message

    def _local(self, message):
        if self.first_local_path is None and message.poses:
            self.first_local_path = message
        self.local_path = message

    def _feedback(self, message):
        if self.first_feedback is None and message.trajectories:
            self.first_feedback = message
        self.feedback = message
        self.feedback_history.append(message)

    def _preview(self, message):
        self.preview_path = message

    def _command(self, message):
        self.commands.append(
            {
                "stamp": rospy.Time.now().to_sec(),
                "vx": message.linear.x,
                "vy": message.linear.y,
                "wz": message.angular.z,
            }
        )

    def _disc_status(self, message):
        try:
            self.disc_status = json.loads(message.data)
        except ValueError:
            pass

    def _corridor_status(self, message):
        try:
            self.corridor_status = json.loads(message.data)
        except ValueError:
            pass

    def _compact_obstacle_status(self, message):
        try:
            self.compact_obstacle_status = json.loads(message.data)
        except ValueError:
            pass

    def _perception_status(self, message):
        try:
            self.perception_status = json.loads(message.data)
        except ValueError:
            pass

    def _rosout(self, message):
        if self.measurement_start_sec is None or message.name != "/move_base":
            return
        self.measurement_logs.append(
            {
                "stamp": message.header.stamp.to_sec(),
                "level": int(message.level),
                "message": message.msg,
            }
        )

    def _global_costmap(self, message):
        self.global_costmap = message

    def _local_costmap(self, message):
        self.local_costmap = message

    @staticmethod
    def _apply_costmap_update(grid, update):
        if grid is None or not grid.data:
            return
        data = list(grid.data)
        for row in range(update.height):
            destination = (update.y + row) * grid.info.width + update.x
            source = row * update.width
            data[destination : destination + update.width] = update.data[
                source : source + update.width
            ]
        grid.data = data

    def _global_costmap_update(self, message):
        self._apply_costmap_update(self.global_costmap, message)

    def _local_costmap_update(self, message):
        self._apply_costmap_update(self.local_costmap, message)


def costmap_disc_values(grid):
    if grid is None:
        return None
    output = []
    for disc_x, disc_y in TRIAL3_DISCS:
        column = int((disc_x - grid.info.origin.position.x) / grid.info.resolution)
        row = int((disc_y - grid.info.origin.position.y) / grid.info.resolution)
        values = []
        radius = max(1, int(math.ceil(0.06 / grid.info.resolution)))
        for y_index in range(
            max(0, row - radius), min(grid.info.height, row + radius + 1)
        ):
            for x_index in range(
                max(0, column - radius), min(grid.info.width, column + radius + 1)
            ):
                values.append(grid.data[y_index * grid.info.width + x_index])
        output.append(max(values) if values else None)
    return output


def teb_metrics(feedback, normal, left_offset):
    if feedback is None or not feedback.trajectories:
        return {"feedback_received": False, "points": 0}
    index = int(feedback.selected_trajectory_idx)
    if index >= len(feedback.trajectories):
        index = 0
    trajectory = feedback.trajectories[index].trajectory
    path = Path()
    path.header = feedback.header
    for point in trajectory:
        stamped = PoseStamped()
        stamped.header = feedback.header
        stamped.pose = point.pose
        path.poses.append(stamped)
    metrics = path_metrics(path, normal, left_offset)
    metrics.update(
        {
            "feedback_received": True,
            "trajectory_count": len(feedback.trajectories),
            "active_obstacle_count": len(feedback.obstacles_msg.obstacles),
            "selected_index": index,
            "time_horizon_sec": (
                trajectory[-1].time_from_start.to_sec() if trajectory else 0.0
            ),
        }
    )
    return metrics


def teb_history_metrics(history, normal, left_offset):
    values = [teb_metrics(item, normal, left_offset) for item in history]
    values = [item for item in values if item.get("feedback_received", False)]
    if not values:
        return {"samples": 0}
    disc_clearances = [item["min_disc_clearance_m"] for item in values]
    yellow_clearances = [item["min_yellow_clearance_m"] for item in values]
    node_disc_clearances = [
        item["node_min_disc_clearance_m"] for item in values
    ]
    node_yellow_clearances = [
        item["node_min_yellow_clearance_m"] for item in values
    ]
    return {
        "samples": len(values),
        "active_obstacle_counts": sorted(
            {item["active_obstacle_count"] for item in values}
        ),
        "trajectory_points_min": min(item["points"] for item in values),
        "trajectory_points_max": max(item["points"] for item in values),
        "min_disc_clearance_m": min(disc_clearances),
        "max_disc_clearance_m": max(disc_clearances),
        "node_min_disc_clearance_m": min(node_disc_clearances),
        "min_yellow_clearance_m": min(yellow_clearances),
        "max_yellow_clearance_m": max(yellow_clearances),
        "node_min_yellow_clearance_m": min(node_yellow_clearances),
        "max_adjacent_translation_m": max(
            item["max_adjacent_translation_m"] for item in values
        ),
    }


def wait_for(predicate, timeout, label):
    deadline = time.monotonic() + timeout
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if predicate():
            return
        rate.sleep()
    raise RuntimeError("timed out waiting for {}".format(label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture-seconds", type=float, default=5.0)
    parser.add_argument("--pre-goal-delay-sec", type=float, default=1.5)
    arguments = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("locked_planner_ab_probe", anonymous=False)
    safety = assert_motion_disconnected()
    planner = rospy.get_param("/move_base/base_global_planner")
    if planner == "global_planner/GlobalPlanner":
        global_topic = "/move_base/GlobalPlanner/plan"
    elif planner == "SBPLLatticePlanner":
        global_topic = "/move_base/SBPLLatticePlanner/plan"
    else:
        raise RuntimeError("unsupported A/B global planner: {}".format(planner))
    local_planner = rospy.get_param("/move_base/base_local_planner")
    if local_planner != "teb_local_planner/TebLocalPlannerROS":
        raise RuntimeError("probe requires the TEB candidate, got {}".format(local_planner))

    static_broadcaster = tf2_ros.StaticTransformBroadcaster()
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = "odom"
    transform.child_frame_id = "base_footprint"
    transform.transform.rotation.w = 1.0
    static_broadcaster.sendTransform(transform)

    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
    boundary_pub = rospy.Publisher(
        "/yellow_track_avoidance/debug/boundaries", String, queue_size=1
    )
    disc_pub = rospy.Publisher(
        "/yellow_track_avoidance/debug/disks", String, queue_size=1
    )
    cancel_pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=1)
    capture = Capture(global_topic)
    tangent, normal, left_offset, left_points = derive_left_line()

    def publish_odom(_event=None):
        message = Odometry()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "odom"
        message.child_frame_id = "base_footprint"
        message.pose.pose.orientation.w = 1.0
        odom_pub.publish(message)

    odom_timer = rospy.Timer(rospy.Duration(0.05), publish_odom)
    rospy.sleep(1.0)
    for _ in range(20):
        stamp = rospy.Time.now().to_sec()
        boundary_pub.publish(
            String(
                data=json.dumps(
                    {
                        "stamp": stamp,
                        "frame_id": "base_link",
                        "left_detected": True,
                        "right_detected": False,
                        "boundary_pair_valid": True,
                        "boundary_inference": "right_from_left_metric",
                        "quality": 1.0,
                        "left_base_m": left_points,
                    }
                )
            )
        )
        raw = []
        confirmed = []
        for track_id, center in enumerate(TRIAL3_DISCS, start=1):
            raw.append(
                {
                    "track_id": track_id,
                    "stamp": stamp,
                    "center_base_m": list(center),
                    "confirmed": True,
                    "estimated_diameter_m": 0.10,
                    "confidence": 1.0,
                }
            )
            confirmed.append({"track_id": track_id})
        disc_pub.publish(
            String(
                data=json.dumps(
                    {"stamp": stamp, "raw": raw, "confirmed_tracks": confirmed}
                )
            )
        )
        rospy.sleep(0.05)

    wait_for(
        lambda: capture.disc_status is not None
        and capture.disc_status.get("disc_count") == 2
        and capture.disc_status.get("costmap_disc_count") == 2,
        5.0,
        "two frozen costmap discs",
    )
    wait_for(
        lambda: capture.corridor_status is not None
        and capture.corridor_status.get("state") == "locked"
        and capture.corridor_status.get("goal_ready", False),
        5.0,
        "locked corridor",
    )
    profile = rospy.get_param("/yellow_track_competition_planner_profile")
    if "compact_obstacles" in profile:
        wait_for(
            lambda: capture.compact_obstacle_status is not None
            and capture.compact_obstacle_status.get("ready", False)
            and capture.compact_obstacle_status.get("obstacle_count") == 4,
            5.0,
            "four compact TEB obstacles",
        )
    rospy.sleep(max(0.0, arguments.pre_goal_delay_sec))

    send_goal = rospy.ServiceProxy(
        "/yellow_track_competition_corridor_map/send_goal", Trigger
    )
    send_goal.wait_for_service(timeout=3.0)
    capture.measurement_start_sec = rospy.Time.now().to_sec()
    result = send_goal()
    if not result.success:
        raise RuntimeError(result.message)

    cpu_before = process_sample()
    capture_start = time.monotonic()
    rospy.sleep(arguments.capture_seconds)
    capture_end = time.monotonic()
    cpu_after = process_sample()
    cancel_pub.publish(GoalID())
    rospy.sleep(0.4)
    odom_timer.shutdown()

    command_intervals = [
        capture.commands[i]["stamp"] - capture.commands[i - 1]["stamp"]
        for i in range(1, len(capture.commands))
        if capture.commands[i]["stamp"] > capture.commands[i - 1]["stamp"]
    ]
    cpu = cpu_delta(cpu_before, cpu_after)
    if cpu:
        cpu["sample_duration_sec"] = capture_end - capture_start
    report = {
        "schema": "yellow_track_locked_planner_ab_v1",
        "profile": profile,
        "global_planner": planner,
        "local_planner": local_planner,
        "physical_motion_guard": safety,
        "scene": {
            "source": "2026-08-09 DWA Trial 3 left-staggered replay",
            "discs_odom_m": [list(item) for item in TRIAL3_DISCS],
            "recorded_goal_approx_m": list(TRIAL3_GOAL),
            "recorded_yaw_rad": TRIAL3_YAW,
            "left_line_tangent": list(tangent),
            "left_line_normal": list(normal),
            "left_line_offset_m": left_offset,
            "locked_disc_status": capture.disc_status,
            "locked_corridor_status": capture.corridor_status,
            "compact_obstacle_status": capture.compact_obstacle_status,
            "perception_status": capture.perception_status,
        },
        "costmaps": {
            "global_disc_cell_max": costmap_disc_values(capture.global_costmap),
            "local_disc_cell_max": costmap_disc_values(capture.local_costmap),
        },
        "global_path": path_metrics(capture.global_path, normal, left_offset),
        "first_global_path": path_metrics(
            capture.first_global_path, normal, left_offset
        ),
        "inactive_make_plan_preview": path_metrics(
            capture.preview_path, normal, left_offset
        ),
        "teb_local_plan": path_metrics(capture.local_path, normal, left_offset),
        "teb_first_local_plan": path_metrics(
            capture.first_local_path, normal, left_offset
        ),
        "teb_selected_trajectory": teb_metrics(
            capture.feedback, normal, left_offset
        ),
        "teb_first_selected_trajectory": teb_metrics(
            capture.first_feedback, normal, left_offset
        ),
        "teb_feedback_history": teb_history_metrics(
            capture.feedback_history, normal, left_offset
        ),
        "commands": {
            "count": len(capture.commands),
            "first_16": capture.commands[:16],
            "max_abs_vx": max(
                (abs(item["vx"]) for item in capture.commands), default=0.0
            ),
            "max_abs_vy": max(
                (abs(item["vy"]) for item in capture.commands), default=0.0
            ),
            "max_abs_wz": max(
                (abs(item["wz"]) for item in capture.commands), default=0.0
            ),
            "positive_vy_count": sum(
                item["vy"] > 1e-5 for item in capture.commands
            ),
            "negative_vy_count": sum(
                item["vy"] < -1e-5 for item in capture.commands
            ),
            "mean_period_sec": (
                statistics.mean(command_intervals) if command_intervals else None
            ),
            "p95_period_sec": percentile(command_intervals, 0.95),
            "periods_over_0_1875_sec": sum(
                value > 0.1875 for value in command_intervals
            ),
        },
        "performance": cpu,
        "planner_events": {
            "trajectory_infeasible_reset_count": sum(
                "trajectory is not feasible. Resetting planner" in item["message"]
                for item in capture.measurement_logs
            ),
            "oscillation_abort_count": sum(
                "appears to be oscillating" in item["message"]
                for item in capture.measurement_logs
            ),
            "control_loop_missed_count": sum(
                "control loop missed" in item["message"].lower()
                for item in capture.measurement_logs
            ),
            "move_base_logs": capture.measurement_logs,
        },
        "teb_parameters": rospy.get_param("/move_base/TebLocalPlannerROS"),
        "goal_service_message": result.message,
    }
    with open(arguments.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
