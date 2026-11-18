#!/usr/bin/env python3
"""Capture Stage-2 walls once and publish a fixed map and compact obstacles."""

import json
import math
import threading

import numpy as np
import rospy
import tf2_ros
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Point, Point32
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.srv import GetMap, GetMapResponse
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

from yellow_track_avoidance.stage2_frozen_scene import (
    geometry_payload,
    rasterize_scene,
    wall_segments,
)
from yellow_track_avoidance.stage2_lidar_goal import (
    build_stage2_geometry,
    voxel_downsample,
)


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class Stage2FrozenSceneNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.scan_topic = str(rospy.get_param("~scan_topic", "/scan"))
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link")).lstrip("/")
        self.map_frame = str(rospy.get_param("~map_frame", "map")).lstrip("/")
        self.scan_frame_count = max(1, int(rospy.get_param("~scan_frame_count", 40)))
        self.voxel_resolution = float(rospy.get_param("~voxel_resolution_m", 0.015))
        self.expected_width = float(rospy.get_param("~expected_width_m", 0.485))
        self.width_tolerance = float(rospy.get_param("~width_tolerance_m", 0.10))
        self.maximum_linear_speed = float(
            rospy.get_param("~maximum_capture_linear_speed_mps", 0.01)
        )
        self.maximum_angular_speed = float(
            rospy.get_param("~maximum_capture_angular_speed_radps", 0.03)
        )
        self.stationary_stable_sec = float(rospy.get_param("~stationary_stable_sec", 0.5))
        self.auto_capture = bool(rospy.get_param("~auto_capture_when_ready", True))
        self.map_resolution = float(rospy.get_param("~map_resolution_m", 0.01))
        self.map_width = float(rospy.get_param("~map_width_m", 5.0))
        self.map_height = float(rospy.get_param("~map_height_m", 4.0))
        self.map_origin_x = float(rospy.get_param("~map_origin_x_m", -0.5))
        self.map_origin_y = float(rospy.get_param("~map_origin_y_m", -2.0))
        self.wall_thickness = float(rospy.get_param("~wall_thickness_m", 0.01))
        self.compact_publish_rate = max(
            1.0, float(rospy.get_param("~compact_publish_rate_hz", 4.0))
        )
        self.fit_options = {
            "minimum_x_m": float(rospy.get_param("~right_wall_minimum_x_m", 0.20)),
            "maximum_x_m": float(rospy.get_param("~right_wall_maximum_x_m", 3.50)),
            "maximum_y_m": float(rospy.get_param("~right_wall_maximum_y_m", -0.10)),
            "ransac_threshold_m": float(
                rospy.get_param("~right_wall_ransac_threshold_m", 0.025)
            ),
            "minimum_inliers": int(rospy.get_param("~right_wall_minimum_inliers", 18)),
            "terminal_minimum_forward_m": float(
                rospy.get_param("~terminal_minimum_forward_m", 1.0)
            ),
            "terminal_cluster_gap_m": float(
                rospy.get_param("~terminal_cluster_gap_m", 0.08)
            ),
            "terminal_minimum_support": int(
                rospy.get_param("~terminal_minimum_support", 8)
            ),
            "terminal_minimum_span_m": float(
                rospy.get_param("~terminal_minimum_span_m", 0.22)
            ),
        }

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.latest_odom = None
        self.stationary_since = None
        self.frames = []
        self.capturing = False
        self.processing = False
        self.frozen_map = None
        self.payload = None
        self.compact_obstacles = None
        self.last_error = None
        self.capture_stamp = None

        self.map_pub = rospy.Publisher("/map", OccupancyGrid, queue_size=1, latch=True)
        self.geometry_pub = rospy.Publisher(
            "/yellow_track_avoidance/stage2_frozen_geometry",
            String,
            queue_size=1,
            latch=True,
        )
        self.obstacle_pub = rospy.Publisher(
            "/move_base/TebLocalPlannerROS/obstacles",
            ObstacleArrayMsg,
            queue_size=1,
            latch=True,
        )
        self.marker_pub = rospy.Publisher(
            "/yellow_track_avoidance/stage2_frozen_scene_markers",
            MarkerArray,
            queue_size=1,
            latch=True,
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.map_service = rospy.Service("/static_map", GetMap, self._map_service)
        self.capture_service = rospy.Service("~capture", Trigger, self._capture_service)
        self.reset_service = rospy.Service("~reset", Trigger, self._reset_service)
        self.scan_sub = rospy.Subscriber(
            self.scan_topic, LaserScan, self._scan_callback, queue_size=30
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_callback, queue_size=1
        )
        self.timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)
        self.compact_timer = rospy.Timer(
            rospy.Duration(1.0 / self.compact_publish_rate), self._publish_compact
        )
        rospy.loginfo(
            "Stage-2 frozen scene ready: capture %d scans once; no cmd_vel publisher",
            self.scan_frame_count,
        )

    def _odom_callback(self, message):
        with self.lock:
            self.latest_odom = message

    def _stationary(self):
        if self.latest_odom is None:
            return False
        twist = self.latest_odom.twist.twist
        return (
            math.hypot(twist.linear.x, twist.linear.y) <= self.maximum_linear_speed
            and abs(twist.angular.z) <= self.maximum_angular_speed
        )

    def _start_capture(self):
        self.frames = []
        self.capturing = True
        self.processing = False
        self.last_error = None
        rospy.loginfo("Stage-2 frozen-map capture started (%d frames)", self.scan_frame_count)

    def _capture_service(self, _request):
        with self.lock:
            if self.frozen_map is not None:
                return TriggerResponse(True, "Stage-2 scene is already frozen")
            if self.capturing or self.processing:
                return TriggerResponse(True, "Stage-2 capture is already running")
            if not self._stationary():
                return TriggerResponse(False, "robot odometry is not stationary")
            self._start_capture()
        return TriggerResponse(True, "Stage-2 frozen-map capture started")

    def _reset_service(self, _request):
        with self.condition:
            self.frames = []
            self.capturing = False
            self.processing = False
            self.frozen_map = None
            self.payload = None
            self.compact_obstacles = None
            self.last_error = None
            self.capture_stamp = None
            self.condition.notify_all()
        return TriggerResponse(True, "Stage-2 frozen scene cleared")

    def _map_service(self, _request):
        with self.condition:
            deadline = rospy.Time.now() + rospy.Duration(5.0)
            while self.frozen_map is None and not rospy.is_shutdown():
                remaining = (deadline - rospy.Time.now()).to_sec()
                if remaining <= 0.0:
                    return GetMapResponse()
                self.condition.wait(timeout=min(0.2, remaining))
            return GetMapResponse(map=self.frozen_map) if self.frozen_map else GetMapResponse()

    def _scan_points(self, message):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            message.header.frame_id,
            message.header.stamp,
            rospy.Duration(0.2),
        )
        ranges = np.asarray(message.ranges, dtype=np.float64)
        angles = message.angle_min + np.arange(len(ranges)) * message.angle_increment
        valid = (
            np.isfinite(ranges)
            & (ranges >= message.range_min)
            & (ranges <= message.range_max)
        )
        laser = np.column_stack(
            (ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid]))
        )
        angle = _yaw(transform.transform.rotation)
        rotation = np.asarray(
            ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))),
            dtype=np.float64,
        )
        translation = np.asarray(
            (transform.transform.translation.x, transform.transform.translation.y),
            dtype=np.float64,
        )
        return laser @ rotation.T + translation

    def _scan_callback(self, message):
        with self.lock:
            if not self.capturing or self.frozen_map is not None:
                return
            if not self._stationary():
                self.frames = []
                self.capturing = False
                self.last_error = "robot moved during Stage-2 frozen-map capture"
                return
        try:
            points = self._scan_points(message)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(1.0, "waiting for Stage-2 lidar TF: %s", error)
            return
        with self.lock:
            if not self.capturing:
                return
            self.frames.append(points)
            self.capture_stamp = message.header.stamp
            if len(self.frames) < self.scan_frame_count:
                return
            frames = self.frames
            stamp = self.capture_stamp
            self.frames = []
            self.capturing = False
            self.processing = True
        self._finish_capture(frames, stamp)

    def _finish_capture(self, frames, stamp):
        try:
            points = voxel_downsample(np.vstack(frames), self.voxel_resolution)
            geometry = build_stage2_geometry(
                points,
                expected_width_m=self.expected_width,
                width_tolerance_m=self.width_tolerance,
                **self.fit_options
            )
            payload = geometry_payload(geometry, self.map_frame)
            payload["capture_stamp"] = stamp.to_sec()
            grid = rasterize_scene(
                payload,
                resolution_m=self.map_resolution,
                width_m=self.map_width,
                height_m=self.map_height,
                origin_x_m=self.map_origin_x,
                origin_y_m=self.map_origin_y,
                wall_thickness_m=self.wall_thickness,
            )
            frozen_map = self._map_message(stamp, grid)
            compact = self._compact_message(stamp, payload)
            markers = self._marker_message(stamp, payload)
        except Exception as error:
            with self.lock:
                self.processing = False
                self.last_error = str(error)
            rospy.logerr("Stage-2 frozen-map capture failed: %s", error)
            return
        with self.condition:
            self.processing = False
            self.frozen_map = frozen_map
            self.payload = payload
            self.compact_obstacles = compact
            self.last_error = None
            self.condition.notify_all()
        self.map_pub.publish(frozen_map)
        self.geometry_pub.publish(String(data=json.dumps(payload, allow_nan=False)))
        self.obstacle_pub.publish(compact)
        self.marker_pub.publish(markers)
        rospy.loginfo(
            "Stage-2 map frozen once: width %.3f m, yaw %.2f deg, 3 wall primitives",
            geometry.measured_width_m,
            math.degrees(geometry.heading_rad),
        )

    def _map_message(self, stamp, grid):
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = self.map_frame
        message.info.map_load_time = stamp
        message.info.resolution = self.map_resolution
        message.info.width = grid.shape[1]
        message.info.height = grid.shape[0]
        message.info.origin.position.x = self.map_origin_x
        message.info.origin.position.y = self.map_origin_y
        message.info.origin.orientation.w = 1.0
        message.data = grid.reshape(-1).tolist()
        return message

    def _compact_message(self, stamp, payload):
        output = ObstacleArrayMsg()
        output.header.stamp = stamp
        output.header.frame_id = self.map_frame
        for obstacle_id, (first, second) in enumerate(wall_segments(payload), start=1):
            obstacle = ObstacleMsg()
            obstacle.header = output.header
            obstacle.id = obstacle_id
            obstacle.orientation.w = 1.0
            for xy in (first, second):
                point = Point32()
                point.x, point.y = float(xy[0]), float(xy[1])
                obstacle.polygon.points.append(point)
            output.obstacles.append(obstacle)
        return output

    @staticmethod
    def _color(red, green, blue):
        return ColorRGBA(red, green, blue, 1.0)

    def _marker_message(self, stamp, payload):
        colors = (
            self._color(0.1, 1.0, 0.2),
            self._color(0.0, 0.9, 1.0),
            self._color(0.75, 0.2, 1.0),
        )
        output = []
        for marker_id, ((first, second), color) in enumerate(
            zip(wall_segments(payload), colors), start=1
        ):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = self.map_frame, stamp
            marker.ns, marker.id = "stage2_frozen_walls", marker_id
            marker.type, marker.action = Marker.LINE_LIST, Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.color = 0.025, color
            for xy in (first, second):
                point = Point(x=float(xy[0]), y=float(xy[1]), z=0.07)
                marker.points.append(point)
            output.append(marker)
        return MarkerArray(markers=output)

    def _publish_compact(self, _event):
        with self.lock:
            message = self.compact_obstacles
        if message is not None:
            message.header.stamp = rospy.Time.now()
            for obstacle in message.obstacles:
                obstacle.header.stamp = message.header.stamp
            self.obstacle_pub.publish(message)

    def _timer_callback(self, _event):
        now = rospy.Time.now()
        with self.lock:
            stationary = self._stationary()
            if stationary:
                if self.stationary_since is None:
                    self.stationary_since = now
            else:
                self.stationary_since = None
            stable = (
                stationary
                and self.stationary_since is not None
                and (now - self.stationary_since).to_sec() >= self.stationary_stable_sec
            )
            if (
                self.auto_capture
                and stable
                and self.frozen_map is None
                and not self.capturing
                and not self.processing
            ):
                self._start_capture()
            status = {
                "mode": "FROZEN" if self.frozen_map is not None else "CAPTURE",
                "stationary": stationary,
                "capturing": self.capturing,
                "captured_frames": len(self.frames),
                "map_frozen": self.frozen_map is not None,
                "scan_updates_costmap": False,
                "compact_obstacle_count": (
                    len(self.compact_obstacles.obstacles)
                    if self.compact_obstacles is not None
                    else 0
                ),
                "last_error": self.last_error,
            }
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))


def main():
    rospy.init_node("yellow_track_stage2_frozen_scene")
    Stage2FrozenSceneNode()
    rospy.spin()


if __name__ == "__main__":
    main()
