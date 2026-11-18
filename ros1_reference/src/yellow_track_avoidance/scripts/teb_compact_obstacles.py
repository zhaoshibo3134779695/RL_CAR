#!/usr/bin/env python3
"""Publish the compact TEB model and optional Stage-1 gap closures."""

import json
import math
import threading

from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Point, Point32
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

from yellow_track_avoidance.compact_obstacles import (
    CompactSceneError,
    build_compact_obstacle_scene,
    sample_filled_disc_points,
    select_teb_obstacles,
)


class TebCompactObstaclePublisher:
    def __init__(self):
        self.lock = threading.RLock()
        self.boundary_payload = None
        self.disc_status = None
        self.publish_count = 0
        self.empty_publish_count = 0
        self.closure_cloud_publish_count = 0
        self.last_ready_generation = None
        self.last_reason = "waiting_for_inputs"

        self.boundary_topic = str(
            rospy.get_param(
                "~boundary_topic", "/yellow_track_avoidance/corridor_boundaries"
            )
        )
        self.disc_status_topic = str(
            rospy.get_param(
                "~disc_status_topic", "/yellow_track_disc_costmap_bridge/status"
            )
        )
        self.output_topic = str(
            rospy.get_param(
                "~output_topic", "/move_base/TebLocalPlannerROS/obstacles"
            )
        )
        self.disc_radius = float(rospy.get_param("~disc_radius_m", 0.05))
        self.disc_collision_radius = float(
            rospy.get_param("~disc_collision_radius_m", self.disc_radius)
        )
        self.teb_disc_collision_radius = float(
            rospy.get_param(
                "~teb_disc_collision_radius_m",
                self.disc_collision_radius,
            )
        )
        # Keep the local feasibility map independently adjustable.  Enlarging
        # this cloud must not move the GlobalPlanner route, change passage-side
        # selection, or silently enlarge TEB's semantic CircularObstacles.
        self.local_disc_collision_radius = float(
            rospy.get_param(
                "~local_disc_collision_radius_m",
                self.disc_collision_radius,
            )
        )
        if (
            not math.isfinite(self.disc_collision_radius)
            or not math.isfinite(self.teb_disc_collision_radius)
            or not math.isfinite(self.local_disc_collision_radius)
            or self.disc_collision_radius < self.disc_radius
            or self.teb_disc_collision_radius < self.disc_radius
            or self.local_disc_collision_radius < self.disc_radius
        ):
            raise ValueError(
                "global, TEB and local disc collision radii must be finite "
                "and no smaller than the physical disc radius"
            )
        self.yellow_half_width = 0.5 * float(
            rospy.get_param("~yellow_line_width_m", 0.03)
        )
        self.publish_rate = max(
            1.0, float(rospy.get_param("~publish_rate_hz", 5.0))
        )
        self.enable_passage_closures = bool(
            rospy.get_param("~enable_passage_closures", False)
        )
        # A passage closure is a global topology hint, not a physical wall.
        # Stage 1 publishes the closure cloud to the global costmap but keeps
        # it out of TEB so the local optimizer only sees real geometry.
        self.include_passage_closures_in_teb = bool(
            rospy.get_param("~include_passage_closures_in_teb", True)
        )
        self.publish_full_disc_cloud = bool(
            rospy.get_param("~publish_full_disc_cloud", False)
        )
        self.full_disc_cloud_topic = str(
            rospy.get_param(
                "~full_disc_cloud_topic",
                "/yellow_track_avoidance/sbpl_disc_cloud",
            )
        )
        self.local_full_disc_cloud_topic = str(
            rospy.get_param(
                "~local_full_disc_cloud_topic",
                "/yellow_track_avoidance/local_disc_cloud",
            )
        )
        self.full_disc_sample_resolution = max(
            0.002,
            float(rospy.get_param("~full_disc_sample_resolution_m", 0.005)),
        )
        self.vehicle_width = float(rospy.get_param("~vehicle_width_m", 0.33))
        self.passage_total_clearance = float(
            rospy.get_param("~passage_total_clearance_m", 0.02)
        )
        self.side_tie_tolerance = float(
            rospy.get_param("~side_tie_tolerance_m", 0.01)
        )
        self.minimum_disc_count = max(
            1, int(rospy.get_param("~minimum_disc_count", 2))
        )
        self.closure_sample_resolution = max(
            0.002, float(rospy.get_param("~closure_sample_resolution_m", 0.005))
        )
        self.closure_cloud_topic = str(
            rospy.get_param(
                "~closure_cloud_topic",
                "/yellow_track_avoidance/passage_closure_cloud",
            )
        )
        self.closure_marker_topic = str(
            rospy.get_param(
                "~closure_marker_topic",
                "/yellow_track_avoidance/passage_closure_markers",
            )
        )

        self.obstacle_pub = rospy.Publisher(
            self.output_topic, ObstacleArrayMsg, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.closure_cloud_pub = rospy.Publisher(
            self.closure_cloud_topic, PointCloud2, queue_size=1, latch=True
        )
        self.full_disc_cloud_pub = rospy.Publisher(
            self.full_disc_cloud_topic, PointCloud2, queue_size=1, latch=True
        )
        self.local_full_disc_cloud_pub = rospy.Publisher(
            self.local_full_disc_cloud_topic,
            PointCloud2,
            queue_size=1,
            latch=True,
        )
        self.closure_marker_pub = rospy.Publisher(
            self.closure_marker_topic, MarkerArray, queue_size=1, latch=True
        )
        self.boundary_sub = rospy.Subscriber(
            self.boundary_topic, String, self._boundary_callback, queue_size=1
        )
        self.disc_sub = rospy.Subscriber(
            self.disc_status_topic, String, self._disc_callback, queue_size=1
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate), self._timer_callback
        )
        rospy.loginfo(
            "TEB compact obstacle publisher ready: %s -> %s, passage closures=%s",
            self.boundary_topic,
            self.output_topic,
            self.enable_passage_closures,
        )

    def _decode(self, message, label):
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("payload is not an object")
            return payload
        except (TypeError, ValueError):
            rospy.logwarn_throttle(1.0, "invalid %s JSON for compact obstacles", label)
            return None

    def _boundary_callback(self, message):
        payload = self._decode(message, "corridor boundary")
        if payload is not None:
            with self.lock:
                self.boundary_payload = payload

    def _disc_callback(self, message):
        payload = self._decode(message, "disc status")
        if payload is not None:
            with self.lock:
                self.disc_status = payload

    @staticmethod
    def _message(scene, stamp, include_passage_closures=True):
        output = ObstacleArrayMsg()
        output.header.stamp = stamp
        output.header.frame_id = scene.frame_id
        for item in select_teb_obstacles(scene, include_passage_closures):
            obstacle = ObstacleMsg()
            obstacle.header = output.header
            obstacle.id = item.obstacle_id
            obstacle.radius = item.radius_m
            obstacle.orientation.w = 1.0
            for x_value, y_value in item.points:
                point = Point32()
                point.x = x_value
                point.y = y_value
                obstacle.polygon.points.append(point)
            output.obstacles.append(obstacle)
        return output

    @staticmethod
    def _empty_message(stamp, frame_id):
        output = ObstacleArrayMsg()
        output.header.stamp = stamp
        output.header.frame_id = frame_id or "odom"
        return output

    def _closure_cloud(self, scene, stamp):
        points = []
        for decision in scene.passage_decisions:
            if not decision.closure:
                continue
            start, end = decision.closure
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = (dx * dx + dy * dy) ** 0.5
            count = max(2, int(length / self.closure_sample_resolution) + 1)
            for index in range(count):
                ratio = float(index) / float(count - 1)
                points.append(
                    (
                        start[0] + ratio * dx,
                        start[1] + ratio * dy,
                        0.0,
                    )
                )
        return point_cloud2.create_cloud_xyz32(
            Header(stamp=stamp, frame_id=scene.frame_id), points
        )

    def _full_disc_cloud(self, scene, stamp):
        points = sample_filled_disc_points(
            scene, resolution_m=self.full_disc_sample_resolution
        )
        return point_cloud2.create_cloud_xyz32(
            Header(stamp=stamp, frame_id=scene.frame_id), points
        )

    @staticmethod
    def _empty_cloud(stamp, frame_id):
        return point_cloud2.create_cloud_xyz32(
            Header(stamp=stamp, frame_id=frame_id or "odom"), []
        )

    @staticmethod
    def _marker_point(xy, z):
        point = Point()
        point.x = float(xy[0])
        point.y = float(xy[1])
        point.z = float(z)
        return point

    def _closure_markers(self, scene, stamp):
        output = MarkerArray()
        for index, decision in enumerate(scene.passage_decisions):
            if decision.closure:
                closure = Marker()
                closure.header.stamp = stamp
                closure.header.frame_id = scene.frame_id
                closure.ns = "blocked_disc_line_gap"
                closure.id = 3000 + decision.memory_id
                closure.type = Marker.LINE_STRIP
                closure.action = Marker.ADD
                closure.pose.orientation.w = 1.0
                closure.scale.x = 0.025
                closure.color.r = 1.0
                closure.color.g = 0.05
                closure.color.b = 0.05
                closure.color.a = 1.0
                closure.points = [
                    self._marker_point(point, 0.09) for point in decision.closure
                ]
                output.markers.append(closure)

            label = Marker()
            label.header.stamp = stamp
            label.header.frame_id = scene.frame_id
            label.ns = "disc_gap_decision"
            label.id = 4000 + decision.memory_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = self._marker_point(decision.center, 0.18)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.075
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = (
                "D{} L={:.2f} R={:.2f} pass {}".format(
                    decision.memory_id,
                    decision.left_gap_m,
                    decision.right_gap_m,
                    decision.pass_side.upper(),
                )
            )
            output.markers.append(label)
        return output

    def _timer_callback(self, _event):
        now = rospy.Time.now()
        with self.lock:
            boundary_payload = self.boundary_payload
            disc_status = self.disc_status
        try:
            scene = build_compact_obstacle_scene(
                boundary_payload,
                disc_status,
                disc_radius_m=self.disc_radius,
                disc_collision_radius_m=self.teb_disc_collision_radius,
                yellow_line_half_width_m=self.yellow_half_width,
                enable_passage_closures=self.enable_passage_closures,
                vehicle_width_m=self.vehicle_width,
                passage_total_clearance_m=self.passage_total_clearance,
                side_tie_tolerance_m=self.side_tie_tolerance,
                minimum_disc_count=self.minimum_disc_count,
            )
            message = self._message(
                scene,
                now,
                include_passage_closures=self.include_passage_closures_in_teb,
            )
            closure_cloud = self._closure_cloud(scene, now)
            if self.publish_full_disc_cloud:
                global_scene = build_compact_obstacle_scene(
                    boundary_payload,
                    disc_status,
                    disc_radius_m=self.disc_radius,
                    disc_collision_radius_m=self.disc_collision_radius,
                    yellow_line_half_width_m=self.yellow_half_width,
                    enable_passage_closures=False,
                    vehicle_width_m=self.vehicle_width,
                    passage_total_clearance_m=self.passage_total_clearance,
                    side_tie_tolerance_m=self.side_tie_tolerance,
                    minimum_disc_count=self.minimum_disc_count,
                )
                full_disc_cloud = self._full_disc_cloud(global_scene, now)
                local_scene = build_compact_obstacle_scene(
                    boundary_payload,
                    disc_status,
                    disc_radius_m=self.disc_radius,
                    disc_collision_radius_m=self.local_disc_collision_radius,
                    yellow_line_half_width_m=self.yellow_half_width,
                    enable_passage_closures=False,
                    vehicle_width_m=self.vehicle_width,
                    passage_total_clearance_m=self.passage_total_clearance,
                    side_tie_tolerance_m=self.side_tie_tolerance,
                    minimum_disc_count=self.minimum_disc_count,
                )
                local_full_disc_cloud = self._full_disc_cloud(local_scene, now)
            else:
                full_disc_cloud = self._empty_cloud(now, scene.frame_id)
                local_full_disc_cloud = self._empty_cloud(now, scene.frame_id)
            closure_markers = self._closure_markers(scene, now)
            ready = True
            reason = "ready"
        except CompactSceneError as exc:
            scene = None
            frame_id = ""
            if isinstance(boundary_payload, dict):
                frame_id = str(boundary_payload.get("frame_id", "")).lstrip("/")
            message = self._empty_message(now, frame_id)
            closure_cloud = self._empty_cloud(now, frame_id)
            full_disc_cloud = self._empty_cloud(now, frame_id)
            local_full_disc_cloud = self._empty_cloud(now, frame_id)
            closure_markers = MarkerArray()
            ready = False
            reason = str(exc)

        self.obstacle_pub.publish(message)
        self.closure_cloud_pub.publish(closure_cloud)
        self.full_disc_cloud_pub.publish(full_disc_cloud)
        self.local_full_disc_cloud_pub.publish(local_full_disc_cloud)
        self.closure_marker_pub.publish(closure_markers)
        with self.lock:
            self.publish_count += 1
            if ready:
                self.last_ready_generation = scene.scene_generation
                self.closure_cloud_publish_count += 1
            else:
                self.empty_publish_count += 1
            self.last_reason = reason
            status = {
                "stamp": now.to_sec(),
                "ready": ready,
                "reason": reason,
                "scene_generation": (
                    scene.scene_generation if scene is not None else None
                ),
                "last_ready_generation": self.last_ready_generation,
                "frame_id": scene.frame_id if scene is not None else message.header.frame_id,
                "obstacle_count": len(message.obstacles),
                "line_count": sum(
                    len(item.polygon.points) == 2 for item in message.obstacles
                ),
                "circle_count": sum(
                    len(item.polygon.points) == 1 and item.radius > 0.0
                    for item in message.obstacles
                ),
                "passage_closures_enabled": self.enable_passage_closures,
                "passage_closures_in_teb": self.include_passage_closures_in_teb,
                "passage_ready": ready and self.enable_passage_closures,
                "closure_count": (
                    sum(bool(item.closure) for item in scene.passage_decisions)
                    if scene is not None
                    else 0
                ),
                "passage_decisions": (
                    [
                        {
                            "memory_id": item.memory_id,
                            "center_m": list(item.center),
                            "longitudinal_m": item.longitudinal_m,
                            "left_gap_m": item.left_gap_m,
                            "right_gap_m": item.right_gap_m,
                            "required_gap_m": item.required_gap_m,
                            "pass_side": item.pass_side,
                            "blocked_side": item.blocked_side,
                            "selected_clearance_m": (
                                max(item.left_gap_m, item.right_gap_m)
                                - item.required_gap_m
                            ),
                        }
                        for item in scene.passage_decisions
                    ]
                    if scene is not None
                    else []
                ),
                "publish_count": self.publish_count,
                "empty_publish_count": self.empty_publish_count,
                "closure_cloud_publish_count": self.closure_cloud_publish_count,
                "output_topic": self.output_topic,
                "closure_cloud_topic": self.closure_cloud_topic,
                "full_disc_cloud_enabled": self.publish_full_disc_cloud,
                "full_disc_cloud_topic": self.full_disc_cloud_topic,
                "full_disc_point_count": (
                    int(full_disc_cloud.width * full_disc_cloud.height)
                    if self.publish_full_disc_cloud
                    else 0
                ),
                "full_disc_sample_resolution_m": self.full_disc_sample_resolution,
                "local_full_disc_cloud_topic": self.local_full_disc_cloud_topic,
                "local_full_disc_point_count": (
                    int(
                        local_full_disc_cloud.width
                        * local_full_disc_cloud.height
                    )
                    if self.publish_full_disc_cloud
                    else 0
                ),
                "closure_marker_topic": self.closure_marker_topic,
                "disc_radius_m": self.disc_radius,
                "disc_collision_radius_m": self.disc_collision_radius,
                "teb_disc_collision_radius_m": (
                    self.teb_disc_collision_radius
                ),
                "local_disc_collision_radius_m": (
                    self.local_disc_collision_radius
                ),
                "disc_hard_clearance_m": (
                    self.disc_collision_radius - self.disc_radius
                ),
                "teb_disc_hard_clearance_m": (
                    self.teb_disc_collision_radius - self.disc_radius
                ),
                "local_disc_hard_clearance_m": (
                    self.local_disc_collision_radius - self.disc_radius
                ),
                "yellow_line_half_width_m": self.yellow_half_width,
                "vehicle_width_m": self.vehicle_width,
                "passage_total_clearance_m": self.passage_total_clearance,
                "side_tie_tolerance_m": self.side_tie_tolerance,
            }
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))


def main():
    rospy.init_node("yellow_track_teb_compact_obstacles")
    TebCompactObstaclePublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
