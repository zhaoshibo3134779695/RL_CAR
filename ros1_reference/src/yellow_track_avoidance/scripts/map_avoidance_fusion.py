#!/usr/bin/env python3
"""Fuse current-frame local avoidance candidates into the static map route."""

import json
import math

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String

from yellow_track_avoidance.config import load_config
from yellow_track_avoidance.map_navigation import (
    avoidance_enabled,
    build_reference_path,
    fuse_candidate_with_reference,
    load_metric_map,
    load_yaml,
    local_to_map,
    remaining_reference_path,
    validate_vehicle_path,
)
from yellow_track_avoidance.planning import polyline_clearance_to_point


class MapAvoidanceFusion:
    def __init__(self):
        route_file = rospy.get_param("~route_file")
        perception_config_file = rospy.get_param("~perception_config_file")
        self.route, self.route_path = load_yaml(route_file)
        self.perception = load_config(perception_config_file)
        self.reference = build_reference_path(self.route)
        (
            self.occupancy,
            self.resolution,
            self.origin,
            _,
            _,
        ) = load_metric_map(self.route, self.route_path)
        start = self.route["start"]
        self.pose = (
            float(start["x_m"]),
            float(start["y_m"]),
            float(start["yaw_rad"]),
        )
        fusion = self.route["fusion"]
        self.pose_source = str(
            rospy.get_param("~pose_source", fusion.get("pose_source", "fixed_start"))
        )
        self.map_frame = str(
            rospy.get_param(
                "~map_frame", fusion.get("map_frame", self.route["frame_id"])
            )
        ).lstrip("/")
        self.base_frame = str(
            rospy.get_param("~base_frame", fusion.get("base_frame", "base_link"))
        ).lstrip("/")
        self.tf_timeout = float(
            rospy.get_param(
                "~tf_lookup_timeout_sec", fusion.get("tf_lookup_timeout_sec", 0.08)
            )
        )
        self.maximum_pose_age = float(
            rospy.get_param(
                "~maximum_pose_age_sec", fusion.get("maximum_pose_age_sec", 0.30)
            )
        )
        self.maximum_route_distance = float(
            rospy.get_param(
                "~maximum_route_distance_m",
                fusion.get("maximum_route_distance_m", 0.45),
            )
        )
        self.forward_offset = float(
            rospy.get_param(
                "~local_forward_origin_offset_m",
                fusion["local_forward_origin_offset_m"],
            )
        )
        self.maximum_age = float(fusion["maximum_message_age_sec"])
        self.latest_disks = None
        self.latest_disk_stamp = None
        self.pending_candidate = None
        self.tf_buffer = None
        self.tf_listener = None
        if self.pose_source == "tf":
            self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        elif self.pose_source != "fixed_start":
            raise ValueError(
                "fusion pose_source must be 'fixed_start' or 'tf', got {}".format(
                    self.pose_source
                )
            )
        self.path_pub = rospy.Publisher(
            str(fusion["fused_path_topic"]), Path, queue_size=1, latch=True
        )
        self.status_pub = rospy.Publisher(
            str(fusion["status_topic"]), String, queue_size=1, latch=True
        )
        self.disk_sub = rospy.Subscriber(
            str(fusion["disk_topic"]), String, self._disk_callback, queue_size=1
        )
        self.candidate_sub = rospy.Subscriber(
            str(fusion["candidate_topic"]),
            String,
            self._candidate_callback,
            queue_size=1,
        )
        if self.pose_source == "fixed_start":
            rospy.loginfo(
                "map avoidance fusion ready: fixed_pose=(%.2f, %.2f, %.1fdeg) motion_output=none",
                self.pose[0], self.pose[1], math.degrees(self.pose[2]),
            )
        else:
            rospy.loginfo(
                "map avoidance fusion ready: pose=tf %s->%s motion_output=none",
                self.map_frame,
                self.base_frame,
            )

    @staticmethod
    def _quaternion_yaw(quaternion):
        sin_yaw = 2.0 * (
            quaternion.w * quaternion.z + quaternion.x * quaternion.y
        )
        cos_yaw = 1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        )
        return math.atan2(sin_yaw, cos_yaw)

    def _pose_for_stamp(self, stamp):
        if self.pose_source == "fixed_start":
            return self.pose, 0.0, None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rospy.Time.from_sec(float(stamp)),
                rospy.Duration(self.tf_timeout),
            )
        except Exception as exc:
            exact_error = str(exc)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.base_frame,
                    rospy.Time(0),
                    rospy.Duration(self.tf_timeout),
                )
            except Exception as latest_exc:
                return None, None, "exact={} latest={}".format(
                    exact_error, latest_exc
                )
        transform_stamp = transform.header.stamp.to_sec()
        pose_age = abs(float(stamp) - transform_stamp)
        if transform_stamp > 0.0 and pose_age > self.maximum_pose_age:
            return None, pose_age, "localization_pose_too_old"
        translation = transform.transform.translation
        self.pose = (
            float(translation.x),
            float(translation.y),
            self._quaternion_yaw(transform.transform.rotation),
        )
        return self.pose, pose_age, None

    def _disk_callback(self, message):
        payload = json.loads(message.data)
        self.latest_disks = payload.get("raw", [])
        stamp = payload.get("stamp")
        if stamp is None:
            stamps = [
                float(item["stamp"])
                for item in self.latest_disks
                if item.get("stamp") is not None
            ]
            stamp = max(stamps) if stamps else None
        self.latest_disk_stamp = None if stamp is None else float(stamp)
        if self.pending_candidate is not None and self.latest_disk_stamp is not None:
            pending_stamp = float(self.pending_candidate.get("stamp", -1.0))
            if abs(pending_stamp - self.latest_disk_stamp) <= 1e-6:
                pending = self.pending_candidate
                self.pending_candidate = None
                self._process_candidate(pending)

    def _publish_path(self, points, stamp):
        message = Path()
        message.header.frame_id = str(self.route["frame_id"])
        message.header.stamp = rospy.Time.from_sec(float(stamp))
        if points is not None:
            values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            if len(values) >= 2:
                headings = np.unwrap(
                    np.arctan2(np.gradient(values[:, 1]), np.gradient(values[:, 0]))
                )
                for (x, y), yaw in zip(values, headings):
                    pose = PoseStamped()
                    pose.header = message.header
                    pose.pose.position.x = float(x)
                    pose.pose.position.y = float(y)
                    pose.pose.orientation.z = math.sin(0.5 * float(yaw))
                    pose.pose.orientation.w = math.cos(0.5 * float(yaw))
                    message.poses.append(pose)
        self.path_pub.publish(message)

    def _candidate_callback(self, message):
        payload = json.loads(message.data)
        stamp = float(payload.get("stamp", rospy.Time.now().to_sec()))
        if self.latest_disk_stamp is None or abs(stamp - self.latest_disk_stamp) > 1e-6:
            self.pending_candidate = payload
            return
        self._process_candidate(payload)

    def _process_candidate(self, payload):
        stamp = float(payload.get("stamp", rospy.Time.now().to_sec()))
        candidates = {
            item.get("name"): item for item in payload.get("candidates", [])
        }
        selected_name = payload.get("selected")
        center = candidates.get("center")
        selected = candidates.get(selected_name)
        raw_disks = self.latest_disks or []
        synchronized = self.latest_disk_stamp is not None and abs(
            stamp - self.latest_disk_stamp
        ) <= 1e-6
        message_age = max(0.0, rospy.Time.now().to_sec() - stamp)
        pose, pose_age, pose_error = self._pose_for_stamp(stamp)

        local_centers = np.asarray(
            [item.get("center_m", [float("nan"), float("nan")]) for item in raw_disks],
            dtype=np.float64,
        ).reshape(-1, 2)
        finite = np.all(np.isfinite(local_centers), axis=1) if len(local_centers) else np.zeros(0, dtype=bool)
        local_centers = local_centers[finite]
        map_centers = (
            np.zeros((0, 2), dtype=np.float64)
            if pose is None
            else local_to_map(local_centers, pose, self.forward_offset)
        )
        handoff_x = float(self.route["avoidance"]["handoff_x_m"])
        in_zone = map_centers[:, 0] > handoff_x if len(map_centers) else np.zeros(0, dtype=bool)
        map_obstacles = map_centers[in_zone]

        mode = "no_safe_local_candidate"
        fused = None
        route_distance = None
        if pose is not None:
            route_distance = float(
                np.min(
                    np.linalg.norm(
                        self.reference - np.asarray(pose[:2], dtype=np.float64),
                        axis=1,
                    )
                )
            )
        if message_age > self.maximum_age:
            mode = "perception_message_too_old"
        elif pose_error is not None:
            mode = "localization_unavailable"
        elif route_distance is not None and route_distance > self.maximum_route_distance:
            mode = "robot_outside_route_corridor"
        elif not synchronized:
            mode = "waiting_for_synchronized_current_frame"
        elif not avoidance_enabled(self.route, pose):
            fused = remaining_reference_path(self.reference, pose)
            mode = "fixed_narrow_route"
        elif payload.get("safe") and center is not None and selected is not None:
            fused, mode = fuse_candidate_with_reference(
                self.route,
                self.reference,
                pose,
                np.asarray(center["points_m"], dtype=np.float64),
                np.asarray(selected["points_m"], dtype=np.float64),
            )

        map_valid = None
        obstacle_valid = None
        vehicle_report = None
        minimum_obstacle_clearance = None
        if fused is not None:
            physical = self.perception["physical"]
            margin = float(self.route["safety"]["map_boundary_margin_m"])
            vehicle_report = validate_vehicle_path(
                self.occupancy,
                self.resolution,
                self.origin,
                fused,
                float(physical["robot_length_m"]),
                float(physical["robot_width_m"]),
                margin,
            )
            map_valid = bool(vehicle_report["collision_free"])
            expanded = (
                0.5 * float(physical["disk_diameter_m"])
                + 0.5 * float(physical["robot_width_m"])
                + float(self.perception["safety"]["obstacle_margin_m"])
                + float(self.perception["planner"]["collision_sample_padding_m"])
            )
            clearances = [
                polyline_clearance_to_point(fused, center_m) - expanded
                for center_m in map_obstacles
            ]
            minimum_obstacle_clearance = min(clearances) if clearances else None
            obstacle_valid = not clearances or minimum_obstacle_clearance >= 0.0
            if not map_valid:
                mode = "map_wall_collision"
                fused = None
            elif not obstacle_valid:
                mode = "map_obstacle_collision"
                fused = None

        safe = fused is not None and map_valid is True and obstacle_valid is True
        self._publish_path(fused if safe else None, stamp)
        status = {
            "stamp": stamp,
            "safe": safe,
            "mode": mode,
            "motion_output": "none",
            "pose_source": self.pose_source,
            "robot_pose_map": None if pose is None else list(pose),
            "pose_age_sec": pose_age,
            "pose_error": pose_error,
            "perception_message_age_sec": message_age,
            "route_distance_m": route_distance,
            "input_synchronized": synchronized,
            "local_selected": selected_name,
            "current_frame_disc_count": len(local_centers),
            "avoidance_zone_disc_count": len(map_obstacles),
            "map_disc_centers_m": map_centers.tolist(),
            "map_wall_valid": map_valid,
            "map_obstacle_valid": obstacle_valid,
            "minimum_obstacle_clearance_m": minimum_obstacle_clearance,
            "fused_path_points": 0 if fused is None else len(fused),
            "vehicle_validation": vehicle_report,
        }
        self.status_pub.publish(
            String(data=json.dumps(status, ensure_ascii=False, allow_nan=False))
        )


if __name__ == "__main__":
    rospy.init_node("yellow_track_map_avoidance_fusion")
    MapAvoidanceFusion()
    rospy.spin()
