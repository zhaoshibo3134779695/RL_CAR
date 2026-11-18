#!/usr/bin/env python3
"""Build a blank-map corridor from the observed left yellow boundary."""

from collections import deque
import json
import math
import threading

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetPlan
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger, TriggerResponse
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from yellow_track_avoidance.corridor_map import (
    CorridorFit,
    blend_heading_reference,
    angular_difference,
    build_corridor_segments,
    build_corridor_stop_goal,
    extend_line_segment,
    fit_left_boundary,
    sample_line_strip,
)


class CompetitionCorridorMap:
    def __init__(self):
        self.lock = threading.RLock()
        self.input_topic = str(
            rospy.get_param(
                "~boundary_topic", "/yellow_track_avoidance/debug/boundaries"
            )
        )
        self.fixed_frame = str(rospy.get_param("~fixed_frame", "odom")).lstrip("/")
        self.planning_frame = str(
            rospy.get_param("~planning_frame", "map")
        ).lstrip("/")
        self.base_frame = str(rospy.get_param("~base_frame", "base_link")).lstrip(
            "/"
        )
        self.cloud_topic = str(
            rospy.get_param(
                "~cloud_topic", "/yellow_track_avoidance/corridor_cloud"
            )
        )
        self.marker_topic = str(
            rospy.get_param(
                "~marker_topic", "/yellow_track_avoidance/corridor_markers"
            )
        )
        self.centerline_topic = str(
            rospy.get_param(
                "~centerline_topic", "/yellow_track_avoidance/corridor_centerline"
            )
        )
        self.locked_boundary_topic = str(
            rospy.get_param(
                "~locked_boundary_topic",
                "/yellow_track_avoidance/corridor_boundaries",
            )
        )
        self.scene_reset_topic = str(
            rospy.get_param(
                "~scene_reset_topic", "/yellow_track_avoidance/reset_scene"
            )
        )
        self.require_lidar_heading_reference = bool(
            rospy.get_param("~require_lidar_heading_reference", False)
        )
        self.lidar_heading_topic = str(
            rospy.get_param(
                "~lidar_heading_topic",
                "/yellow_track_avoidance/stage1_lidar_heading",
            )
        )
        self.lidar_heading_weight = float(
            rospy.get_param("~lidar_heading_reference_weight", 0.65)
        )
        self.maximum_lidar_heading_disagreement = math.radians(
            float(
                rospy.get_param(
                    "~maximum_lidar_heading_disagreement_deg", 3.0
                )
            )
        )
        self.maximum_lidar_heading_correction = math.radians(
            float(
                rospy.get_param(
                    "~maximum_lidar_heading_correction_deg", 1.0
                )
            )
        )
        self.minimum_quality = float(rospy.get_param("~minimum_quality", 0.40))
        self.center_separation = float(
            rospy.get_param("~center_separation_m", 0.91)
        )
        self.line_width = float(rospy.get_param("~yellow_line_width_m", 0.03))
        self.vehicle_front_offset = float(
            rospy.get_param("~vehicle_front_offset_m", 0.17)
        )
        self.corridor_length = float(
            rospy.get_param("~corridor_length_m", 1.85)
        )
        self.publish_goal_on_lock = bool(
            rospy.get_param("~publish_goal_on_lock", False)
        )
        self.goal_topic = str(
            rospy.get_param("~goal_topic", "/move_base_simple/goal")
        )
        self.goal_pose_topic = str(
            rospy.get_param(
                "~goal_pose_topic", "/yellow_track_avoidance/corridor_goal"
            )
        )
        self.goal_left_offset = float(
            rospy.get_param("~goal_left_from_right_endpoint_m", 0.24)
        )
        self.goal_forward_offset = float(
            rospy.get_param("~goal_forward_from_right_endpoint_m", 0.54)
        )
        self.vehicle_width = float(rospy.get_param("~vehicle_width_m", 0.33))
        self.preview_plan_topic = str(
            rospy.get_param(
                "~preview_plan_topic",
                "/yellow_track_avoidance/corridor_goal_plan",
            )
        )
        self.publish_goal_preview = bool(
            rospy.get_param("~publish_goal_preview", True)
        )
        self.make_plan_service = str(
            rospy.get_param("~make_plan_service", "/move_base/make_plan")
        )
        self.preview_plan_tolerance = float(
            rospy.get_param("~preview_plan_tolerance_m", 0.08)
        )
        self.preview_retry = max(
            0.2, float(rospy.get_param("~preview_plan_retry_sec", 1.0))
        )
        self.sample_resolution = float(
            rospy.get_param("~sample_resolution_m", 0.01)
        )
        self.costmap_extension_before = float(
            rospy.get_param("~costmap_extension_before_m", 0.0)
        )
        self.costmap_extension_after = float(
            rospy.get_param("~costmap_extension_after_m", 0.0)
        )
        if (
            self.costmap_extension_before < 0.0
            or self.costmap_extension_after < 0.0
        ):
            raise ValueError("costmap corridor extensions must be non-negative")
        self.required_frames = max(1, int(rospy.get_param("~required_frames", 5)))
        self.maximum_yaw_spread = math.radians(
            float(rospy.get_param("~maximum_yaw_spread_deg", 2.0))
        )
        self.maximum_lateral_spread = float(
            rospy.get_param("~maximum_lateral_spread_m", 0.025)
        )
        self.maximum_fit_residual = float(
            rospy.get_param("~maximum_fit_residual_m", 0.02)
        )
        self.publish_rate = float(rospy.get_param("~publish_rate_hz", 5.0))
        self.candidates = deque(maxlen=self.required_frames)
        self.locked_segments = None
        self.locked_fit = None
        self.locked_goal = None
        self.locked_quality = 0.0
        self.locked_source_stamp = None
        self.goal_sent = False
        self.preview_plan_pose_count = 0
        self.next_preview_attempt = rospy.Time(0)
        self.scene_generation = 0
        self.cloud_publish_count = 0
        self.last_cloud_publish_sec = None
        self.lidar_heading_yaw = None
        self.lidar_heading_source_stamp = None
        self.camera_heading_before_reference = None
        self.heading_correction_rad = None
        self.lidar_heading_reference_accepted = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cloud_pub = rospy.Publisher(
            self.cloud_topic, PointCloud2, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1, latch=True
        )
        self.centerline_pub = rospy.Publisher(
            self.centerline_topic, Path, queue_size=1, latch=True
        )
        self.locked_boundary_pub = rospy.Publisher(
            self.locked_boundary_topic, String, queue_size=1, latch=True
        )
        self.goal_pose_pub = rospy.Publisher(
            self.goal_pose_topic, PoseStamped, queue_size=1, latch=True
        )
        self.navigation_goal_pub = rospy.Publisher(
            self.goal_topic, PoseStamped, queue_size=1
        )
        self.preview_plan_pub = rospy.Publisher(
            self.preview_plan_topic, Path, queue_size=1, latch=True
        )
        self.make_plan_client = rospy.ServiceProxy(
            self.make_plan_service, GetPlan, persistent=False
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.send_goal_service = rospy.Service(
            "~send_goal", Trigger, self._send_goal
        )
        self.boundary_sub = rospy.Subscriber(
            self.input_topic, String, self._boundary_callback, queue_size=1
        )
        if self.require_lidar_heading_reference:
            self.lidar_heading_sub = rospy.Subscriber(
                self.lidar_heading_topic,
                PoseStamped,
                self._lidar_heading_callback,
                queue_size=1,
            )
        else:
            self.lidar_heading_sub = None
        self.scene_reset_sub = rospy.Subscriber(
            self.scene_reset_topic,
            String,
            self._scene_reset_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(1.0, self.publish_rate)), self._timer_callback
        )
        rospy.loginfo(
            "competition corridor map ready: left-only -> %.3f m pair in %s",
            self.center_separation,
            self.fixed_frame,
        )
        if self.require_lidar_heading_reference:
            rospy.loginfo(
                "corridor lock waits for one frozen lidar right-wall heading on %s",
                self.lidar_heading_topic,
            )

    def _reset_scene(self, scene_generation):
        """Clear the latched corridor only for an explicit new mission."""
        self.scene_generation = int(scene_generation)
        self.candidates.clear()
        self.locked_segments = None
        self.locked_fit = None
        self.locked_goal = None
        self.locked_quality = 0.0
        self.locked_source_stamp = None
        self.goal_sent = False
        self.preview_plan_pose_count = 0
        self.next_preview_attempt = rospy.Time(0)
        self.cloud_publish_count = 0
        self.last_cloud_publish_sec = None
        self.lidar_heading_yaw = None
        self.lidar_heading_source_stamp = None
        self.camera_heading_before_reference = None
        self.heading_correction_rad = None
        self.lidar_heading_reference_accepted = None

    @staticmethod
    def _normalize(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _lidar_heading_callback(self, message):
        source = str(message.header.frame_id).lstrip("/")
        if not source:
            return
        pose_yaw = self._yaw(message.pose.orientation)
        try:
            if source == self.fixed_frame:
                heading = pose_yaw
            else:
                transform = self.tf_buffer.lookup_transform(
                    self.fixed_frame,
                    source,
                    message.header.stamp,
                    rospy.Duration(0.15),
                )
                heading = self._normalize(
                    self._yaw(transform.transform.rotation) + pose_yaw
                )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return
        with self.lock:
            if self.locked_segments is not None:
                return
            self.lidar_heading_yaw = float(heading)
            self.lidar_heading_source_stamp = message.header.stamp.to_sec()

    def _scene_reset_callback(self, message):
        try:
            payload = json.loads(message.data)
            generation = int(payload["scene_generation"])
        except (KeyError, TypeError, ValueError, OverflowError):
            rospy.logwarn_throttle(
                1.0, "invalid competition corridor scene-reset event"
            )
            return
        with self.lock:
            if generation <= self.scene_generation:
                return
            self._reset_scene(generation)
        rospy.logwarn(
            "competition corridor reset to collecting at scene generation=%d",
            generation,
        )

    @staticmethod
    def _yaw(quaternion):
        return math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )

    @staticmethod
    def _transform_xy(transform, point):
        yaw = CompetitionCorridorMap._yaw(transform.transform.rotation)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return (
            transform.transform.translation.x + cosine * point[0] - sine * point[1],
            transform.transform.translation.y + sine * point[0] + cosine * point[1],
        )

    def _stable_fit(self):
        if len(self.candidates) < self.required_frames:
            return None
        reference = self.candidates[0][0]
        for fit, _, _, _ in self.candidates:
            if fit.residual_m > self.maximum_fit_residual:
                return None
            if angular_difference(reference.tangent, fit.tangent) > self.maximum_yaw_spread:
                return None
            offset = abs(
                (fit.anchor[0] - reference.anchor[0]) * reference.left_normal[0]
                + (fit.anchor[1] - reference.anchor[1]) * reference.left_normal[1]
            )
            if offset > self.maximum_lateral_spread:
                return None
        tangent = np.mean(
            np.asarray([item[0].tangent for item in self.candidates]), axis=0
        )
        tangent /= np.linalg.norm(tangent)
        normal = np.asarray((-tangent[1], tangent[0]))
        if float(np.dot(normal, reference.left_normal)) < 0.0:
            normal = -normal
        anchor = np.mean(
            np.asarray([item[0].anchor for item in self.candidates]), axis=0
        )
        residual = max(item[0].residual_m for item in self.candidates)
        return CorridorFit(
            tangent=(float(tangent[0]), float(tangent[1])),
            left_normal=(float(normal[0]), float(normal[1])),
            anchor=(float(anchor[0]), float(anchor[1])),
            residual_m=float(residual),
        )

    def _boundary_callback(self, message):
        with self.lock:
            if self.locked_segments is not None:
                return
        try:
            payload = json.loads(message.data)
            if not bool(payload.get("left_detected", False)):
                return
            quality = float(payload.get("quality", 0.0))
            if quality < self.minimum_quality:
                return
            source = str(payload.get("frame_id", "")).lstrip("/")
            stamp = rospy.Time.from_sec(float(payload["stamp"]))
            raw_points = payload.get("left_base_m", [])
            points = [
                (float(item[0]), float(item[1]))
                for item in raw_points
                if isinstance(item, list) and len(item) >= 2
            ]
            if len(points) < 2 or not source:
                return
            transform = self.tf_buffer.lookup_transform(
                self.fixed_frame, source, stamp, rospy.Duration(0.08)
            )
        except (KeyError, TypeError, ValueError):
            return
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return

        transformed = [self._transform_xy(transform, point) for point in points]
        yaw = self._yaw(transform.transform.rotation)
        forward_hint = math.cos(yaw), math.sin(yaw)
        left_hint = -math.sin(yaw), math.cos(yaw)
        try:
            fit = fit_left_boundary(transformed, forward_hint, left_hint)
        except ValueError:
            return
        robot_xy = (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
        )
        with self.lock:
            self.candidates.append((fit, robot_xy, quality, stamp.to_sec()))
            stable = self._stable_fit()
            if stable is None:
                return
            if (
                self.require_lidar_heading_reference
                and self.lidar_heading_yaw is None
            ):
                return
            camera_heading = math.atan2(
                stable.tangent[1], stable.tangent[0]
            )
            if self.lidar_heading_yaw is not None:
                self.camera_heading_before_reference = camera_heading
                stable, correction, accepted = blend_heading_reference(
                    stable,
                    self.lidar_heading_yaw,
                    reference_weight=self.lidar_heading_weight,
                    maximum_disagreement_rad=(
                        self.maximum_lidar_heading_disagreement
                    ),
                    maximum_correction_rad=(
                        self.maximum_lidar_heading_correction
                    ),
                )
                self.heading_correction_rad = correction
                self.lidar_heading_reference_accepted = accepted
            robot_mean = np.mean(
                np.asarray([item[1] for item in self.candidates]), axis=0
            )
            self.locked_fit = stable
            self.locked_segments = build_corridor_segments(
                stable,
                (float(robot_mean[0]), float(robot_mean[1])),
                self.center_separation,
                self.vehicle_front_offset,
                self.corridor_length,
            )
            self.locked_goal = build_corridor_stop_goal(
                self.locked_segments,
                self.goal_left_offset,
                self.goal_forward_offset,
                self.vehicle_front_offset,
            )
            self.locked_quality = float(
                np.mean([item[2] for item in self.candidates])
            )
            self.locked_source_stamp = max(item[3] for item in self.candidates)
            rospy.loginfo(
                "competition corridor locked: quality=%.3f residual=%.4f m, "
                "heading source=%s correction=%.3f deg",
                self.locked_quality,
                stable.residual_m,
                "camera_lidar_fused"
                if self.lidar_heading_reference_accepted
                else "camera",
                math.degrees(self.heading_correction_rad or 0.0),
            )

    @staticmethod
    def _point(value, z=0.04):
        point = Point()
        point.x = float(value[0])
        point.y = float(value[1])
        point.z = float(z)
        return point

    def _markers(self, stamp, segments, goal):
        output = MarkerArray()
        specifications = (
            ("left_yellow_live", 1, segments.left, (1.0, 0.82, 0.0, 1.0), self.line_width),
            ("right_yellow_inferred", 2, segments.right, (1.0, 0.62, 0.0, 0.85), self.line_width),
            ("corridor_centerline", 3, segments.center, (0.0, 0.95, 1.0, 0.95), 0.018),
        )
        for namespace, marker_id, segment, color, width in specifications:
            marker = Marker()
            marker.header.frame_id = self.fixed_frame
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(width)
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            marker.points = [self._point(value) for value in segment]
            output.markers.append(marker)

        goal_reference = Marker()
        goal_reference.header.frame_id = self.fixed_frame
        goal_reference.header.stamp = stamp
        goal_reference.ns = "corridor_goal_reference"
        goal_reference.id = 10
        goal_reference.type = Marker.SPHERE
        goal_reference.action = Marker.ADD
        goal_reference.pose.position = self._point(goal.reference, z=0.07)
        goal_reference.pose.orientation.w = 1.0
        goal_reference.scale.x = 0.08
        goal_reference.scale.y = 0.08
        goal_reference.scale.z = 0.08
        goal_reference.color.r = 1.0
        goal_reference.color.g = 0.1
        goal_reference.color.b = 0.8
        goal_reference.color.a = 1.0
        output.markers.append(goal_reference)

        offset_guide = Marker()
        offset_guide.header = goal_reference.header
        offset_guide.ns = "corridor_goal_offset"
        offset_guide.id = 11
        offset_guide.type = Marker.LINE_STRIP
        offset_guide.action = Marker.ADD
        offset_guide.pose.orientation.w = 1.0
        offset_guide.scale.x = 0.018
        offset_guide.color.r = 0.9
        offset_guide.color.g = 0.2
        offset_guide.color.b = 1.0
        offset_guide.color.a = 1.0
        offset_guide.points = [
            self._point(goal.reference, z=0.065),
            self._point(goal.front_center, z=0.065),
        ]
        output.markers.append(offset_guide)

        half_length = self.vehicle_front_offset
        half_width = 0.5 * self.vehicle_width
        cosine = math.cos(goal.yaw)
        sine = math.sin(goal.yaw)
        forward = np.asarray((cosine, sine), dtype=np.float64)
        left = np.asarray((-sine, cosine), dtype=np.float64)
        center = np.asarray(goal.base_center, dtype=np.float64)
        corners = [
            center + forward * half_length + left * half_width,
            center + forward * half_length - left * half_width,
            center - forward * half_length - left * half_width,
            center - forward * half_length + left * half_width,
        ]
        footprint = Marker()
        footprint.header = goal_reference.header
        footprint.ns = "corridor_goal_vehicle"
        footprint.id = 12
        footprint.type = Marker.LINE_STRIP
        footprint.action = Marker.ADD
        footprint.pose.orientation.w = 1.0
        footprint.scale.x = 0.025
        footprint.color.r = 0.1
        footprint.color.g = 1.0
        footprint.color.b = 0.35
        footprint.color.a = 1.0
        footprint.points = [
            self._point(value, z=0.075) for value in corners + [corners[0]]
        ]
        output.markers.append(footprint)
        return output

    def _goal_pose(self, stamp, goal):
        message = PoseStamped()
        message.header.frame_id = self.fixed_frame
        message.header.stamp = stamp
        message.pose.position.x = float(goal.base_center[0])
        message.pose.position.y = float(goal.base_center[1])
        message.pose.orientation.z = math.sin(0.5 * goal.yaw)
        message.pose.orientation.w = math.cos(0.5 * goal.yaw)
        return message

    def _send_goal(self, _request):
        with self.lock:
            goal = self.locked_goal
        if goal is None:
            return TriggerResponse(False, "corridor goal is not ready")
        if self.navigation_goal_pub.get_num_connections() <= 0:
            return TriggerResponse(False, "move_base goal subscriber is missing")
        message = self._goal_pose(rospy.Time.now(), goal)
        self.navigation_goal_pub.publish(message)
        with self.lock:
            self.goal_sent = True
        return TriggerResponse(
            True,
            "corridor goal sent: base=(%.3f, %.3f), yaw=%.3f rad"
            % (goal.base_center[0], goal.base_center[1], goal.yaw),
        )

    @staticmethod
    def _pose(stamp, frame_id, xy, yaw):
        message = PoseStamped()
        message.header.frame_id = frame_id
        message.header.stamp = stamp
        message.pose.position.x = float(xy[0])
        message.pose.position.y = float(xy[1])
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        return message

    def _publish_preview_plan(self, now, goal):
        with self.lock:
            if self.preview_plan_pose_count > 0 or now < self.next_preview_attempt:
                return
            self.next_preview_attempt = now + rospy.Duration(self.preview_retry)
        try:
            fixed_to_plan = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.fixed_frame,
                rospy.Time(0),
                rospy.Duration(0.08),
            )
            robot_to_plan = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.08),
            )
            goal_xy = self._transform_xy(fixed_to_plan, goal.base_center)
            goal_yaw = goal.yaw + self._yaw(fixed_to_plan.transform.rotation)
            robot_xy = (
                robot_to_plan.transform.translation.x,
                robot_to_plan.transform.translation.y,
            )
            robot_yaw = self._yaw(robot_to_plan.transform.rotation)
            start = self._pose(now, self.planning_frame, robot_xy, robot_yaw)
            target = self._pose(now, self.planning_frame, goal_xy, goal_yaw)
            response = self.make_plan_client(
                start, target, self.preview_plan_tolerance
            )
        except (
            rospy.ServiceException,
            rospy.ROSException,
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return
        if not response.plan.poses:
            return
        response.plan.header.frame_id = self.planning_frame
        response.plan.header.stamp = now
        for pose in response.plan.poses:
            pose.header = response.plan.header
        self.preview_plan_pub.publish(response.plan)
        with self.lock:
            self.preview_plan_pose_count = len(response.plan.poses)
        rospy.loginfo(
            "published latched corridor goal preview: %d poses on %s",
            len(response.plan.poses),
            self.preview_plan_topic,
        )

    def _centerline(self, stamp, segment):
        message = Path()
        message.header.frame_id = self.fixed_frame
        message.header.stamp = stamp
        yaw = math.atan2(
            segment[1][1] - segment[0][1], segment[1][0] - segment[0][0]
        )
        for value in segment:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(value[0])
            pose.pose.position.y = float(value[1])
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            message.poses.append(pose)
        return message

    def _locked_boundary_message(
        self, stamp, segments, quality, scene_generation
    ):
        payload = {
            "stamp": stamp.to_sec(),
            "scene_generation": int(scene_generation),
            "frame_id": self.fixed_frame,
            "left_detected": True,
            "right_detected": False,
            "boundary_detection_mode": "locked_left_only",
            "boundary_pair_valid": True,
            "boundary_inference": "right_from_locked_left_metric_width",
            "left_base_m": [list(point) for point in segments.left],
            "right_base_m": [list(point) for point in segments.right],
            "quality": float(quality),
            "expected_center_separation_m": self.center_separation,
            "expected_inner_clear_width_m": (
                self.center_separation - self.line_width
            ),
            "yellow_line_width_m": self.line_width,
        }
        return String(data=json.dumps(payload, allow_nan=False))

    def _timer_callback(self, _event):
        with self.lock:
            segments = self.locked_segments
            fit = self.locked_fit
            goal = self.locked_goal
            candidate_count = len(self.candidates)
            quality = self.locked_quality
            source_stamp = self.locked_source_stamp
            goal_sent = self.goal_sent
            preview_plan_pose_count = self.preview_plan_pose_count
            scene_generation = self.scene_generation
            cloud_publish_count = self.cloud_publish_count
            last_cloud_publish_sec = self.last_cloud_publish_sec
            lidar_heading_yaw = self.lidar_heading_yaw
            lidar_heading_source_stamp = self.lidar_heading_source_stamp
            camera_heading_before_reference = self.camera_heading_before_reference
            heading_correction_rad = self.heading_correction_rad
            lidar_heading_reference_accepted = self.lidar_heading_reference_accepted
        now = rospy.Time.now()
        if segments is not None:
            # Keep the measured segments and semantic gap calculations exact.
            # Only the costmap rails are extended so GlobalPlanner cannot
            # escape the corridor by wrapping around a finite yellow-line end.
            costmap_left = extend_line_segment(
                segments.left,
                self.costmap_extension_before,
                self.costmap_extension_after,
            )
            costmap_right = extend_line_segment(
                segments.right,
                self.costmap_extension_before,
                self.costmap_extension_after,
            )
            left = sample_line_strip(
                costmap_left, self.line_width, self.sample_resolution
            )
            right = sample_line_strip(
                costmap_right, self.line_width, self.sample_resolution
            )
            cloud = point_cloud2.create_cloud_xyz32(
                Header(stamp=now, frame_id=self.fixed_frame),
                np.vstack((left, right)).tolist(),
            )
            markers = self._markers(now, segments, goal)
            centerline = self._centerline(now, segments.center)
            locked_boundary = self._locked_boundary_message(
                now, segments, quality, scene_generation
            )
            goal_pose = self._goal_pose(now, goal)
            published_current_generation = False
            goal_published_now = False
            with self.lock:
                if (
                    self.scene_generation == scene_generation
                    and self.locked_segments is segments
                ):
                    # Keep the generation check and downstream publications in
                    # one critical section.  An explicit reset can therefore
                    # never be followed by a late publication from the old
                    # corridor timer cycle.
                    self.cloud_pub.publish(cloud)
                    self.marker_pub.publish(markers)
                    self.centerline_pub.publish(centerline)
                    self.locked_boundary_pub.publish(locked_boundary)
                    self.goal_pose_pub.publish(goal_pose)
                    self.cloud_publish_count += 1
                    self.last_cloud_publish_sec = now.to_sec()
                    cloud_publish_count = self.cloud_publish_count
                    last_cloud_publish_sec = self.last_cloud_publish_sec
                    published_current_generation = True
                    if (
                        self.publish_goal_on_lock
                        and not self.goal_sent
                        and self.navigation_goal_pub.get_num_connections() > 0
                    ):
                        self.navigation_goal_pub.publish(goal_pose)
                        self.goal_sent = True
                        goal_sent = True
                        goal_published_now = True
            if published_current_generation:
                # A make_plan service call may block, so keep it outside the
                # state lock after the generation-safe publications finish.
                if self.publish_goal_preview:
                    self._publish_preview_plan(now, goal)
                if goal_published_now:
                    rospy.loginfo(
                        "published corridor goal: right endpoint + left %.3f m + "
                        "forward %.3f m; base=(%.3f, %.3f), yaw=%.3f rad",
                        self.goal_left_offset,
                        self.goal_forward_offset,
                        goal.base_center[0],
                        goal.base_center[1],
                        goal.yaw,
                    )
        # Re-read state after publication so a reset concurrent with message
        # construction produces a collecting/new-generation status, never a
        # stale locked status from the preceding mission.
        with self.lock:
            segments = self.locked_segments
            fit = self.locked_fit
            goal = self.locked_goal
            candidate_count = len(self.candidates)
            quality = self.locked_quality
            source_stamp = self.locked_source_stamp
            goal_sent = self.goal_sent
            preview_plan_pose_count = self.preview_plan_pose_count
            scene_generation = self.scene_generation
            cloud_publish_count = self.cloud_publish_count
            last_cloud_publish_sec = self.last_cloud_publish_sec
        status = {
            "stamp": now.to_sec(),
            "state": "locked" if segments is not None else "collecting",
            "scene_generation": scene_generation,
            "corridor_cloud_ready": bool(
                segments is not None and cloud_publish_count > 0
            ),
            "costmap_extension_before_m": self.costmap_extension_before,
            "costmap_extension_after_m": self.costmap_extension_after,
            "cloud_publish_count": cloud_publish_count,
            "last_cloud_publish_sec": last_cloud_publish_sec,
            "scene_reset_topic": self.scene_reset_topic,
            "fixed_frame": self.fixed_frame,
            "candidate_count": candidate_count,
            "required_frames": self.required_frames,
            "left_source": "camera_live",
            "right_source": "left_plus_metric_width",
            "center_separation_m": self.center_separation,
            "inner_clear_width_m": self.center_separation - self.line_width,
            "yellow_line_width_m": self.line_width,
            "vehicle_front_offset_m": self.vehicle_front_offset,
            "corridor_length_m": self.corridor_length,
            "goal_left_from_right_endpoint_m": self.goal_left_offset,
            "goal_forward_from_right_endpoint_m": self.goal_forward_offset,
            "goal_reference": None if goal is None else list(goal.reference),
            "goal_front_center": None if goal is None else list(goal.front_center),
            "goal_base_center": None if goal is None else list(goal.base_center),
            "goal_yaw_rad": None if goal is None else goal.yaw,
            "publish_goal_on_lock": self.publish_goal_on_lock,
            "goal_ready": goal is not None,
            "send_goal_service": rospy.resolve_name("~send_goal"),
            "goal_sent": goal_sent,
            "preview_plan_topic": self.preview_plan_topic,
            "publish_goal_preview": self.publish_goal_preview,
            "preview_plan_pose_count": preview_plan_pose_count,
            "locked_boundary_topic": self.locked_boundary_topic,
            "quality": quality if segments is not None else None,
            "fit_residual_m": None if fit is None else fit.residual_m,
            "source_stamp": source_stamp,
            "heading_source": (
                "camera_lidar_fused"
                if segments is not None and lidar_heading_reference_accepted
                else "camera"
            ),
            "lidar_heading_reference_required": self.require_lidar_heading_reference,
            "lidar_heading_reference_ready": lidar_heading_yaw is not None,
            "lidar_heading_reference_applied": bool(
                segments is not None and lidar_heading_reference_accepted
            ),
            "lidar_heading_reference_accepted": lidar_heading_reference_accepted,
            "lidar_heading_reference_weight": self.lidar_heading_weight,
            "maximum_lidar_heading_disagreement_deg": math.degrees(
                self.maximum_lidar_heading_disagreement
            ),
            "maximum_lidar_heading_correction_deg": math.degrees(
                self.maximum_lidar_heading_correction
            ),
            "lidar_heading_yaw_rad": lidar_heading_yaw,
            "lidar_heading_source_stamp": lidar_heading_source_stamp,
            "camera_heading_before_reference_rad": camera_heading_before_reference,
            "heading_correction_rad": heading_correction_rad,
        }
        self.status_pub.publish(String(data=json.dumps(status, allow_nan=False)))


def main():
    rospy.init_node("yellow_track_competition_corridor_map")
    CompetitionCorridorMap()
    rospy.spin()


if __name__ == "__main__":
    main()
