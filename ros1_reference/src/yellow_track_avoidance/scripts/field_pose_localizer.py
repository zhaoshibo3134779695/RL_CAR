#!/usr/bin/env python3
"""One-shot field localization using lidar and a left-line-inferred wall."""

import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rospy
import tf2_ros
import yaml
from actionlib_msgs.msg import GoalID
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import (
    Empty,
    SetBool,
    Trigger,
    TriggerResponse,
)

from yellow_track_avoidance.field_localization import (
    OccupancyDistanceField,
    search_field_pose,
)


class FieldPoseLocalizer:
    def __init__(self):
        self.lock = threading.RLock()
        self.map_yaml_path = str(rospy.get_param("~map_yaml"))
        self.base_frame = str(
            rospy.get_param("~base_frame", "base_link")
        ).lstrip("/")
        self.map_frame = str(
            rospy.get_param("~map_frame", "map")
        ).lstrip("/")
        self.scan_topic = str(rospy.get_param("~scan_topic", "/scan"))
        self.boundary_topic = str(
            rospy.get_param(
                "~boundary_topic",
                "/yellow_track_avoidance/debug/boundaries",
            )
        )
        self.initial_pose_topic = str(
            rospy.get_param("~initial_pose_topic", "/initialpose")
        )
        self.move_base_cancel_topic = str(
            rospy.get_param("~move_base_cancel_topic", "/move_base/cancel")
        )
        self.gate_service_name = str(
            rospy.get_param(
                "~gate_service",
                "/yellow_track_navigation_gate/set_enabled",
            )
        )
        self.clear_costmaps_service_name = str(
            rospy.get_param(
                "~clear_costmaps_service",
                "/move_base/clear_costmaps",
            )
        )
        self.input_timeout = float(
            rospy.get_param("~input_timeout_sec", 0.80)
        )
        self.minimum_boundary_quality = float(
            rospy.get_param("~minimum_boundary_quality", 0.75)
        )
        self.maximum_match_score = float(
            rospy.get_param("~maximum_match_score", 0.08)
        )
        self.maximum_wall_score = float(
            rospy.get_param("~maximum_wall_score", 0.02)
        )
        self.minimum_score_margin = float(
            rospy.get_param("~minimum_score_margin", 0.025)
        )
        self.search_bounds_m = rospy.get_param(
            "~search_bounds_m", None
        )
        if self.search_bounds_m is not None:
            self.search_bounds_m = tuple(
                float(value) for value in self.search_bounds_m
            )
            if (
                len(self.search_bounds_m) != 4
                or self.search_bounds_m[0] >= self.search_bounds_m[2]
                or self.search_bounds_m[1] >= self.search_bounds_m[3]
            ):
                raise ValueError(
                    "search_bounds_m must be [xmin, ymin, xmax, ymax]"
                )
        self.yellow_to_wall_distance_m = float(
            rospy.get_param("~yellow_to_wall_distance_m", 0.0)
        )
        self.yellow_weight = float(
            rospy.get_param("~yellow_weight", 3.5)
        )
        self.lidar_weight = float(
            rospy.get_param("~lidar_weight", 1.0)
        )
        if (
            self.yellow_to_wall_distance_m < 0.0
            or self.yellow_weight <= 0.0
            or self.lidar_weight <= 0.0
        ):
            raise ValueError("invalid yellow-line localization weighting")
        self.auto_localize_on_start = bool(
            rospy.get_param("~auto_localize_on_start", False)
        )
        self.startup_delay_sec = float(
            rospy.get_param("~startup_delay_sec", 2.0)
        )
        self.startup_timeout_sec = float(
            rospy.get_param("~startup_timeout_sec", 90.0)
        )
        self.startup_retry_interval_sec = float(
            rospy.get_param("~startup_retry_interval_sec", 1.0)
        )
        if (
            self.startup_delay_sec < 0.0
            or self.startup_timeout_sec <= 0.0
            or self.startup_retry_interval_sec <= 0.0
        ):
            raise ValueError("invalid startup localization timing")
        self.latest_scan = None
        self.latest_scan_receipt_sec = None
        self.latest_boundary = None
        self.latest_boundary_receipt_sec = None
        self.running = False
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.distance_field = self._load_map(self.map_yaml_path)

        self.initial_pose_pub = rospy.Publisher(
            self.initial_pose_topic,
            PoseWithCovarianceStamped,
            queue_size=1,
        )
        self.cancel_pub = rospy.Publisher(
            self.move_base_cancel_topic, GoalID, queue_size=1
        )
        self.status_pub = rospy.Publisher(
            "~status", String, queue_size=1, latch=True
        )
        self.overlay_pub = rospy.Publisher(
            "~debug_overlay", Image, queue_size=1, latch=True
        )
        self.scan_sub = rospy.Subscriber(
            self.scan_topic, LaserScan, self._scan_callback, queue_size=1
        )
        self.boundary_sub = rospy.Subscriber(
            self.boundary_topic,
            String,
            self._boundary_callback,
            queue_size=1,
        )
        self.localize_service = rospy.Service(
            "~localize", Trigger, self._localize
        )
        if self.auto_localize_on_start:
            self._publish_status(
                "startup_waiting",
                "waiting to run automatic field localization",
            )
            self.startup_thread = threading.Thread(
                target=self._auto_localize_startup,
                name="field-localization-startup",
                daemon=True,
            )
            self.startup_thread.start()
        else:
            self.startup_thread = None
            self._publish_status("idle", "waiting for one-shot request")
        rospy.loginfo(
            "field pose localizer ready: scan=%s boundary=%s map=%s "
            "auto_start=%s",
            self.scan_topic,
            self.boundary_topic,
            self.map_yaml_path,
            self.auto_localize_on_start,
        )

    @staticmethod
    def _load_map(map_yaml_path):
        with open(map_yaml_path, "r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
        resolution = float(metadata["resolution"])
        origin = list(metadata.get("origin", [0.0, 0.0, 0.0]))
        if len(origin) < 3 or abs(float(origin[2])) > 1e-6:
            raise ValueError("rotated map origins are not supported")
        image_path = str(metadata["image"])
        if not os.path.isabs(image_path):
            image_path = os.path.join(
                os.path.dirname(os.path.abspath(map_yaml_path)),
                image_path,
            )
        occupied_threshold = float(metadata.get("occupied_thresh", 0.65))
        pixel_threshold = int(
            round(255.0 * (1.0 - occupied_threshold))
        )
        return OccupancyDistanceField.from_pgm(
            image_path,
            resolution,
            (float(origin[0]), float(origin[1])),
            pixel_threshold,
        )

    def _scan_callback(self, message):
        with self.lock:
            self.latest_scan = message
            self.latest_scan_receipt_sec = rospy.Time.now().to_sec()

    def _boundary_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            rospy.logwarn_throttle(
                1.0, "invalid boundary JSON for field localization"
            )
            return
        with self.lock:
            self.latest_boundary = payload
            self.latest_boundary_receipt_sec = rospy.Time.now().to_sec()

    def _publish_status(
        self, state, message, match=None, running=None
    ):
        payload = {
            "stamp": rospy.Time.now().to_sec(),
            "state": str(state),
            "message": str(message),
            "running": (
                self.running if running is None else bool(running)
            ),
        }
        if match is not None:
            payload["match"] = {
                "x_m": match.x_m,
                "y_m": match.y_m,
                "yaw_rad": match.yaw_rad,
                "yaw_deg": math.degrees(match.yaw_rad),
                "score": match.score,
                "lidar_score": match.lidar_score,
                "wall_score": match.wall_score,
                "ray_penalty": match.ray_penalty,
                "alternative_score": (
                    None
                    if not np.isfinite(match.alternative_score)
                    else match.alternative_score
                ),
                "score_margin": (
                    None
                    if not np.isfinite(match.score_margin)
                    else match.score_margin
                ),
            }
        self.status_pub.publish(
            String(
                data=json.dumps(
                    payload, ensure_ascii=False, allow_nan=False
                )
            )
        )

    def _disable_navigation(self):
        self.cancel_pub.publish(GoalID())
        try:
            rospy.wait_for_service(self.gate_service_name, timeout=2.0)
            rospy.ServiceProxy(self.gate_service_name, SetBool)(False)
        except (rospy.ROSException, rospy.ServiceException) as error:
            raise RuntimeError(
                "could not lock navigation gate: {}".format(error)
            )

    def _scan_points_base(self, scan):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            scan.header.frame_id.lstrip("/"),
            rospy.Time(0),
            rospy.Duration(3.0),
        ).transform
        quaternion = transform.rotation
        laser_yaw = math.atan2(
            2.0
            * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0
            - 2.0
            * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )
        origin = np.asarray(
            [transform.translation.x, transform.translation.y],
            dtype=np.float64,
        )
        points = []
        maximum_range = min(float(scan.range_max), 5.0)
        for index, distance in enumerate(scan.ranges):
            value = float(distance)
            if (
                not np.isfinite(value)
                or value < float(scan.range_min)
                or value > maximum_range
            ):
                continue
            angle = (
                float(scan.angle_min)
                + index * float(scan.angle_increment)
                + laser_yaw
            )
            points.append(
                [
                    origin[0] + value * math.cos(angle),
                    origin[1] + value * math.sin(angle),
                ]
            )
        return np.asarray(points, dtype=np.float64), origin

    def _validate_inputs(self, now_sec):
        if self.latest_scan is None or self.latest_scan_receipt_sec is None:
            raise RuntimeError("lidar scan is missing")
        if (
            now_sec - self.latest_scan_receipt_sec
            > self.input_timeout
        ):
            raise RuntimeError("lidar scan is stale")
        if (
            self.latest_boundary is None
            or self.latest_boundary_receipt_sec is None
        ):
            raise RuntimeError("yellow boundary data is missing")
        if (
            now_sec - self.latest_boundary_receipt_sec
            > self.input_timeout
        ):
            raise RuntimeError("yellow boundary data is stale")
        boundary = self.latest_boundary
        boundary_mode = str(
            boundary.get("boundary_detection_mode", "both")
        ).strip().lower()
        if boundary_mode == "left_only":
            if not bool(boundary.get("left_detected", False)):
                raise RuntimeError("left yellow line is not detected")
            if (
                str(boundary.get("boundary_inference", "none"))
                != "right_from_left_metric_width"
                or not bool(boundary.get("boundary_pair_valid", False))
            ):
                raise RuntimeError(
                    "right wall was not inferred from the left yellow line"
                )
            quality_label = "left yellow-line"
        else:
            if not bool(boundary.get("right_detected", False)):
                raise RuntimeError("right yellow line is not detected")
            quality_label = "right yellow-line"
        quality = float(boundary.get("quality", 0.0))
        if quality < self.minimum_boundary_quality:
            raise RuntimeError(
                "{} quality {:.3f} is below {:.3f}".format(
                    quality_label, quality, self.minimum_boundary_quality
                )
            )
        wall = np.asarray(
            boundary.get("right_wall_base_m", []),
            dtype=np.float64,
        ).reshape(-1, 2)
        if len(wall) < 4 or not np.all(np.isfinite(wall)):
            raise RuntimeError("inferred right-wall points are invalid")
        return self.latest_scan, wall

    def _match_valid(self, match):
        if match.score > self.maximum_match_score:
            return False, "match score {:.4f} exceeds {:.4f}".format(
                match.score, self.maximum_match_score
            )
        if match.wall_score > self.maximum_wall_score:
            return False, "wall score {:.4f} exceeds {:.4f}".format(
                match.wall_score, self.maximum_wall_score
            )
        if (
            np.isfinite(match.score_margin)
            and match.score_margin < self.minimum_score_margin
        ):
            return False, "pose is ambiguous: score margin {:.4f}".format(
                match.score_margin
            )
        return True, "field geometry match accepted"

    def _publish_initial_pose(self, match):
        message = PoseWithCovarianceStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.map_frame
        message.pose.pose.position.x = match.x_m
        message.pose.pose.position.y = match.y_m
        message.pose.pose.orientation.z = math.sin(0.5 * match.yaw_rad)
        message.pose.pose.orientation.w = math.cos(0.5 * match.yaw_rad)
        message.pose.covariance[0] = 0.0009
        message.pose.covariance[7] = 0.0009
        message.pose.covariance[14] = 1e-6
        message.pose.covariance[21] = 1e-6
        message.pose.covariance[28] = 1e-6
        message.pose.covariance[35] = math.radians(3.0) ** 2
        self.initial_pose_pub.publish(message)

    def _publish_overlay(self, match, lidar_points, wall_points):
        canvas = cv2.cvtColor(
            self.distance_field.image, cv2.COLOR_GRAY2BGR
        )
        height = self.distance_field.height
        rotation = np.asarray(
            [
                [math.cos(match.yaw_rad), -math.sin(match.yaw_rad)],
                [math.sin(match.yaw_rad), math.cos(match.yaw_rad)],
            ]
        )
        translation = np.asarray([match.x_m, match.y_m])

        def pixel(point):
            column = int(
                round(
                    (point[0] - self.distance_field.origin_x)
                    / self.distance_field.resolution_m
                )
            )
            row = height - 1 - int(
                round(
                    (point[1] - self.distance_field.origin_y)
                    / self.distance_field.resolution_m
                )
            )
            return column, row

        for point in translation + lidar_points @ rotation.T:
            column, row = pixel(point)
            if (
                0 <= column < self.distance_field.width
                and 0 <= row < self.distance_field.height
            ):
                cv2.circle(canvas, (column, row), 1, (0, 0, 255), -1)
        for point in translation + wall_points @ rotation.T:
            column, row = pixel(point)
            if (
                0 <= column < self.distance_field.width
                and 0 <= row < self.distance_field.height
            ):
                cv2.circle(canvas, (column, row), 2, (255, 0, 255), -1)
        center = pixel(translation)
        cv2.circle(canvas, center, 6, (0, 255, 0), -1)
        heading = (
            int(round(center[0] + 20.0 * math.cos(match.yaw_rad))),
            int(round(center[1] - 20.0 * math.sin(match.yaw_rad))),
        )
        cv2.arrowedLine(
            canvas, center, heading, (0, 255, 0), 2, tipLength=0.3
        )
        message = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.map_frame
        self.overlay_pub.publish(message)

    def _clear_costmaps(self):
        try:
            rospy.wait_for_service(
                self.clear_costmaps_service_name, timeout=3.0
            )
            rospy.ServiceProxy(
                self.clear_costmaps_service_name, Empty
            )()
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn(
                "field localization could not clear costmaps: %s", error
            )

    def _auto_localize_startup(self):
        if self.startup_delay_sec > 0.0:
            rospy.sleep(self.startup_delay_sec)
        started = time.monotonic()
        attempt = 0
        last_error = "automatic field localization did not run"
        while not rospy.is_shutdown():
            attempt += 1
            result = self._localize(None)
            if result.success:
                rospy.loginfo(
                    "automatic field localization succeeded on attempt %d: %s",
                    attempt,
                    result.message,
                )
                return
            last_error = str(result.message)
            elapsed = time.monotonic() - started
            if elapsed >= self.startup_timeout_sec:
                message = (
                    "automatic field localization timed out after "
                    "{:.1f}s: {}"
                ).format(elapsed, last_error)
                rospy.logerr(message)
                self._publish_status(
                    "startup_failed", message, running=False
                )
                return
            rospy.logwarn(
                "automatic field localization attempt %d failed: %s; "
                "retrying",
                attempt,
                last_error,
            )
            rospy.sleep(self.startup_retry_interval_sec)

    def _localize(self, _request):
        with self.lock:
            if self.running:
                return TriggerResponse(
                    False, "field localization is already running"
                )
            self.running = True
            try:
                self._publish_status(
                    "matching",
                    "locking navigation and matching field geometry",
                )
                self._disable_navigation()
                scan, wall = self._validate_inputs(
                    rospy.Time.now().to_sec()
                )
                lidar, sensor_origin = self._scan_points_base(scan)
                match = search_field_pose(
                    self.distance_field,
                    lidar,
                    wall,
                    sensor_origin,
                    position_bounds_m=self.search_bounds_m,
                    wall_target_distance_m=(
                        self.yellow_to_wall_distance_m
                    ),
                    lidar_weight=self.lidar_weight,
                    wall_weight=self.yellow_weight,
                )
                accepted, reason = self._match_valid(match)
                self._publish_overlay(match, lidar, wall)
                if not accepted:
                    self._publish_status(
                        "rejected", reason, match, running=False
                    )
                    return TriggerResponse(False, reason)
                self._publish_initial_pose(match)
                rospy.sleep(1.0)
                self._clear_costmaps()
                message = (
                    "localized at x={:.3f}, y={:.3f}, yaw={:.1f} deg; "
                    "score={:.4f}, margin={:.4f}"
                ).format(
                    match.x_m,
                    match.y_m,
                    math.degrees(match.yaw_rad),
                    match.score,
                    match.score_margin,
                )
                self._publish_status(
                    "localized", message, match, running=False
                )
                return TriggerResponse(True, message)
            except Exception as error:
                message = str(error)
                rospy.logerr("field localization failed: %s", message)
                self._publish_status(
                    "failed", message, running=False
                )
                return TriggerResponse(False, message)
            finally:
                self.running = False


if __name__ == "__main__":
    rospy.init_node("yellow_track_field_localization")
    FieldPoseLocalizer()
    rospy.spin()
