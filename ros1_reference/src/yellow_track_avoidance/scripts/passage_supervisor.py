#!/usr/bin/env python3
"""Unified single-disc, multi-obstacle and narrow-gate ROS supervisor."""

import copy
from dataclasses import replace
import json
import math
import threading
import time

import numpy as np
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalID, GoalStatus, GoalStatusArray
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray

from yellow_track_avoidance.narrow_gate import DiscTrack, RobotPose, wrap_angle
from yellow_track_avoidance.passage_navigation import (
    BoundaryLines,
    PassageConfig,
    PassageStateMachine,
    SCENE_BLOCKED,
    SCENE_NORMAL,
    SweptSafety,
    STATE_DEGRADED_HOLD,
    STATE_FAILED_STOP,
    STATE_NORMAL,
    STATE_RECOVERABLE_HOLD,
    odom_corridor_cache_valid,
    passage_attempt_key,
    point_in_polygon,
    register_passage_attempt,
    route_tangent_at_point,
    swept_command_safety,
)


class PassageSupervisorNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.enabled = bool(rospy.get_param("~enabled", True))
        self.fixed_frame = str(
            rospy.get_param("~fixed_frame", "odom")
        ).lstrip("/")
        self.global_path_topic = str(
            rospy.get_param(
                "~global_path_topic",
                "/move_base/SBPLLatticePlanner/plan",
            )
        )
        self.disc_status_topic = str(
            rospy.get_param(
                "~disc_status_topic",
                "/yellow_track_disc_costmap_bridge/status",
            )
        )
        self.disc_scene_reset_service = str(
            rospy.get_param(
                "~disc_scene_reset_service",
                "/yellow_track_disc_costmap_bridge/reset_scene",
            )
        )
        self.boundary_topic = str(
            rospy.get_param(
                "~boundary_topic",
                "/yellow_track_avoidance/debug/boundaries",
            )
        )
        self.odom_topic = str(rospy.get_param("~odom_topic", "/odom"))
        self.move_base_status_topic = str(
            rospy.get_param("~move_base_status_topic", "/move_base/status")
        )
        self.current_goal_topic = str(
            rospy.get_param(
                "~current_goal_topic", "/move_base/current_goal"
            )
        )
        self.cancel_topic = str(
            rospy.get_param("~cancel_topic", "/move_base/cancel")
        )
        self.resume_goal_topic = str(
            rospy.get_param(
                "~resume_goal_topic", "/move_base_simple/goal"
            )
        )
        self.command_topic = str(
            rospy.get_param(
                "~command_topic",
                "/yellow_track_avoidance/passage_cmd_vel",
            )
        )
        self.move_base_cmd_topic = str(
            rospy.get_param(
                "~move_base_cmd_topic",
                "/yellow_track_avoidance/move_base_cmd_vel",
            )
        )
        self.status_topic = str(
            rospy.get_param(
                "~status_topic",
                "/yellow_track_avoidance/passage/status",
            )
        )
        self.marker_topic = str(
            rospy.get_param(
                "~marker_topic",
                "/yellow_track_avoidance/passage/markers",
            )
        )
        self.last_failure_topic = str(
            rospy.get_param(
                "~last_failure_topic",
                "/yellow_track_avoidance/passage/last_failure",
            )
        )
        self.event_log_path = str(
            rospy.get_param(
                "~event_log_path",
                "/tmp/yellow_track_avoidance_passage_events.jsonl",
            )
        )
        self.auto_resume_enabled = bool(
            rospy.get_param("~auto_resume_enabled", True)
        )
        self.auto_resume_cooldown_sec = float(
            rospy.get_param("~auto_resume_cooldown_sec", 0.50)
        )
        self.max_auto_resume_per_goal = int(
            rospy.get_param("~max_auto_resume_per_goal", 1)
        )
        self.retry_navigation_failures = bool(
            rospy.get_param("~retry_navigation_failures", True)
        )
        self.navigation_failure_retry_cooldown_sec = float(
            rospy.get_param(
                "~navigation_failure_retry_cooldown_sec", 0.75
            )
        )
        self.max_navigation_failure_retries_per_goal = int(
            rospy.get_param(
                "~max_navigation_failure_retries_per_goal", 0
            )
        )
        self.finish_recovery_enabled = bool(
            rospy.get_param("~finish_recovery_enabled", True)
        )
        self.navigation_no_progress_sec = float(
            rospy.get_param("~navigation_no_progress_sec", 4.0)
        )
        self.navigation_progress_distance_m = float(
            rospy.get_param("~navigation_progress_distance_m", 0.05)
        )
        self.recovery_short_goal_distance_m = float(
            rospy.get_param("~recovery_short_goal_distance_m", 0.40)
        )
        self.recovery_creep_distance_m = float(
            rospy.get_param("~recovery_creep_distance_m", 0.40)
        )
        self.recovery_creep_speed_mps = float(
            rospy.get_param("~recovery_creep_speed_mps", 0.04)
        )
        self.clear_costmaps_service = str(
            rospy.get_param(
                "~clear_costmaps_service", "/move_base/clear_costmaps"
            )
        )
        self.recovery_lidar_topic = str(
            rospy.get_param("~recovery_lidar_topic", "/scan")
        )
        self.recovery_lidar_timeout_sec = float(
            rospy.get_param("~recovery_lidar_timeout_sec", 0.30)
        )
        self.recovery_lidar_stop_distance_m = float(
            rospy.get_param("~recovery_lidar_stop_distance_m", 0.30)
        )
        self.recovery_lidar_half_angle_rad = math.radians(
            float(rospy.get_param("~recovery_lidar_half_angle_deg", 35.0))
        )
        self.control_rate_hz = float(
            rospy.get_param("~control_rate_hz", 20.0)
        )
        self.robot_timeout_sec = float(
            rospy.get_param("~robot_pose_timeout_sec", 0.30)
        )
        self.disc_timeout_sec = float(
            rospy.get_param("~disc_status_timeout_sec", 0.40)
        )
        self.boundary_timeout_sec = float(
            rospy.get_param("~boundary_timeout_sec", 0.45)
        )
        self.boundary_source_timeout_sec = float(
            rospy.get_param("~boundary_source_timeout_sec", 1.20)
        )
        self.last_corridor_max_age_sec = float(
            rospy.get_param("~last_corridor_max_age_sec", 1.20)
        )
        self.odom_propagation_max_sec = float(
            rospy.get_param("~odom_propagation_max_sec", 0.90)
        )
        self.max_blind_distance_m = float(
            rospy.get_param("~max_blind_distance_m", 0.05)
        )
        self.route_cache_timeout_sec = float(
            rospy.get_param("~route_cache_timeout_sec", 30.0)
        )
        self.navigation_failure_grace_sec = float(
            rospy.get_param(
                "~navigation_failure_takeover_grace_sec", 5.0
            )
        )
        self.minimum_boundary_quality = float(
            rospy.get_param("~minimum_boundary_quality", 0.70)
        )
        self.minimum_single_boundary_quality = float(
            rospy.get_param(
                "~minimum_single_boundary_quality", 0.35
            )
        )
        self.allow_single_boundary_inference = bool(
            rospy.get_param(
                "~allow_single_boundary_inference", True
            )
        )
        self.move_base_cmd_timeout_sec = float(
            rospy.get_param("~move_base_cmd_timeout_sec", 0.25)
        )
        self.activation_zone_enabled = bool(
            rospy.get_param("~activation_zone_enabled", False)
        )
        self.activation_zone_frame = str(
            rospy.get_param("~activation_zone_frame", "map")
        ).lstrip("/")
        raw_zone = rospy.get_param("~activation_zone_polygon", [])
        try:
            self.activation_zone_polygon = tuple(
                (float(point[0]), float(point[1]))
                for point in raw_zone
                if isinstance(point, (list, tuple)) and len(point) >= 2
            )
        except (TypeError, ValueError):
            raise ValueError("invalid passage activation polygon")
        if (
            self.activation_zone_enabled
            and len(self.activation_zone_polygon) < 3
        ):
            raise ValueError("passage activation polygon needs three points")
        self.stationary_linear_limit = float(
            rospy.get_param("~reset_linear_speed_max_mps", 0.01)
        )
        self.stationary_angular_limit = float(
            rospy.get_param("~reset_angular_speed_max_radps", 0.03)
        )
        timing = (
            self.control_rate_hz,
            self.robot_timeout_sec,
            self.disc_timeout_sec,
            self.boundary_timeout_sec,
            self.boundary_source_timeout_sec,
            self.last_corridor_max_age_sec,
            self.odom_propagation_max_sec,
            self.max_blind_distance_m,
            self.route_cache_timeout_sec,
            self.navigation_failure_grace_sec,
            self.auto_resume_cooldown_sec,
            self.navigation_failure_retry_cooldown_sec,
            self.move_base_cmd_timeout_sec,
            self.navigation_no_progress_sec,
            self.navigation_progress_distance_m,
            self.recovery_short_goal_distance_m,
            self.recovery_creep_distance_m,
            self.recovery_creep_speed_mps,
            self.recovery_lidar_timeout_sec,
            self.recovery_lidar_stop_distance_m,
            self.recovery_lidar_half_angle_rad,
        )
        if any(value <= 0.0 for value in timing):
            raise ValueError("invalid passage supervisor timing")
        if self.max_auto_resume_per_goal < 0:
            raise ValueError("max auto resume count cannot be negative")
        if self.max_navigation_failure_retries_per_goal < 0:
            raise ValueError(
                "max navigation failure retry count cannot be negative"
            )

        radians = math.radians
        self.config = PassageConfig(
            vehicle_half_length_m=float(
                rospy.get_param("~vehicle_half_length_m", 0.165)
            ),
            vehicle_half_width_m=float(
                rospy.get_param("~vehicle_half_width_m", 0.150)
            ),
            disc_radius_m=float(
                rospy.get_param("~disc_radius_m", 0.050)
            ),
            decision_horizon_m=float(
                rospy.get_param("~decision_horizon_m", 1.20)
            ),
            decision_min_forward_m=float(
                rospy.get_param("~decision_min_forward_m", 0.10)
            ),
            bypass_takeover_distance_m=float(
                rospy.get_param("~bypass_takeover_distance_m", 1.05)
            ),
            gate_takeover_distance_m=float(
                rospy.get_param("~gate_takeover_distance_m", 0.70)
            ),
            track_fresh_sec=float(
                rospy.get_param("~track_fresh_sec", 2.00)
            ),
            gate_visual_update_max_age_sec=float(
                rospy.get_param(
                    "~gate_visual_update_max_age_sec", 0.60
                )
            ),
            locked_track_drift_stop_m=float(
                rospy.get_param(
                    "~locked_track_drift_stop_m", 0.030
                )
            ),
            scene_confirm_sec=float(
                rospy.get_param("~scene_confirm_sec", 0.50)
            ),
            scene_release_sec=float(
                rospy.get_param("~scene_release_sec", 0.80)
            ),
            scene_position_variation_max_m=float(
                rospy.get_param(
                    "~scene_position_variation_max_m", 0.015
                )
            ),
            scene_progress_variation_max_m=float(
                rospy.get_param(
                    "~scene_progress_variation_max_m", 0.030
                )
            ),
            scene_angle_variation_max_rad=radians(
                float(
                    rospy.get_param(
                        "~scene_angle_variation_max_deg", 1.5
                    )
                )
            ),
            route_projection_max_distance_m=float(
                rospy.get_param(
                    "~route_projection_max_distance_m", 0.40
                )
            ),
            line_clearance_margin_m=float(
                rospy.get_param("~line_clearance_margin_m", 0.030)
            ),
            disc_clearance_margin_m=float(
                rospy.get_param("~disc_clearance_margin_m", 0.030)
            ),
            geometry_uncertainty_m=float(
                rospy.get_param("~geometry_uncertainty_m", 0.005)
            ),
            minimum_residual_clearance_m=float(
                rospy.get_param(
                    "~minimum_residual_clearance_m", 0.020
                )
            ),
            minimum_predicted_gate_clearance_m=float(
                rospy.get_param(
                    "~minimum_predicted_gate_clearance_m", 0.020
                )
            ),
            emergency_direct_clearance_m=float(
                rospy.get_param(
                    "~emergency_direct_clearance_m", 0.020
                )
            ),
            emergency_line_clearance_m=float(
                rospy.get_param(
                    "~emergency_line_clearance_m", 0.010
                )
            ),
            boundary_start_extrapolation_m=float(
                rospy.get_param(
                    "~boundary_start_extrapolation_m", 0.45
                )
            ),
            boundary_end_extrapolation_m=float(
                rospy.get_param(
                    "~boundary_end_extrapolation_m", 0.30
                )
            ),
            enable_gate_special_mode=bool(
                rospy.get_param("~enable_gate_special_mode", True)
            ),
            claim_all_road_discs=bool(
                rospy.get_param("~claim_all_road_discs", False)
            ),
            force_between_disc_pair=bool(
                rospy.get_param("~force_between_disc_pair", False)
            ),
            competition_two_disc_mode=bool(
                rospy.get_param("~competition_two_disc_mode", False)
            ),
            swept_prediction_horizon_sec=float(
                rospy.get_param(
                    "~swept_prediction_horizon_sec", 0.35
                )
            ),
            swept_translation_step_m=float(
                rospy.get_param(
                    "~swept_translation_step_m", 0.005
                )
            ),
            swept_rotation_step_rad=radians(
                float(
                    rospy.get_param(
                        "~swept_rotation_step_deg", 1.0
                    )
                )
            ),
            side_choice_min_advantage_m=float(
                rospy.get_param(
                    "~side_choice_min_advantage_m", 0.040
                )
            ),
            side_switch_required_advantage_m=float(
                rospy.get_param(
                    "~side_switch_required_advantage_m", 0.060
                )
            ),
            side_invalid_confirm_sec=float(
                rospy.get_param("~side_invalid_confirm_sec", 0.30)
            ),
            minimum_shift_start_distance_m=float(
                rospy.get_param(
                    "~minimum_shift_start_distance_m", 0.40
                )
            ),
            late_detection_stop_distance_m=float(
                rospy.get_param(
                    "~late_detection_stop_distance_m", 0.30
                )
            ),
            single_shift_length_m=float(
                rospy.get_param("~single_shift_length_m", 0.45)
            ),
            single_hold_after_disc_m=float(
                rospy.get_param("~single_hold_after_disc_m", 0.30)
            ),
            single_merge_length_m=float(
                rospy.get_param("~single_merge_length_m", 0.45)
            ),
            passage_sample_step_m=float(
                rospy.get_param("~passage_sample_step_m", 0.05)
            ),
            passage_lateral_grid_step_m=float(
                rospy.get_param(
                    "~passage_lateral_grid_step_m", 0.01
                )
            ),
            maximum_lateral_slope=float(
                rospy.get_param("~maximum_lateral_slope", 0.60)
            ),
            bypass_speed_mps=float(
                rospy.get_param("~bypass_speed_mps", 0.050)
            ),
            gate_approach_speed_mps=float(
                rospy.get_param("~approach_speed_mps", 0.080)
            ),
            gate_speed_mps=float(
                rospy.get_param("~gate_speed_mps", 0.030)
            ),
            gate_exit_speed_mps=float(
                rospy.get_param("~gate_exit_speed_mps", 0.030)
            ),
            lateral_gain=float(
                rospy.get_param("~lateral_gain", 1.00)
            ),
            yaw_gain=float(rospy.get_param("~yaw_gain", 1.50)),
            maximum_lateral_speed_mps=float(
                rospy.get_param(
                    "~maximum_lateral_speed_mps", 0.025
                )
            ),
            maximum_angular_speed_radps=float(
                rospy.get_param(
                    "~maximum_angular_speed_radps", 0.10
                )
            ),
            stop_lateral_error_m=float(
                rospy.get_param("~stop_lateral_error_m", 0.080)
            ),
            stop_yaw_error_rad=radians(
                float(rospy.get_param("~stop_yaw_error_deg", 15.0))
            ),
            completion_lateral_tolerance_m=float(
                rospy.get_param(
                    "~completion_lateral_tolerance_m", 0.025
                )
            ),
            gate_align_distance_m=float(
                rospy.get_param("~gate_align_distance_m", 0.50)
            ),
            gate_align_lateral_tolerance_m=float(
                rospy.get_param(
                    "~gate_align_lateral_tolerance_m", 0.005
                )
            ),
            gate_align_yaw_tolerance_rad=radians(
                float(
                    rospy.get_param(
                        "~gate_align_yaw_tolerance_deg", 1.5
                    )
                )
            ),
            gate_align_hold_sec=float(
                rospy.get_param("~gate_align_hold_sec", 0.30)
            ),
            gate_max_align_speed_mps=float(
                rospy.get_param("~gate_max_align_speed_mps", 0.020)
            ),
            gate_lateral_abort_m=float(
                rospy.get_param("~gate_lateral_abort_m", 0.15)
            ),
            gate_yaw_abort_rad=radians(
                float(rospy.get_param("~gate_yaw_abort_deg", 45.0))
            ),
            gate_exit_distance_m=float(
                rospy.get_param("~gate_exit_distance_m", 0.35)
            ),
            gate_exit_clearance_margin_m=float(
                rospy.get_param(
                    "~gate_exit_clearance_margin_m", 0.05
                )
            ),
            handoff_linear_speed_max_mps=float(
                rospy.get_param(
                    "~handoff_linear_speed_max_mps", 0.010
                )
            ),
            handoff_angular_speed_max_radps=float(
                rospy.get_param(
                    "~handoff_angular_speed_max_radps", 0.030
                )
            ),
            handoff_stop_hold_sec=float(
                rospy.get_param("~handoff_stop_hold_sec", 0.25)
            ),
            degraded_hold_sec=float(
                rospy.get_param("~degraded_hold_sec", 0.40)
            ),
            exit_stop_hold_sec=float(
                rospy.get_param("~exit_stop_hold_sec", 0.30)
            ),
            takeover_ack_timeout_sec=float(
                rospy.get_param("~takeover_ack_timeout_sec", 0.50)
            ),
            special_control_max_sec=float(
                rospy.get_param("~special_control_max_sec", 20.0)
            ),
            special_no_progress_timeout_sec=float(
                rospy.get_param(
                    "~special_no_progress_timeout_sec", 1.20
                )
            ),
            special_min_progress_m=float(
                rospy.get_param("~special_min_progress_m", 0.015)
            ),
            special_min_yaw_progress_rad=radians(
                float(
                    rospy.get_param(
                        "~special_min_yaw_progress_deg", 0.5
                    )
                )
            ),
            handoff_stop_timeout_sec=float(
                rospy.get_param("~handoff_stop_timeout_sec", 1.50)
            ),
            approach_align_timeout_sec=float(
                rospy.get_param("~approach_align_timeout_sec", 10.0)
            ),
            cross_exit_timeout_sec=float(
                rospy.get_param("~cross_exit_timeout_sec", 22.0)
            ),
            competition_overall_timeout_sec=float(
                rospy.get_param("~overall_timeout_sec", 30.0)
            ),
            special_retry_cooldown_sec=float(
                rospy.get_param("~special_retry_cooldown_sec", 2.0)
            ),
            special_max_attempts=int(
                rospy.get_param("~special_max_attempts", 2)
            ),
            competition_finish_priority=bool(
                rospy.get_param(
                    "~competition_finish_priority", False
                )
            ),
            acceleration_x_mps2=float(
                rospy.get_param("~acceleration_x_mps2", 0.10)
            ),
            acceleration_y_mps2=float(
                rospy.get_param("~acceleration_y_mps2", 0.08)
            ),
            acceleration_yaw_radps2=float(
                rospy.get_param(
                    "~acceleration_yaw_radps2", 0.25
                )
            ),
            gate_special_mode_max_gap_m=float(
                rospy.get_param(
                    "~gate_special_mode_max_gap_m", 0.45
                )
            ),
            gate_release_gap_m=float(
                rospy.get_param("~gate_release_gap_m", 0.48)
            ),
            gate_longitudinal_mismatch_max_m=float(
                rospy.get_param(
                    "~gate_longitudinal_mismatch_max_m", 0.25
                )
            ),
            gate_midpoint_path_distance_max_m=float(
                rospy.get_param(
                    "~gate_midpoint_path_distance_max_m", 0.12
                )
            ),
            gate_normal_angle_tolerance_rad=radians(
                float(
                    rospy.get_param(
                        "~gate_normal_angle_tolerance_deg", 30.0
                    )
                )
            ),
        )
        self.machine = PassageStateMachine(self.config)

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.robot_pose = None
        self.robot_receipt_sec = None
        self.robot_linear_speed = math.inf
        self.robot_angular_speed = math.inf
        self.tracks = []
        self.live_gate_tracks = []
        self.disc_receipt_sec = None
        self.disc_input_fresh = False
        self.disc_fixed_transform_available = False
        self.disc_timestamp_transform_drop_count = 0
        self.disc_scene_generation = 0
        self.boundary = BoundaryLines()
        self.boundary_receipt_sec = None
        self.boundary_source_sec = None
        self.boundary_inference = "none"
        self.last_trusted_boundary = None
        self.last_trusted_boundary_source_sec = None
        self.last_trusted_boundary_robot_xy = None
        self.last_trusted_boundary_inference = "none"
        self.live_boundary_fresh = False
        self.boundary_reference_mode = "none"
        self.path = ()
        self.path_receipt_sec = None
        self.route_source = "none"
        self.visualization_path = ()
        self.navigation_active = False
        self.navigation_failure_receipt_sec = None
        self.navigation_failure_key = None
        self.navigation_failure_retry_count = 0
        self.last_navigation_failure_retry_sec = None
        self.latest_move_base_status = None
        self.current_goal_id = ""
        self.current_goal = None
        self.current_goal_signature = None
        self.resume_goal = None
        self.resume_in_flight = False
        self.last_cancel_owner = ""
        self.last_cancel_reason = ""
        self.supervisor_cancel_outstanding = False
        self.supervisor_cancel_sec = None
        self.cancelled_goal_id = ""
        self.auto_resume_count = 0
        self.logical_goal_epoch = 0
        self.attempted_passages = set()
        self.suppressed_passage_count = 0
        self.finish_recovery_stage = 0
        self.finish_recovery_action = "idle"
        self.navigation_progress_xy = None
        self.navigation_progress_sec = None
        self.recovery_goal_in_flight = False
        self.recovery_short_goal_active = False
        self.recovery_resume_original_pending = False
        self.recovery_creep_active = False
        self.recovery_creep_handoff_complete = False
        self.recovery_creep_start_xy = None
        self.recovery_creep_travel_xy = None
        self.recovery_creep_requested_sec = None
        self.recovery_count = 0
        self.recovery_lidar_receipt_sec = None
        self.recovery_lidar_forward_min_m = None
        self.last_failure_snapshot = None
        self.last_failure_key = None
        self.last_decision = None
        self.cancel_count = 0
        self.resume_count = 0
        self.latest_move_base_command = None
        self.move_base_cmd_receipt_sec = None
        self.activation_zone_pose = None
        self.activation_zone_pose_available = (
            not self.activation_zone_enabled
        )
        self.activation_zone_active = False
        self.normal_command_safety = SweptSafety(
            True, "activation_zone_not_required"
        )

        self.command_pub = rospy.Publisher(
            self.command_topic, Twist, queue_size=1
        )
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=1, latch=True
        )
        self.last_failure_pub = rospy.Publisher(
            self.last_failure_topic, String, queue_size=1, latch=True
        )
        self.marker_pub = rospy.Publisher(
            self.marker_topic, MarkerArray, queue_size=1
        )
        self.cancel_pub = rospy.Publisher(
            self.cancel_topic, GoalID, queue_size=1
        )
        self.resume_pub = rospy.Publisher(
            self.resume_goal_topic, PoseStamped, queue_size=1
        )
        self.clear_costmaps_client = rospy.ServiceProxy(
            self.clear_costmaps_service, Empty
        )
        self.disc_scene_reset_client = rospy.ServiceProxy(
            self.disc_scene_reset_service, Trigger
        )
        self.path_sub = rospy.Subscriber(
            self.global_path_topic,
            Path,
            self._path_callback,
            queue_size=1,
        )
        self.disc_sub = rospy.Subscriber(
            self.disc_status_topic,
            String,
            self._disc_callback,
            queue_size=1,
        )
        self.boundary_sub = rospy.Subscriber(
            self.boundary_topic,
            String,
            self._boundary_callback,
            queue_size=1,
        )
        self.odom_sub = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self._odom_callback,
            queue_size=1,
        )
        self.move_base_status_sub = rospy.Subscriber(
            self.move_base_status_topic,
            GoalStatusArray,
            self._move_base_status_callback,
            queue_size=1,
        )
        self.current_goal_sub = rospy.Subscriber(
            self.current_goal_topic,
            PoseStamped,
            self._current_goal_callback,
            queue_size=1,
        )
        self.move_base_cmd_sub = rospy.Subscriber(
            self.move_base_cmd_topic,
            Twist,
            self._move_base_cmd_callback,
            queue_size=1,
        )
        self.recovery_lidar_sub = rospy.Subscriber(
            self.recovery_lidar_topic,
            LaserScan,
            self._recovery_lidar_callback,
            queue_size=1,
        )
        self.reset_service = rospy.Service(
            "~reset", Trigger, self._reset_callback
        )
        self.reset_scene_service = rospy.Service(
            "~reset_scene", Trigger, self._reset_scene_callback
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate_hz),
            self._timer_callback,
        )
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "passage supervisor ready: bypass<=%.2fm gate_gap<=%.2fm "
            "command=%s fixed_frame=%s enabled=%s",
            self.config.bypass_takeover_distance_m,
            self.config.gate_special_mode_max_gap_m,
            self.command_topic,
            self.fixed_frame,
            self.enabled,
        )

    @staticmethod
    def _yaw(quaternion):
        return math.atan2(
            2.0 * (
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

    @staticmethod
    def _transform_xy(transform, point):
        yaw = PassageSupervisorNode._yaw(
            transform.transform.rotation
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        translation = transform.transform.translation
        return (
            float(translation.x)
            + cosine * float(point[0])
            - sine * float(point[1]),
            float(translation.y)
            + sine * float(point[0])
            + cosine * float(point[1]),
        )

    def _lookup_transform(self, target, source, stamp):
        if target == source:
            return None
        try:
            return self.tf_buffer.lookup_transform(
                target, source, stamp, rospy.Duration(0.05)
            )
        except tf2_ros.ExtrapolationException:
            return self.tf_buffer.lookup_transform(
                target, source, rospy.Time(0), rospy.Duration(0.05)
            )

    def _activation_zone_state(self):
        if not self.activation_zone_enabled:
            return True, False, None
        if self.robot_pose is None:
            return False, False, None
        point = self.robot_pose.xy
        try:
            transform = self._lookup_transform(
                self.activation_zone_frame,
                self.fixed_frame,
                rospy.Time(0),
            )
            if transform is not None:
                point = self._transform_xy(transform, point)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return False, False, None
        if not all(np.isfinite(value) for value in point):
            return False, False, None
        return (
            True,
            point_in_polygon(point, self.activation_zone_polygon),
            point,
        )

    def _move_base_cmd_callback(self, message):
        with self.lock:
            self.latest_move_base_command = copy.deepcopy(message)
            self.move_base_cmd_receipt_sec = rospy.Time.now().to_sec()

    def _recovery_lidar_callback(self, message):
        forward_ranges = []
        angle = float(message.angle_min)
        for value in message.ranges:
            distance = float(value)
            if (
                abs(wrap_angle(angle)) <= self.recovery_lidar_half_angle_rad
                and math.isfinite(distance)
                and distance >= float(message.range_min)
                and distance <= float(message.range_max)
            ):
                forward_ranges.append(distance)
            angle += float(message.angle_increment)
        with self.lock:
            self.recovery_lidar_forward_min_m = (
                min(forward_ranges) if forward_ranges else None
            )
            self.recovery_lidar_receipt_sec = rospy.Time.now().to_sec()

    def _normal_swept_safety(
        self,
        now_sec,
        robot_fresh,
        boundary,
        boundary_fresh,
        path,
        route_valid,
    ):
        if not self.activation_zone_enabled:
            return SweptSafety(True, "activation_zone_not_active")
        if not self.activation_zone_pose_available:
            return SweptSafety(False, "activation_zone_pose_unavailable")
        if not self.activation_zone_active:
            return SweptSafety(True, "activation_zone_not_active")
        if not robot_fresh or self.robot_pose is None:
            return SweptSafety(False, "swept_robot_pose_unavailable")
        if not boundary_fresh:
            return SweptSafety(False, "swept_boundary_unavailable")
        if not route_valid:
            return SweptSafety(False, "swept_route_unavailable")
        command_fresh = (
            self.latest_move_base_command is not None
            and self.move_base_cmd_receipt_sec is not None
            and now_sec - self.move_base_cmd_receipt_sec
            <= self.move_base_cmd_timeout_sec
        )
        if not command_fresh:
            return SweptSafety(False, "move_base_command_stale")
        command = self.latest_move_base_command
        return swept_command_safety(
            robot=self.robot_pose,
            command_body=(
                command.linear.x,
                command.linear.y,
                command.angular.z,
            ),
            path=path,
            boundary=boundary,
            tracks=self.tracks,
            config=self.config,
        )

    def _odom_callback(self, message):
        source = str(message.header.frame_id).lstrip("/")
        if not source:
            return
        point = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
        )
        yaw = self._yaw(message.pose.pose.orientation)
        try:
            transform = self._lookup_transform(
                self.fixed_frame, source, message.header.stamp
            )
            if transform is not None:
                point = self._transform_xy(transform, point)
                yaw = wrap_angle(
                    self._yaw(transform.transform.rotation) + yaw
                )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return
        values = point[0], point[1], yaw
        if not all(np.isfinite(value) for value in values):
            return
        with self.lock:
            self.robot_pose = RobotPose(*values)
            self.robot_receipt_sec = rospy.Time.now().to_sec()
            self.robot_linear_speed = math.hypot(
                float(message.twist.twist.linear.x),
                float(message.twist.twist.linear.y),
            )
            self.robot_angular_speed = abs(
                float(message.twist.twist.angular.z)
            )

    def _transform_points(self, points, transform):
        output = []
        for item in points:
            if not isinstance(item, list) or len(item) < 2:
                continue
            try:
                point = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(value) for value in point):
                continue
            output.append(
                point
                if transform is None
                else self._transform_xy(transform, point)
            )
        return tuple(output)

    def _path_callback(self, message):
        source = str(message.header.frame_id).lstrip("/")
        if not source and message.poses:
            source = str(message.poses[0].header.frame_id).lstrip("/")
        if not source:
            return
        try:
            transform = self._lookup_transform(
                self.fixed_frame, source, message.header.stamp
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return
        points = []
        for pose in message.poses:
            point = (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
            )
            if transform is not None:
                point = self._transform_xy(transform, point)
            if not all(np.isfinite(value) for value in point):
                continue
            if points and math.hypot(
                point[0] - points[-1][0],
                point[1] - points[-1][1],
            ) <= 1e-4:
                continue
            points.append(point)
        if len(points) < 2:
            return
        with self.lock:
            self.path = tuple(points)
            self.path_receipt_sec = rospy.Time.now().to_sec()
            self.route_source = "sbpl"

    def _boundary_callback(self, message):
        try:
            payload = json.loads(message.data)
            source = str(payload.get("frame_id", "")).lstrip("/")
            stamp = rospy.Time.from_sec(float(payload["stamp"]))
            source_sec = stamp.to_sec()
            quality = float(payload.get("quality", 0.0))
            line_width = float(
                payload.get("yellow_line_width_m", 0.030)
            )
        except (KeyError, TypeError, ValueError):
            return
        if not source:
            return
        try:
            transform = self._lookup_transform(
                self.fixed_frame, source, stamp
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return
        left = self._transform_points(
            payload.get("left_base_m", []), transform
        )
        right = self._transform_points(
            payload.get("right_base_m", []), transform
        )
        both_detected = (
            bool(payload.get("left_detected", False))
            and bool(payload.get("right_detected", False))
        )
        inference = str(payload.get("boundary_inference", "none"))
        inferred_pair = (
            inference != "none"
            and bool(payload.get("boundary_pair_valid", False))
            and len(left) >= 2
            and len(right) >= 2
        )
        detected = both_detected or (
            self.allow_single_boundary_inference and inferred_pair
        )
        with self.lock:
            self.boundary = BoundaryLines(
                left_world=left,
                right_world=right,
                yellow_line_width_m=line_width,
                quality=quality,
                fresh=detected,
            )
            self.boundary_inference = (
                inference if inferred_pair else "none"
            )
            self.boundary_receipt_sec = rospy.Time.now().to_sec()
            self.boundary_source_sec = source_sec
            quality_threshold = (
                self.minimum_single_boundary_quality
                if self.boundary_inference != "none"
                else self.minimum_boundary_quality
            )
            if (
                self.boundary.valid
                and quality >= quality_threshold
                and self.robot_pose is not None
            ):
                self.last_trusted_boundary = self.boundary
                self.last_trusted_boundary_source_sec = source_sec
                self.last_trusted_boundary_robot_xy = (
                    float(self.robot_pose.x),
                    float(self.robot_pose.y),
                )
                self.last_trusted_boundary_inference = (
                    self.boundary_inference
                )

    def _disc_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if str(payload.get("fixed_frame", "")).lstrip("/") != self.fixed_frame:
            return
        try:
            scene_generation = int(payload.get("scene_generation", 0))
        except (TypeError, ValueError):
            return
        with self.lock:
            if scene_generation < self.disc_scene_generation:
                return
        tracks = []
        for item in payload.get("active_tracks", []):
            fixed_xy = item.get("fixed_xy")
            if not isinstance(fixed_xy, list) or len(fixed_xy) < 2:
                continue
            try:
                values = float(fixed_xy[0]), float(fixed_xy[1])
                memory_id = int(item["memory_id"])
                age = float(item.get("last_seen_age_sec", math.inf))
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not all(np.isfinite(value) for value in values)
                or not np.isfinite(age)
                or age < 0.0
            ):
                continue
            tracks.append(
                DiscTrack(
                    memory_id=memory_id,
                    fixed_xy=values,
                    confirmed=bool(item.get("confirmed", False)),
                    phase=str(item.get("phase", "unconfirmed")),
                    last_seen_age_sec=age,
                    estimated_diameter_m=(
                        float(item["estimated_diameter_m"])
                        if item.get("estimated_diameter_m") is not None
                        and np.isfinite(float(item["estimated_diameter_m"]))
                        else None
                    ),
                    confidence=(
                        float(item["confidence"])
                        if item.get("confidence") is not None
                        and np.isfinite(float(item["confidence"]))
                        else None
                    ),
                )
            )
        live_gate_tracks = []
        for item in payload.get("live_tracks", []):
            fixed_xy = item.get("fixed_xy")
            if not isinstance(fixed_xy, list) or len(fixed_xy) < 2:
                continue
            try:
                values = float(fixed_xy[0]), float(fixed_xy[1])
                memory_id = int(item["memory_id"])
                age = float(item.get("last_seen_age_sec", math.inf))
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not all(np.isfinite(value) for value in values)
                or not np.isfinite(age)
                or age < 0.0
            ):
                continue
            live_gate_tracks.append(
                DiscTrack(
                    memory_id=memory_id,
                    fixed_xy=values,
                    confirmed=bool(item.get("confirmed", False)),
                    phase=str(item.get("phase", "unconfirmed")),
                    last_seen_age_sec=age,
                    estimated_diameter_m=(
                        float(item["estimated_diameter_m"])
                        if item.get("estimated_diameter_m") is not None
                        and np.isfinite(float(item["estimated_diameter_m"]))
                        else None
                    ),
                    confidence=(
                        float(item["confidence"])
                        if item.get("confidence") is not None
                        and np.isfinite(float(item["confidence"]))
                        else None
                    ),
                )
            )
        with self.lock:
            if scene_generation < self.disc_scene_generation:
                return
            self.disc_scene_generation = scene_generation
            self.tracks = tracks
            self.live_gate_tracks = live_gate_tracks
            self.disc_receipt_sec = rospy.Time.now().to_sec()
            self.disc_input_fresh = bool(
                payload.get("input_fresh", False)
            )
            self.disc_fixed_transform_available = bool(
                payload.get("fixed_transform_available", False)
            )
            self.disc_timestamp_transform_drop_count = int(
                payload.get("timestamp_transform_drop_count", 0)
            )

    def _publish_supervisor_cancel(self, reason):
        self.cancel_pub.publish(GoalID())
        self.cancel_count += 1
        self.last_cancel_owner = "passage_supervisor"
        self.last_cancel_reason = str(reason)
        self.supervisor_cancel_outstanding = True
        self.supervisor_cancel_sec = rospy.Time.now().to_sec()
        self.cancelled_goal_id = self.current_goal_id

    def _publish_resume_goal(self, now_sec, automatic):
        if self.resume_goal is None:
            return False
        goal = copy.deepcopy(self.resume_goal)
        goal.header.stamp = rospy.Time.from_sec(now_sec)
        self.path = ()
        self.path_receipt_sec = None
        self.route_source = "waiting_for_sbpl"
        self.resume_in_flight = True
        self.resume_pub.publish(goal)
        self.resume_count += 1
        if automatic:
            self.auto_resume_count += 1
        self.supervisor_cancel_outstanding = False
        self.last_cancel_owner = "passage_supervisor_resumed"
        return True

    def _clear_costmaps_and_retry(self, now_sec):
        try:
            self.clear_costmaps_client()
        except (rospy.ServiceException, rospy.ROSException) as error:
            rospy.logwarn("finish recovery costmap clear failed: %s", error)
        published = self._publish_resume_goal(now_sec, automatic=False)
        if published:
            self.finish_recovery_action = "clear_costmaps_and_retry"
            self.recovery_count += 1
        return published

    def _publish_recovery_short_goal(self, now_sec, tangent):
        if self.robot_pose is None or self.resume_goal is None:
            return False
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.from_sec(now_sec)
        goal.header.frame_id = self.fixed_frame
        goal.pose.position.x = (
            self.robot_pose.x
            + self.recovery_short_goal_distance_m * tangent[0]
        )
        goal.pose.position.y = (
            self.robot_pose.y
            + self.recovery_short_goal_distance_m * tangent[1]
        )
        yaw = math.atan2(tangent[1], tangent[0])
        goal.pose.orientation.z = math.sin(0.5 * yaw)
        goal.pose.orientation.w = math.cos(0.5 * yaw)
        self.path = ()
        self.path_receipt_sec = None
        self.route_source = "waiting_for_recovery_short_goal"
        self.recovery_goal_in_flight = True
        self.recovery_short_goal_active = True
        self.resume_pub.publish(goal)
        self.finish_recovery_action = "forward_short_goal"
        self.recovery_count += 1
        return True

    def _reset_finish_recovery(self, now_sec=None, clear_stage=True):
        if clear_stage:
            self.finish_recovery_stage = 0
        self.finish_recovery_action = "idle"
        self.recovery_goal_in_flight = False
        self.recovery_short_goal_active = False
        self.recovery_resume_original_pending = False
        self.recovery_creep_active = False
        self.recovery_creep_handoff_complete = False
        self.recovery_creep_start_xy = None
        self.recovery_creep_travel_xy = None
        self.recovery_creep_requested_sec = None
        self.navigation_progress_xy = (
            None
            if self.robot_pose is None
            else (self.robot_pose.x, self.robot_pose.y)
        )
        self.navigation_progress_sec = now_sec

    def _finish_recovery_override(
        self,
        now_sec,
        decision,
        robot_fresh,
        boundary,
        boundary_fresh,
        path,
        route_valid,
    ):
        """Apply clear, short-goal and bounded corridor-creep recovery."""
        if (
            not self.finish_recovery_enabled
            or not self.config.competition_finish_priority
            or decision.authority_requested
            or decision.cancel_move_base
            or decision.resume_move_base
        ):
            return decision
        if self.recovery_resume_original_pending:
            self.recovery_resume_original_pending = False
            self.recovery_short_goal_active = False
            self._publish_resume_goal(now_sec, automatic=False)
            self._reset_finish_recovery(now_sec)
            return decision

        if self.recovery_creep_active:
            if (
                not robot_fresh
                or self.robot_pose is None
                or not boundary_fresh
                or not route_valid
                or self.recovery_creep_start_xy is None
                or self.recovery_creep_travel_xy is None
            ):
                self.recovery_creep_handoff_complete = False
                return replace(
                    decision,
                    state="finish_recovery_hard_stop",
                    authority_requested=True,
                    command_body=(0.0, 0.0, 0.0),
                    reason="finish_recovery_localization_or_corridor_lost",
                    swept_safety=SweptSafety(
                        False, "recovery_reference_unavailable"
                    ),
                )
            lidar_fresh = (
                self.recovery_lidar_receipt_sec is not None
                and now_sec - self.recovery_lidar_receipt_sec
                <= self.recovery_lidar_timeout_sec
            )
            lidar_blocked = (
                self.recovery_lidar_forward_min_m is not None
                and self.recovery_lidar_forward_min_m
                < self.recovery_lidar_stop_distance_m
            )
            if not lidar_fresh or lidar_blocked:
                self.recovery_creep_handoff_complete = False
                reason = (
                    "recovery_lidar_stale"
                    if not lidar_fresh
                    else "recovery_lidar_wall_clearance"
                )
                return replace(
                    decision,
                    state="finish_recovery_hard_stop",
                    authority_requested=True,
                    command_body=(0.0, 0.0, 0.0),
                    reason="finish_recovery_hard_stop:" + reason,
                    swept_safety=SweptSafety(False, reason),
                )
            # The stop check belongs only to the one-time ownership handoff.
            # Rechecking it after creep motion starts creates a stop/go loop:
            # the command makes odometry non-stationary, which immediately
            # sends the next cycle back to handoff_stop.
            if not self.recovery_creep_handoff_complete:
                stopped = (
                    self.robot_linear_speed <= self.stationary_linear_limit
                    and self.robot_angular_speed
                    <= self.stationary_angular_limit
                )
                cancel_waiting = (
                    self.navigation_active
                    and self.recovery_creep_requested_sec is not None
                    and now_sec - self.recovery_creep_requested_sec
                    < self.config.handoff_stop_timeout_sec
                )
                if not stopped or cancel_waiting:
                    return replace(
                        decision,
                        state="finish_recovery_handoff_stop",
                        authority_requested=True,
                        command_body=(0.0, 0.0, 0.0),
                        reason="finish_recovery_waiting_for_stop",
                        swept_safety=SweptSafety(
                            True, "recovery_handoff_stop"
                        ),
                    )
                self.recovery_creep_handoff_complete = True
            delta = (
                self.robot_pose.x - self.recovery_creep_start_xy[0],
                self.robot_pose.y - self.recovery_creep_start_xy[1],
            )
            progress = (
                delta[0] * self.recovery_creep_travel_xy[0]
                + delta[1] * self.recovery_creep_travel_xy[1]
            )
            if progress >= self.recovery_creep_distance_m:
                self.command_pub.publish(Twist())
                self.recovery_creep_active = False
                self._publish_resume_goal(now_sec, automatic=False)
                self._reset_finish_recovery(now_sec)
                return decision
            yaw_ref = math.atan2(
                self.recovery_creep_travel_xy[1],
                self.recovery_creep_travel_xy[0],
            )
            yaw_error = wrap_angle(yaw_ref - self.robot_pose.yaw)
            cosine = math.cos(self.robot_pose.yaw)
            sine = math.sin(self.robot_pose.yaw)
            world_x = (
                self.recovery_creep_speed_mps
                * self.recovery_creep_travel_xy[0]
            )
            world_y = (
                self.recovery_creep_speed_mps
                * self.recovery_creep_travel_xy[1]
            )
            command = (
                cosine * world_x + sine * world_y,
                -sine * world_x + cosine * world_y,
                max(-0.10, min(0.10, 1.5 * yaw_error)),
            )
            # Discs are a score penalty in this last-resort stage. Yellow-line
            # contact remains a hard veto via the full-rectangle sweep.
            safety = swept_command_safety(
                robot=self.robot_pose,
                command_body=command,
                path=path,
                boundary=boundary,
                tracks=(),
                config=self.config,
            )
            if not safety.safe:
                command = (0.0, 0.0, 0.0)
            return replace(
                decision,
                state=(
                    "finish_recovery_creep"
                    if safety.safe
                    else "finish_recovery_hard_stop"
                ),
                authority_requested=True,
                command_body=command,
                reason=(
                    "finish_recovery_corridor_creep"
                    if safety.safe
                    else "finish_recovery_hard_stop:" + safety.reason
                ),
                swept_safety=safety,
            )

        if self.robot_pose is None or not robot_fresh:
            return decision
        current_xy = self.robot_pose.x, self.robot_pose.y
        if self.navigation_progress_xy is None:
            self.navigation_progress_xy = current_xy
            self.navigation_progress_sec = now_sec
            return decision
        moved = math.hypot(
            current_xy[0] - self.navigation_progress_xy[0],
            current_xy[1] - self.navigation_progress_xy[1],
        )
        if moved >= self.navigation_progress_distance_m:
            self.navigation_progress_xy = current_xy
            self.navigation_progress_sec = now_sec
            if not self.recovery_short_goal_active:
                self.finish_recovery_stage = 0
            return decision
        terminal_failure = self.navigation_failure_receipt_sec is not None
        no_progress = (
            self.navigation_active
            and self.navigation_progress_sec is not None
            and now_sec - self.navigation_progress_sec
            >= self.navigation_no_progress_sec
        )
        if not terminal_failure and not no_progress:
            return decision
        tangent = route_tangent_at_point(
            current_xy,
            path,
            self.config.route_projection_max_distance_m,
        )
        if self.finish_recovery_stage == 0:
            if self._clear_costmaps_and_retry(now_sec):
                self.finish_recovery_stage = 1
                self.navigation_progress_xy = current_xy
                self.navigation_progress_sec = now_sec
                self.navigation_failure_receipt_sec = None
            return decision
        if self.finish_recovery_stage == 1 and tangent is not None:
            if self._publish_recovery_short_goal(now_sec, tangent):
                self.finish_recovery_stage = 2
                self.navigation_progress_xy = current_xy
                self.navigation_progress_sec = now_sec
                self.navigation_failure_receipt_sec = None
            return decision
        if tangent is None or not boundary_fresh or not route_valid:
            return replace(
                decision,
                state="finish_recovery_hard_stop",
                authority_requested=True,
                command_body=(0.0, 0.0, 0.0),
                reason="finish_recovery_localization_or_corridor_lost",
                swept_safety=SweptSafety(
                    False, "recovery_reference_unavailable"
                ),
            )
        self._publish_supervisor_cancel("finish_recovery_corridor_creep")
        self.recovery_creep_active = True
        self.recovery_creep_handoff_complete = False
        self.recovery_creep_start_xy = current_xy
        self.recovery_creep_travel_xy = tangent
        self.recovery_creep_requested_sec = now_sec
        self.finish_recovery_action = "corridor_creep"
        self.recovery_count += 1
        self.navigation_failure_receipt_sec = None
        return replace(
            decision,
            state="finish_recovery_creep",
            authority_requested=True,
            command_body=(0.0, 0.0, 0.0),
            reason="finish_recovery_creep_handoff_stop",
            swept_safety=SweptSafety(True, "recovery_handoff_stop"),
        )

    def _move_base_status_callback(self, message):
        active_states = {
            GoalStatus.PENDING,
            GoalStatus.ACTIVE,
            GoalStatus.PREEMPTING,
            GoalStatus.RECALLING,
        }
        active_items = [
            status
            for status in message.status_list
            if status.status in active_states and status.goal_id.id
        ]
        failure_states = {
            GoalStatus.ABORTED,
            GoalStatus.REJECTED,
            GoalStatus.LOST,
        }
        with self.lock:
            self.navigation_active = bool(active_items)
            if active_items:
                self.current_goal_id = active_items[-1].goal_id.id
                self.latest_move_base_status = int(
                    active_items[-1].status
                )
                if (
                    self.supervisor_cancel_outstanding
                    and self.current_goal_id
                    and self.current_goal_id != self.cancelled_goal_id
                ):
                    self.supervisor_cancel_outstanding = False
                self.navigation_failure_key = None
                self.navigation_failure_receipt_sec = None
            identified = [
                status
                for status in message.status_list
                if status.goal_id.id
            ]
            latest = next(
                (
                    status
                    for status in reversed(identified)
                    if status.goal_id.id == self.current_goal_id
                ),
                identified[-1] if identified else None,
            )
            if latest is not None:
                self.latest_move_base_status = int(latest.status)
            if active_items:
                return
            if (
                latest is not None
                and latest.goal_id.id
                and latest.status in failure_states
            ):
                key = latest.goal_id.id, int(latest.status)
                if key != self.navigation_failure_key:
                    self.navigation_failure_key = key
                    self.navigation_failure_receipt_sec = (
                        rospy.Time.now().to_sec()
                    )
                    if self.recovery_short_goal_active:
                        self.recovery_short_goal_active = False
            elif (
                latest is not None
                and latest.status == GoalStatus.SUCCEEDED
            ):
                if self.recovery_short_goal_active:
                    self.recovery_resume_original_pending = True
                self.navigation_failure_key = None
                self.navigation_failure_receipt_sec = None
                self.supervisor_cancel_outstanding = False

    def _current_goal_callback(self, message):
        signature = (
            str(message.header.frame_id).lstrip("/"),
            round(float(message.pose.position.x), 4),
            round(float(message.pose.position.y), 4),
            round(float(message.pose.position.z), 4),
            round(float(message.pose.orientation.x), 5),
            round(float(message.pose.orientation.y), 5),
            round(float(message.pose.orientation.z), 5),
            round(float(message.pose.orientation.w), 5),
        )
        with self.lock:
            internally_resumed = self.resume_in_flight
            self.resume_in_flight = False
            internally_recovery = self.recovery_goal_in_flight
            self.recovery_goal_in_flight = False
            retrying_recoverable_failure = (
                self.machine.state
                in (STATE_RECOVERABLE_HOLD, STATE_FAILED_STOP)
                and self.machine.failure_recoverable
                and self.robot_linear_speed
                <= self.stationary_linear_limit
                and self.robot_angular_speed
                <= self.stationary_angular_limit
            )
            if retrying_recoverable_failure:
                # A fresh operator goal is an explicit retry. Release only
                # pre-motion scene failures while stationary; runtime/control
                # failures still require the reset service.
                self.machine.reset()
                self.command_pub.publish(Twist())
            self.current_goal = copy.deepcopy(message)
            changed = signature != self.current_goal_signature
            self.current_goal_signature = signature
            if not internally_resumed and not internally_recovery:
                self.logical_goal_epoch += 1
                self.attempted_passages.clear()
                self.auto_resume_count = 0
                self.navigation_failure_retry_count = 0
                self.last_navigation_failure_retry_sec = None
                self.supervisor_cancel_outstanding = False
                self.last_cancel_owner = ""
                self.last_cancel_reason = ""
                self._reset_finish_recovery(
                    rospy.Time.now().to_sec(), clear_stage=True
                )
            if (
                (changed or retrying_recoverable_failure)
                and not internally_recovery
            ):
                self.resume_goal = copy.deepcopy(message)
            if (
                (changed or retrying_recoverable_failure)
                and not self.machine.authority_requested
            ):
                # A new goal must not be classified against an older goal's
                # cached route while SBPL is still computing its first plan.
                self.path = ()
                self.path_receipt_sec = None
                self.route_source = "waiting_for_sbpl"
            elif changed:
                # A newly clicked goal becomes the post-passage destination,
                # but move_base must remain stopped until special authority
                # has been released.
                self._publish_supervisor_cancel(
                    "new_goal_during_special_authority"
                )

    def _reset_callback(self, _request):
        with self.lock:
            if (
                self.robot_linear_speed > self.stationary_linear_limit
                or self.robot_angular_speed > self.stationary_angular_limit
            ):
                return TriggerResponse(
                    False, "robot must be stationary before passage reset"
                )
            self.machine.reset(clear_competition_latch=True)
            self._reset_finish_recovery(
                rospy.Time.now().to_sec(), clear_stage=True
            )
            self.supervisor_cancel_outstanding = False
            self.last_cancel_owner = ""
            self.last_cancel_reason = ""
            self.command_pub.publish(Twist())
            return TriggerResponse(True, "passage supervisor reset")

    def _reset_scene_callback(self, _request):
        with self.lock:
            if self.navigation_active:
                return TriggerResponse(
                    False, "scene reset rejected: navigation_goal_active"
                )
            if (
                self.robot_linear_speed > self.stationary_linear_limit
                or self.robot_angular_speed > self.stationary_angular_limit
            ):
                return TriggerResponse(
                    False, "scene reset rejected: robot_not_stationary"
                )
        try:
            bridge_response = self.disc_scene_reset_client()
        except rospy.ServiceException as exc:
            return TriggerResponse(
                False, "scene reset bridge call failed: {}".format(exc)
            )
        if not bridge_response.success:
            return TriggerResponse(False, bridge_response.message)
        with self.lock:
            self.disc_scene_generation += 1
            self.machine.reset(clear_competition_latch=True)
            self.tracks = []
            self.live_gate_tracks = []
            self.disc_receipt_sec = None
            self.disc_input_fresh = False
            self.attempted_passages.clear()
            self.suppressed_passage_count = 0
            self.current_goal = None
            self.current_goal_id = ""
            self.current_goal_signature = None
            self.resume_goal = None
            self.supervisor_cancel_outstanding = False
            self.last_cancel_owner = ""
            self.last_cancel_reason = ""
            self.last_decision = None
            self.command_pub.publish(Twist())
            generation = self.disc_scene_generation
        return TriggerResponse(
            True,
            "passage/disc scene reset; expected_generation={}".format(
                generation
            ),
        )

    def _boundary_route(self):
        left = self.boundary.left_world
        right = self.boundary.right_world
        if len(left) < 2 or len(right) < 2:
            return ()
        count = min(len(left), len(right))
        centers = [
            (
                0.5 * (left[index][0] + right[index][0]),
                0.5 * (left[index][1] + right[index][1]),
            )
            for index in range(count)
        ]
        if self.robot_pose is not None:
            centers.sort(
                key=lambda point: (
                    point[0] - self.robot_pose.x
                ) ** 2
                + (
                    point[1] - self.robot_pose.y
                ) ** 2
            )
            centers.insert(0, self.robot_pose.xy)
        output = []
        for point in centers:
            if not output or math.hypot(
                point[0] - output[-1][0],
                point[1] - output[-1][1],
            ) > 0.01:
                output.append(point)
        return tuple(output) if len(output) >= 2 else ()

    def _selected_route(self, now_sec, boundary_fresh):
        if self.config.competition_two_disc_mode and boundary_fresh:
            boundary_route = self._boundary_route()
            if len(boundary_route) >= 2:
                return boundary_route, "yellow_centerline", True
        path_fresh = (
            self.path_receipt_sec is not None
            and now_sec - self.path_receipt_sec
            <= self.route_cache_timeout_sec
            and len(self.path) >= 2
        )
        if path_fresh:
            return self.path, self.route_source, True
        if boundary_fresh:
            boundary_route = self._boundary_route()
            if len(boundary_route) >= 2:
                return boundary_route, "yellow_centerline", True
        return (), "none", False

    @staticmethod
    def _twist(command):
        message = Twist()
        message.linear.x = float(command[0])
        message.linear.y = float(command[1])
        message.angular.z = float(command[2])
        return message

    @staticmethod
    def _point(x_value, y_value, z_value=0.04):
        point = Point()
        point.x = float(x_value)
        point.y = float(y_value)
        point.z = float(z_value)
        return point

    def _markers(self, stamp, decision):
        message = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        message.markers.append(clear)

        if (
            self.activation_zone_enabled
            and len(self.activation_zone_polygon) >= 3
        ):
            zone = Marker()
            zone.header.frame_id = self.activation_zone_frame
            zone.header.stamp = stamp
            zone.ns = "passage_activation_zone"
            zone.id = 0
            zone.type = Marker.LINE_STRIP
            zone.action = Marker.ADD
            zone.scale.x = 0.025
            zone.color.a = 0.95
            if not self.activation_zone_pose_available:
                zone.color.r = 0.6
                zone.color.g = 0.6
                zone.color.b = 0.6
            elif self.activation_zone_active:
                zone.color.r = 0.1
                zone.color.g = 1.0
                zone.color.b = 0.2
            else:
                zone.color.r = 1.0
                zone.color.g = 0.7
                zone.color.b = 0.0
            zone.points = [
                self._point(*point, z_value=0.03)
                for point in (
                    list(self.activation_zone_polygon)
                    + [self.activation_zone_polygon[0]]
                )
            ]
            message.markers.append(zone)

        plan = decision.bypass_plan
        reference_path = (
            self.machine.locked_path
            if len(self.machine.locked_path) >= 2
            else self.visualization_path
        )
        if plan is not None and len(reference_path) >= 2:
            from yellow_track_avoidance.disc_passage import cumulative_lengths
            from yellow_track_avoidance.passage_navigation import (
                _path_tangent_at_s,
                _point_at_s,
            )

            cumulative = cumulative_lengths(reference_path)
            marker = Marker()
            marker.header.frame_id = self.fixed_frame
            marker.header.stamp = stamp
            marker.ns = "passage_reference"
            marker.id = 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.018
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 1.0
            marker.color.a = 0.95
            for progress, lateral in zip(
                plan.reference_s_m, plan.reference_d_m
            ):
                base = _point_at_s(
                    reference_path, cumulative, progress
                )
                tangent = _path_tangent_at_s(
                    reference_path, cumulative, progress
                )
                normal = -tangent[1], tangent[0]
                marker.points.append(
                    self._point(
                        base[0] + lateral * normal[0],
                        base[1] + lateral * normal[1],
                    )
                )
            message.markers.append(marker)

        frame = decision.gate_frame
        if frame is not None:
            pair = Marker()
            pair.header.frame_id = self.fixed_frame
            pair.header.stamp = stamp
            pair.ns = "passage_gate"
            pair.id = 2
            pair.type = Marker.LINE_STRIP
            pair.action = Marker.ADD
            pair.scale.x = 0.020
            pair.color.r = 1.0
            pair.color.g = 0.7
            pair.color.b = 0.0
            pair.color.a = 0.95
            pair.points = [
                self._point(*frame.disc_first_xy),
                self._point(*frame.disc_second_xy),
            ]
            message.markers.append(pair)

        assessment = decision.assessment
        if assessment is not None:
            for index, projected in enumerate(
                assessment.projected_discs
            ):
                disc = Marker()
                disc.header.frame_id = self.fixed_frame
                disc.header.stamp = stamp
                disc.ns = "passage_obstacles"
                disc.id = 100 + index
                disc.type = Marker.CYLINDER
                disc.action = Marker.ADD
                disc.pose.position = self._point(
                    *projected.track.fixed_xy, z_value=0.025
                )
                disc.pose.orientation.w = 1.0
                disc.scale.x = disc.scale.y = 2.0 * self.config.disc_radius_m
                disc.scale.z = 0.05
                disc.color.r = 1.0
                disc.color.g = 0.2
                disc.color.b = 0.1
                disc.color.a = 0.70
                message.markers.append(disc)

        label = Marker()
        label.header.frame_id = self.fixed_frame
        label.header.stamp = stamp
        label.ns = "passage_state"
        label.id = 10
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        if self.robot_pose is not None:
            label.pose.position = self._point(
                self.robot_pose.x, self.robot_pose.y, 0.35
            )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.08
        label.color.r = (
            1.0
            if decision.state in (STATE_FAILED_STOP, STATE_DEGRADED_HOLD)
            else 0.2
        )
        label.color.g = (
            0.1
            if decision.state == STATE_FAILED_STOP
            else 1.0
        )
        label.color.b = 0.3
        label.color.a = 1.0
        zone_state = (
            "zone:off"
            if not self.activation_zone_enabled
            else (
                "zone:on"
                if self.activation_zone_active
                else (
                    "zone:outside"
                    if self.activation_zone_pose_available
                    else "zone:no_pose"
                )
            )
        )
        label.text = "{} / {} / {}".format(
            decision.state, decision.scene, zone_state
        )
        message.markers.append(label)
        return message

    @staticmethod
    def _finite_or_none(value):
        value = float(value)
        return value if math.isfinite(value) else None

    @staticmethod
    def _gate_pair_diagnostic_status(diagnostic):
        return {
            "disc_ids": list(diagnostic.pair_ids),
            "first": {
                "fixed_xy": list(diagnostic.first_xy),
                "estimated_diameter_m": diagnostic.first_diameter_m,
                "confidence": diagnostic.first_confidence,
                "age_sec": diagnostic.first_age_sec,
                "route_s_m": diagnostic.first_route_s_m,
                "route_d_m": diagnostic.first_route_d_m,
            },
            "second": {
                "fixed_xy": list(diagnostic.second_xy),
                "estimated_diameter_m": diagnostic.second_diameter_m,
                "confidence": diagnostic.second_confidence,
                "age_sec": diagnostic.second_age_sec,
                "route_s_m": diagnostic.second_route_s_m,
                "route_d_m": diagnostic.second_route_d_m,
            },
            "center_distance_m": diagnostic.center_distance_m,
            "estimated_edge_gap_m": diagnostic.edge_gap_m,
            "midpoint_xy": list(diagnostic.midpoint_xy),
            "midpoint_route_s_m": diagnostic.midpoint_route_s_m,
            "midpoint_route_offset_m": (
                diagnostic.midpoint_route_offset_m
            ),
            "longitudinal_mismatch_m": (
                diagnostic.longitudinal_mismatch_m
            ),
            "pair_angle_rad": diagnostic.pair_angle_rad,
            "route_angle_rad": diagnostic.route_angle_rad,
            "angle_error_rad": diagnostic.angle_error_rad,
            "takeover_distance_m": diagnostic.path_progress_ahead_m,
            "robot_progress_m": diagnostic.robot_progress_m,
            "predicted_left_clearance_m": (
                diagnostic.predicted_left_clearance_m
            ),
            "predicted_right_clearance_m": (
                diagnostic.predicted_right_clearance_m
            ),
            "predicted_min_clearance_m": (
                diagnostic.predicted_min_clearance_m
            ),
            "checks": [dict(item) for item in diagnostic.checks],
            "reason_codes": list(diagnostic.reason_codes),
            "accepted": diagnostic.accepted,
        }

    @staticmethod
    def _assessment_status(assessment):
        if assessment is None:
            return None
        return {
            "scene": assessment.scene,
            "key": list(assessment.key),
            "reason": assessment.reason,
            "nearest_ahead_m":
                PassageSupervisorNode._finite_or_none(
                    assessment.nearest_ahead_m
                ),
            "affecting_ids": list(assessment.affecting_ids),
            "projected_discs": [
                {
                    "memory_id": item.track.memory_id,
                    "s_m": item.path_s_m,
                    "d_m": item.lateral_m,
                    "ahead_m": item.ahead_m,
                    "age_sec": item.track.last_seen_age_sec,
                }
                for item in assessment.projected_discs
            ],
            "gate_pair_diagnostics": [
                PassageSupervisorNode._gate_pair_diagnostic_status(item)
                for item in assessment.gate_pair_diagnostics
            ],
        }

    @staticmethod
    def _plan_status(plan):
        if plan is None:
            return None
        return {
            "scene": plan.scene,
            "obstacle_ids": list(plan.obstacle_ids),
            "selected_side": (
                "left" if plan.selected_side > 0 else "right"
            ),
            "left_residual_m":
                PassageSupervisorNode._finite_or_none(
                    plan.left_residual_m
                ),
            "right_residual_m":
                PassageSupervisorNode._finite_or_none(
                    plan.right_residual_m
                ),
            "minimum_channel_width_m":
                PassageSupervisorNode._finite_or_none(
                    plan.minimum_channel_width_m
                ),
            "start_s_m": plan.start_s_m,
            "end_s_m": plan.end_s_m,
            "sample_count": len(plan.reference_s_m),
        }

    @staticmethod
    def _errors_status(errors):
        if errors is None:
            return None
        return {
            "lateral_m": errors.lateral_m,
            "yaw_rad": errors.yaw_rad,
            "progress_m": errors.progress_m,
            "reference_lateral_m": errors.reference_lateral_m,
            "minimum_direct_clearance_m":
                PassageSupervisorNode._finite_or_none(
                    errors.minimum_direct_clearance_m
            ),
        }

    @staticmethod
    def _swept_status(safety):
        if safety is None:
            return None
        return {
            "safe": safety.safe,
            "reason": safety.reason,
            "minimum_disc_clearance_m":
                PassageSupervisorNode._finite_or_none(
                    safety.minimum_disc_clearance_m
                ),
            "minimum_line_clearance_m":
                PassageSupervisorNode._finite_or_none(
                    safety.minimum_line_clearance_m
                ),
            "sample_count": safety.sample_count,
        }

    def _record_transition_and_failure(
        self,
        now_sec,
        decision,
        boundary_fresh,
        route_valid,
    ):
        previous_state = (
            "startup"
            if self.last_decision is None
            else self.last_decision.state
        )
        previous_reason = (
            ""
            if self.last_decision is None
            else self.last_decision.reason
        )
        if (
            previous_state != decision.state
            or previous_reason != decision.reason
        ):
            event = {
                "stamp": now_sec,
                "event": "PASSAGE_TRANSITION",
                "from": previous_state,
                "to": decision.state,
                "reason": decision.reason,
                "goal_id": self.current_goal_id,
                "navigation_active": self.navigation_active,
                "move_base_status": self.latest_move_base_status,
                "cancel_sent": decision.cancel_move_base,
                "resume_requested": decision.resume_move_base,
                "cancel_count": self.cancel_count,
                "resume_count": self.resume_count,
            }
            rospy.loginfo(
                "PASSAGE_TRANSITION from=%s to=%s reason=%s "
                "goal=%s cancel=%s resume=%s",
                previous_state,
                decision.state,
                decision.reason,
                self.current_goal_id,
                decision.cancel_move_base,
                decision.resume_move_base,
            )
            try:
                with open(self.event_log_path, "a") as stream:
                    stream.write(
                        json.dumps(
                            event, ensure_ascii=False, allow_nan=False
                        )
                        + "\n"
                    )
            except (IOError, OSError) as error:
                rospy.logwarn_throttle(
                    5.0,
                    "cannot append passage event log: %s",
                    error,
                )

        failure_states = {
            STATE_RECOVERABLE_HOLD,
            STATE_DEGRADED_HOLD,
            STATE_FAILED_STOP,
        }
        if decision.state not in failure_states:
            return
        failure_reason = (
            self.machine.failure_reason
            if self.machine.failure_reason
            else decision.reason
        )
        failure_key = (
            decision.state,
            failure_reason,
            self.current_goal_id,
        )
        if failure_key == self.last_failure_key:
            return
        self.last_failure_key = failure_key
        snapshot = {
            "stamp": now_sec,
            "failure_reason": failure_reason,
            "state": decision.state,
            "decision_reason": decision.reason,
            "scene_class": decision.scene,
            "goal_id": self.current_goal_id,
            "goal": (
                None
                if self.resume_goal is None
                else {
                    "frame_id": self.resume_goal.header.frame_id,
                    "x": self.resume_goal.pose.position.x,
                    "y": self.resume_goal.pose.position.y,
                    "orientation_z":
                        self.resume_goal.pose.orientation.z,
                    "orientation_w":
                        self.resume_goal.pose.orientation.w,
                }
            ),
            "navigation_active": self.navigation_active,
            "move_base_status": self.latest_move_base_status,
            "reference_source": self.route_source,
            "route_valid": route_valid,
            "boundary_fresh": boundary_fresh,
            "boundary_live_fresh": self.live_boundary_fresh,
            "boundary_reference_mode": self.boundary_reference_mode,
            "boundary": {
                "quality": self.boundary.quality,
                "inference": self.boundary_inference,
                "left_first": (
                    list(self.boundary.left_world[0])
                    if self.boundary.left_world
                    else None
                ),
                "right_first": (
                    list(self.boundary.right_world[0])
                    if self.boundary.right_world
                    else None
                ),
            },
            "tracks": [
                {
                    "memory_id": track.memory_id,
                    "fixed_xy": list(track.fixed_xy),
                    "confirmed": track.confirmed,
                    "phase": track.phase,
                    "last_seen_age_sec": track.last_seen_age_sec,
                }
                for track in self.tracks
            ],
            "assessment": self._assessment_status(
                decision.assessment
            ),
            "cancel_owner": self.last_cancel_owner,
            "cancel_reason": self.last_cancel_reason,
            "cancel_count": self.cancel_count,
            "resume_count": self.resume_count,
            "auto_resume_count": self.auto_resume_count,
        }
        self.last_failure_snapshot = snapshot
        self.last_failure_pub.publish(
            String(
                data=json.dumps(
                    snapshot, ensure_ascii=False, allow_nan=False
                )
            )
        )

    def _publish_status(
        self,
        now_sec,
        decision,
        robot_fresh,
        disc_fresh,
        boundary_fresh,
        route_valid,
        locked_tracks_live,
        locked_track_geometry_stable,
        failure_takeover_active,
        control_cycle_ms,
    ):
        activation_allowed = (
            not self.activation_zone_enabled
            or (
                self.activation_zone_pose_available
                and self.activation_zone_active
            )
        )
        command_is_zero = all(
            abs(value) <= 1e-9 for value in decision.command_body
        )
        locked_competition_pair = (
            self.config.competition_two_disc_mode
            and decision.authority_requested
        )
        safe_to_move = (
            decision.state
            not in (
                STATE_RECOVERABLE_HOLD,
                STATE_FAILED_STOP,
                STATE_DEGRADED_HOLD,
            )
            and robot_fresh
            and boundary_fresh
            and route_valid
            and activation_allowed
            and (disc_fresh or locked_competition_pair)
            and (
                locked_tracks_live
                or not decision.authority_requested
                or locked_competition_pair
            )
            and (
                locked_track_geometry_stable
                or not decision.authority_requested
                or locked_competition_pair
            )
            and (
                not decision.authority_requested
                or command_is_zero
                or (
                    decision.swept_safety is not None
                    and decision.swept_safety.safe
                )
            )
        )
        move_base_command_fresh = (
            self.latest_move_base_command is not None
            and self.move_base_cmd_receipt_sec is not None
            and now_sec - self.move_base_cmd_receipt_sec
            <= self.move_base_cmd_timeout_sec
        )
        status = {
            "stamp": now_sec,
            "enabled": self.enabled,
            "state": decision.state,
            "scene_class": decision.scene,
            "authority_requested": decision.authority_requested,
            "safe_to_move": safe_to_move,
            "reason": decision.reason,
            "stop_reason": (
                (
                    self.machine.failure_reason
                    if self.machine.failure_reason
                    else decision.reason
                )
                if decision.state
                in (
                    STATE_RECOVERABLE_HOLD,
                    STATE_FAILED_STOP,
                    STATE_DEGRADED_HOLD,
                )
                else ""
            ),
            "failure_recoverable": self.machine.failure_recoverable,
            "command_fresh": True,
            "robot_pose_fresh": robot_fresh,
            "disc_status_fresh": disc_fresh,
            "disc_timestamp_transform_drop_count": (
                self.disc_timestamp_transform_drop_count
            ),
            "disc_scene_generation": self.disc_scene_generation,
            "gate_visual_reference": {
                "updated_this_cycle": (
                    self.machine.gate_frame_live_updated
                ),
                "update_count": (
                    self.machine.gate_frame_live_update_count
                ),
                "last_update_age_sec": (
                    None
                    if self.machine.gate_frame_live_last_update_sec is None
                    else max(
                        0.0,
                        now_sec
                        - self.machine.gate_frame_live_last_update_sec,
                    )
                ),
            },
            "boundary_fresh": boundary_fresh,
            "boundary_live_fresh": self.live_boundary_fresh,
            "boundary_inference": self.boundary_inference,
            "boundary_reference_mode": self.boundary_reference_mode,
            "boundary_source_age_sec": (
                None
                if self.boundary_source_sec is None
                else max(0.0, now_sec - self.boundary_source_sec)
            ),
            "boundary_receipt_age_sec": (
                None
                if self.boundary_receipt_sec is None
                else max(0.0, now_sec - self.boundary_receipt_sec)
            ),
            "cached_corridor_age_sec": (
                None
                if self.last_trusted_boundary_source_sec is None
                else max(
                    0.0,
                    now_sec - self.last_trusted_boundary_source_sec,
                )
            ),
            "cached_corridor_inference": (
                self.last_trusted_boundary_inference
            ),
            "route_valid": route_valid,
            "reference_source": self.route_source,
            "activation_zone": {
                "enabled": self.activation_zone_enabled,
                "frame_id": self.activation_zone_frame,
                "polygon": [
                    list(point) for point in self.activation_zone_polygon
                ],
                "pose_available": self.activation_zone_pose_available,
                "active": self.activation_zone_active,
                "robot_xy": (
                    None
                    if self.activation_zone_pose is None
                    else list(self.activation_zone_pose)
                ),
            },
            "move_base_command_fresh": move_base_command_fresh,
            "normal_command_safety": self._swept_status(
                self.normal_command_safety
            ),
            "locked_tracks_live": locked_tracks_live,
            "locked_track_geometry_stable":
                locked_track_geometry_stable,
            "navigation_active": self.navigation_active,
            "navigation_failure_takeover_active":
                failure_takeover_active,
            "navigation_goal_id": self.current_goal_id,
            "move_base_status": self.latest_move_base_status,
            "cached_goal_valid": self.resume_goal is not None,
            "cancel_owner": self.last_cancel_owner,
            "cancel_reason": self.last_cancel_reason,
            "supervisor_cancel_outstanding":
                self.supervisor_cancel_outstanding,
            "scene_stable_sec": decision.scene_stable_sec,
            "control_cycle_ms": control_cycle_ms,
            "cancel_count": self.cancel_count,
            "resume_count": self.resume_count,
            "auto_resume_enabled": self.auto_resume_enabled,
            "auto_resume_count": self.auto_resume_count,
            "max_auto_resume_per_goal":
                self.max_auto_resume_per_goal,
            "retry_navigation_failures":
                self.retry_navigation_failures,
            "navigation_failure_retry_count":
                self.navigation_failure_retry_count,
            "max_navigation_failure_retries_per_goal":
                self.max_navigation_failure_retries_per_goal,
            "finish_recovery_enabled": self.finish_recovery_enabled,
            "finish_recovery_stage": self.finish_recovery_stage,
            "finish_recovery_action": self.finish_recovery_action,
            "finish_recovery_count": self.recovery_count,
            "recovery_short_goal_active": (
                self.recovery_short_goal_active
            ),
            "recovery_creep_active": self.recovery_creep_active,
            "recovery_lidar_age_sec": (
                None
                if self.recovery_lidar_receipt_sec is None
                else max(0.0, now_sec - self.recovery_lidar_receipt_sec)
            ),
            "recovery_lidar_forward_min_m": (
                self.recovery_lidar_forward_min_m
            ),
            "last_navigation_failure_retry_age_sec": (
                None
                if self.last_navigation_failure_retry_sec is None
                else max(
                    0.0,
                    now_sec - self.last_navigation_failure_retry_sec,
                )
            ),
            "logical_goal_epoch": self.logical_goal_epoch,
            "attempted_passage_count": len(self.attempted_passages),
            "suppressed_passage_count": self.suppressed_passage_count,
            "competition_finish_priority": (
                self.config.competition_finish_priority
            ),
            "competition_two_disc_mode": (
                self.config.competition_two_disc_mode
            ),
            "competition_pair_consumed": self.machine.pair_consumed,
            "competition_pair_disabled": self.machine.pair_disabled,
            "competition_special_attempt_count": (
                self.machine.special_attempt_count
            ),
            "resume_goal_available": self.resume_goal is not None,
            "last_failure_reason": (
                ""
                if self.last_failure_snapshot is None
                else self.last_failure_snapshot["failure_reason"]
            ),
            "last_failure": self.last_failure_snapshot,
            "assessment": self._assessment_status(decision.assessment),
            "channel": self._plan_status(decision.bypass_plan),
            "errors": self._errors_status(decision.errors),
            "swept_safety": self._swept_status(
                decision.swept_safety
            ),
            "gate": (
                None
                if decision.gate_frame is None
                else {
                    "pair_ids": list(decision.gate_frame.pair_ids),
                    "edge_gap_m": decision.gate_frame.edge_gap_m,
                    "midpoint_xy": list(
                        decision.gate_frame.midpoint_xy
                    ),
                    "yaw_ref": decision.gate_frame.yaw_ref,
                    "control": (
                        None
                        if decision.gate_diagnostics is None
                        else {
                            "projected_width_m":
                                decision.gate_diagnostics.projected_width_m,
                            "predicted_clearance_m":
                                decision.gate_diagnostics.predicted_clearance_m,
                            "direct_clearance_first_m":
                                decision.gate_diagnostics.direct_clearance_first_m,
                            "direct_clearance_second_m":
                                decision.gate_diagnostics.direct_clearance_second_m,
                        }
                    ),
                }
            ),
            "command": {
                "linear_x": decision.command_body[0],
                "linear_y": decision.command_body[1],
                "angular_z": decision.command_body[2],
            },
        }
        self.status_pub.publish(
            String(
                data=json.dumps(
                    status, ensure_ascii=False, allow_nan=False
                )
            )
        )

    def _timer_callback(self, _event):
        if rospy.is_shutdown():
            return
        cycle_start = time.perf_counter()
        with self.lock:
            now_sec = rospy.Time.now().to_sec()
            robot_fresh = (
                self.robot_pose is not None
                and self.robot_receipt_sec is not None
                and now_sec - self.robot_receipt_sec
                <= self.robot_timeout_sec
            )
            disc_fresh = (
                self.disc_receipt_sec is not None
                and now_sec - self.disc_receipt_sec
                <= self.disc_timeout_sec
                and self.disc_input_fresh
                and self.disc_fixed_transform_available
            )
            boundary_quality_threshold = (
                self.minimum_single_boundary_quality
                if self.boundary_inference != "none"
                else self.minimum_boundary_quality
            )
            live_boundary_fresh = (
                self.boundary_receipt_sec is not None
                and self.boundary_source_sec is not None
                and now_sec - self.boundary_receipt_sec
                <= self.boundary_timeout_sec
                and max(0.0, now_sec - self.boundary_source_sec)
                <= self.boundary_source_timeout_sec
                and self.boundary.fresh
                and self.boundary.quality >= boundary_quality_threshold
            )
            self.live_boundary_fresh = live_boundary_fresh
            boundary_fresh = live_boundary_fresh
            boundary = replace(self.boundary, fresh=live_boundary_fresh)
            self.boundary_reference_mode = (
                "live" if live_boundary_fresh else "none"
            )
            cached_corridor_valid = odom_corridor_cache_valid(
                now_sec=now_sec,
                source_sec=self.last_trusted_boundary_source_sec,
                cached_robot_xy=self.last_trusted_boundary_robot_xy,
                robot=self.robot_pose if robot_fresh else None,
                maximum_age_sec=self.last_corridor_max_age_sec,
                maximum_propagation_sec=self.odom_propagation_max_sec,
                maximum_distance_m=self.max_blind_distance_m,
            )
            if (
                not live_boundary_fresh
                and self.config.competition_finish_priority
                and self.machine.authority_requested
                and cached_corridor_valid
                and self.last_trusted_boundary is not None
            ):
                boundary = replace(
                    self.last_trusted_boundary, fresh=True
                )
                boundary_fresh = True
                self.boundary_reference_mode = "odom_cached"
            (
                self.activation_zone_pose_available,
                self.activation_zone_active,
                self.activation_zone_pose,
            ) = self._activation_zone_state()
            activation_allowed = (
                not self.activation_zone_enabled
                or (
                    self.activation_zone_pose_available
                    and self.activation_zone_active
                )
            )
            selected_path, route_source, route_valid = (
                self._selected_route(now_sec, boundary_fresh)
            )
            self.route_source = route_source
            self.visualization_path = selected_path
            self.normal_command_safety = self._normal_swept_safety(
                now_sec=now_sec,
                robot_fresh=robot_fresh,
                boundary=boundary,
                boundary_fresh=boundary_fresh,
                path=selected_path,
                route_valid=route_valid,
            )
            locked_ids = {
                track.memory_id for track in self.machine.locked_obstacles
            }
            live_by_id = {
                track.memory_id: track for track in self.tracks
            }
            live_ids = set(live_by_id)
            locked_tracks_live = (
                not locked_ids or locked_ids.issubset(live_ids)
            )
            locked_track_geometry_stable = locked_tracks_live
            if locked_tracks_live:
                for locked in self.machine.locked_obstacles:
                    current = live_by_id.get(locked.memory_id)
                    if (
                        current is not None
                        and current.last_seen_age_sec
                        <= self.config.track_fresh_sec
                        and math.hypot(
                            current.fixed_xy[0] - locked.fixed_xy[0],
                            current.fixed_xy[1] - locked.fixed_xy[1],
                        )
                        > self.config.locked_track_drift_stop_m
                    ):
                        locked_track_geometry_stable = False
                        break
            critical_inputs_fresh = (
                robot_fresh
                and boundary_fresh
                and activation_allowed
                and (
                    disc_fresh
                    if not (
                        self.config.competition_two_disc_mode
                        and self.machine.authority_requested
                    )
                    else True
                )
                and (
                    locked_tracks_live
                    if not self.config.competition_two_disc_mode
                    else True
                )
                and (
                    locked_track_geometry_stable
                    if not self.config.competition_two_disc_mode
                    else True
                )
            )
            failure_takeover_active = (
                not self.navigation_active
                and self.navigation_failure_receipt_sec is not None
                and now_sec - self.navigation_failure_receipt_sec
                <= self.navigation_failure_grace_sec
                and self.resume_goal is not None
            )
            mission_active = (
                self.enabled
                and activation_allowed
                and (
                    self.navigation_active
                    or failure_takeover_active
                )
            )
            decision = self.machine.update(
                now_sec=now_sec,
                robot=self.robot_pose if robot_fresh else None,
                tracks=self.tracks,
                path=selected_path,
                boundary=boundary,
                navigation_active=self.navigation_active,
                measured_linear_speed_mps=self.robot_linear_speed,
                measured_angular_speed_radps=self.robot_angular_speed,
                critical_inputs_fresh=critical_inputs_fresh,
                mission_active=mission_active,
                live_gate_tracks=(
                    self.live_gate_tracks if disc_fresh else ()
                ),
            )
            decision = self._finish_recovery_override(
                now_sec=now_sec,
                decision=decision,
                robot_fresh=robot_fresh,
                boundary=boundary,
                boundary_fresh=boundary_fresh,
                path=selected_path,
                route_valid=route_valid,
            )

            orphaned_goal = (
                decision.state == STATE_NORMAL
                and self.supervisor_cancel_outstanding
                and self.resume_goal is not None
                and not self.navigation_active
                and not decision.resume_move_base
            )
            if orphaned_goal:
                self.machine.fail("orphaned_navigation_goal")
                decision = self.machine.update(
                    now_sec,
                    self.robot_pose if robot_fresh else None,
                    self.tracks,
                    selected_path,
                    boundary,
                    False,
                    self.robot_linear_speed,
                    self.robot_angular_speed,
                    critical_inputs_fresh,
                )

            if decision.cancel_move_base:
                if self.resume_goal is None and self.current_goal is not None:
                    self.resume_goal = copy.deepcopy(self.current_goal)
                if self.resume_goal is None:
                    self.machine.fail("resume_goal_missing")
                    decision = self.machine.update(
                        now_sec,
                        self.robot_pose if robot_fresh else None,
                        self.tracks,
                        (),
                        boundary,
                        False,
                        self.robot_linear_speed,
                        self.robot_angular_speed,
                        critical_inputs_fresh,
                    )
                else:
                    attempt_key = passage_attempt_key(
                        self.logical_goal_epoch, decision.assessment
                    )
                    repeated = register_passage_attempt(
                        self.attempted_passages, attempt_key
                    )
                    if self.config.competition_two_disc_mode:
                        # The state machine owns the explicit two-attempt
                        # budget and 2 s cooldown for the one competition pair.
                        repeated = False
                    if repeated:
                        self.suppressed_passage_count += 1
                        self.machine.reset()
                        decision = replace(
                            decision,
                            state=STATE_NORMAL,
                            scene=SCENE_NORMAL,
                            authority_requested=False,
                            command_body=(0.0, 0.0, 0.0),
                            reason=(
                                "same_goal_passage_takeover_suppressed"
                            ),
                            cancel_move_base=False,
                            resume_move_base=False,
                        )
                    else:
                        self._publish_supervisor_cancel(decision.reason)

            if (
                decision.resume_move_base
                and self.resume_goal is not None
            ):
                automatic = (
                    decision.reason
                    == "recoverable_hold_cleared_resuming_goal"
                )
                cooldown_ready = (
                    self.supervisor_cancel_sec is None
                    or now_sec - self.supervisor_cancel_sec
                    >= self.auto_resume_cooldown_sec
                )
                auto_resume_allowed = (
                    self.auto_resume_enabled
                    and self.auto_resume_count
                    < self.max_auto_resume_per_goal
                    and cooldown_ready
                )
                if automatic and not auto_resume_allowed:
                    reason = (
                        "auto_resume_disabled"
                        if not self.auto_resume_enabled
                        else (
                            "auto_resume_budget_exhausted"
                            if self.auto_resume_count
                            >= self.max_auto_resume_per_goal
                            else "auto_resume_cooldown"
                        )
                    )
                    self.machine.fail(reason)
                    decision = self.machine.update(
                        now_sec,
                        self.robot_pose if robot_fresh else None,
                        self.tracks,
                        selected_path,
                        boundary,
                        False,
                        self.robot_linear_speed,
                        self.robot_angular_speed,
                        critical_inputs_fresh,
                    )
                else:
                    self._publish_resume_goal(
                        now_sec, automatic=automatic
                    )

            # move_base treats an exhausted global-planning attempt as a
            # terminal goal failure. In finish-first competition mode that is
            # worse than retrying: transient scan marks or localization jitter
            # would otherwise leave the vehicle stopped forever. Reissue only
            # terminal planner failures; an operator cancel/preempt and a
            # successful goal are intentionally excluded by the status callback.
            retry_budget_available = (
                self.max_navigation_failure_retries_per_goal == 0
                or self.navigation_failure_retry_count
                < self.max_navigation_failure_retries_per_goal
            )
            navigation_failure_retry_due = (
                self.retry_navigation_failures
                and not self.finish_recovery_enabled
                and self.config.competition_finish_priority
                and retry_budget_available
                and not self.navigation_active
                and self.navigation_failure_receipt_sec is not None
                and now_sec - self.navigation_failure_receipt_sec
                >= self.navigation_failure_retry_cooldown_sec
                and self.resume_goal is not None
                and not self.supervisor_cancel_outstanding
                and not decision.authority_requested
                and not decision.cancel_move_base
                and not decision.resume_move_base
            )
            if navigation_failure_retry_due:
                if self._publish_resume_goal(now_sec, automatic=False):
                    self.navigation_failure_retry_count += 1
                    self.last_navigation_failure_retry_sec = now_sec
                    self.last_cancel_owner = "navigation_failure_retry"
                    self.last_cancel_reason = "move_base_failed_retrying_goal"
                    # Keep the old key as a debounce guard until move_base
                    # reports the newly submitted goal active. Otherwise its
                    # still-latched ABORTED status would cause a 20 Hz retry.
                    self.navigation_failure_receipt_sec = None

            command = (
                decision.command_body
                if self.enabled and decision.authority_requested
                else (0.0, 0.0, 0.0)
            )
            self.command_pub.publish(self._twist(command))

            stamp = rospy.Time.from_sec(now_sec)
            self.marker_pub.publish(self._markers(stamp, decision))
            control_cycle_ms = 1000.0 * (
                time.perf_counter() - cycle_start
            )
            self._record_transition_and_failure(
                now_sec,
                decision,
                boundary_fresh,
                route_valid,
            )
            self._publish_status(
                now_sec,
                decision,
                robot_fresh,
                disc_fresh,
                boundary_fresh,
                route_valid,
                locked_tracks_live,
                locked_track_geometry_stable,
                failure_takeover_active,
                control_cycle_ms,
            )
            self.last_decision = decision

    def _shutdown(self):
        try:
            self.timer.shutdown()
        except Exception:
            pass
        for _ in range(5):
            try:
                self.command_pub.publish(Twist())
                rospy.sleep(0.03)
            except Exception:
                break


def main():
    rospy.init_node("yellow_track_passage_supervisor")
    PassageSupervisorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
