"""Shared data models. This module has no ROS dependency."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class MotionState(str, Enum):
    NORMAL = "NORMAL"
    APPROACH = "APPROACH"
    AVOID = "AVOID"
    RETURN = "RETURN"
    STOP = "STOP"


@dataclass
class DiskDetection:
    center_px: Tuple[float, float]
    center_m: Tuple[float, float]
    ellipse: Tuple[Tuple[float, float], Tuple[float, float], float]
    contour: np.ndarray
    estimated_diameter_m: float
    confidence: float
    stamp_sec: float
    track_id: int = -1
    confirmed: bool = False
    red_annulus_ratio: float = 0.0
    white_center_ratio: float = 0.0
    red_sector_coverage: float = 0.0
    white_center_contrast: float = 0.0
    proposal_source: str = ""
    candidate_id: int = -1
    contour_area_px: float = 0.0
    red_area_px: int = 0
    white_area_px: int = 0
    red_white_center_distance_px: float = 0.0
    inside_lane: Optional[bool] = None
    distance_to_left_line_px: float = float("nan")
    distance_to_right_line_px: float = float("nan")
    used_history: bool = False
    projection_diagnostics: Dict = field(default_factory=dict)


@dataclass
class TrackedDisk:
    track_id: int
    center_m: Tuple[float, float]
    center_px: Tuple[float, float]
    estimated_diameter_m: float
    confidence: float
    stamp_sec: float
    hits: int
    misses: int = 0


@dataclass
class BoundaryResult:
    rows_px: np.ndarray
    left_px: np.ndarray
    right_px: np.ndarray
    center_px: np.ndarray
    left_detected: bool
    right_detected: bool
    used_history: bool
    both_missing_frames: int
    lane_width_px: float
    quality: float
    observed_min_row_px: int
    observed_max_row_px: int


@dataclass
class SafeCorridor:
    forward_m: np.ndarray
    safe_left_m: np.ndarray
    safe_right_m: np.ndarray
    center_m: np.ndarray
    valid: bool
    minimum_width_m: float
    reason: str = ""


@dataclass
class PathCandidate:
    name: str
    points_m: np.ndarray
    offset_m: float
    direction: int
    collision_free: bool
    score: float
    costs: Dict[str, float] = field(default_factory=dict)
    minimum_boundary_clearance_m: float = 0.0
    minimum_obstacle_clearance_m: float = 0.0
    max_curvature: float = 0.0
    reject_reason: str = ""


@dataclass
class PlanningResult:
    candidates: List[PathCandidate]
    selected: Optional[PathCandidate]
    safe: bool
    left_space_m: float
    right_space_m: float
    reason: str = ""


@dataclass
class StateDecision:
    state: MotionState
    stop_reason: str
    locked_direction: int
    changed: bool


@dataclass
class ControlCommand:
    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0


@dataclass
class PipelineResult:
    stamp_sec: float
    state: MotionState
    stop_reason: str
    boundary: Optional[BoundaryResult]
    raw_disks: List[DiskDetection]
    tracked_disks: List[TrackedDisk]
    corridor: Optional[SafeCorridor]
    planning: Optional[PlanningResult]
    command: ControlCommand
    debug_images: Dict[str, np.ndarray]
    diagnostics: Dict[str, object]
