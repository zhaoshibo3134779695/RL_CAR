"""Ideal planar odometry used by the locked Stage-1 planner test.

This module intentionally has no ROS dependency.  The ROS wrapper integrates
move_base velocity commands with this kinematic model while the physical
chassis command consumer is absent.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlanarState:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0


def normalized_angle(angle_rad):
    """Return an angle in [-pi, pi)."""

    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def integrate_body_twist(state, vx_mps, vy_mps, wz_radps, dt_sec):
    """Integrate one holonomic body-frame twist into the odom frame."""

    dt = max(0.0, float(dt_sec))
    if dt == 0.0:
        return state
    yaw = float(state.yaw_rad)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    world_vx = cosine * float(vx_mps) - sine * float(vy_mps)
    world_vy = sine * float(vx_mps) + cosine * float(vy_mps)
    return PlanarState(
        x_m=float(state.x_m) + world_vx * dt,
        y_m=float(state.y_m) + world_vy * dt,
        yaw_rad=normalized_angle(yaw + float(wz_radps) * dt),
    )
