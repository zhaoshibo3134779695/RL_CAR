"""Pure readiness helpers for the one-command Stage-2 mission runner."""


def scene_is_ready(payload):
    """Return whether the lidar scene is frozen and complete."""

    payload = payload or {}
    return (
        payload.get("mode") == "FROZEN"
        and bool(payload.get("map_frozen", False))
        and not bool(payload.get("scan_updates_costmap", True))
        and int(payload.get("compact_obstacle_count", 0)) == 3
        and not payload.get("last_error")
    )


def goal_is_ready(payload):
    """Return whether the goal from the frozen scene is ready for move_base."""

    payload = payload or {}
    return (
        payload.get("state") == "goal_ready"
        and bool(payload.get("goal_frozen", False))
        and not bool(payload.get("goal_sent", False))
        and int(payload.get("preview_plan_pose_count", 0)) > 1
        and not payload.get("last_error")
    )


def command_topology_reason(publishers, subscribers, motion_enabled):
    """Validate the single-authority Stage-2 ``/cmd_vel`` topology."""

    publisher_set = set(publishers or ())
    subscriber_set = set(subscribers or ())
    if publisher_set != {"/move_base"}:
        return "cmd_vel_publishers_not_move_base_only"
    if motion_enabled and "/cmd_vel_listener" not in subscriber_set:
        return "chassis_subscriber_missing"
    if not motion_enabled and "/cmd_vel_listener" in subscriber_set:
        return "chassis_present_in_dry_run"
    return "ready"
