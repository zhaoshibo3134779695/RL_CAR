"""Pure readiness helpers for the one-command Stage-1 mission runner."""


def goal_is_ready(payload):
    """Return whether the frozen lidar goal is safe to hand to move_base."""

    payload = payload or {}
    return (
        payload.get("state") == "goal_ready"
        and bool(payload.get("world_ready", False))
        and bool(payload.get("world_stable", False))
        and bool(payload.get("goal_frozen", False))
        and not bool(payload.get("goal_sent", False))
        and int(payload.get("preview_plan_pose_count", 0)) > 1
        and not payload.get("last_error")
    )


def command_topology_reason(publishers, subscribers, motion_enabled):
    """Validate the single-authority Stage-1 /cmd_vel topology.

    ``publishers`` and ``subscribers`` are the node-name sequences returned by
    ``rosgraph.Master.getSystemState`` for ``/cmd_vel``.
    """

    publisher_set = set(publishers or ())
    subscriber_set = set(subscribers or ())
    if publisher_set != {"/move_base"}:
        return "cmd_vel_publishers_not_move_base_only"
    if motion_enabled and "/cmd_vel_listener" not in subscriber_set:
        return "chassis_subscriber_missing"
    if not motion_enabled and "/cmd_vel_listener" in subscriber_set:
        return "chassis_present_in_dry_run"
    return "ready"

