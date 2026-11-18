"""Pure helpers for the sequential Stage-1 -> Stage-2 mission launcher."""


TERMINAL_SUCCESS = "SUCCEEDED"
TERMINAL_FAILURES = frozenset(("FAILED", "ABORTED", "REJECTED", "LOST"))


def runner_outcome(payload):
    """Return ``success``, ``failure`` or ``None`` for a runner status payload."""

    payload = payload or {}
    state = str(payload.get("state", "")).upper()
    if state == TERMINAL_SUCCESS:
        return "success"
    if state in TERMINAL_FAILURES:
        return "failure"
    return None


def build_stage_command(
    launch_file,
    startup_timeout_sec,
    navigation_timeout_sec,
    start_rviz=True,
    start_perception_view=None,
):
    """Build a command that reuses one formal, already-validated stage."""

    command = [
        "roslaunch",
        "--sigint-timeout=5",
        "--sigterm-timeout=2",
        "yellow_track_avoidance",
        str(launch_file),
        "enable_motion:=true",
        "start_rviz:=" + ("true" if start_rviz else "false"),
        "startup_timeout_sec:=" + str(float(startup_timeout_sec)),
        "navigation_timeout_sec:=" + str(float(navigation_timeout_sec)),
    ]
    if start_perception_view is not None:
        command.append(
            "start_perception_view:="
            + ("true" if start_perception_view else "false")
        )
    return command
