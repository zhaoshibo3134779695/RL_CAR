#!/usr/bin/env python3
"""Run the validated Stage-1 and Stage-2 missions as isolated ROS sessions.

This process is deliberately not a ROS velocity controller.  It starts the
existing one-command launch for Stage 1, watches the latched runner status,
shuts that complete ROS graph down, and only then starts the independently
validated Stage-2 launch.  Therefore the two move_base instances, maps, TF
trees, lidar captures and chassis listeners can never overlap.

The combined runner owns sequencing only.  All perception, planning and
motion parameters continue to come from the two formal stage launch files so
that a successfully tested single-stage configuration cannot silently diverge
from the Stage-1 + Stage-2 configuration.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import yaml

from yellow_track_avoidance.stage12_auto import (
    build_stage_command,
    runner_outcome,
)


PACKAGE = "yellow_track_avoidance"
STAGES = (
    {
        "name": "stage1",
        "launch": "competition_stage1_auto.launch",
        "status_topic": "/yellow_track_stage1_auto_runner/status",
        # Exposure selection, corridor locking, two-disc confirmation and the
        # lidar goal capture all happen before motion is released.  The old
        # 35 s limit could kill a healthy visual startup while RViz still had
        # no frozen yellow-line markers.  This is a maximum failure timeout,
        # not a fixed delay: a ready scene starts navigating immediately.
        "startup_timeout": 90.0,
        "navigation_timeout": 120.0,
        "outer_timeout": 230.0,
        "perception_view": True,
    },
    {
        "name": "stage2",
        "launch": "competition_stage2_auto.launch",
        "status_topic": "/yellow_track_stage2_auto_runner/status",
        "startup_timeout": 45.0,
        "navigation_timeout": 90.0,
        "outer_timeout": 145.0,
        "perception_view": None,
    },
)


class MissionError(RuntimeError):
    pass


def _status_from_rostopic(topic, timeout_sec=2.0):
    """Read one latched std_msgs/String status without keeping a ROS node alive."""

    try:
        result = subprocess.run(
            ["rostopic", "echo", "-n", "1", topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        document = next(yaml.safe_load_all(result.stdout))
        data = document.get("data") if isinstance(document, dict) else None
        return json.loads(data) if isinstance(data, str) else None
    except (StopIteration, TypeError, ValueError, yaml.YAMLError):
        return None


def _master_is_up():
    try:
        result = subprocess.run(
            ["rosnode", "list"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _tail(path, lines=35):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return "".join(stream.readlines()[-lines:])
    except OSError:
        return ""


class SequentialMission:
    def __init__(self, args):
        self.args = args
        self.child = None
        self.child_log = None
        self.interrupted = False
        self.results = []
        self.log_dir = tempfile.mkdtemp(prefix="yellow_track_stage12_")

    def request_stop(self, _signum=None, _frame=None):
        self.interrupted = True
        self.stop_child()

    def start_child(self, stage):
        command = build_stage_command(
            stage["launch"],
            stage["startup_timeout"],
            stage["navigation_timeout"],
            start_rviz=self.args.start_rviz,
            start_perception_view=stage["perception_view"],
        )
        log_path = os.path.join(self.log_dir, stage["name"] + ".log")
        self.child_log = open(log_path, "w", encoding="utf-8")
        print("[总控] 启动 {}: {}".format(stage["name"].upper(), " ".join(command)))
        print("[总控] 子流程日志: {}".format(log_path))
        sys.stdout.flush()
        self.child = subprocess.Popen(
            command,
            stdout=self.child_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop_child(self):
        child = self.child
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=2.0)
        self.child = None
        if self.child_log is not None:
            self.child_log.close()
            self.child_log = None

    def wait_master_down(self, timeout_sec=8.0):
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if not _master_is_up():
                return True
            time.sleep(0.2)
        return not _master_is_up()

    def run_stage(self, stage):
        self.start_child(stage)
        started = time.monotonic()
        last_state = None
        latest = None
        try:
            while not self.interrupted:
                if self.child.poll() is not None:
                    raise MissionError(
                        "{}进程提前退出(code={})\n{}".format(
                            stage["name"],
                            self.child.returncode,
                            _tail(os.path.join(self.log_dir, stage["name"] + ".log")),
                        )
                    )
                latest = _status_from_rostopic(stage["status_topic"])
                if latest:
                    state = str(latest.get("state", "UNKNOWN"))
                    if state != last_state:
                        print(
                            "[总控] {} 状态: {} ({})".format(
                                stage["name"].upper(),
                                state,
                                latest.get("reason", ""),
                            )
                        )
                        sys.stdout.flush()
                        last_state = state
                    outcome = runner_outcome(latest)
                    if outcome == "success":
                        elapsed = time.monotonic() - started
                        self.results.append(
                            {
                                "stage": stage["name"],
                                "outcome": "SUCCEEDED",
                                "elapsed_sec": elapsed,
                                "runner_status": latest,
                            }
                        )
                        print(
                            "[总控] {} 完成，用时 {:.1f}s".format(
                                stage["name"].upper(), elapsed
                            )
                        )
                        sys.stdout.flush()
                        # Give move_base's terminal zero command time to reach
                        # the chassis before tearing down this isolated graph.
                        time.sleep(self.args.success_settle_sec)
                        return
                    if outcome == "failure":
                        raise MissionError(
                            "{}失败: {}".format(
                                stage["name"], latest.get("reason", "unknown")
                            )
                        )
                if time.monotonic() - started > stage["outer_timeout"]:
                    raise MissionError("{}总控等待超时".format(stage["name"]))
                time.sleep(self.args.poll_sec)
        finally:
            self.stop_child()

    def run(self):
        for index, stage in enumerate(STAGES):
            if self.interrupted:
                raise KeyboardInterrupt
            self.run_stage(stage)
            if self.interrupted:
                raise KeyboardInterrupt
            if not self.wait_master_down():
                raise MissionError(
                    "{}关闭后ROS master仍在线，拒绝启动下一阶段".format(
                        stage["name"]
                    )
                )
            if index + 1 < len(STAGES):
                print(
                    "[总控] Stage 1 ROS图已清空；等待硬件端口释放后启动Stage 2"
                )
                sys.stdout.flush()
                time.sleep(self.args.transition_sec)

        summary_path = os.path.join(self.log_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as stream:
            json.dump(self.results, stream, ensure_ascii=False, indent=2)
        print("[总控] Stage 1 + Stage 2 全流程成功")
        print("[总控] 汇总: {}".format(summary_path))
        return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Stage 1 and Stage 2 sequentially with isolated ROS graphs"
    )
    parser.add_argument(
        "--no-rviz", dest="start_rviz", action="store_false", default=True
    )
    parser.add_argument("--poll-sec", type=float, default=0.35)
    parser.add_argument("--success-settle-sec", type=float, default=1.0)
    parser.add_argument("--transition-sec", type=float, default=0.5)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the two commands without starting ROS or the chassis",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.plan_only:
        for stage in STAGES:
            print(
                " ".join(
                    build_stage_command(
                        stage["launch"],
                        stage["startup_timeout"],
                        stage["navigation_timeout"],
                        start_rviz=args.start_rviz,
                        start_perception_view=stage["perception_view"],
                    )
                )
            )
        return 0

    if _master_is_up():
        print(
            "拒绝启动：当前已有ROS master。请先关闭旧Stage/导航进程，避免重复速度源。",
            file=sys.stderr,
        )
        return 2

    mission = SequentialMission(args)
    signal.signal(signal.SIGINT, mission.request_stop)
    signal.signal(signal.SIGTERM, mission.request_stop)
    try:
        return mission.run()
    except KeyboardInterrupt:
        print("[总控] 用户中止，当前阶段已关闭", file=sys.stderr)
        return 130
    except MissionError as error:
        mission.stop_child()
        print("[总控] 任务失败: {}".format(error), file=sys.stderr)
        print("[总控] 日志目录: {}".format(mission.log_dir), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
