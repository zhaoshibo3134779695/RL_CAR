#!/usr/bin/env python3
"""Search Stage-1 planner parameters with real sensing and no chassis motion."""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import yaml


PACKAGE = "yellow_track_avoidance"
LAUNCH = "competition_stage1_shadow_auto.launch"
RUNNER_STATUS = "/yellow_track_stage1_auto_runner/status"
SHADOW_STATUS = "/cmd_vel_listener/status"
OBSERVER_STATUS = "/yellow_track_stage1_shadow_observer/status"


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            return stream.read()
    except OSError:
        return ""


def _terminal_from_log(log_text):
    if "Stage-1 automatic mission reached its frozen goal" in log_text:
        return {"state": "SUCCEEDED", "reason": "stage1_goal_reached"}
    if "Stage-1 automatic mission failed in move_base" in log_text:
        return {"state": "FAILED", "reason": "move_base_failed"}
    if "Stage-1 navigation timed out" in log_text:
        return {"state": "FAILED", "reason": "navigation_timeout"}
    if "Stage-1 startup timed out" in log_text:
        return {"state": "FAILED", "reason": "startup_timeout"}
    if "Aborting because a valid control could not be found" in log_text:
        return {"state": "FAILED", "reason": "move_base_no_valid_control"}
    if "Aborting because a valid plan could not be found" in log_text:
        return {"state": "FAILED", "reason": "move_base_no_valid_plan"}
    return None


def _final_odom_from_log(log_text):
    matches = re.findall(r"automatic mission failed in move_base; odom=(\{[^\n]+\})", log_text)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except (TypeError, ValueError):
        return None


def _last_json_log(log_text, label):
    matches = re.findall(label + r" (\{[^\n]+\})", log_text)
    for encoded in reversed(matches):
        try:
            return json.loads(encoded)
        except (TypeError, ValueError):
            continue
    return None


def _topic_json(topic, timeout_sec=1.2):
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
        value = document.get("data") if isinstance(document, dict) else None
        return json.loads(value) if isinstance(value, str) else None
    except (StopIteration, TypeError, ValueError, yaml.YAMLError):
        return None


def _master_up():
    try:
        return subprocess.run(
            ["rosnode", "list"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _wait_master_down(timeout_sec=10.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not _master_up():
            return True
        time.sleep(0.2)
    return not _master_up()


def _stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=2.0)


def _candidate_table():
    # Follow-up to the controlled topology A/B.  Synchronized closures changed
    # one failure into two successes and one failure.  Geometry logging showed
    # the failing predicted path repeatedly sitting at ~9 mm from the 90 mm
    # collision circle: exactly the former 5+5 mm soft edge and one 10 mm
    # costmap cell.  This candidate changes only that desired separation to
    # ~20 mm; no collision radius or hard safety envelope is reduced.
    common_teb = {
        "max_vel_y": 0.15,
        "acc_lim_y": 0.30,
        "max_vel_theta": 0.05,
        "acc_lim_theta": 0.10,
        "max_global_plan_lookahead_dist": 0.90,
        "weight_viapoint": 1.0,
        "feasibility_check_no_poses": 5,
        "dt_ref": 0.15,
        "no_inner_iterations": 10,
        "no_outer_iterations": 6,
        "min_obstacle_dist": 0.015,
        "inflation_dist": 0.015,
    }
    return (
        {
            "name": "topology_synced_soft_clearance020",
            "teb_radius": 0.09,
            "local_radius": 0.09,
            "global_inflation": 0.24,
            "global_scaling": 20.0,
            "include_passage_closures_in_teb": True,
            "teb": dict(common_teb),
        },
    )


class Search:
    def __init__(self, args):
        self.args = args
        self.root = tempfile.mkdtemp(prefix="yellow_track_shadow_search_")
        self.results = []
        self.child = None
        self.stop_requested = False
        with open(args.base_config, "r", encoding="utf-8") as stream:
            self.base_config = yaml.safe_load(stream) or {}

    def request_stop(self, _signum=None, _frame=None):
        self.stop_requested = True
        _stop_process(self.child)

    def _write_config(self, candidate, run_index):
        payload = json.loads(json.dumps(self.base_config))
        teb = payload.setdefault("TebLocalPlannerROS", {})
        teb.update(candidate["teb"])
        path = os.path.join(
            self.root, "{}_run{}.yaml".format(candidate["name"], run_index)
        )
        with open(path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, default_flow_style=False, sort_keys=False)
        return path

    def run_once(self, candidate, run_index):
        config_path = self._write_config(candidate, run_index)
        log_path = os.path.join(
            self.root, "{}_run{}.log".format(candidate["name"], run_index)
        )
        command = [
            "roslaunch",
            "--sigint-timeout=5",
            "--sigterm-timeout=2",
            PACKAGE,
            LAUNCH,
            "start_rviz:=false",
            "start_perception_view:=false",
            "startup_timeout_sec:=" + str(self.args.startup_timeout_sec),
            "navigation_timeout_sec:=" + str(self.args.navigation_timeout_sec),
            "stage1_teb_guard_config:=" + config_path,
            "global_costmap_inflation_radius:="
            + str(candidate["global_inflation"]),
            "global_costmap_cost_scaling_factor:="
            + str(candidate["global_scaling"]),
            "teb_disc_collision_radius_m:=" + str(candidate["teb_radius"]),
            "local_disc_collision_radius_m:=" + str(candidate["local_radius"]),
            "include_passage_closures_in_teb:="
            + str(candidate["include_passage_closures_in_teb"]).lower(),
        ]
        print(
            "[影子搜索] {} 第{}轮: {}".format(
                candidate["name"], run_index, " ".join(command)
            )
        )
        sys.stdout.flush()
        started = time.monotonic()
        maximum_abs_yaw = 0.0
        latest_shadow = None
        terminal = None
        next_shadow_sample = started
        with open(log_path, "w", encoding="utf-8") as log_stream:
            child_environment = os.environ.copy()
            child_environment["PYTHONUNBUFFERED"] = "1"
            self.child = subprocess.Popen(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_environment,
            )
            deadline = started + self.args.outer_timeout_sec
            while not self.stop_requested and time.monotonic() < deadline:
                if self.child.poll() is not None:
                    terminal = {"state": "PROCESS_EXITED", "reason": "roslaunch_exit"}
                    break
                log_text = _read_text(log_path)
                terminal = _terminal_from_log(log_text)
                if terminal is not None:
                    break
                # Topic sampling is diagnostic only; terminal detection above
                # comes directly from the runner log and cannot be delayed by
                # a short-lived rostopic XMLRPC connection.
                if time.monotonic() >= next_shadow_sample:
                    next_shadow_sample = time.monotonic() + 2.0
                    shadow = _topic_json(SHADOW_STATUS, timeout_sec=0.8)
                    if shadow:
                        latest_shadow = shadow
                        maximum_abs_yaw = max(
                            maximum_abs_yaw,
                            abs(float(shadow.get("yaw_rad", 0.0))),
                        )
                time.sleep(0.25)
        _stop_process(self.child)
        self.child = None
        _wait_master_down()
        log_text = _read_text(log_path)
        final_odom = _final_odom_from_log(log_text)
        logged_shadow = _last_json_log(log_text, "SHADOW_STATE(?:_FINAL)?")
        geometry = _last_json_log(log_text, "SHADOW_GEOMETRY(?:_FINAL)?")
        if logged_shadow is not None:
            latest_shadow = logged_shadow
            maximum_abs_yaw = max(
                maximum_abs_yaw,
                abs(float(logged_shadow.get("maximum_abs_yaw_rad", 0.0))),
            )
        if final_odom is not None:
            maximum_abs_yaw = max(
                maximum_abs_yaw, abs(float(final_odom.get("yaw_rad", 0.0)))
            )
        result = {
            "candidate": candidate,
            "run_index": run_index,
            "state": str((terminal or {}).get("state", "TIMEOUT")).upper(),
            "reason": (terminal or {}).get("reason", "outer_timeout"),
            "elapsed_sec": time.monotonic() - started,
            "infeasible_count": log_text.count("trajectory is not feasible"),
            "maximum_abs_yaw_deg": math.degrees(maximum_abs_yaw),
            "shadow": latest_shadow,
            "geometry": geometry,
            "final_odom": final_odom,
            "runner": terminal,
            "log_path": log_path,
            "config_path": config_path,
        }
        self.results.append(result)
        print(
            "[影子搜索] 结果 {}: state={} infeasible={} max_yaw={:.2f}deg elapsed={:.1f}s".format(
                candidate["name"],
                result["state"],
                result["infeasible_count"],
                result["maximum_abs_yaw_deg"],
                result["elapsed_sec"],
            )
        )
        sys.stdout.flush()
        return result

    @staticmethod
    def _rank(result):
        candidate = result["candidate"]
        return (
            -float(candidate["local_radius"]),
            int(result["infeasible_count"]),
            float(result["maximum_abs_yaw_deg"]),
            float(result["elapsed_sec"]),
        )

    def run(self):
        if _master_up():
            raise RuntimeError("已有ROS master，拒绝开始影子搜索")
        table = _candidate_table()
        radii = []
        for candidate in table:
            radius = float(candidate["local_radius"])
            if radius not in radii:
                radii.append(radius)
        for radius in radii:
            successes = []
            for candidate in table:
                if self.stop_requested:
                    break
                if float(candidate["local_radius"]) != radius:
                    continue
                result = self.run_once(candidate, 1)
                if result["state"] == "SUCCEEDED":
                    successes.append(result)
            for first_success in sorted(successes, key=self._rank):
                candidate = first_success["candidate"]
                confirmations = [first_success]
                for run_index in range(2, self.args.confirmation_runs + 1):
                    if self.stop_requested:
                        break
                    result = self.run_once(candidate, run_index)
                    confirmations.append(result)
                    if result["state"] != "SUCCEEDED":
                        break
                if len(confirmations) == self.args.confirmation_runs and all(
                    item["state"] == "SUCCEEDED" for item in confirmations
                ):
                    summary_path = self._save_summary(first_success)
                    print(
                        "[影子搜索] 最佳参数连续{}轮成功: {}".format(
                            len(confirmations), candidate["name"]
                        )
                    )
                    print("[影子搜索] 最佳配置: " + first_success["config_path"])
                    print("[影子搜索] 汇总: " + summary_path)
                    return 0
        summary_path = self._save_summary(None)
        print("[影子搜索] 没有候选通过重复性确认，汇总: " + summary_path)
        return 1

    def _save_summary(self, best):
        path = os.path.join(self.root, "summary.json")
        payload = {
            "physical_chassis_enabled": False,
            "best": best,
            "results": self.results,
        }
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        return path


def parse_args(argv=None):
    default_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "competition_stage1_teb_guard_override.yaml",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=default_config)
    parser.add_argument("--startup-timeout-sec", type=float, default=90.0)
    parser.add_argument("--navigation-timeout-sec", type=float, default=90.0)
    parser.add_argument("--outer-timeout-sec", type=float, default=150.0)
    parser.add_argument("--confirmation-runs", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    search = Search(args)
    signal.signal(signal.SIGINT, search.request_stop)
    signal.signal(signal.SIGTERM, search.request_stop)
    try:
        return search.run()
    except (KeyboardInterrupt, RuntimeError) as error:
        search.request_stop()
        print("[影子搜索] 中止: {}".format(error), file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
