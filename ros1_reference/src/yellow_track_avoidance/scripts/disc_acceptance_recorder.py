#!/usr/bin/env python3
"""Record current-frame disc detections and generate acceptance statistics."""

import csv
import json
import math
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from yellow_track_avoidance.config import load_config


CSV_FIELDS = [
    "timestamp",
    "frame_id",
    "scene_session_id",
    "scene_name",
    "disc_present",
    "expected_disc_count",
    "position_left_center_right",
    "distance_near_middle_far",
    "occlusion_level",
    "reflection_level",
    "near_yellow_line",
    "left_line_detected",
    "right_line_detected",
    "disc_detected",
    "disc_count",
    "disc_center_x_px",
    "disc_center_y_px",
    "disc_major_axis_px",
    "disc_minor_axis_px",
    "disc_angle_deg",
    "disc_area_px",
    "disc_confidence",
    "red_area_px",
    "white_area_px",
    "red_mask_area_px",
    "white_mask_area_px",
    "red_white_center_distance_px",
    "candidate_count",
    "rejected_candidate_count",
    "disc_inside_lane",
    "distance_to_left_line_px",
    "distance_to_right_line_px",
    "processing_time_ms",
    "used_history",
    "motion_enabled",
    "detections_json",
    "candidates_json",
]


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_float(value):
    if value in (None, "", "None", "null"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values else None


def _std(values):
    return float(np.std(values, ddof=0)) if values else None


class DiscAcceptanceRecorder:
    def __init__(self):
        default_output = os.path.expanduser("~/.ros/yellow_disc_acceptance")
        self.output_dir = Path(rospy.get_param("~output_dir", default_output)).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "disc_frames.csv"
        self.report_path = self.output_dir / "disc_summary.json"
        # The launch file always supplies a package-relative config path. Do
        # not fall back to the commissioning workstation's absolute path.
        config_path = rospy.get_param("~config_file")
        self.config = load_config(config_path)
        self.thresholds = dict(self.config.get("disc_acceptance", {}))
        self.lock = threading.RLock()
        self.recording = False
        self.stream = None
        self.writer = None
        self.labels = {}
        self.scene_session_id = ""
        self.last_frame_key = None
        self.frames_this_session = 0

        topic = rospy.get_param(
            "~disc_status_topic", "/yellow_track_avoidance/debug/disc_status"
        )
        self.subscriber = rospy.Subscriber(topic, String, self._status_callback, queue_size=100)
        self.start_service = rospy.Service("~start", Trigger, self._start)
        self.stop_service = rospy.Service("~stop", Trigger, self._stop)
        self.report_service = rospy.Service("~report", Trigger, self._report)
        rospy.on_shutdown(self._close)
        if _as_bool(rospy.get_param("~auto_start", False)):
            response = self._start(None)
            if not response.success:
                rospy.logerr("disc recorder auto-start failed: %s", response.message)
        rospy.loginfo(
            "disc acceptance recorder ready: topic=%s output=%s",
            topic,
            self.output_dir,
        )

    def _read_labels(self):
        scene_name = str(rospy.get_param("~scene_name", "")).strip()
        if not scene_name:
            raise ValueError("~scene_name must be set before starting")
        return {
            "scene_name": scene_name,
            "disc_present": _as_bool(rospy.get_param("~disc_present", False)),
            "expected_disc_count": max(
                0, int(rospy.get_param("~expected_disc_count", 1))
            ),
            "position_left_center_right": str(
                rospy.get_param("~position_left_center_right", "unspecified")
            ),
            "distance_near_middle_far": str(
                rospy.get_param("~distance_near_middle_far", "unspecified")
            ),
            "occlusion_level": str(rospy.get_param("~occlusion_level", "0")),
            "reflection_level": str(
                rospy.get_param("~reflection_level", "normal")
            ),
            "near_yellow_line": _as_bool(
                rospy.get_param("~near_yellow_line", False)
            ),
        }

    def _start(self, _request):
        with self.lock:
            if self.recording:
                return TriggerResponse(False, "already recording")
            try:
                self.labels = self._read_labels()
            except ValueError as exc:
                return TriggerResponse(False, str(exc))
            file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            self.stream = self.csv_path.open("a", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.stream, fieldnames=CSV_FIELDS)
            if not file_exists:
                self.writer.writeheader()
            self.scene_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.last_frame_key = None
            self.frames_this_session = 0
            self.recording = True
            message = "recording {} -> {}".format(
                self.labels["scene_name"], self.csv_path
            )
            rospy.loginfo(message)
            return TriggerResponse(True, message)

    def _stop(self, _request):
        with self.lock:
            if not self.recording:
                return TriggerResponse(False, "not recording")
            scene = self.labels.get("scene_name", "")
            count = self.frames_this_session
            self.recording = False
            self._close_stream()
            self._generate_report_locked()
            message = "stopped {} after {} frames; report={}".format(
                scene, count, self.report_path
            )
            rospy.loginfo(message)
            return TriggerResponse(True, message)

    def _report(self, _request):
        with self.lock:
            if self.stream is not None:
                self.stream.flush()
            try:
                self._generate_report_locked()
            except ValueError as exc:
                return TriggerResponse(False, str(exc))
            return TriggerResponse(True, "report={}".format(self.report_path))

    def _close_stream(self):
        if self.stream is not None:
            self.stream.flush()
            self.stream.close()
        self.stream = None
        self.writer = None

    def _close(self):
        with self.lock:
            self.recording = False
            self._close_stream()

    def _status_callback(self, message):
        with self.lock:
            if not self.recording or self.writer is None:
                return
            try:
                status = json.loads(message.data)
            except (TypeError, ValueError) as exc:
                rospy.logwarn_throttle(2.0, "invalid disc_status JSON: %s", exc)
                return
            frame_key = (status.get("timestamp"), status.get("frame_id"))
            if frame_key == self.last_frame_key:
                return
            self.last_frame_key = frame_key
            row = {
                "timestamp": status.get("timestamp"),
                "frame_id": status.get("frame_id"),
                "scene_session_id": self.scene_session_id,
                **self.labels,
                "left_line_detected": status.get("left_line_detected", False),
                "right_line_detected": status.get("right_line_detected", False),
                "disc_detected": status.get("disc_detected", False),
                "disc_count": status.get("disc_count", 0),
                "disc_center_x_px": status.get("disc_center_x_px"),
                "disc_center_y_px": status.get("disc_center_y_px"),
                "disc_major_axis_px": status.get("disc_major_axis_px"),
                "disc_minor_axis_px": status.get("disc_minor_axis_px"),
                "disc_angle_deg": status.get("disc_angle_deg"),
                "disc_area_px": status.get("disc_area_px"),
                "disc_confidence": status.get("disc_confidence"),
                "red_area_px": status.get("red_area_px", 0),
                "white_area_px": status.get("white_area_px", 0),
                "red_mask_area_px": status.get("red_mask_area_px", 0),
                "white_mask_area_px": status.get("white_mask_area_px", 0),
                "red_white_center_distance_px": status.get(
                    "red_white_center_distance_px"
                ),
                "candidate_count": status.get("candidate_count", 0),
                "rejected_candidate_count": status.get(
                    "rejected_candidate_count", 0
                ),
                "disc_inside_lane": status.get("disc_inside_lane"),
                "distance_to_left_line_px": status.get(
                    "distance_to_left_line_px"
                ),
                "distance_to_right_line_px": status.get(
                    "distance_to_right_line_px"
                ),
                "processing_time_ms": status.get("processing_time_ms"),
                "used_history": status.get("used_history", False),
                "motion_enabled": status.get("motion_enabled", False),
                "detections_json": json.dumps(
                    status.get("detections", []), ensure_ascii=False, separators=(",", ":")
                ),
                "candidates_json": json.dumps(
                    status.get("candidates", []), ensure_ascii=False, separators=(",", ":")
                ),
            }
            self.writer.writerow(row)
            self.stream.flush()
            self.frames_this_session += 1

    @staticmethod
    def _frame_success(row):
        if not _as_bool(row["disc_present"]):
            return not _as_bool(row["disc_detected"])
        expected = max(1, int(row.get("expected_disc_count", 1) or 1))
        return int(row.get("disc_count", 0) or 0) >= expected

    @staticmethod
    def _longest_miss_run(rows):
        longest = 0
        current = 0
        for row in rows:
            missed = _as_bool(row["disc_present"]) and not DiscAcceptanceRecorder._frame_success(row)
            current = current + 1 if missed else 0
            longest = max(longest, current)
        return longest

    @staticmethod
    def _center_steps(rows):
        points = []
        for row in rows:
            if not (_as_bool(row["disc_present"]) and _as_bool(row["disc_detected"])):
                points.append(None)
                continue
            x = _optional_float(row["disc_center_x_px"])
            y = _optional_float(row["disc_center_y_px"])
            points.append(None if x is None or y is None else (x, y))
        steps = []
        for first, second in zip(points[:-1], points[1:]):
            if first is not None and second is not None:
                steps.append(math.hypot(second[0] - first[0], second[1] - first[1]))
        return steps

    @staticmethod
    def _multi_disc_stability(rows, expected_count):
        per_disc = [[] for _ in range(expected_count)]
        for row in rows:
            if not DiscAcceptanceRecorder._frame_success(row):
                continue
            try:
                detections = json.loads(row.get("detections_json", "[]"))
            except (TypeError, ValueError):
                continue
            detections = sorted(
                detections,
                key=lambda item: float(item.get("center_x_px", float("inf"))),
            )
            if len(detections) < expected_count:
                continue
            for index in range(expected_count):
                detection = detections[index]
                x = _optional_float(detection.get("center_x_px"))
                y = _optional_float(detection.get("center_y_px"))
                major = _optional_float(detection.get("major_axis_px"))
                minor = _optional_float(detection.get("minor_axis_px"))
                if x is not None and y is not None:
                    per_disc[index].append((x, y, major, minor))
        output = []
        for index, samples in enumerate(per_disc):
            x = [item[0] for item in samples]
            y = [item[1] for item in samples]
            major = [item[2] for item in samples if item[2] is not None]
            minor = [item[3] for item in samples if item[3] is not None]
            steps = [
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(samples[:-1], samples[1:])
            ]
            output.append(
                {
                    "left_to_right_index": index,
                    "samples": len(samples),
                    "center_std_x_px": _std(x),
                    "center_std_y_px": _std(y),
                    "center_step_p95_px": _percentile(steps, 95),
                    "center_step_max_px": max(steps) if steps else None,
                    "major_axis_std_px": _std(major),
                    "minor_axis_std_px": _std(minor),
                }
            )
        return output

    def _summarize(self, rows, include_position_statistics=True):
        positives = [row for row in rows if _as_bool(row["disc_present"])]
        negatives = [row for row in rows if not _as_bool(row["disc_present"])]
        true_positives = sum(self._frame_success(row) for row in positives)
        false_positives = sum(not self._frame_success(row) for row in negatives)
        detected_positive = [
            row for row in positives if self._frame_success(row)
        ]
        processing = [
            value
            for value in (_optional_float(row["processing_time_ms"]) for row in rows)
            if value is not None
        ]
        confidence = [
            value
            for value in (_optional_float(row["disc_confidence"]) for row in detected_positive)
            if value is not None
        ]
        summary = {
            "frames": len(rows),
            "positive_frames": len(positives),
            "negative_frames": len(negatives),
            "true_positive_frames": true_positives,
            "false_positive_frames": false_positives,
            "recall": true_positives / len(positives) if positives else None,
            "false_positive_rate": false_positives / len(negatives) if negatives else None,
            "longest_consecutive_misses": self._longest_miss_run(rows),
            "confidence_distribution": {
                "count": len(confidence),
                "min": min(confidence) if confidence else None,
                "p05": _percentile(confidence, 5),
                "p50": _percentile(confidence, 50),
                "p95": _percentile(confidence, 95),
                "max": max(confidence) if confidence else None,
                "histogram_bins": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "histogram_counts": (
                    np.histogram(
                        confidence, bins=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
                    )[0].astype(int).tolist()
                    if confidence
                    else [0, 0, 0, 0, 0, 0]
                ),
            },
            "processing_time_ms": {
                "p50": _percentile(processing, 50),
                "p95": _percentile(processing, 95),
                "max": max(processing) if processing else None,
                "fps_from_p50": (
                    1000.0 / _percentile(processing, 50)
                    if processing and _percentile(processing, 50) > 0.0
                    else None
                ),
            },
            "history_used_frames": sum(_as_bool(row["used_history"]) for row in rows),
            "motion_enabled_frames": sum(
                _as_bool(row["motion_enabled"]) for row in rows
            ),
            "both_lines_detected_rate": (
                sum(
                    _as_bool(row["left_line_detected"])
                    and _as_bool(row["right_line_detected"])
                    for row in rows
                )
                / len(rows)
                if rows
                else None
            ),
        }
        if include_position_statistics:
            expected_counts = {
                max(1, int(row.get("expected_disc_count", 1) or 1))
                for row in positives
            }
            expected_count = next(iter(expected_counts)) if len(expected_counts) == 1 else 1
            center_x = [
                value
                for value in (
                    _optional_float(row["disc_center_x_px"])
                    for row in detected_positive
                )
                if value is not None
            ]
            center_y = [
                value
                for value in (
                    _optional_float(row["disc_center_y_px"])
                    for row in detected_positive
                )
                if value is not None
            ]
            major = [
                value
                for value in (
                    _optional_float(row["disc_major_axis_px"])
                    for row in detected_positive
                )
                if value is not None
            ]
            minor = [
                value
                for value in (
                    _optional_float(row["disc_minor_axis_px"])
                    for row in detected_positive
                )
                if value is not None
            ]
            steps = self._center_steps(rows)
            summary["center_stability_px"] = {
                "std_x": _std(center_x),
                "std_y": _std(center_y),
                "p95_frame_step": _percentile(steps, 95),
                "max_frame_step": max(steps) if steps else None,
            }
            summary["ellipse_stability_px"] = {
                "major_axis_std": _std(major),
                "minor_axis_std": _std(minor),
            }
            if expected_count > 1:
                per_disc = self._multi_disc_stability(rows, expected_count)
                summary["multi_disc_stability"] = per_disc
                valid = [item for item in per_disc if item["samples"] > 0]
                summary["center_stability_px"] = {
                    "std_x": max(
                        item["center_std_x_px"] for item in valid
                        if item["center_std_x_px"] is not None
                    ) if valid else None,
                    "std_y": max(
                        item["center_std_y_px"] for item in valid
                        if item["center_std_y_px"] is not None
                    ) if valid else None,
                    "p95_frame_step": max(
                        item["center_step_p95_px"] for item in valid
                        if item["center_step_p95_px"] is not None
                    ) if any(item["center_step_p95_px"] is not None for item in valid) else None,
                    "max_frame_step": max(
                        item["center_step_max_px"] for item in valid
                        if item["center_step_max_px"] is not None
                    ) if any(item["center_step_max_px"] is not None for item in valid) else None,
                }
        return summary

    def _scene_target(self, rows, summary):
        first = rows[0]
        if not _as_bool(first["disc_present"]):
            target = float(self.thresholds.get("false_positive_rate", 0.01))
            value = summary["false_positive_rate"]
            return {
                "metric": "false_positive_rate",
                "target_max": target,
                "value": value,
                "passed": value is not None and value <= target,
            }
        occlusion = _optional_float(first["occlusion_level"]) or 0.0
        if occlusion > 1.0:
            occlusion /= 100.0
        reflection = str(first["reflection_level"]).strip().lower()
        if occlusion >= 0.40:
            target = float(self.thresholds.get("occlusion_40_recall", 0.85))
        elif occlusion >= 0.20:
            target = float(self.thresholds.get("occlusion_20_recall", 0.95))
        elif _as_bool(first["near_yellow_line"]):
            target = float(self.thresholds.get("near_yellow_recall", 0.95))
        elif reflection not in {"", "0", "none", "normal", "low"}:
            target = float(self.thresholds.get("reflection_recall", 0.90))
        else:
            target = float(self.thresholds.get("normal_recall", 0.98))
        value = summary["recall"]
        distance = str(first["distance_near_middle_far"]).strip().lower()
        jitter_limit = float(
            self.thresholds.get(
                "center_std_{}_px".format(distance),
                self.thresholds.get("center_std_middle_px", 4.0),
            )
        )
        stability = summary.get("center_stability_px", {})
        std_x = stability.get("std_x")
        std_y = stability.get("std_y")
        jitter_pass = (
            std_x is not None
            and std_y is not None
            and std_x <= jitter_limit
            and std_y <= jitter_limit
        )
        return {
            "metric": "recall",
            "target_min": target,
            "value": value,
            "recall_passed": value is not None and value >= target,
            "center_std_limit_px": jitter_limit,
            "center_stability_passed": jitter_pass,
            "passed": value is not None and value >= target and jitter_pass,
        }

    def _generate_report_locked(self):
        if not self.csv_path.exists():
            raise ValueError("no CSV data at {}".format(self.csv_path))
        with self.csv_path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError("CSV has no frames")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["scene_name"]].append(row)
        scenes = {}
        for scene_name, scene_rows in sorted(grouped.items()):
            summary = self._summarize(scene_rows, include_position_statistics=True)
            summary["labels"] = {
                key: scene_rows[0][key]
                for key in (
                    "disc_present",
                    "expected_disc_count",
                    "position_left_center_right",
                    "distance_near_middle_far",
                    "occlusion_level",
                    "reflection_level",
                    "near_yellow_line",
                )
            }
            summary["acceptance"] = self._scene_target(scene_rows, summary)
            scenes[scene_name] = summary

        overall = self._summarize(rows, include_position_statistics=False)
        scene_miss_max = max(
            scene["longest_consecutive_misses"] for scene in scenes.values()
        )
        overall["longest_consecutive_misses"] = scene_miss_max
        recall_target = float(self.thresholds.get("overall_recall", 0.95))
        fp_target = float(self.thresholds.get("false_positive_rate", 0.01))
        miss_target = int(self.thresholds.get("max_consecutive_misses", 3))
        fps_target = float(self.thresholds.get("processing_fps", 10.0))
        checks = {
            "recall": overall["recall"] is not None
            and overall["recall"] >= recall_target,
            "false_positive_rate": overall["false_positive_rate"] is not None
            and overall["false_positive_rate"] <= fp_target,
            "max_consecutive_misses": scene_miss_max <= miss_target,
            "processing_fps": overall["processing_time_ms"]["fps_from_p50"]
            is not None
            and overall["processing_time_ms"]["fps_from_p50"] >= fps_target,
            "used_history_false": overall["history_used_frames"] == 0,
            "motion_enabled_false": overall["motion_enabled_frames"] == 0,
            "all_scenes": all(
                bool(scene["acceptance"]["passed"]) for scene in scenes.values()
            ),
        }
        report = {
            "generated_at": datetime.now().isoformat(),
            "csv_path": str(self.csv_path),
            "thresholds": self.thresholds,
            "overall": overall,
            "scenes": scenes,
            "acceptance": {
                "checks": checks,
                "passed": all(checks.values()),
                "note": (
                    "overall pass requires both positive and negative samples; "
                    "position stability is evaluated per static scene"
                ),
            },
        }
        temporary = self.report_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(str(temporary), str(self.report_path))


def main():
    rospy.init_node("disc_acceptance_recorder")
    DiscAcceptanceRecorder()
    rospy.spin()


if __name__ == "__main__":
    main()
