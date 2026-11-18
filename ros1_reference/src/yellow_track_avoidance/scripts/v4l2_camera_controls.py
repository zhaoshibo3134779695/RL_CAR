#!/usr/bin/env python3
"""Select fixed UVC photometry once, then freeze it for the mission.

The camera does not expose a hardware metering-window control, so its automatic
exposure can be biased by bright areas outside the competition detection ROI.
The formal profile therefore keeps exposure manual, tries bounded fixed levels
from low to high, and freezes the first level for which the real detector sees
the yellow line and both metric discs in consecutive frames.  AWB is allowed a
short stationary warm-up and is frozen before the exposure sweep.  Hardware
AE commissioning remains available only as an explicit diagnostic option.
"""

import json
import math
import subprocess
import threading
import time

import rospy
from std_msgs.msg import Bool, String


def set_control(device, name, value):
    subprocess.run(
        [
            "v4l2-ctl",
            "-d",
            device,
            "--set-ctrl={}={}".format(name, value),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def get_controls(device, names):
    """Return integer UVC control values reported by v4l2-ctl."""
    result = subprocess.run(
        [
            "v4l2-ctl",
            "-d",
            device,
            "--get-ctrl={}".format(",".join(names)),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return parse_control_values(result.stdout)


def parse_control_values(output):
    values = {}
    for line in str(output).splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator:
            continue
        try:
            values[name.strip()] = int(raw_value.strip().split()[0])
        except (IndexError, TypeError, ValueError):
            continue
    return values


def parse_exposures(value):
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).replace(";", ",").split(",")
    result = []
    for item in items:
        try:
            exposure = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if exposure > 0 and exposure not in result:
            result.append(exposure)
    return sorted(result)


def detection_metrics(payload, expected_discs, minimum_line_quality, minimum_disc_confidence):
    line_detected = bool(payload.get("left_line_detected", False))
    try:
        line_quality = float(payload.get("left_line_quality", 0.0))
    except (TypeError, ValueError):
        line_quality = 0.0
    try:
        disc_count = int(payload.get("disc_count", 0))
    except (TypeError, ValueError):
        disc_count = 0
    confidences = []
    for item in payload.get("detections", []):
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError, AttributeError):
            continue
        projection = item.get("metric_projection") if isinstance(item, dict) else None
        base_m = projection.get("base_m") if isinstance(projection, dict) else None
        try:
            valid_base = (
                isinstance(base_m, list)
                and len(base_m) >= 2
                and all(math.isfinite(float(value)) for value in base_m[:2])
            )
        except (TypeError, ValueError):
            valid_base = False
        if (
            valid_base
        ):
            confidences.append(confidence)
    confidences.sort(reverse=True)
    enough_metric_discs = (
        disc_count >= expected_discs and len(confidences) >= expected_discs
    )
    minimum_confidence = (
        min(confidences[:expected_discs]) if enough_metric_discs else 0.0
    )
    accepted = bool(
        line_detected
        and line_quality >= minimum_line_quality
        and enough_metric_discs
        and minimum_confidence >= minimum_disc_confidence
    )
    score = (
        min(disc_count, expected_discs),
        int(line_detected),
        min(minimum_confidence, 1.0),
        min(max(line_quality, 0.0), 1.0),
    )
    return {
        "accepted": accepted,
        "line_detected": line_detected,
        "line_quality": line_quality,
        "metric_disc_count": disc_count,
        "minimum_disc_confidence": minimum_confidence,
        "score": score,
    }


class DetectionMonitor:
    def __init__(self, topic):
        self.condition = threading.Condition()
        self.serial = 0
        self.payload = None
        self.subscriber = rospy.Subscriber(
            topic, String, self._callback, queue_size=1
        )

    def _callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        with self.condition:
            self.serial += 1
            self.payload = payload
            self.condition.notify_all()

    def marker(self):
        with self.condition:
            return self.serial

    def wait_after(self, marker, timeout_sec):
        deadline = time.monotonic() + max(0.0, timeout_sec)
        with self.condition:
            while self.serial <= marker and not rospy.is_shutdown():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return marker, None
                self.condition.wait(remaining)
            return self.serial, self.payload


def publish_status(publisher, state, **fields):
    payload = {"state": state, **fields}
    publisher.publish(
        String(data=json.dumps(payload, ensure_ascii=False, allow_nan=False))
    )


def scene_requirements():
    return {
        "expected_discs": int(
            rospy.get_param("~expected_metric_disc_count", 2)
        ),
        "minimum_line_quality": float(
            rospy.get_param("~minimum_left_line_quality", 0.70)
        ),
        "minimum_disc_confidence": float(
            rospy.get_param("~minimum_disc_confidence", 0.68)
        ),
        "required_hits": max(
            1, int(rospy.get_param("~required_detection_hits", 2))
        ),
    }


def wait_for_stable_scene(
    monitor,
    status_pub,
    state,
    timeout_sec,
    requirements,
    status_fields=None,
):
    """Wait for consecutive complete metric-scene detections."""
    marker = monitor.marker()
    deadline = time.monotonic() + max(0.2, float(timeout_sec))
    valid_hits = 0
    latest = None
    best = None
    static_fields = dict(status_fields or {})
    while time.monotonic() < deadline and not rospy.is_shutdown():
        marker, payload = monitor.wait_after(
            marker, min(0.35, max(0.0, deadline - time.monotonic()))
        )
        if payload is None:
            continue
        metrics = detection_metrics(
            payload,
            requirements["expected_discs"],
            requirements["minimum_line_quality"],
            requirements["minimum_disc_confidence"],
        )
        latest = metrics
        if best is None or metrics["score"] > best["score"]:
            best = metrics
        # A single good frame is not enough to freeze mission photometry.
        # Reset on a miss so the required hits are genuinely consecutive.
        valid_hits = valid_hits + 1 if metrics["accepted"] else 0
        publish_status(
            status_pub,
            state,
            valid_detection_hits=valid_hits,
            required_detection_hits=requirements["required_hits"],
            **static_fields,
            **{key: value for key, value in metrics.items() if key != "score"},
        )
        if valid_hits >= requirements["required_hits"]:
            return True, metrics, best
    return False, latest or {}, best or {}


def freeze_automatic_photometry(device, snapshot):
    """Disable AE/AWB without discarding the values they converged to."""
    manual_exposure_mode = int(
        rospy.get_param("~manual_exposure_mode", 1)
    )
    configured_exposure = int(
        rospy.get_param("~exposure_time_absolute", 40)
    )
    configured_gain = int(rospy.get_param("~gain", 0))
    configured_white_balance = int(
        rospy.get_param("~white_balance_temperature", 4600)
    )
    exposure = int(snapshot.get("exposure_time_absolute", configured_exposure))
    gain = int(snapshot.get("gain", configured_gain))
    white_balance = int(
        snapshot.get("white_balance_temperature", configured_white_balance)
    )

    set_control(device, "auto_exposure", manual_exposure_mode)
    set_control(device, "exposure_time_absolute", exposure)
    set_control(device, "gain", gain)
    set_control(device, "white_balance_automatic", 0)
    set_control(device, "white_balance_temperature", white_balance)
    return {
        "exposure_time_absolute": exposure,
        "gain": gain,
        "white_balance_temperature": white_balance,
    }


def select_automatic_photometry(device, monitor, status_pub):
    """Run the camera's AE and AWB once and freeze a detector-approved result."""
    requirements = scene_requirements()
    automatic_exposure_mode = int(
        rospy.get_param("~automatic_exposure_mode", 0)
    )
    automatic_settle_sec = max(
        0.0, float(rospy.get_param("~automatic_photometry_settle_sec", 0.8))
    )
    automatic_evaluate_sec = max(
        0.5, float(rospy.get_param("~automatic_photometry_evaluate_sec", 3.2))
    )
    post_freeze_validate_sec = max(
        0.2, float(rospy.get_param("~post_freeze_validate_sec", 0.8))
    )

    set_control(device, "auto_exposure", automatic_exposure_mode)
    set_control(device, "white_balance_automatic", 1)
    publish_status(
        status_pub,
        "automatic_settling",
        automatic_exposure_mode=automatic_exposure_mode,
    )
    rospy.sleep(automatic_settle_sec)
    passed, metrics, best = wait_for_stable_scene(
        monitor,
        status_pub,
        "automatic_evaluating",
        automatic_evaluate_sec,
        requirements,
        {"automatic_exposure_mode": automatic_exposure_mode},
    )

    snapshot = get_controls(
        device,
        ["exposure_time_absolute", "gain", "white_balance_temperature"],
    )
    frozen = freeze_automatic_photometry(device, snapshot)
    publish_status(status_pub, "automatic_frozen", **frozen)

    # Confirm that switching the UVC loops to manual did not restore stale
    # pre-auto values.  Use the same consecutive detector gate after freezing.
    if passed:
        frozen_passed, frozen_metrics, frozen_best = wait_for_stable_scene(
            monitor,
            status_pub,
            "validating_frozen_automatic",
            post_freeze_validate_sec,
            requirements,
            frozen,
        )
        if frozen_passed:
            rospy.loginfo(
                "automatic camera photometry accepted and frozen: "
                "exposure=%d gain=%d white_balance=%d",
                frozen["exposure_time_absolute"],
                frozen["gain"],
                frozen["white_balance_temperature"],
            )
            return frozen, True, frozen_metrics
        if frozen_best and (
            not best or frozen_best.get("score", ()) > best.get("score", ())
        ):
            best = frozen_best
        metrics = frozen_metrics

    rospy.logwarn(
        "automatic camera photometry did not retain a stable complete metric "
        "scene; frozen exposure=%d gain=%d white_balance=%d before fallback",
        frozen["exposure_time_absolute"],
        frozen["gain"],
        frozen["white_balance_temperature"],
    )
    return frozen, False, metrics or best


def select_startup_exposure(device, monitor, status_pub, candidates):
    requirements = scene_requirements()
    settle_sec = max(0.0, float(rospy.get_param("~exposure_settle_sec", 0.70)))
    evaluate_sec = max(0.2, float(rospy.get_param("~exposure_evaluate_sec", 1.50)))
    best = None

    for exposure in candidates:
        if rospy.is_shutdown():
            break
        set_control(device, "exposure_time_absolute", exposure)
        publish_status(status_pub, "settling", exposure=exposure)
        rospy.sleep(settle_sec)
        passed, latest, candidate_best = wait_for_stable_scene(
            monitor,
            status_pub,
            "evaluating",
            evaluate_sec,
            requirements,
            {"exposure": exposure},
        )
        if candidate_best:
            candidate = (
                candidate_best["score"],
                -exposure,
                exposure,
                candidate_best,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if passed:
            rospy.loginfo(
                "startup exposure selected=%d after %d consecutive valid "
                "scene hits (line_quality=%.3f metric_discs=%d "
                "min_disc_confidence=%.3f)",
                exposure,
                requirements["required_hits"],
                latest["line_quality"],
                latest["metric_disc_count"],
                latest["minimum_disc_confidence"],
            )
            return exposure, True, latest
        rospy.loginfo(
            "startup exposure %d did not pass: %s",
            exposure,
            "no detector samples" if latest is None else latest,
        )

    fallback = int(rospy.get_param("~exposure_time_absolute", 40))
    metrics = {}
    if best is not None:
        fallback = best[2]
        metrics = best[3]
    set_control(device, "exposure_time_absolute", fallback)
    rospy.logwarn(
        "no exposure produced a stable complete metric scene; using best "
        "candidate exposure=%d and leaving the normal two-disc scene gate active",
        fallback,
    )
    return fallback, False, metrics


def main():
    rospy.init_node("yellow_track_v4l2_controls")
    device = rospy.get_param("~video_device")
    ready_topic = str(
        rospy.get_param(
            "~photometry_ready_topic",
            "/yellow_track_avoidance/camera_photometry_ready",
        )
    )
    preview_topic = str(
        rospy.get_param(
            "~photometry_preview_topic",
            "/yellow_track_avoidance/camera_photometry_preview",
        )
    )
    status_topic = str(
        rospy.get_param(
            "~photometry_status_topic",
            "/yellow_track_avoidance/camera_photometry_status",
        )
    )
    detector_topic = str(
        rospy.get_param(
            "~detector_status_topic",
            "/yellow_track_avoidance/debug/disc_status",
        )
    )
    ready_pub = rospy.Publisher(ready_topic, Bool, queue_size=1, latch=True)
    preview_pub = rospy.Publisher(preview_topic, Bool, queue_size=1, latch=True)
    status_pub = rospy.Publisher(status_topic, String, queue_size=1, latch=True)
    monitor = DetectionMonitor(detector_topic)
    ready_pub.publish(Bool(data=False))
    preview_pub.publish(Bool(data=False))
    publish_status(status_pub, "starting")

    delay_sec = float(rospy.get_param("~startup_delay_sec", 1.0))
    white_balance_warmup_sec = float(
        rospy.get_param("~white_balance_warmup_sec", 4.0)
    )
    automatic_commissioning = bool(
        rospy.get_param("~automatic_photometry_commissioning", True)
    )
    automatic_tuning = bool(rospy.get_param("~auto_tune_exposure", True))
    static_controls = [
        (
            "exposure_dynamic_framerate",
            int(rospy.get_param("~exposure_dynamic_framerate", 0)),
        ),
        ("sharpness", int(rospy.get_param("~sharpness", 3))),
    ]
    rospy.sleep(max(0.0, delay_sec))
    try:
        for name, value in static_controls:
            set_control(device, name, value)

        configured_exposure = int(
            rospy.get_param("~exposure_time_absolute", 40)
        )
        preview_pub.publish(
            Bool(data=bool(automatic_commissioning or automatic_tuning))
        )

        selected = configured_exposure
        passed = False
        metrics = {}
        frozen = {}
        if automatic_commissioning:
            frozen, passed, metrics = select_automatic_photometry(
                device, monitor, status_pub
            )
            selected = int(
                frozen.get("exposure_time_absolute", configured_exposure)
            )
        else:
            # Formal path: do not leave the previous process' UVC AE state
            # active during AWB warm-up.  Exposure stays manual throughout the
            # subsequent low-to-high detector-gated sweep.
            initial_exposure = configured_exposure
            if automatic_tuning:
                startup_candidates = parse_exposures(
                    rospy.get_param(
                        "~exposure_candidates",
                        [30, 40, 50, 54, 56, 58, 60],
                    )
                )
                if startup_candidates:
                    initial_exposure = startup_candidates[0]
            set_control(
                device,
                "auto_exposure",
                int(rospy.get_param("~manual_exposure_mode", 1)),
            )
            set_control(device, "exposure_time_absolute", initial_exposure)
            set_control(device, "gain", int(rospy.get_param("~gain", 0)))
            if white_balance_warmup_sec > 0.0:
                set_control(device, "white_balance_automatic", 1)
                rospy.sleep(white_balance_warmup_sec)
            snapshot = get_controls(
                device, ["white_balance_temperature"]
            )
            snapshot.update({
                "exposure_time_absolute": initial_exposure,
                "gain": int(rospy.get_param("~gain", 0)),
            })
            frozen = freeze_automatic_photometry(device, snapshot)
            selected = initial_exposure

        if not passed and automatic_tuning:
            candidates = parse_exposures(
                rospy.get_param(
                    "~exposure_candidates",
                    [30, 40, 50, 54, 56, 58, 60],
                )
            )
            if not candidates:
                candidates = [configured_exposure]
            publish_status(
                status_pub,
                "fallback_exposure_tuning",
                candidates=candidates,
                frozen_white_balance_temperature=frozen.get(
                    "white_balance_temperature"
                ),
            )
            selected, passed, metrics = select_startup_exposure(
                device, monitor, status_pub, candidates
            )
        elif not automatic_commissioning and not automatic_tuning:
            selected = configured_exposure
            passed = True
            metrics = {}
            set_control(device, "exposure_time_absolute", selected)

        final_names = [
            "auto_exposure",
            "exposure_dynamic_framerate",
            "exposure_time_absolute",
            "gain",
            "white_balance_automatic",
            "white_balance_temperature",
            "brightness",
            "contrast",
            "saturation",
            "gamma",
            "sharpness",
        ]
        final_controls = get_controls(device, final_names)
        result_text = "\n".join(
            "{}: {}".format(name, final_controls[name])
            for name in final_names
            if name in final_controls
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        preview_pub.publish(Bool(data=False))
        rospy.logfatal("failed to commission UVC camera controls: %s", exc)
        raise SystemExit(1)

    rospy.loginfo(
        "locked UVC camera controls on %s:\n%s", device, result_text
    )
    preview_pub.publish(Bool(data=False))
    rospy.sleep(0.05)
    ready_pub.publish(Bool(data=True))
    publish_status(
        status_pub,
        "selected" if passed else "fallback_best_scene",
        exposure=selected,
        gain=final_controls.get("gain"),
        white_balance_temperature=final_controls.get(
            "white_balance_temperature"
        ),
        stable_complete_scene=passed,
        metrics=metrics,
    )
    rospy.loginfo(
        "camera photometry ready: %s exposure=%d white_balance=%s "
        "stable_complete_scene=%s",
        ready_topic,
        selected,
        final_controls.get("white_balance_temperature"),
        passed,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
