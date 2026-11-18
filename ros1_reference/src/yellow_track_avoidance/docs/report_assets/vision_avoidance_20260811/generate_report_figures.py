#!/usr/bin/env python3
"""Generate RAICOM report figures only from recorded project data.

This script is a reporting utility. It does not start ROS nodes, publish commands,
change navigation/perception parameters, or modify the formal runtime chain.
"""

import hashlib
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import rosbag
from cv_bridge import CvBridge


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = OUT / "sources"
CAPTURE_BAG = Path("/tmp/vision_report_capture_20260811.bag")
TRIAL_BAG = Path(
    "/home/mowen/Sebastian/"
    "stage1_global_teb_compact_standard_ekf_20260810_trial7.bag"
)
CONFIG = ROOT / "config/current_car_recalibrated_640x480_20260809.yaml"
ACQUIRE_REPORT = ROOT / "docs/navigation/PERCEPTION_ACQUIRE_LOCKED_REPORT_20260810.md"
DISC_REPORT = ROOT / "docs/perception/DISC_DETECTION_ACCEPTANCE.md"
CALIBRATION_REPORT = (
    ROOT
    / "calibration_runs/current_car_640x480_20260809_repair/calibration_report.json"
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        # Matplotlib 3.1 on this vehicle cannot address individual faces in the
        # installed Noto TTC collection, while DroidSansFallback lacks Latin.
        # AR PL UKai CN is the installed single face verified to contain both
        # Chinese and Latin glyphs, so mixed labels remain complete.
        "font.sans-serif": ["AR PL UKai CN"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.edgecolor": "#4b5563",
        "axes.linewidth": 0.8,
    }
)

COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "teal": "#0f766e",
    "green": "#16a34a",
    "orange": "#ea580c",
    "yellow": "#eab308",
    "red": "#dc2626",
    "purple": "#9333ea",
    "gray": "#6b7280",
    "light": "#f3f4f6",
    "dark": "#111827",
}


def ensure_dirs():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return path


def panel_label(ax, text, dark=False):
    ax.text(
        0.018,
        0.975,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white" if dark else COLORS["dark"],
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#111827" if dark else "white",
            edgecolor="none",
            alpha=0.88,
        ),
        zorder=20,
    )


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def yaw_from_quaternion(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def image_msg_to_bgr(bridge, message):
    return bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")


def extract_capture():
    """Extract one timestamp-consistent live frame and debug bundle."""
    if not CAPTURE_BAG.exists():
        raise FileNotFoundError(CAPTURE_BAG)

    image_topics = {
        "/yellow_track_avoidance/debug/raw_overlay": "raw_overlay.png",
        "/yellow_track_avoidance/debug/yellow_score": "yellow_score.png",
        "/yellow_track_avoidance/debug/yellow_mask": "yellow_mask.png",
        "/yellow_track_avoidance/debug/disc_overlay": "disc_overlay.png",
        "/yellow_track_avoidance/debug/disc_candidates": "disc_candidates.png",
        "/yellow_track_avoidance/debug/disc_red_mask": "disc_red_mask.png",
        "/yellow_track_avoidance/debug/disc_white_mask": "disc_white_mask.png",
        "/yellow_track_avoidance/debug/bev": "bev.png",
    }
    required = set(image_topics)
    by_stamp = {}
    raw_by_stamp = {}
    json_messages = []
    bridge = CvBridge()

    with rosbag.Bag(str(CAPTURE_BAG)) as bag:
        for topic, message, bag_time in bag.read_messages():
            if topic == "/usb_cam/image_raw":
                raw_by_stamp[round(message.header.stamp.to_sec(), 6)] = message
            elif topic in image_topics:
                stamp = round(message.header.stamp.to_sec(), 6)
                by_stamp.setdefault(stamp, {})[topic] = message
            elif hasattr(message, "data"):
                try:
                    data = json.loads(message.data)
                except Exception:
                    continue
                stamp = float(data.get("stamp", bag_time.to_sec()))
                json_messages.append((topic, stamp, data))

    candidates = [stamp for stamp, items in by_stamp.items() if required <= set(items)]
    if not candidates:
        raise RuntimeError("no complete synchronized debug frame in capture bag")
    target = max(candidates)
    if target not in raw_by_stamp:
        nearest = min(raw_by_stamp, key=lambda value: abs(value - target))
    else:
        nearest = target

    raw = image_msg_to_bgr(bridge, raw_by_stamp[nearest])
    cv2.imwrite(str(SOURCES / "raw_rgb.png"), raw)
    for topic, filename in image_topics.items():
        cv2.imwrite(
            str(SOURCES / filename),
            image_msg_to_bgr(bridge, by_stamp[target][topic]),
        )

    selected_json = {}
    for topic in [
        "/yellow_track_avoidance/debug/status",
        "/yellow_track_avoidance/debug/disks",
        "/yellow_track_avoidance/debug/boundaries",
        "/yellow_track_avoidance/debug/candidates",
    ]:
        matching = [item for item in json_messages if item[0] == topic]
        if not matching:
            continue
        _, stamp, data = min(matching, key=lambda item: abs(item[1] - target))
        selected_json[topic] = {"message_stamp": stamp, "data": data}

    with (SOURCES / "live_frame_messages.json").open("w", encoding="utf-8") as stream:
        json.dump(selected_json, stream, ensure_ascii=False, indent=2)

    metadata = {
        "capture_bag": str(CAPTURE_BAG),
        "capture_bag_sha256": sha256(CAPTURE_BAG),
        "selected_image_stamp": target,
        "selected_raw_stamp": nearest,
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "formal_motion_started": False,
        "capture_nodes": ["usb_cam", "yellow_track_avoidance"],
        "motion_enabled": False,
        "planning_enabled": False,
    }
    with (SOURCES / "live_capture_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    return raw, selected_json, metadata


def load_project_pipeline(raw_bgr):
    import sys

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from yellow_track_avoidance.config import load_config
    from yellow_track_avoidance.pipeline import AvoidancePipeline

    config = load_config(str(CONFIG))
    config["control"]["enable_motion"] = False
    pipeline = AvoidancePipeline(config)
    working = (
        pipeline.undistorter.undistort(raw_bgr)
        if pipeline.undistorter is not None
        else raw_bgr.copy()
    )
    masks = pipeline.segmenter.masks(working)
    detector = pipeline.profile_boundary_detector
    profiles = detector._profiles(
        "left",
        masks["yellow_confidence"],
        masks["yellow_opponent"],
        masks["yellow_roi"],
    )
    selected = detector._dynamic_path(profiles)
    boundary = detector.detect(
        masks["yellow_confidence"],
        masks["yellow_opponent"],
        masks["yellow_roi"],
    )
    detections = pipeline.disk_detector.detect(
        masks["red"],
        masks["white"],
        0.0,
        0.0,
        0.0,
        lambda points: np.zeros((len(points), 2), dtype=np.float64),
        image_bgr=working,
    )
    return config, pipeline, working, masks, profiles, selected, boundary, detections


def figure_1_architecture():
    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, color, subtitle=None):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * (0.59 if subtitle else 0.5), text,
                ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["dark"])
        if subtitle:
            ax.text(x + w / 2, y + h * 0.28, subtitle,
                    ha="center", va="center", fontsize=9.5, color=COLORS["gray"])

    def arrow(x1, y1, x2, y2, color=COLORS["gray"], text=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, linewidth=1.7, color=color))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2+0.17, text, ha="center", va="bottom",
                    fontsize=9.5, color=color)

    box(0.25, 2.45, 1.45, 1.15, "RGB图像", COLORS["blue"], "640×480")
    box(2.25, 3.65, 2.15, 1.12, "黄色边界检测", COLORS["yellow"], "置信度·截面·DP")
    box(2.25, 1.25, 2.15, 1.12, "红白圆片检测", COLORS["red"], "椭圆·扇区·白心")
    box(5.0, 2.45, 1.85, 1.15, "米制BEV", COLORS["cyan"], "地面坐标（m）")
    box(7.45, 2.45, 2.0, 1.15, "odom世界模型", COLORS["teal"], "走廊与圆片冻结")
    box(10.1, 3.7, 2.05, 1.08, "全局/局部costmap", COLORS["orange"], "完整栅格约束")
    box(10.1, 1.2, 2.05, 1.08, "紧凑几何障碍", COLORS["purple"], "2条线 + 2个圆")
    box(12.8, 3.7, 1.55, 1.08, "GlobalPlanner", COLORS["blue"], "全局路径")
    box(12.8, 1.2, 1.55, 1.08, "TEB", COLORS["purple"], "局部轨迹")
    box(14.65, 2.45, 0.7, 1.15, "麦\n轮\n底\n盘", COLORS["green"])

    arrow(1.7, 3.02, 2.25, 4.2)
    arrow(1.7, 3.02, 2.25, 1.82)
    arrow(4.4, 4.2, 5.0, 3.15)
    arrow(4.4, 1.82, 5.0, 2.9)
    arrow(6.85, 3.02, 7.45, 3.02)
    arrow(9.45, 3.02, 10.1, 4.24)
    arrow(9.45, 3.02, 10.1, 1.74)
    arrow(12.15, 4.24, 12.8, 4.24)
    arrow(12.15, 1.74, 12.8, 1.74)
    arrow(13.58, 3.7, 13.58, 2.28, COLORS["blue"], "路径参考")
    arrow(14.35, 1.74, 14.84, 2.45, COLORS["green"], "vx/vy/wz")

    ax.text(8.45, 0.25, "ACQUIRE完成后锁存静态世界模型，视觉进入LOCKED并停止重型计算",
            ha="center", va="center", color=COLORS["teal"], fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ecfdf5", edgecolor="#99f6e4"))
    return save(fig, "fig01_visual_avoidance_architecture.png")


def figure_2_yellow(raw, analysis):
    config, pipeline, working, masks, profiles, selected, boundary, _ = analysis
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    for ax in axes.flat:
        ax.axis("off")

    axes[0, 0].imshow(bgr_to_rgb(raw))
    panel_label(axes[0, 0], "A  原始RGB（同一帧）", dark=True)

    confidence = masks["yellow_confidence"]
    im = axes[0, 1].imshow(confidence, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 1].contour(masks["yellow_roi"] > 0, levels=[0.5], colors=["#22d3ee"], linewidths=1.5)
    panel_label(axes[0, 1], "B  黄色连续置信度", dark=True)
    cbar = fig.colorbar(im, ax=axes[0, 1], fraction=0.035, pad=0.02)
    cbar.set_label("置信度")

    roi_canvas = working.copy()
    overlay = roi_canvas.copy()
    overlay[masks["yellow_roi"] > 0] = (45, 105, 45)
    roi_canvas = cv2.addWeighted(overlay, 0.25, roi_canvas, 0.75, 0)
    for profile in profiles:
        points = np.rint(profile.coordinates[[0, -1]]).astype(np.int32)
        cv2.line(roi_canvas, tuple(points[0]), tuple(points[1]), (255, 220, 0), 1, cv2.LINE_AA)
    axes[1, 0].imshow(bgr_to_rgb(roi_canvas))
    panel_label(axes[1, 0], "C  ROI与28条法向截面", dark=True)

    dp_canvas = working.copy()
    selected_points = np.asarray(
        [profiles[i].coordinates[selected[i]] for i in range(len(profiles))], dtype=np.float64
    )
    selected_score = np.asarray(
        [profiles[i].score[selected[i]] for i in range(len(profiles))], dtype=np.float64
    )
    selected_base = np.asarray(
        [profiles[i].base_confidence[selected[i]] for i in range(len(profiles))], dtype=np.float64
    )
    selected_prom = np.asarray(
        [profiles[i].prominence[selected[i]] for i in range(len(profiles))], dtype=np.float64
    )
    selected_inside = np.asarray(
        [profiles[i].valid_roi[selected[i]] for i in range(len(profiles))], dtype=bool
    )
    cfg = config["normal_profile_detector"]
    valid = (
        selected_inside
        & (selected_score >= float(cfg["minimum_section_confidence"]))
        & (selected_base >= float(cfg["minimum_base_confidence"]))
        & (selected_prom >= float(cfg["minimum_section_prominence"]))
    )
    for point, is_valid in zip(selected_points, valid):
        cv2.circle(dp_canvas, tuple(np.rint(point).astype(int)), 4,
                   (40, 220, 40) if is_valid else (0, 120, 255), -1, cv2.LINE_AA)
    finite = np.isfinite(boundary.left_px)
    fit_points = np.column_stack([boundary.left_px[finite], boundary.rows_px[finite]])
    if len(fit_points) >= 2:
        cv2.polylines(dp_canvas, [np.rint(fit_points).astype(np.int32)], False,
                      (0, 255, 255), 3, cv2.LINE_AA)
    axes[1, 1].imshow(bgr_to_rgb(dp_canvas))
    metrics = pipeline.profile_boundary_detector.last_metrics
    axes[1, 1].text(
        0.98, 0.04,
        "有效截面 {}/28\n质量 {:.3f}\n拟合RMS {:.2f} px".format(
            metrics["left_valid_sections"], metrics["left_quality"], metrics["left_fit_rms_px"]
        ),
        transform=axes[1, 1].transAxes, ha="right", va="bottom", color="white",
        fontsize=10.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", alpha=0.82, edgecolor="none")
    )
    panel_label(axes[1, 1], "D  DP支持点与最终拟合线", dark=True)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    return save(fig, "fig02_yellow_line_detection.png")


def ellipse_point(ellipse, radius, phi):
    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    theta = math.radians(angle_deg)
    lx = 0.5 * axis_a * radius * math.cos(phi)
    ly = 0.5 * axis_b * radius * math.sin(phi)
    return (
        int(round(cx + lx * math.cos(theta) - ly * math.sin(theta))),
        int(round(cy + lx * math.sin(theta) + ly * math.cos(theta))),
    )


def draw_disc_geometry(image, detection, red_mask):
    canvas = image.copy()
    red_overlay = np.zeros_like(canvas)
    red_overlay[:, :, 2] = red_mask
    canvas = cv2.addWeighted(canvas, 1.0, red_overlay, 0.32, 0.0)
    cv2.drawContours(canvas, [detection.contour], -1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, detection.ellipse, (0, 255, 0), 2, cv2.LINE_AA)
    inner = (
        detection.ellipse[0],
        (detection.ellipse[1][0] * 0.48, detection.ellipse[1][1] * 0.48),
        detection.ellipse[2],
    )
    center_region = (
        detection.ellipse[0],
        (detection.ellipse[1][0] * 0.40, detection.ellipse[1][1] * 0.40),
        detection.ellipse[2],
    )
    cv2.ellipse(canvas, inner, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, center_region, (255, 255, 0), 2, cv2.LINE_AA)
    for index in range(20):
        phi = 2.0 * math.pi * index / 20.0
        p0 = ellipse_point(detection.ellipse, 0.48, phi)
        p1 = ellipse_point(detection.ellipse, 1.08, phi)
        cv2.line(canvas, p0, p1, (0, 210, 0), 1, cv2.LINE_AA)
    center = tuple(np.rint(detection.ellipse[0]).astype(int))
    cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    return canvas


def crop_around(image, ellipse, scale=2.2):
    cx, cy = ellipse[0]
    radius = max(ellipse[1]) * scale / 2.0
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(image.shape[1], int(math.ceil(cx + radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(image.shape[0], int(math.ceil(cy + radius)))
    return image[y0:y1, x0:x1]


def figure_3_discs(analysis):
    _, _, working, masks, _, _, _, detections = analysis
    detections = sorted(detections, key=lambda item: item.center_px[0])
    if len(detections) != 2:
        raise RuntimeError("live current-source frame did not yield exactly two disc detections")

    overview = working.copy()
    red_overlay = np.zeros_like(overview)
    red_overlay[:, :, 2] = masks["red"]
    overview = cv2.addWeighted(overview, 1.0, red_overlay, 0.28, 0.0)
    for index, detection in enumerate(detections, 1):
        cv2.drawContours(overview, [detection.contour], -1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.ellipse(overview, detection.ellipse, (0, 255, 0), 2, cv2.LINE_AA)
        center = tuple(np.rint(detection.center_px).astype(int))
        cv2.drawMarker(overview, center, (255, 255, 255), cv2.MARKER_CROSS, 13, 2)
        cv2.putText(overview, "D{}  conf={:.3f}".format(index, detection.confidence),
                    (center[0] + 8, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 255, 0), 1, cv2.LINE_AA)

    fig = plt.figure(figsize=(15.5, 8.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.65, 1.0], hspace=0.12, wspace=0.08)
    ax0 = fig.add_subplot(grid[:, 0])
    ax0.imshow(bgr_to_rgb(overview))
    ax0.axis("off")
    panel_label(ax0, "A  红色候选轮廓、拟合椭圆与最终中心", dark=True)

    for row, detection in enumerate(detections):
        ax = fig.add_subplot(grid[row, 1])
        geometry = draw_disc_geometry(working, detection, masks["red"])
        crop = crop_around(geometry, detection.ellipse)
        ax.imshow(bgr_to_rgb(crop))
        ax.axis("off")
        ax.text(
            0.98, 0.04,
            "置信度 {:.3f}\n红环比例 {:.3f}\n扇区覆盖 {:.0f}/20\n白心比例 {:.3f}".format(
                detection.confidence,
                detection.red_annulus_ratio,
                detection.red_sector_coverage * 20.0,
                detection.white_center_ratio,
            ),
            transform=ax.transAxes, ha="right", va="bottom", color="white", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", alpha=0.82, edgecolor="none")
        )
        panel_label(ax, "{}  20扇区、红环与白心区域".format(chr(ord("B") + row)), dark=True)
    return save(fig, "fig03_disc_detection_geometry.png")


def live_json(messages, topic):
    return messages[topic]["data"]


def figure_4_bev(raw, messages, config):
    bev = cv2.imread(str(SOURCES / "bev.png"), cv2.IMREAD_COLOR)
    disks = live_json(messages, "/yellow_track_avoidance/debug/disks")["raw"]
    boundaries = live_json(messages, "/yellow_track_avoidance/debug/boundaries")
    if len(disks) != 2:
        raise RuntimeError("selected live frame does not contain two raw discs")

    raw_marked = raw.copy()
    for index, disk in enumerate(disks, 1):
        px = disk["metric_projection"]["distorted_full_image_px"]
        center = tuple(np.rint(px).astype(int))
        cv2.drawMarker(raw_marked, center, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(raw_marked, "D{}".format(index), (center[0] + 8, center[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    width_m = float(config["bev"]["real_width_m"])
    length_m = float(config["bev"]["real_length_m"])
    offset_m = float(config["physical"]["local_forward_origin_offset_m"])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), gridspec_kw={"width_ratios": [1.0, 1.05]})
    axes[0].imshow(bgr_to_rgb(raw_marked))
    axes[0].axis("off")
    panel_label(axes[0], "A  原始透视图与检测中心", dark=True)

    ax = axes[1]
    ax.imshow(
        np.flipud(bgr_to_rgb(bev)),
        origin="lower",
        extent=[width_m / 2.0, -width_m / 2.0, offset_m, offset_m + length_m],
        aspect="auto",
    )
    centers = np.asarray([disk["center_base_m"] for disk in disks], dtype=np.float64)
    for index, (forward, lateral) in enumerate(centers, 1):
        ax.scatter(lateral, forward, s=80, c=COLORS["green"], edgecolors="white", linewidths=1.4, zorder=8)
        ax.text(lateral + 0.018, forward + 0.018, "D{}\n({:.3f}, {:.3f}) m".format(index, forward, lateral),
                color=COLORS["dark"], fontsize=9.5, zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d1d5db", alpha=0.88))
    p0, p1 = centers[0], centers[1]
    distance = float(np.linalg.norm(p1 - p0))
    ax.plot([p0[1], p1[1]], [p0[0], p1[0]], "--", color=COLORS["red"], linewidth=2.0, zorder=7)
    mid = 0.5 * (p0 + p1)
    ax.text(mid[1], mid[0], "本帧中心距 {:.3f} m".format(distance),
            ha="center", va="bottom", color=COLORS["red"], fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#fecaca", alpha=0.92), zorder=10)

    for key, color, label in [("left_base_m", COLORS["yellow"], "左边界"), ("right_base_m", "#f59e0b", "推算右边界")]:
        line = np.asarray(boundaries[key], dtype=np.float64)
        ax.plot(line[:, 1], line[:, 0], color=color, linewidth=3.0, label=label, zorder=6)
    left = np.asarray(boundaries["left_base_m"], dtype=np.float64)
    start, end = left[2], left[-3]
    ax.annotate("黄线方向", xy=(end[1], end[0]), xytext=(start[1], start[0]),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["blue"], lw=2.0),
                color=COLORS["blue"], fontsize=10.5, ha="left", va="bottom")
    ax.set_xlabel("横向 y（向左为正，m）")
    ax.set_ylabel("前向 x（m）")
    ax.grid(True, color="white", alpha=0.42, linewidth=0.7)
    ax.legend(loc="upper right", framealpha=0.9)
    panel_label(ax, "B  米制BEV与地面几何")

    fig.subplots_adjust(wspace=0.13)
    return save(fig, "fig04_perspective_to_metric_bev.png")


def figure_5_scale_validation():
    data = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    stats = data["locked_live_validation"]["center_distance_m"]
    actual = 500.0
    mean = float(stats["mean"]) * 1000.0
    std = float(stats["standard_deviation"]) * 1000.0
    minimum = float(stats["minimum"]) * 1000.0
    maximum = float(stats["maximum"]) * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    bars = ax.bar(["物理基准", "视觉测量均值"], [actual, mean], color=["#94a3b8", COLORS["blue"]], width=0.56)
    ax.errorbar([1], [mean], yerr=[std], fmt="none", ecolor=COLORS["red"], elinewidth=2, capsize=7, label="±1σ")
    ax.set_ylim(495.0, 505.0)
    ax.set_ylabel("两圆片中心距（mm）")
    ax.grid(axis="y", alpha=0.24)
    for bar, value in zip(bars, [actual, mean]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.35, "{:.3f}".format(value), ha="center", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    panel_label(ax, "A  物理基准与60帧均值")

    ax = axes[1]
    ax.hlines(1, minimum, maximum, color=COLORS["blue"], linewidth=5, alpha=0.42)
    ax.scatter([minimum, maximum], [1, 1], s=75, color=COLORS["blue"], zorder=3)
    ax.scatter([mean], [1], s=130, marker="D", color=COLORS["red"], edgecolor="white", linewidth=1.2, zorder=4)
    ax.axvline(actual, color=COLORS["gray"], linestyle="--", linewidth=1.7)
    ax.text(actual, 1.12, "500.000 mm基准", ha="center", color=COLORS["gray"])
    ax.text(mean, 0.84, "均值501.828 mm\n标准差1.010 mm", ha="center", va="top", color=COLORS["red"], fontweight="bold")
    ax.text(minimum, 1.06, "最小\n{:.3f}".format(minimum), ha="center", va="bottom", fontsize=9.5)
    ax.text(maximum, 1.06, "最大\n{:.3f}".format(maximum), ha="center", va="bottom", fontsize=9.5)
    ax.set_xlim(498.5, 504.5)
    ax.set_ylim(0.65, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("两圆片中心距（mm）")
    ax.grid(axis="x", alpha=0.24)
    panel_label(ax, "B  60帧测量范围")
    fig.subplots_adjust(wspace=0.23)
    return save(fig, "fig05_scale_validation_500mm.png")


def read_path(message):
    return np.asarray([(pose.pose.position.x, pose.pose.position.y) for pose in message.poses], dtype=np.float64)


def read_trial_bag():
    if not TRIAL_BAG.exists():
        raise FileNotFoundError(TRIAL_BAG)
    data = {
        "odom": [], "global_plans": [], "local_plans": [], "obstacles": [],
        "global_maps": [], "local_maps": [], "result": None,
    }
    with rosbag.Bag(str(TRIAL_BAG)) as bag:
        start = bag.get_start_time()
        data["start"] = start
        data["end"] = bag.get_end_time()
        for topic, message, stamp in bag.read_messages(topics=[
            "/odom",
            "/move_base/GlobalPlanner/plan",
            "/move_base/TebLocalPlannerROS/local_plan",
            "/move_base/TebLocalPlannerROS/obstacles",
            "/move_base/global_costmap/costmap",
            "/move_base/local_costmap/costmap",
            "/move_base/result",
        ]):
            t = stamp.to_sec()
            if topic == "/odom":
                pose = message.pose.pose
                data["odom"].append((t, pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation)))
            elif topic == "/move_base/GlobalPlanner/plan":
                path = read_path(message)
                if len(path): data["global_plans"].append((t, path))
            elif topic == "/move_base/TebLocalPlannerROS/local_plan":
                path = read_path(message)
                if len(path): data["local_plans"].append((t, path))
            elif topic == "/move_base/TebLocalPlannerROS/obstacles":
                primitives = []
                for obstacle in message.obstacles:
                    points = np.asarray([(point.x, point.y) for point in obstacle.polygon.points], dtype=np.float64)
                    primitives.append({"id": int(obstacle.id), "radius": float(obstacle.radius), "points": points})
                data["obstacles"].append((t, primitives))
            elif topic == "/move_base/global_costmap/costmap":
                data["global_maps"].append((t, message))
            elif topic == "/move_base/local_costmap/costmap":
                data["local_maps"].append((t, message))
            elif topic == "/move_base/result":
                data["result"] = {"time": t, "status": int(message.status.status), "text": message.status.text}
    data["odom"] = np.asarray(data["odom"], dtype=np.float64)
    return data


def nearest(sequence, target, before=False):
    usable = [item for item in sequence if (not before or item[0] <= target)]
    if not usable:
        usable = sequence
    return min(usable, key=lambda item: abs(item[0] - target))


def robot_at(trial, target):
    index = int(np.argmin(np.abs(trial["odom"][:, 0] - target)))
    return trial["odom"][index]


def draw_costmap(ax, message, alpha=0.72):
    grid = np.asarray(message.data, dtype=np.float32).reshape(message.info.height, message.info.width)
    masked = np.ma.masked_where(grid <= 0, grid)
    x0 = message.info.origin.position.x
    y0 = message.info.origin.position.y
    x1 = x0 + message.info.width * message.info.resolution
    y1 = y0 + message.info.height * message.info.resolution
    cmap = LinearSegmentedColormap.from_list("costmap", ["#dbeafe", "#60a5fa", "#f59e0b", "#111827"])
    cmap.set_bad((1, 1, 1, 0))
    ax.imshow(masked, origin="lower", extent=[x0, x1, y0, y1], cmap=cmap, vmin=1, vmax=100,
              interpolation="nearest", alpha=alpha, zorder=0)


def draw_obstacles(ax, primitives, labels=False):
    for obstacle in primitives:
        points = obstacle["points"]
        if len(points) == 2 and obstacle["radius"] == 0:
            ax.plot(points[:, 0], points[:, 1], color=COLORS["yellow"], linewidth=3.0,
                    label="黄色边界" if labels and obstacle["id"] == 1001 else None, zorder=4)
        elif len(points) == 1 and obstacle["radius"] > 0:
            ax.add_patch(Circle(tuple(points[0]), obstacle["radius"], facecolor="#ef4444",
                                edgecolor="#991b1b", linewidth=1.5, alpha=0.88, zorder=5,
                                label="圆片障碍" if labels and obstacle["id"] == 2002 else None))


def footprint_polygon(x, y, yaw, length=0.34, width=0.33):
    corners = np.asarray([[length/2, width/2], [length/2, -width/2], [-length/2, -width/2], [-length/2, width/2]])
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return corners.dot(rotation.T) + np.asarray([x, y])


def draw_robot(ax, odom_row, label=None):
    _, x, y, yaw = odom_row
    polygon = footprint_polygon(x, y, yaw)
    ax.add_patch(Polygon(polygon, closed=True, facecolor="#10b981", edgecolor="#065f46",
                         linewidth=1.8, alpha=0.76, zorder=8, label=label))
    ax.arrow(x, y, 0.17*math.cos(yaw), 0.17*math.sin(yaw), width=0.008,
             head_width=0.045, head_length=0.055, color="#064e3b", zorder=9,
             length_includes_head=True)


def figure_6_world_model(trial):
    odom = trial["odom"]
    index = int(np.argmin(np.abs(odom[:, 1] - 0.80)))
    target = odom[index, 0]
    row = odom[index]
    _, global_map = nearest(trial["global_maps"], target)
    _, local_map = nearest(trial["local_maps"], target)
    _, global_path = nearest(trial["global_plans"], target, before=True)
    _, local_path = nearest(trial["local_plans"], target)
    _, obstacles = nearest(trial["obstacles"], target)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    ax = axes[0]
    draw_costmap(ax, global_map, alpha=0.70)
    draw_obstacles(ax, obstacles, labels=True)
    ax.plot(global_path[:, 0], global_path[:, 1], color=COLORS["blue"], linewidth=2.2, label="GlobalPlanner路径", zorder=6)
    ax.plot(odom[:index+1, 1], odom[:index+1, 2], color=COLORS["green"], linewidth=1.6, linestyle="--", label="实车odom轨迹", zorder=6)
    draw_robot(ax, row, label="机器人footprint")
    ax.set_xlim(-0.10, 2.25)
    ax.set_ylim(-0.62, 0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("odom x（m）")
    ax.set_ylabel("odom y（m）")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    panel_label(ax, "A  全局costmap与全局路径")

    ax = axes[1]
    draw_costmap(ax, local_map, alpha=0.72)
    draw_obstacles(ax, obstacles, labels=True)
    ax.plot(global_path[:, 0], global_path[:, 1], color=COLORS["blue"], linewidth=1.6, alpha=0.75, label="全局路径", zorder=5)
    ax.plot(local_path[:, 0], local_path[:, 1], color=COLORS["purple"], linewidth=3.0, label="TEB局部轨迹", zorder=7)
    ax.plot(odom[:index+1, 1], odom[:index+1, 2], color=COLORS["green"], linewidth=1.5, linestyle="--", label="已行驶轨迹", zorder=6)
    draw_robot(ax, row, label="机器人footprint")
    ax.set_xlim(row[1]-0.75, row[1]+0.95)
    ax.set_ylim(-0.58, 0.58)
    ax.set_aspect("equal")
    ax.set_xlabel("odom x（m）")
    ax.set_ylabel("odom y（m）")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    panel_label(ax, "B  局部costmap与TEB轨迹")
    fig.subplots_adjust(wspace=0.18)
    return save(fig, "fig06_navigation_world_model.png")


def figure_7_s_curve(trial):
    odom = trial["odom"]
    x_targets = [0.10, 0.52, 0.80, 1.10, 1.55]
    labels = ["进入", "第一段横移", "两圆片之间", "第二段横移", "离开"]
    indices = [int(np.argmin(np.abs(odom[:, 1] - value))) for value in x_targets]
    _, obstacles = nearest(trial["obstacles"], odom[indices[2], 0])
    _, global_path = nearest(trial["global_plans"], odom[indices[2], 0], before=True)

    # A wide, shallow layout keeps the five true bag snapshots readable in a
    # Word/PDF column without leaving a large unused band below the axes.
    fig, axes = plt.subplots(1, 5, figsize=(18.0, 3.45), sharex=True, sharey=True)
    for panel, (ax, index, label) in enumerate(zip(axes, indices, labels), 1):
        target = odom[index, 0]
        _, local_path = nearest(trial["local_plans"], target)
        draw_obstacles(ax, obstacles)
        ax.plot(global_path[:, 0], global_path[:, 1], color="#94a3b8", linewidth=1.2, label="全局路径")
        ax.plot(local_path[:, 0], local_path[:, 1], color=COLORS["purple"], linewidth=2.1, label="TEB局部轨迹")
        ax.plot(odom[:index+1, 1], odom[:index+1, 2], color=COLORS["green"], linewidth=2.0, label="实车轨迹")
        draw_robot(ax, odom[index])
        ax.set_xlim(-0.06, 1.72)
        ax.set_ylim(-0.53, 0.55)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.18)
        ax.set_xlabel("x（m）")
        ax.text(0.5, 0.97, label, transform=ax.transAxes, ha="center", va="top",
                fontweight="bold", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d1d5db", alpha=0.9))
        ax.text(0.04, 0.04, "x={:.2f} m\ny={:.2f} m".format(odom[index,1], odom[index,2]),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=COLORS["dark"],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.83))
        if panel == 1:
            ax.set_ylabel("y（m）")
    handles = [
        plt.Line2D([0], [0], color=COLORS["green"], lw=2, label="实车odom轨迹"),
        plt.Line2D([0], [0], color=COLORS["purple"], lw=2, label="TEB局部轨迹"),
        plt.Line2D([0], [0], color="#94a3b8", lw=2, label="GlobalPlanner路径"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#ef4444", markeredgecolor="#991b1b", label="圆片"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.05, bottom=0.23, top=0.98)
    return save(fig, "fig07_stage1_s_curve_five_moments.png")


def figure_8_cpu():
    means = [283.4, 8.87]
    p95 = [294.9, 12.37]
    maximum = [301.0, 12.77]
    labels = ["ACQUIRE\n完整感知", "LOCKED\n跳过重型处理"]
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    x = np.arange(2)
    bars = ax.bar(x, means, color=[COLORS["orange"], COLORS["teal"]], width=0.56, label="均值")
    ax.scatter(x, p95, color=COLORS["blue"], marker="D", s=70, label="p95", zorder=4)
    ax.scatter(x, maximum, color=COLORS["red"], marker="^", s=75, label="最大值", zorder=4)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, value+7, "{:.2f}%".format(value), ha="center", fontweight="bold", fontsize=12)
    ax.annotate("CPU均值下降约96.9%", xy=(1, means[1]), xytext=(0.52, 185),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color=COLORS["teal"]),
                color=COLORS["teal"], fontsize=12, fontweight="bold", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("视觉进程CPU占用（%）")
    ax.set_ylim(0, 325)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right")
    return save(fig, "fig08_acquire_locked_cpu_ab.png")


def figure_9_disc_acceptance():
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    ax = axes[0]
    ax.barh([0], [183], color=COLORS["green"], height=0.42, label="双圆片检出")
    ax.barh([0], [1], left=[183], color=COLORS["red"], height=0.42, label="漏检")
    ax.set_xlim(0, 184)
    ax.set_yticks([0])
    ax.set_yticklabels(["正样本184帧"])
    ax.set_xlabel("帧数")
    ax.text(91.5, 0, "183", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax.text(181.5, 0.22, "漏检1帧", color=COLORS["red"], ha="right", fontsize=10)
    ax.text(0.98, 0.10, "双圆片帧级召回率 99.46%", transform=ax.transAxes,
            ha="right", color=COLORS["green"], fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.20)
    ax.legend(loc="upper left", frameon=False)
    panel_label(ax, "A  有圆片正样本")

    ax = axes[1]
    ax.barh([0], [336], color=COLORS["blue"], height=0.42, label="正确无检出")
    ax.set_xlim(0, 336)
    ax.set_yticks([0])
    ax.set_yticklabels(["负样本336帧"])
    ax.set_xlabel("帧数")
    ax.text(168, 0, "336", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax.text(0.98, 0.10, "误检 0帧，误检率 0%", transform=ax.transAxes,
            ha="right", color=COLORS["blue"], fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.20)
    ax.legend(loc="upper left", frameon=False)
    panel_label(ax, "B  无圆片负样本")
    fig.subplots_adjust(wspace=0.26)
    return save(fig, "fig09_disc_520frame_acceptance.png")


def draw_nav_overview(ax, trial):
    odom = trial["odom"]
    index = int(np.argmin(np.abs(odom[:, 1] - 0.80)))
    target = odom[index, 0]
    _, global_path = nearest(trial["global_plans"], target, before=True)
    _, local_path = nearest(trial["local_plans"], target)
    _, obstacles = nearest(trial["obstacles"], target)
    draw_obstacles(ax, obstacles)
    ax.plot(global_path[:, 0], global_path[:, 1], color=COLORS["blue"], linewidth=2.0)
    ax.plot(local_path[:, 0], local_path[:, 1], color=COLORS["purple"], linewidth=2.8)
    ax.plot(odom[:index+1, 1], odom[:index+1, 2], color=COLORS["green"], linewidth=1.8, linestyle="--")
    draw_robot(ax, odom[index])
    ax.set_xlim(-0.05, 1.72)
    ax.set_ylim(-0.54, 0.58)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.16)
    ax.set_xlabel("odom x（m）")
    ax.set_ylabel("odom y（m）")


def figure_10_closed_loop(messages, config, trial):
    perception = cv2.imread(str(SOURCES / "disc_overlay.png"), cv2.IMREAD_COLOR)
    bev = cv2.imread(str(SOURCES / "bev.png"), cv2.IMREAD_COLOR)
    disks = live_json(messages, "/yellow_track_avoidance/debug/disks")["raw"]
    width_m = float(config["bev"]["real_width_m"])
    length_m = float(config["bev"]["real_length_m"])
    offset_m = float(config["physical"]["local_forward_origin_offset_m"])

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.0), gridspec_kw={"width_ratios": [1.2, 0.78, 1.15]})
    axes[0].imshow(bgr_to_rgb(perception))
    axes[0].axis("off")
    panel_label(axes[0], "A  感知：黄线与圆片", dark=True)

    ax = axes[1]
    ax.imshow(np.flipud(bgr_to_rgb(bev)), origin="lower",
              extent=[width_m/2, -width_m/2, offset_m, offset_m+length_m], aspect="auto")
    for disk in disks:
        forward, lateral = disk["center_base_m"]
        ax.scatter(lateral, forward, s=60, color=COLORS["green"], edgecolor="white", zorder=6)
    ax.set_xlabel("y（m）")
    ax.set_ylabel("x（m）")
    ax.grid(True, color="white", alpha=0.35)
    panel_label(ax, "B  建模：米制BEV")

    draw_nav_overview(axes[2], trial)
    panel_label(axes[2], "C  规划：全局与局部轨迹")

    for left, right, text in [(axes[0], axes[1], "地面投影"), (axes[1], axes[2], "世界坐标")]:
        p0 = left.get_position(); p1 = right.get_position()
        fig.add_artist(FancyArrowPatch((p0.x1+0.006, (p0.y0+p0.y1)/2),
                                       (p1.x0-0.006, (p1.y0+p1.y1)/2),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=18, linewidth=2.0, color=COLORS["gray"]))
        fig.text((p0.x1+p1.x0)/2, (p0.y0+p0.y1)/2+0.04, text,
                 ha="center", va="bottom", fontsize=10, color=COLORS["gray"])
    fig.subplots_adjust(wspace=0.23)
    return save(fig, "fig10_visual_navigation_closed_loop.png")


def write_manifest(metadata, messages, trial, outputs):
    disks = live_json(messages, "/yellow_track_avoidance/debug/disks")["raw"]
    boundaries = live_json(messages, "/yellow_track_avoidance/debug/boundaries")
    centers = np.asarray([item["center_base_m"] for item in disks], dtype=np.float64)
    manifest = {
        "generated_on": "2026-08-11",
        "generation_script": str(Path(__file__).resolve()),
        "live_capture": metadata,
        "live_selected_frame": {
            "disc_count": len(disks),
            "disc_confidences": [item["confidence"] for item in disks],
            "disc_estimated_diameters_m": [item["estimated_diameter_m"] for item in disks],
            "disc_base_centers_m": centers.tolist(),
            "disc_center_distance_m": float(np.linalg.norm(centers[1] - centers[0])),
            "left_boundary_quality": boundaries["quality"],
            "boundary_detection_mode": boundaries["boundary_detection_mode"],
        },
        "stage1_bag": {
            "path": str(TRIAL_BAG),
            "sha256": sha256(TRIAL_BAG),
            "duration_sec": trial["end"] - trial["start"],
            "move_base_result": trial["result"],
        },
        "reported_statistics": {
            "scale_validation_source": str(CALIBRATION_REPORT),
            "acquire_locked_source": str(ACQUIRE_REPORT),
            "disc_acceptance_source": str(DISC_REPORT),
        },
        "generated_images": [str(path) for path in outputs],
        "truth_constraints": [
            "No vehicle motion was started for the live image capture.",
            "Figures 2-4 and 10 use a real synchronized RGB/debug frame from the current source and config.",
            "Figures 6-7 use messages recorded in the successful Stage1 Trial 7 bag.",
            "Figures 5, 8 and 9 use only values already present in checked project reports.",
            "No missing camera sequence was synthesized for the five-moment S-curve figure.",
        ],
    }
    with (OUT / "source_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main():
    ensure_dirs()
    raw, messages, metadata = extract_capture()
    analysis = load_project_pipeline(raw)
    config = analysis[0]
    trial = read_trial_bag()

    outputs = [
        figure_1_architecture(),
        figure_2_yellow(raw, analysis),
        figure_3_discs(analysis),
        figure_4_bev(raw, messages, config),
        figure_5_scale_validation(),
        figure_6_world_model(trial),
        figure_7_s_curve(trial),
        figure_8_cpu(),
        figure_9_disc_acceptance(),
        figure_10_closed_loop(messages, config, trial),
    ]
    write_manifest(metadata, messages, trial, outputs)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
