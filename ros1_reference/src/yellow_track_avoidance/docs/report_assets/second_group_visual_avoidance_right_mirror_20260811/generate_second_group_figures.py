#!/usr/bin/env python3
"""Generate the second team's right-mirrored visual-avoidance report figures.

Every raster measurement in this script comes from the live 2026-08-11 capture,
the lock-state planner capture, or an existing acceptance report.  The script
does not start ROS, publish a goal, or control the chassis.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import rosbag

# Ubuntu 20.04 ships an older Matplotlib polar projection that still accesses
# NumPy's removed compatibility alias.
if "float" not in np.__dict__:
    np.float = float


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCES = OUT / "sources"
CAPTURE_BAG = Path("/tmp/second_group_right_mirror_20260811.bag")
PLAN_BAG = Path("/tmp/second_group_right_mirror_plans_20260811.bag")
COMMON_GENERATOR = (
    ROOT / "docs/report_assets/vision_avoidance_20260811/generate_report_figures.py"
)
CONFIG = ROOT / "config/current_car_recalibrated_640x480_20260809.yaml"
DISC_REPORT = ROOT / "docs/perception/DISC_DETECTION_ACCEPTANCE.md"
PERCEPTION_REPORT = ROOT / "docs/navigation/PERCEPTION_ACQUIRE_LOCKED_REPORT_20260810.md"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["AR PL UKai CN"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.8,
    }
)

C = {
    "ink": "#172033",
    "muted": "#64748b",
    "paper": "#f8fafc",
    "blue": "#1d4ed8",
    "cyan": "#0891b2",
    "teal": "#0f766e",
    "green": "#15803d",
    "yellow": "#d6a700",
    "orange": "#ea580c",
    "red": "#dc2626",
    "rose": "#be123c",
    "purple": "#7e22ce",
    "slate": "#334155",
}


def load_common():
    spec = importlib.util.spec_from_file_location("first_report_common", COMMON_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.SOURCES = SOURCES
    module.CAPTURE_BAG = CAPTURE_BAG
    return module


COMMON = load_common()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig, filename: str) -> Path:
    path = OUT / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return path


def bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_json_messages(bag_path: Path, topics=None):
    rows = []
    with rosbag.Bag(str(bag_path)) as bag:
        for topic, message, bag_time in bag.read_messages(topics=topics):
            if not hasattr(message, "data"):
                continue
            try:
                data = json.loads(message.data)
            except Exception:
                continue
            stamp = float(data.get("stamp", data.get("timestamp", bag_time.to_sec())))
            rows.append((topic, stamp, data))
    return rows


def nearest_json(rows, topic, stamp):
    choices = [item for item in rows if item[0] == topic]
    if not choices:
        return {}
    return min(choices, key=lambda item: abs(item[1] - stamp))[2]


def path_xy(message):
    return np.asarray(
        [[pose.pose.position.x, pose.pose.position.y] for pose in message.poses],
        dtype=float,
    )


def load_plan_data():
    data = {
        "global_paths": [],
        "local_paths": [],
        "teb_global_paths": [],
        "preview_paths": [],
        "obstacles": [],
        "global_costmap": None,
        "local_costmap": None,
        "footprints": [],
        "odom": [],
    }
    with rosbag.Bag(str(PLAN_BAG)) as bag:
        for topic, message, bag_time in bag.read_messages():
            if topic == "/move_base/GlobalPlanner/plan":
                data["global_paths"].append(path_xy(message))
            elif topic == "/move_base/TebLocalPlannerROS/local_plan":
                data["local_paths"].append(path_xy(message))
            elif topic == "/move_base/TebLocalPlannerROS/global_plan":
                data["teb_global_paths"].append(path_xy(message))
            elif topic == "/yellow_track_avoidance/corridor_goal_plan":
                data["preview_paths"].append(path_xy(message))
            elif topic == "/move_base/TebLocalPlannerROS/obstacles":
                data["obstacles"].append(message)
            elif topic == "/move_base/global_costmap/costmap":
                data["global_costmap"] = message
            elif topic == "/move_base/local_costmap/costmap":
                data["local_costmap"] = message
            elif topic.endswith("/footprint"):
                points = np.asarray([[p.x, p.y] for p in message.polygon.points], dtype=float)
                if len(points):
                    data["footprints"].append(points)
            elif topic == "/odom":
                data["odom"].append(
                    (bag_time.to_sec(), message.pose.pose.position.x, message.pose.pose.position.y)
                )
    return data


def load_capture_data():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    raw, selected, metadata = COMMON.extract_capture()
    analysis = COMMON.load_project_pipeline(raw)
    target = float(metadata["selected_image_stamp"])
    topics = [
        "/yellow_track_avoidance/debug/disc_status",
        "/yellow_track_avoidance/debug/disks",
        "/yellow_track_disc_costmap_bridge/status",
        "/yellow_track_competition_corridor_map/status",
        "/yellow_track_avoidance/corridor_boundaries",
        "/yellow_track_teb_compact_obstacles/status",
    ]
    rows = read_json_messages(CAPTURE_BAG, topics)
    snapshot = {topic: nearest_json(rows, topic, target) for topic in topics}
    with (SOURCES / "second_group_snapshot.json").open("w", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2)
    rviz = Path("/tmp/second_group_rviz_window.png")
    if rviz.exists():
        shutil.copy2(rviz, SOURCES / "rviz_lock_state.png")
    return raw, analysis, metadata, rows, snapshot


def accepted_detections(analysis):
    return sorted(analysis[-1], key=lambda item: item.center_px[0])


def crop_bounds(shape, ellipse, scale=2.2):
    cx, cy = ellipse[0]
    radius = max(ellipse[1]) * scale / 2.0
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(shape[1], int(math.ceil(cx + radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(shape[0], int(math.ceil(cy + radius)))
    return x0, y0, x1, y1


def sector_ratios(detection, red_mask, sectors=20):
    (cx, cy), (axis_a, axis_b), angle_deg = detection.ellipse
    yy, xx = np.indices(red_mask.shape[:2], dtype=float)
    theta = math.radians(float(angle_deg))
    dx, dy = xx - cx, yy - cy
    local_x = dx * math.cos(theta) + dy * math.sin(theta)
    local_y = -dx * math.sin(theta) + dy * math.cos(theta)
    nx = local_x / max(1e-6, 0.5 * axis_a)
    ny = local_y / max(1e-6, 0.5 * axis_b)
    radius = np.sqrt(nx * nx + ny * ny)
    phi = (np.arctan2(ny, nx) + 2.0 * math.pi) % (2.0 * math.pi)
    ratios = []
    for index in range(sectors):
        lo = 2.0 * math.pi * index / sectors
        hi = 2.0 * math.pi * (index + 1) / sectors
        region = (radius >= 0.48) & (radius <= 1.08) & (phi >= lo) & (phi < hi)
        ratios.append(float(np.mean(red_mask[region] > 0)) if np.any(region) else 0.0)
    return np.asarray(ratios)


def figure_01_layered_architecture():
    fig, ax = plt.subplots(figsize=(14.8, 8.0))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 8)
    ax.axis("off")

    bands = [
        (5.55, 7.35, "感知层", "640×480 RGB", "黄线连续置信度与法向截面", "红环—白心几何验证", "#eff6ff", C["blue"]),
        (3.15, 5.1, "米制建模层", "去畸变与地面单应性", "车体坐标（m）", "双圆片与走廊冻结到 odom", "#ecfeff", C["teal"]),
        (0.7, 2.7, "规划执行层", "完整 costmap", "2条线 + 2个圆语义障碍", "GlobalPlanner + TEB → 麦轮底盘", "#fff7ed", C["orange"]),
    ]
    for y0, y1, title, a, b, c, face, edge in bands:
        ax.add_patch(FancyBboxPatch((0.6, y0), 10.7, y1-y0, boxstyle="round,pad=0.18", facecolor=face, edgecolor=edge, linewidth=1.8))
        ax.text(0.95, (y0+y1)/2, title, color=edge, fontsize=16, fontweight="bold", va="center")
        xs = [3.1, 5.85, 8.6]
        for x, text in zip(xs, [a, b, c]):
            ax.add_patch(FancyBboxPatch((x, y0+0.42), 2.35, y1-y0-0.84, boxstyle="round,pad=0.13", facecolor="white", edgecolor="#cbd5e1", linewidth=1.0))
            ax.text(x+1.175, (y0+y1)/2, text, ha="center", va="center", color=C["ink"], fontsize=11.2, wrap=True)
        for x in [5.53, 8.28]:
            ax.add_patch(FancyArrowPatch((x, (y0+y1)/2), (x+0.27, (y0+y1)/2), arrowstyle="-|>", mutation_scale=13, color=edge, linewidth=1.4))
    for y0, y1 in [(5.1, 5.55), (2.7, 3.15)]:
        ax.add_patch(FancyArrowPatch((5.95, y1), (5.95, y0), arrowstyle="-|>", mutation_scale=16, color=C["slate"], linewidth=1.6))

    ax.add_patch(FancyBboxPatch((11.7, 0.7), 2.45, 6.65, boxstyle="round,pad=0.18", facecolor="#f1f5f9", edgecolor="#64748b", linewidth=1.5))
    ax.text(12.925, 6.9, "任务态资源调度", ha="center", va="center", fontsize=14.5, fontweight="bold", color=C["slate"])
    ax.add_patch(FancyBboxPatch((12.05, 4.65), 1.75, 1.35, boxstyle="round,pad=0.12", facecolor="#fee2e2", edgecolor=C["red"], linewidth=1.5))
    ax.text(12.925, 5.33, "ACQUIRE", ha="center", fontweight="bold", color=C["red"], fontsize=13)
    ax.text(12.925, 4.94, "完整识别\n建立世界模型", ha="center", va="center", fontsize=10.5)
    ax.add_patch(FancyArrowPatch((12.925, 4.5), (12.925, 3.55), arrowstyle="-|>", mutation_scale=17, color=C["slate"], linewidth=1.8))
    ax.text(13.12, 4.03, "走廊+双圆片\n连续稳定0.5 s", fontsize=9.3, va="center", color=C["muted"])
    ax.add_patch(FancyBboxPatch((12.05, 1.95), 1.75, 1.35, boxstyle="round,pad=0.12", facecolor="#dcfce7", edgecolor=C["green"], linewidth=1.5))
    ax.text(12.925, 2.62, "LOCKED", ha="center", fontweight="bold", color=C["green"], fontsize=13)
    ax.text(12.925, 2.23, "跳过重计算\n下游持续发布", ha="center", va="center", fontsize=10.5)
    ax.text(0.65, 7.72, "右侧镜像布局的视觉避障数据生命周期", fontsize=18, fontweight="bold", color=C["ink"])
    return save(fig, "B01-视觉避障分层架构与任务态调度.png")


def figure_02_scene_card(raw, snapshot):
    tracks = snapshot["/yellow_track_disc_costmap_bridge/status"].get("active_tracks", [])
    fig = plt.figure(figsize=(14.5, 7.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.7], wspace=0.03)
    ax = fig.add_subplot(grid[0, 0]); ax.imshow(bgr(raw)); ax.axis("off")
    ax.text(0.02, 0.96, "车载RGB实景（右侧镜像错列）", transform=ax.transAxes, va="top", color="white", fontsize=15, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#111827", alpha=0.78, edgecolor="none"))
    ax2 = fig.add_subplot(grid[0, 1]); ax2.axis("off"); ax2.set_facecolor(C["paper"])
    ax2.text(0.08, 0.93, "现场状态", fontsize=17, fontweight="bold", color=C["ink"], transform=ax2.transAxes)
    items = [
        ("黄线走廊", "已锁定"),
        ("图像圆片", "2枚"),
        ("odom冻结", "2枚"),
        ("costmap写入", "2枚"),
        ("车辆运动", "锁止采集"),
    ]
    y = 0.83
    for key, value in items:
        ax2.text(0.08, y, key, color=C["muted"], fontsize=11, transform=ax2.transAxes)
        ax2.text(0.92, y, value, color=C["green"], fontsize=12, fontweight="bold", ha="right", transform=ax2.transAxes)
        ax2.plot([0.08,0.92],[y-0.035,y-0.035],transform=ax2.transAxes,color="#e2e8f0",lw=0.8)
        y -= 0.105
    ax2.text(0.08, y-0.01, "冻结圆片坐标 / m", fontsize=12.5, fontweight="bold", color=C["ink"], transform=ax2.transAxes)
    y -= 0.08
    for index, track in enumerate(sorted(tracks, key=lambda t: t["fixed_xy"][0]), 1):
        x_m, y_m = track["fixed_xy"]
        ax2.add_patch(Circle((0.14, y+0.015), 0.026, transform=ax2.transAxes, facecolor="#fee2e2", edgecolor=C["red"], lw=1.3))
        ax2.text(0.14, y+0.015, str(index), transform=ax2.transAxes, ha="center", va="center", fontsize=9, color=C["red"], fontweight="bold")
        ax2.text(0.22, y+0.015, f"x={x_m:.3f}\ny={y_m:+.3f}", transform=ax2.transAxes, va="center", fontsize=11.2, color=C["slate"])
        y -= 0.13
    ax2.text(0.08, 0.08, "注：本图仅记录真实采集场景，\n未补画车辆轨迹或检测结果。", transform=ax2.transAxes, fontsize=9.5, color=C["muted"], va="bottom")
    return save(fig, "B02-右侧镜像双圆片实景与锁定状态.png")


def figure_03_red_segmentation(raw, analysis):
    masks = analysis[3]; detections = accepted_detections(analysis)
    red = masks["red"]
    dim = (raw.astype(np.float32) * 0.27).astype(np.uint8)
    isolated = np.zeros_like(raw)
    isolated[:,:,2] = red
    composite = cv2.addWeighted(dim, 1.0, isolated, 0.95, 0.0)
    for index, det in enumerate(detections, 1):
        cv2.drawContours(composite, [det.contour], -1, (30, 230, 255), 2, cv2.LINE_AA)
        cv2.ellipse(composite, det.ellipse, (0, 255, 120), 2, cv2.LINE_AA)
        c = tuple(np.rint(det.ellipse[0]).astype(int))
        cv2.putText(composite, f"D{index}", (c[0]+7,c[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    fig, ax = plt.subplots(figsize=(13.8, 8.5))
    ax.imshow(bgr(composite)); ax.axis("off")
    ax.text(0.02,0.96,"红色像素保留，非红区域降亮度",transform=ax.transAxes,va="top",color="white",fontsize=15,fontweight="bold",bbox=dict(boxstyle="round,pad=0.3",facecolor="#111827",alpha=0.78,edgecolor="none"))
    for index, det in enumerate(detections, 1):
        x0,y0,x1,y1=crop_bounds(raw.shape,det.ellipse,2.5)
        inset=ax.inset_axes([0.64,0.62-(index-1)*0.31,0.31,0.25])
        crop=raw[y0:y1,x0:x1].copy(); mask=red[y0:y1,x0:x1]
        crop[mask==0]=(crop[mask==0]*0.18).astype(np.uint8)
        inset.imshow(bgr(crop)); inset.axis("off")
        inset.set_title(f"D{index} 红环连通域  面积={det.red_area_px:.0f}px",fontsize=10.5,color=C["ink"],loc="left")
    ax.text(0.02,0.035,"红色候选仅负责提出可能目标；最终还需白心、同心性、20扇区覆盖和米制尺寸共同确认。",transform=ax.transAxes,color="white",fontsize=11.5,bbox=dict(boxstyle="round,pad=0.35",facecolor="#111827",alpha=0.75,edgecolor="none"))
    return save(fig, "B03-红色环带分割与连通域提取.png")


def figure_04_white_center_and_sectors(raw, analysis):
    masks = analysis[3]; detections = accepted_detections(analysis)
    fig = plt.figure(figsize=(14.8, 9.3))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.55,1.15,1.15,1.6], hspace=0.32, wspace=0.20)
    for row, det in enumerate(detections[:2]):
        x0,y0,x1,y1=crop_bounds(raw.shape,det.ellipse,2.4)
        crop=raw[y0:y1,x0:x1].copy()
        local_ellipse=((det.ellipse[0][0]-x0,det.ellipse[0][1]-y0),det.ellipse[1],det.ellipse[2])
        cv2.ellipse(crop,local_ellipse,(0,255,0),2,cv2.LINE_AA)
        center=(int(round(local_ellipse[0][0])),int(round(local_ellipse[0][1])))
        cv2.drawMarker(crop,center,(255,255,255),cv2.MARKER_CROSS,12,2)
        ax=fig.add_subplot(grid[row,0]);ax.imshow(bgr(crop));ax.axis("off");ax.set_title(f"D{row+1} 原始局部与拟合椭圆",loc="left",fontweight="bold")
        ax=fig.add_subplot(grid[row,1]);ax.imshow(masks["red"][y0:y1,x0:x1],cmap="Reds",vmin=0,vmax=255);ax.axis("off");ax.set_title("红环掩膜",loc="left")
        ax=fig.add_subplot(grid[row,2]);ax.imshow(masks["white"][y0:y1,x0:x1],cmap="gray",vmin=0,vmax=255);ax.axis("off");ax.set_title("低饱和亮白中心",loc="left")
        ratios=sector_ratios(det,masks["red"])
        ax=fig.add_subplot(grid[row,3],projection="polar")
        theta=np.linspace(0,2*math.pi,20,endpoint=False);width=2*math.pi/20
        colors=[C["green"] if value>=0.12 else "#fecaca" for value in ratios]
        ax.bar(theta,ratios,width=width*0.86,bottom=0,color=colors,edgecolor="white",linewidth=0.5)
        ax.axhline(0.12,color=C["red"],ls="--",lw=1.0);ax.set_ylim(0,1);ax.set_yticks([0.12,0.5,1.0]);ax.set_xticks([])
        ax.set_title(f"20扇区覆盖={det.red_sector_coverage:.2f}",fontsize=11,pad=12)
        ax.text(1.23,0.08,
                f"红环比 {det.red_annulus_ratio:.2f}\n白心比 {det.white_center_ratio:.2f}\n亮度对比 {det.white_center_contrast:.1f}\n置信度 {det.confidence:.3f}",
                transform=ax.transAxes,va="bottom",fontsize=9.6,color=C["slate"],bbox=dict(boxstyle="round,pad=0.3",facecolor="#f8fafc",edgecolor="#cbd5e1"))
    fig.suptitle("红环完整性、白心中性特征与同心性联合验证",fontsize=17,fontweight="bold",x=0.47)
    return save(fig, "B04-白心中性特征与20扇区验证.png")


def figure_05_candidate_funnel(snapshot):
    status=snapshot["/yellow_track_avoidance/debug/disc_status"]
    candidates=status.get("candidates",[]); detections=status.get("detections",[])
    accepted=[c for c in candidates if c.get("accepted")]
    projected=[c for c in accepted if c.get("metric_projection")]
    rejected=Counter(c.get("reject_reason") or "accepted" for c in candidates if not c.get("accepted"))
    stages=[("初始候选",len(candidates)),("颜色+几何通过",len(accepted)),("米制投影有效",len(projected)),("本帧正式输出",len(detections)),("连续确认轨迹",2)]
    fig=plt.figure(figsize=(14.2,8.0));grid=fig.add_gridspec(1,2,width_ratios=[1.1,1.35],wspace=0.25)
    ax=fig.add_subplot(grid[0,0]);ax.set_xlim(-1.1,1.1);ax.set_ylim(-0.2,len(stages)+0.45);ax.axis("off")
    max_count=max(1,stages[0][1])
    widths=[1.0,0.65,0.53,0.43,0.34]
    colors=["#dbeafe","#bfdbfe","#99f6e4","#86efac","#4ade80"]
    for i,((label,count),width,color) in enumerate(zip(stages,widths,colors)):
        y=len(stages)-i-0.8
        next_width=widths[min(i+1,len(widths)-1)]
        poly=np.array([[-width,y+0.65],[width,y+0.65],[next_width,y-0.05],[-next_width,y-0.05]])
        ax.add_patch(Polygon(poly,closed=True,facecolor=color,edgecolor="white",linewidth=2))
        ax.text(0,y+0.30,f"{label}  {count}",ha="center",va="center",fontsize=12.2,fontweight="bold",color=C["ink"])
    ax.set_title("候选逐级收敛",fontsize=16,fontweight="bold",pad=12)
    ax2=fig.add_subplot(grid[0,1])
    top=rejected.most_common(8);labels=[item[0].replace("_","\n") for item in top][::-1];values=[item[1] for item in top][::-1]
    ax2.barh(np.arange(len(top)),values,color=plt.cm.Reds(np.linspace(0.35,0.85,len(top))),edgecolor="white")
    ax2.set_yticks(np.arange(len(top)));ax2.set_yticklabels(labels,fontsize=9.5);ax2.set_xlabel("候选数量");ax2.set_title("未通过候选的真实原因分布",fontsize=16,fontweight="bold")
    ax2.grid(axis="x",alpha=0.22)
    for y,value in enumerate(values):ax2.text(value+0.2,y,str(value),va="center",fontsize=10)
    fig.suptitle("本帧圆片候选筛选证据",fontsize=18,fontweight="bold",y=1.01)
    return save(fig,"B05-圆片候选筛选漏斗与拒绝原因.png")


def figure_06_yellow_confidence(raw, analysis):
    config,pipeline,working,masks,profiles,selected,boundary,_=analysis
    confidence=masks["yellow_confidence"]
    heat=cv2.applyColorMap(np.uint8(np.clip(confidence,0,1)*255),cv2.COLORMAP_INFERNO)
    overlay=cv2.addWeighted(working,0.68,heat,0.58,0)
    points=np.asarray([profiles[i].coordinates[selected[i]] for i in range(len(profiles))])
    scores=np.asarray([profiles[i].score[selected[i]] for i in range(len(profiles))])
    cfg=config["normal_profile_detector"]
    valid=scores>=float(cfg["minimum_section_confidence"])
    for p,ok in zip(points,valid):cv2.circle(overlay,tuple(np.rint(p).astype(int)),4,(30,230,60) if ok else (0,120,255),-1,cv2.LINE_AA)
    finite=np.isfinite(boundary.left_px)
    fit=np.column_stack([boundary.left_px[finite],boundary.rows_px[finite]])
    if len(fit)>1:cv2.polylines(overlay,[np.rint(fit).astype(np.int32)],False,(0,255,255),3,cv2.LINE_AA)
    metrics=pipeline.profile_boundary_detector.last_metrics
    fig=plt.figure(figsize=(14.5,7.8));grid=fig.add_gridspec(2,2,width_ratios=[3.2,1.25],height_ratios=[1,1],wspace=0.18,hspace=0.27)
    ax=fig.add_subplot(grid[:,0]);ax.imshow(bgr(overlay));ax.axis("off");ax.set_title("连续黄色置信度叠加、DP支持点与拟合线",fontsize=15,fontweight="bold",loc="left")
    ax1=fig.add_subplot(grid[0,1]);ax1.plot(np.arange(len(scores))+1,scores,"o-",color=C["blue"],lw=1.5,ms=4);ax1.axhline(float(cfg["minimum_section_confidence"]),color=C["red"],ls="--",label="截面阈值");ax1.set_xlabel("法向截面编号");ax1.set_ylabel("DP候选得分");ax1.set_ylim(0,1.05);ax1.grid(alpha=0.22);ax1.legend(frameon=False,fontsize=9)
    ax2=fig.add_subplot(grid[1,1]);ax2.axis("off")
    rows=[("法向截面",str(len(profiles))),("有效截面",str(metrics.get("left_valid_sections"))),("整线质量",f"{metrics.get('left_quality',0):.3f}"),("拟合RMS",f"{metrics.get('left_fit_rms_px',0):.2f} px"),("输出模式","仅检测左黄线")]
    for i,(k,v) in enumerate(rows):
        y=0.88-i*0.17;ax2.text(0.03,y,k,color=C["muted"],fontsize=11,transform=ax2.transAxes);ax2.text(0.97,y,v,color=C["ink"],fontsize=12,fontweight="bold",ha="right",transform=ax2.transAxes);ax2.plot([0.03,0.97],[y-0.06,y-0.06],transform=ax2.transAxes,color="#e2e8f0",lw=0.8)
    return save(fig,"B06-黄线连续置信度与截面得分诊断.png")


def figure_07_coordinate_correspondence(raw, snapshot):
    disks=snapshot["/yellow_track_avoidance/debug/disks"]
    tracks=sorted(disks.get("raw",[]),key=lambda d:d["center_base_m"][0])
    canvas=raw.copy()
    colors=[(255,170,0),(0,120,255)]
    for i,(d,color) in enumerate(zip(tracks,colors),1):
        px=d["metric_projection"]["undistorted_full_image_px"]
        c=tuple(np.rint(px).astype(int));cv2.circle(canvas,c,8,color,2,cv2.LINE_AA);cv2.putText(canvas,f"P{i}",(c[0]+10,c[1]-8),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2,cv2.LINE_AA)
    fig=plt.figure(figsize=(14.2,8.2));grid=fig.add_gridspec(2,1,height_ratios=[3.3,1.3],hspace=0.08)
    ax=fig.add_subplot(grid[0]);ax.imshow(bgr(canvas));ax.axis("off");ax.set_title("像素检测中心",loc="left",fontsize=15,fontweight="bold")
    ax2=fig.add_subplot(grid[1]);ax2.axis("off")
    columns=["目标","图像中心 (u,v)/px","车体坐标 (x,y)/m","估计直径/m","置信度"]
    values=[]
    for i,d in enumerate(tracks,1):
        uv=d["metric_projection"]["undistorted_full_image_px"];xy=d["center_base_m"]
        values.append([f"P{i}",f"({uv[0]:.1f}, {uv[1]:.1f})",f"({xy[0]:.3f}, {xy[1]:+.3f})",f"{d['estimated_diameter_m']:.3f}",f"{d['confidence']:.3f}"])
    table=ax2.table(cellText=values,colLabels=columns,cellLoc="center",colLoc="center",loc="center",colWidths=[0.10,0.25,0.25,0.18,0.16])
    table.auto_set_font_size(False);table.set_fontsize(11.5);table.scale(1,2.1)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1");cell.set_linewidth(0.8)
        if row==0:cell.set_facecolor("#e0f2fe");cell.set_text_props(weight="bold",color=C["ink"])
        else:cell.set_facecolor("#f8fafc")
    ax2.text(0.5,-0.03,"同一像素中心经去畸变与地面单应性转换为规划器可用的米制坐标",transform=ax2.transAxes,ha="center",fontsize=11,color=C["muted"])
    return save(fig,"B07-圆片像素中心到车体米制坐标对应.png")


def collect_track_series(rows):
    series=defaultdict(list)
    topic="/yellow_track_avoidance/debug/disks"
    for _,stamp,data in rows:
        if _ != topic:
            continue
        for track in data.get("confirmed_tracks",[]):
            xy=track.get("center_base_m")
            if not xy:continue
            series[int(track["track_id"])].append((stamp,float(xy[0]),float(xy[1]),float(track.get("estimated_diameter_m",np.nan)),float(track.get("confidence",np.nan))))
    useful={key:np.asarray(value,float) for key,value in series.items() if len(value)>=5}
    return useful


def confidence_ellipse(ax, x, y, color):
    if len(x)<3:return
    cov=np.cov(np.vstack([x,y]));vals,vecs=np.linalg.eigh(cov);order=vals.argsort()[::-1];vals=vals[order];vecs=vecs[:,order]
    angle=math.degrees(math.atan2(vecs[1,0],vecs[0,0]));width,height=2*2.4477*np.sqrt(np.maximum(vals,0))
    ax.add_patch(Ellipse((np.mean(x),np.mean(y)),width,height,angle=angle,facecolor="none",edgecolor=color,lw=2,ls="--"))


def figure_08_stability(rows):
    series=collect_track_series(rows)
    ordered=sorted(series.items(),key=lambda item:np.mean(item[1][:,1]))[:2]
    colors=[C["blue"],C["orange"]]
    fig=plt.figure(figsize=(14.4,8.5));grid=fig.add_gridspec(2,2,width_ratios=[1.2,1.5],hspace=0.30,wspace=0.24)
    ax=fig.add_subplot(grid[:,0])
    summary=[]
    t0=min(arr[0,0] for _,arr in ordered)
    for idx,((track_id,arr),color) in enumerate(zip(ordered,colors),1):
        ax.scatter(arr[:,1],arr[:,2],s=14,alpha=0.34,color=color,label=f"轨迹{idx} ({len(arr)}帧)")
        confidence_ellipse(ax,arr[:,1],arr[:,2],color)
        ax.plot(np.mean(arr[:,1]),np.mean(arr[:,2]),marker="X",ms=10,color=color)
        summary.append((idx,np.mean(arr[:,1]),np.std(arr[:,1])*1000,np.mean(arr[:,2]),np.std(arr[:,2])*1000,np.mean(arr[:,3])*1000,np.mean(arr[:,4])))
    ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=0.22);ax.set_xlabel("车体前向 x / m");ax.set_ylabel("车体左向 y / m");ax.set_title("连续帧中心散点与95%统计椭圆",fontweight="bold");ax.legend(frameon=False)
    ax1=fig.add_subplot(grid[0,1]);ax2=fig.add_subplot(grid[1,1])
    for idx,((_,arr),color) in enumerate(zip(ordered,colors),1):
        ax1.plot(arr[:,0]-t0,arr[:,4],color=color,lw=1.6,label=f"轨迹{idx}")
        ax2.plot(arr[:,0]-t0,arr[:,3]*1000,color=color,lw=1.6,label=f"轨迹{idx}")
    ax1.set_ylabel("检测置信度");ax1.set_ylim(0.55,1.02);ax1.grid(alpha=0.2);ax1.legend(frameon=False,ncol=2);ax1.set_title("静止场景中的连续检测稳定性",fontweight="bold")
    ax2.set_xlabel("采集时间 / s");ax2.set_ylabel("投影直径 / mm");ax2.grid(alpha=0.2)
    text="\n".join(f"轨迹{i}: x={mx:.3f}±{sx:.1f}mm, y={my:+.3f}±{sy:.1f}mm, 直径均值={diam:.1f}mm" for i,mx,sx,my,sy,diam,conf in summary)
    fig.text(0.55,0.015,text,ha="center",fontsize=9.7,color=C["muted"])
    return save(fig,"B08-双圆片连续帧定位稳定性.png")


def occupancy_points(grid):
    if grid is None:return np.empty((0,2))
    values=np.asarray(grid.data,dtype=np.int16).reshape(grid.info.height,grid.info.width)
    rows,cols=np.where(values>=90)
    x=grid.info.origin.position.x+(cols+0.5)*grid.info.resolution
    y=grid.info.origin.position.y+(rows+0.5)*grid.info.resolution
    return np.column_stack([x,y])


def obstacle_primitives(message):
    lines=[];circles=[]
    if message is None:return lines,circles
    for obstacle in message.obstacles:
        pts=np.asarray([[p.x,p.y] for p in obstacle.polygon.points],float)
        radius=float(obstacle.radius)
        if radius>0 and len(pts):circles.append((pts[0,0],pts[0,1],radius))
        elif len(pts)>=2:lines.append(pts)
    return lines,circles


def figure_09_semantic_obstacles(plan):
    occupied=occupancy_points(plan["global_costmap"])
    obstacle_msg=plan["obstacles"][0] if plan["obstacles"] else None
    lines,circles=obstacle_primitives(obstacle_msg)
    fig,ax=plt.subplots(figsize=(13.8,7.6))
    if len(occupied):ax.scatter(occupied[:,0],occupied[:,1],s=5,c="#cbd5e1",alpha=0.65,label=f"costmap占用单元 {len(occupied)}个")
    for i,line in enumerate(lines):ax.plot(line[:,0],line[:,1],color=C["yellow"],lw=5,solid_capstyle="round",label="LineObstacle" if i==0 else None)
    for i,(x,y,r) in enumerate(circles):ax.add_patch(Circle((x,y),r,facecolor="#fecaca",edgecolor=C["red"],lw=2.2,label="CircularObstacle" if i==0 else None));ax.plot(x,y,"+",color=C["red"],ms=9,mew=2)
    ax.set_aspect("equal");ax.grid(alpha=0.20);ax.set_xlabel("odom x / m");ax.set_ylabel("odom y / m");ax.legend(frameon=True,loc="upper right")
    ax.set_title("完整栅格安全检查与4个语义优化障碍叠加",fontsize=16,fontweight="bold")
    ax.text(0.02,0.04,f"优化器输入：{len(lines)}条边界线 + {len(circles)}个圆片\n硬可行性检查：仍保留完整costmap",transform=ax.transAxes,fontsize=11.5,color=C["slate"],bbox=dict(boxstyle="round,pad=0.4",facecolor="white",edgecolor="#94a3b8",alpha=0.94))
    return save(fig,"B09-costmap栅格与紧凑语义障碍对照.png")


def figure_10_planning(plan):
    occupied=occupancy_points(plan["global_costmap"])
    global_path=max(plan["global_paths"],key=len) if plan["global_paths"] else max(plan["preview_paths"],key=len)
    local_paths=[p for p in plan["local_paths"] if len(p)>=2]
    obstacle_msg=plan["obstacles"][0] if plan["obstacles"] else None
    lines,circles=obstacle_primitives(obstacle_msg)
    fig,ax=plt.subplots(figsize=(12.2,8.3))
    if len(occupied):ax.scatter(occupied[:,0],occupied[:,1],s=5,c="#e2e8f0",alpha=0.62)
    for line in lines:ax.plot(line[:,0],line[:,1],color=C["yellow"],lw=4,solid_capstyle="round")
    for x,y,r in circles:ax.add_patch(Circle((x,y),r,facecolor="#fca5a5",edgecolor=C["red"],lw=2));ax.plot(x,y,"o",color=C["red"],ms=4)
    ax.plot(global_path[:,0],global_path[:,1],color=C["blue"],lw=2.3,label="GlobalPlanner路径")
    for p in local_paths:ax.plot(p[:,0],p[:,1],color=C["purple"],alpha=0.10,lw=1.0)
    if local_paths:ax.plot(local_paths[-1][:,0],local_paths[-1][:,1],color=C["purple"],lw=2.3,label=f"TEB局部轨迹（{len(local_paths)}周期）")
    if plan["footprints"]:
        fp=plan["footprints"][-1];ax.add_patch(Polygon(fp,closed=True,facecolor="#34d399",edgecolor=C["teal"],alpha=0.6,lw=1.8,label="0.34×0.33m footprint"))
    ax.set_aspect("equal");ax.grid(alpha=0.22);ax.set_xlabel("odom x / m");ax.set_ylabel("odom y / m");ax.legend(frameon=True,loc="upper right")
    ax.set_title("右侧镜像布局的锁止规划结果",fontsize=17,fontweight="bold")
    ax.text(0.02,0.03,"底盘物理断开，仅采集规划器输出；图中不代表实车已按该轨迹运动。",transform=ax.transAxes,fontsize=10.5,color=C["muted"],bbox=dict(boxstyle="round,pad=0.3",facecolor="white",edgecolor="#cbd5e1",alpha=0.93))
    return save(fig,"B10-右侧镜像布局GlobalPlanner与TEB锁止规划.png")


def figure_11_cpu_dumbbell():
    acquire=283.4;locked=8.87;drop=(1-locked/acquire)*100
    fig,ax=plt.subplots(figsize=(13.5,5.7));ax.set_xlim(0,310);ax.set_ylim(-0.65,0.65);ax.set_yticks([])
    ax.plot([locked,acquire],[0,0],color="#cbd5e1",lw=10,solid_capstyle="round")
    ax.scatter([acquire],[0],s=480,color=C["red"],zorder=3);ax.scatter([locked],[0],s=480,color=C["green"],zorder=3)
    ax.text(acquire,0.17,f"ACQUIRE\n{acquire:.1f}%",ha="center",fontsize=14,fontweight="bold",color=C["red"])
    ax.text(locked,0.17,f"LOCKED\n{locked:.2f}%",ha="center",fontsize=14,fontweight="bold",color=C["green"])
    ax.annotate(f"CPU均值下降 {drop:.1f}%",xy=((acquire+locked)/2,0),xytext=((acquire+locked)/2,-0.38),ha="center",fontsize=14,fontweight="bold",color=C["teal"],arrowprops=dict(arrowstyle="-[,widthB=8.0,lengthB=0.8",lw=2,color=C["teal"]))
    ax.set_xlabel("视觉进程CPU占用 / %（100%约为一个逻辑核心）");ax.grid(axis="x",alpha=0.18);ax.set_title("世界模型完成后的感知计算卸载",fontsize=17,fontweight="bold")
    for side in ["left","right","top"]:ax.spines[side].set_visible(False)
    return save(fig,"B11-ACQUIRE至LOCKED视觉CPU哑铃对比.png")


def figure_12_acceptance_matrix():
    matrix=np.array([[183,1],[0,336]])
    fig,ax=plt.subplots(figsize=(10.5,7.2));im=ax.imshow(matrix,cmap="YlGn",vmin=0,vmax=336)
    ax.set_xticks([0,1]);ax.set_xticklabels(["满足专项验收输出","未满足/无输出"])
    ax.set_yticks([0,1]);ax.set_yticklabels(["有双圆片 184帧","无圆片 336帧"])
    labels=[["双圆片检出\n183","漏检1帧"],["误检0帧","正确无检出\n336"]]
    for i in range(2):
        for j in range(2):
            ax.text(j,i,labels[i][j],ha="center",va="center",fontsize=15,fontweight="bold",color="white" if matrix[i,j]>170 else C["ink"])
    ax.set_title("520帧红白圆片专项验收结果矩阵",fontsize=17,fontweight="bold",pad=14)
    ax.text(0.5,-0.14,"该统计仅对应验收报告所述场景，不外推为任意环境性能。",transform=ax.transAxes,ha="center",fontsize=10.5,color=C["muted"])
    for spine in ax.spines.values():spine.set_visible(False)
    return save(fig,"B12-红白圆片520帧专项验收矩阵.png")


def make_contact_sheet(files):
    thumbs=[]
    for path in files:
        image=cv2.imread(str(path));
        if image is None:continue
        h,w=image.shape[:2];scale=min(520/w,340/h);res=cv2.resize(image,(max(1,int(w*scale)),max(1,int(h*scale))),interpolation=cv2.INTER_AREA)
        canvas=np.full((390,550,3),255,np.uint8);x=(550-res.shape[1])//2;y=8+(340-res.shape[0])//2;canvas[y:y+res.shape[0],x:x+res.shape[1]]=res
        # OpenCV's built-in font has no Chinese glyphs.  Keep contact-sheet
        # labels to the stable ASCII figure index; full Chinese names remain
        # available in the files and README.
        index_label = path.name.split("-", 1)[0]
        cv2.putText(canvas,index_label,(18,372),cv2.FONT_HERSHEY_SIMPLEX,0.58,(25,35,55),1,cv2.LINE_AA);thumbs.append(canvas)
    rows=[]
    for i in range(0,len(thumbs),3):
        row=thumbs[i:i+3]
        while len(row)<3:row.append(np.full_like(thumbs[0],255))
        rows.append(np.hstack(row))
    sheet=np.vstack(rows);path=OUT/"contact_sheet.png";cv2.imwrite(str(path),sheet);return path


def write_manifest(files, metadata, snapshot, plan):
    manifest={
        "purpose":"第二组技术报告，右侧镜像错列布局，图片版式与第一组分离",
        "capture_bag":{"path":str(CAPTURE_BAG),"sha256":sha256(CAPTURE_BAG),"included_in_zip":False},
        "plan_bag":{"path":str(PLAN_BAG),"sha256":sha256(PLAN_BAG),"included_in_zip":False},
        "config":{"path":str(CONFIG),"sha256":sha256(CONFIG)},
        "selected_image_stamp":metadata["selected_image_stamp"],
        "motion_during_static_capture":False,
        "dynamic_trial_bag_available":False,
        "disc_count":snapshot["/yellow_track_disc_costmap_bridge/status"].get("disc_count"),
        "costmap_disc_count":snapshot["/yellow_track_disc_costmap_bridge/status"].get("costmap_disc_count"),
        "planning_samples":{"global":len(plan["global_paths"]),"local":len(plan["local_paths"])},
        "figures":[{"file":path.name,"sha256":sha256(path)} for path in files],
    }
    with (OUT/"source_manifest.json").open("w",encoding="utf-8") as stream:json.dump(manifest,stream,ensure_ascii=False,indent=2)


def main():
    if not CAPTURE_BAG.exists():raise FileNotFoundError(CAPTURE_BAG)
    if not PLAN_BAG.exists():raise FileNotFoundError(PLAN_BAG)
    raw,analysis,metadata,rows,snapshot=load_capture_data();plan=load_plan_data()
    files=[
        figure_01_layered_architecture(),
        figure_02_scene_card(raw,snapshot),
        figure_03_red_segmentation(raw,analysis),
        figure_04_white_center_and_sectors(raw,analysis),
        figure_05_candidate_funnel(snapshot),
        figure_06_yellow_confidence(raw,analysis),
        figure_07_coordinate_correspondence(raw,snapshot),
        figure_08_stability(rows),
        figure_09_semantic_obstacles(plan),
        figure_10_planning(plan),
        figure_11_cpu_dumbbell(),
        figure_12_acceptance_matrix(),
    ]
    make_contact_sheet(files)
    write_manifest(files,metadata,snapshot,plan)
    print("\n".join(str(path) for path in files))


if __name__=="__main__":
    main()
