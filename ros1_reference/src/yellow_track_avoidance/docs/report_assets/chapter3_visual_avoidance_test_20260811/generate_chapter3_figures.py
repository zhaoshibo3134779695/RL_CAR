#!/usr/bin/env python3
"""Assemble Chapter 3 report figures from existing verified assets only."""

from pathlib import Path
import shutil

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/report_assets/vision_avoidance_20260811"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["AR PL UKai CN"],
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)


def read_rgb(name):
    image = cv2.imread(str(SOURCE / "sources" / name), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(SOURCE / "sources" / name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save(fig, name):
    target = OUT / name
    fig.savefig(target, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return target


def panel_label(ax, text, dark=False):
    ax.text(
        0.018,
        0.975,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="white" if dark else "#111827",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="#111827" if dark else "white",
            edgecolor="none",
            alpha=0.88,
        ),
    )


def test_scene():
    raw = read_rgb("raw_rgb.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(raw)
    ax.axis("off")
    return save(fig, "3.2.4.0-反坦克雷区域视觉避障测试场景.png")


def detection_panels():
    panels = [
        ("raw_rgb.png", "A  原始RGB", False),
        ("raw_overlay.png", "B  黄线与道路走廊", True),
        ("disc_overlay.png", "C  双圆片检测结果", True),
        ("bev.png", "D  米制BEV投影", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    for ax, (filename, label, dark) in zip(axes.flat, panels):
        ax.imshow(read_rgb(filename))
        ax.axis("off")
        panel_label(ax, label, dark=dark)
    fig.subplots_adjust(wspace=0.025, hspace=0.04)
    return save(fig, "3.2.4.1-视觉避障现场检测效果图.png")


def copy_verified(source_name, target_name):
    source = SOURCE / source_name
    target = OUT / target_name
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)
    return target


def main():
    outputs = [
        test_scene(),
        detection_panels(),
        copy_verified(
            "fig04_perspective_to_metric_bev.png",
            "3.2.4.2-双圆片米制BEV定位结果.png",
        ),
        copy_verified(
            "fig07_stage1_s_curve_five_moments.png",
            "3.2.4.3-双圆片S形避障五个关键时刻.png",
        ),
        copy_verified(
            "fig06_navigation_world_model.png",
            "3.2.4.4-视觉障碍建模与规划结果.png",
        ),
        copy_verified(
            "fig08_acquire_locked_cpu_ab.png",
            "3.2.4.5-ACQUIRE_LOCKED视觉CPU对比.png",
        ),
        copy_verified(
            "fig09_disc_520frame_acceptance.png",
            "3.2.4.6-红白圆片520帧专项验收统计.png",
        ),
        copy_verified(
            "fig05_scale_validation_500mm.png",
            "3.2.4.7-米制BEV的500mm尺度验证.png",
        ),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
