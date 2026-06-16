#!/usr/bin/env python3
import cv2
import numpy as np
import csv
from pathlib import Path

# ====================== 你只需要改这里 ======================
PNG_PATH = str((Path(__file__).resolve().parent.parent.parent / "nav_demo" / "map" / "mymap.png"))
OUTPUT_CSV = "map_grid.csv"                # 输出的地图CSV
THRESHOLD = 127  # 二值化阈值（不用改）
# ==========================================================

# 1. 读取图片（灰度图）
img = cv2.imread(PNG_PATH, 0)
if img is None:
    print("❌ 找不到图片，请检查路径！")
    exit()

# 2. 二值化：白色=0（可通行），黑色=1（障碍物）
_, binary = cv2.threshold(img, THRESHOLD, 255, cv2.THRESH_BINARY)
grid = (binary == 0).astype(np.uint8)  # 0=可走，1=障碍

# 3. 保存到 CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    for row in grid:
        writer.writerow(row)

print(f"✅ PNG 地图已转换成 CSV：{OUTPUT_CSV}")
print(f"📊 地图尺寸：{grid.shape[0]} 行 × {grid.shape[1]} 列")
