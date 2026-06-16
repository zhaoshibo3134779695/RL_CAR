#!/usr/bin/env python3
import heapq
import csv
import cv2
import numpy as np
from pathlib import Path


def load_map_and_image(map_path, threshold=127):
    """加载地图：返回A*用栅格和用于标注显示的彩色图"""
    gray = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"无法加载地图文件: {map_path}")

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    grid = (binary < threshold).astype(np.uint8)  # 1=障碍, 0=可走
    return grid, color


class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

    @staticmethod
    def heuristic(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def plan(self, start, goal):
        open_heap = []
        heapq.heappush(open_heap, (0.0, start[0], start[1]))

        g = np.full(self.grid.shape, np.inf, dtype=np.float64)
        g[start] = 0.0

        parent = np.full((self.rows, self.cols, 2), -1, dtype=np.int32)
        closed = set()

        while open_heap:
            _, x, y = heapq.heappop(open_heap)

            if (x, y) in closed:
                continue
            closed.add((x, y))

            if (x, y) == goal:
                path = []
                cx, cy = x, y
                while (cx, cy) != start:
                    path.append((cx, cy))
                    cx, cy = parent[cx, cy]
                path.append(start)
                path.reverse()
                return path

            for dx, dy in self.dirs:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.rows and 0 <= ny < self.cols):
                    continue
                if self.grid[nx, ny] == 1:
                    continue

                step = 1.414 if dx != 0 and dy != 0 else 1.0
                new_g = g[x, y] + step
                if new_g < g[nx, ny]:
                    g[nx, ny] = new_g
                    parent[nx, ny] = (x, y)
                    f = new_g + self.heuristic((nx, ny), goal)
                    heapq.heappush(open_heap, (f, nx, ny))

        return None


def grid_to_world(row, col, resolution, origin):
    """栅格(row,col)->世界坐标(x,y)"""
    x = origin[0] + (col + 0.5) * resolution
    y = origin[1] + (row + 0.5) * resolution
    return x, y


def export_xy_csv(path, resolution, origin, filename):
    """导出纯x,y两列，无表头"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r, c in path:
            x, y = grid_to_world(r, c, resolution, origin)
            writer.writerow([x, y])


def plan_multi_segments(grid, start, waypoints):
    """按顺序规划多段路径并拼接"""
    astar = AStar(grid)
    full_path = []
    cur = start

    for i, wp in enumerate(waypoints, start=1):
        print(f"规划第{i}段: {cur} -> {wp}")
        seg = astar.plan(cur, wp)
        if seg is None:
            raise RuntimeError(f"无法到达目标点: {wp}")
        if full_path:
            full_path.pop()  # 去重连接点
        full_path.extend(seg)
        cur = wp

    return full_path


def draw_path_overlay(base_img, path, start, waypoints, out_img_path):
    """
    在原图上标注路径并保存:
    - 路径: 红色线
    - 起点: 绿色圆
    - 目标点: 蓝色圆 + 编号
    """
    vis = base_img.copy()

    # 画路径线（col=x, row=y）
    pts = np.array([[c, r] for r, c in path], dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(vis, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    # 起点
    cv2.circle(vis, (start[1], start[0]), 5, (0, 255, 0), -1)
    cv2.putText(vis, "S", (start[1] + 6, start[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1, cv2.LINE_AA)

    # 目标点
    for i, wp in enumerate(waypoints, start=1):
        cv2.circle(vis, (wp[1], wp[0]), 5, (255, 0, 0), -1)
        cv2.putText(vis, f"W{i}", (wp[1] + 6, wp[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 0), 1, cv2.LINE_AA)

    ok = cv2.imwrite(out_img_path, vis)
    if not ok:
        raise RuntimeError(f"路径标注图保存失败: {out_img_path}")


if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    NAV_MAP_DIR = THIS_DIR.parent.parent / "nav_demo" / "map"
    # ====== 只改这几项 ======
    MAP_PATH = str(NAV_MAP_DIR / "mymap.png")
    THRESHOLD = 127
    RESOLUTION = 0.01
    ORIGIN = (-7.0, -14.44, 0.0)

    START = (292, 57)
    WAYPOINTS = [
        (292, 128),
        (267, 96),
        (226, 70),
        (292, 57),
    ]

    OUTPUT_CSV = "ros_path.csv"
    OUTPUT_OVERLAY_IMG = "path_overlay.png"
    # ========================

    grid, map_img = load_map_and_image(MAP_PATH, THRESHOLD)
    path = plan_multi_segments(grid, START, WAYPOINTS)

    export_xy_csv(path, RESOLUTION, ORIGIN, OUTPUT_CSV)
    draw_path_overlay(map_img, path, START, WAYPOINTS, OUTPUT_OVERLAY_IMG)

    print(f"已生成CSV: {OUTPUT_CSV}，点数: {len(path)}")
    print(f"已生成路径标注图: {OUTPUT_OVERLAY_IMG}")
