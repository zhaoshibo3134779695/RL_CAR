#!/usr/bin/env python3
import heapq
import cv2
import numpy as np
import csv
import math
from pathlib import Path

# ====================== 1. ROS地图加载与预处理 ======================
def load_ros_map(map_path, threshold=127):
    """
    加载ROS风格PNG栅格地图，转为二值化栅格
    :param map_path: 地图文件路径，如"map/mymap.png"
    :param threshold: 二值化阈值，>threshold为可通行(0)，<为障碍(1)
    :return: grid(二值化栅格), img_shape(地图尺寸)
    """
    # 以灰度图加载
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载地图文件: {map_path}")
    
    # 二值化：ROS地图中白色为可通行，黑色为障碍
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # 转为0(可走)/1(障碍)的栅格
    grid = (binary < threshold).astype(np.int8)
    
    print(f"地图加载成功，尺寸: {img.shape[1]}×{img.shape[0]} 像素")
    return grid, img.shape

# ====================== 2. A*算法核心（适配ROS地图） ======================
class AStarROS:
    def __init__(self, grid):
        self.grid = grid
        self.rows, self.cols = grid.shape
        # 8方向移动（支持斜向，更平滑），ROS小车常用
        self.dirs = [(-1,0), (1,0), (0,-1), (0,1),
                     (-1,-1), (-1,1), (1,-1), (1,1)]

    def heuristic(self, a, b):
        """启发函数：欧几里得距离（ROS连续空间更适配）"""
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def plan_path(self, start_grid, goal_grid):
        """
        栅格坐标下的A*路径规划
        :param start_grid: 起点栅格坐标 (row, col)
        :param goal_grid: 终点栅格坐标 (row, col)
        :return: 栅格路径列表 [(row,col), ...]
        """
        open_heap = []
        heapq.heappush(open_heap, (0, *start_grid))

        # g值：起点到当前节点的实际代价
        g = np.full_like(self.grid, np.inf, dtype=np.float64)
        g[start_grid[0], start_grid[1]] = 0.0

        # 父节点记录，用于回溯路径
        parent = np.zeros((self.rows, self.cols, 2), dtype=np.int32)
        parent[:] = -1  # 初始化无父节点
        closed = set()

        while open_heap:
            f, x, y = heapq.heappop(open_heap)

            # 到达终点，回溯路径
            if (x, y) == goal_grid:
                path = []
                while (x, y) != start_grid:
                    path.append((x, y))
                    x, y = parent[x, y]
                path.append(start_grid)
                return path[::-1]  # 反转得到起点→终点

            # 已处理过的节点跳过
            if (x, y) in closed:
                continue
            closed.add((x, y))

            # 遍历8个方向
            for dx, dy in self.dirs:
                nx, ny = x + dx, y + dy
                # 边界检查
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    # 障碍检查
                    if self.grid[nx, ny] == 1:
                        continue
                    # 移动代价：斜向为√2≈1.414，正交为1
                    cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    new_g = g[x, y] + cost

                    # 找到更优路径，更新
                    if new_g < g[nx, ny]:
                        g[nx, ny] = new_g
                        h = self.heuristic((nx, ny), goal_grid)
                        f = new_g + h
                        parent[nx, ny] = (x, y)
                        heapq.heappush(open_heap, (f, nx, ny))

        return None  # 无可行路径

# ====================== 3. 栅格坐标 ↔ ROS世界坐标转换 ======================
def grid_to_world(grid_x, grid_y, resolution, origin):
    """
    ROS标准坐标转换：栅格坐标 → 世界坐标（米）
    ROS地图坐标规则：
    - 栅格(row, col)对应世界(x,y)
    - 原点(origin)为地图左下角在世界坐标系的坐标
    :param grid_x: 栅格行坐标 (row)
    :param grid_y: 栅格列坐标 (col)
    :param resolution: 地图分辨率 (m/像素)
    :param origin: 地图原点 (origin_x, origin_y, origin_yaw)
    :return: 世界坐标 (world_x, world_y)
    """
    # ROS标准转换公式
    world_x = origin[0] + (grid_y + 0.5) * resolution
    world_y = origin[1] + (grid_x + 0.5) * resolution
    return round(world_x, 4), round(world_y, 4)

def calculate_yaw(path):
    """
    计算路径点的航向角yaw（弧度），用于小车循迹
    :param path: 栅格路径列表
    :return: yaw列表（弧度）
    """
    yaws = []
    for i in range(len(path)):
        if i == len(path) - 1:
            # 最后一个点，沿用前一个点的航向角
            yaws.append(yaws[-1] if i > 0 else 0.0)
        else:
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            # 计算方向角（ROS右手坐标系，x向前，y向左）
            dx = x2 - x1
            dy = y2 - y1
            yaw = math.atan2(dy, dx)
            yaws.append(round(yaw, 4))
    return yaws

# ====================== 4. 导出ROS标准CSV路径 ======================
def export_ros_path_csv(path, resolution, origin, filename="ros_path.csv"):
    """
    导出ROS格式的路径CSV文件，包含x,y,yaw
    :param path: 栅格路径列表
    :param resolution: 地图分辨率
    :param origin: 地图原点
    :param filename: 输出文件名
    """
    if not path:
        print("路径为空，无法导出")
        return

    # 计算航向角
    yaws = calculate_yaw(path)

    # 写入CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ROS导航标准表头
        writer.writerow(["x", "y", "yaw"])
        for i, (grid_x, grid_y) in enumerate(path):
            world_x, world_y = grid_to_world(grid_x, grid_y, resolution, origin)
            writer.writerow([world_x, world_y, yaws[i]])

    print(f"✅ ROS路径已成功导出到: {filename}")
    print(f"📊 路径总点数: {len(path)}")

# ====================== 5. 多目标点顺序导航（ROS适配） ======================
def multi_waypoint_navigation(grid, start_grid, waypoints_grid):
    """
    多目标点顺序导航，拼接完整路径
    :param grid: 栅格地图
    :param start_grid: 起点栅格坐标
    :param waypoints_grid: 目标点栅格坐标列表
    :return: 完整栅格路径
    """
    astar = AStarROS(grid)
    full_path = []
    current = start_grid

    for idx, wp in enumerate(waypoints_grid):
        print(f"🔄 规划第 {idx+1} 段路径: {current} → {wp}")
        path = astar.plan_path(current, wp)
        if not path:
            print(f"❌ 无法到达目标点 {wp}，导航终止")
            return None
        # 拼接路径，去重（避免重复经过中间点）
        if full_path:
            full_path.pop()
        full_path.extend(path)
        current = wp

    return full_path

# ====================== 6. 主程序（你只需要改这里！） ======================
if __name__ == "__main__":
    THIS_DIR = Path(__file__).resolve().parent
    NAV_MAP_DIR = THIS_DIR.parent.parent / "nav_demo" / "map"
    # -------------------------- 1. 配置你的ROS地图参数 --------------------------
    MAP_PATH = str(NAV_MAP_DIR / "mymap.png")
    RESOLUTION = 0.010000  # 地图分辨率，单位m/像素（ROS常用0.05或0.1）
    # 地图原点：ROS标准为地图左下角在世界坐标系的坐标
    # 352×0.05=17.6m，设原点为(-8.8, -8.8, 0.0)，让地图中心为(0,0)
    ORIGIN = (-7.000000, -14.440000, 0.000000)
    THRESHOLD = 127  # 二值化阈值，根据你的地图调整

    # -------------------------- 2. 设置起点和目标点（栅格坐标） --------------------------
    # 栅格坐标格式：(row, col)，对应图像的(行, 列)
    # 你可以用图像工具查看目标点的像素坐标，直接填这里
    START_GRID = (292,57)  # 起点栅格坐标
    WAYPOINTS_GRID = [
        (292,128),  # 目标点1
        (267,96),  # 目标点2
        (226,70),    # 目标点3
        (292,57)
    ]

    # -------------------------- 3. 执行完整流程 --------------------------
    try:
        # 步骤1：加载ROS地图
        grid, img_shape = load_ros_map(MAP_PATH, THRESHOLD)
        rows, cols = img_shape

        # 步骤2：多目标点A*路径规划
        full_path = multi_waypoint_navigation(grid, START_GRID, WAYPOINTS_GRID)
        if not full_path:
            exit(1)

        # 步骤3：导出ROS标准CSV路径（x,y,yaw）
        export_ros_path_csv(full_path, RESOLUTION, ORIGIN, filename="ros_path.csv")

        # （可选）可视化路径
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,8))
        plt.imshow(grid, cmap="gray_r", origin="upper")
        # 画路径
        path_x = [p[1] for p in full_path]
        path_y = [p[0] for p in full_path]
        plt.plot(path_x, path_y, "r-", linewidth=2, label="规划路径")
        # 画起点/目标点
        plt.scatter(START_GRID[1], START_GRID[0], c="g", s=100, label="起点")
        for i, wp in enumerate(WAYPOINTS_GRID):
            plt.scatter(wp[1], wp[0], c="b", s=100, label=f"目标点{i+1}")
        plt.grid(True)
        plt.legend()
        plt.title("ROS地图A*路径规划结果")
        plt.show()

    except Exception as e:
        print(f"❌ 程序出错: {str(e)}")
