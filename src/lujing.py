#!/usr/bin/env python3
import pygame
import math
import csv
from queue import PriorityQueue

# ====================== 全局配置 ======================
WIDTH = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A*小车路径规划 + 坐标提取 + CSV保存")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)    # 终点
GREEN = (0, 255, 0)  # 起点
BLUE = (0, 0, 255)   # 小车/路径
GREY = (128, 128, 128)

# ====================== 网格节点类 ======================
class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == BLUE

    def is_end(self):
        return self.color == RED

    def reset(self):
        self.color = WHITE

    def make_closed(self):
        self.color = (200, 200, 200)

    def make_open(self):
        self.color = (220, 220, 220)

    def make_barrier(self):
        self.color = BLACK

    def make_start(self):
        self.color = GREEN

    def make_end(self):
        self.color = RED

    def make_path(self):
        self.color = BLUE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    # 更新相邻节点
    def update_neighbors(self, grid):
        self.neighbors = []
        # 下
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])
        # 上
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])
        # 右
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])
        # 左
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

# ====================== A*核心算法 ======================
# 曼哈顿距离（启发函数）
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

# 重构最优路径
def reconstruct_path(came_from, current, draw):
    path = []
    while current in came_from:
        current = came_from[current]
        current.make_path()
        # 保存路径坐标点 (行, 列)
        path.append(current.get_pos())
        draw()
    # 反转得到正序路径
    path.reverse()
    return path

# A*主算法
def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    
    # 初始化g和f值
    g_score = {node: float("inf") for row in grid for node in row}
    f_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        # 找到终点，返回路径
        if current == end:
            path = reconstruct_path(came_from, end, draw)
            end.make_end()
            return path

        # 遍历邻居
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return None  # 无路径

# ====================== 坐标保存到CSV ======================
def save_path_to_csv(path, filename="car_path.csv"):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(["序号", "行坐标", "列坐标"])
        # 写入所有路径点
        for idx, (row, col) in enumerate(path):
            writer.writerow([idx + 1, row, col])
    print(f"✅ 小车坐标已保存到 {filename}")
    print(f"📊 总路径点数：{len(path)}")

# ====================== 界面初始化 ======================
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

# 获取鼠标点击位置
def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

# ====================== 主函数 ======================
def main(win, width):
    ROWS = 30
    grid = make_grid(ROWS, width)
    start = None
    end = None
    run = True
    path = []

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            # 关闭窗口
            if event.type == pygame.QUIT:
                run = False

            # 左键设置：起点→终点→障碍物
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]

                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != end and node != start:
                    node.make_barrier()

            # 右键清除
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                if node == end:
                    end = None

            # 空格键：启动A*算法
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                    # 执行算法，获取路径坐标
                    path = algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    # 保存坐标到CSV
                    if path:
                        save_path_to_csv(path)
                    else:
                        print("❌ 无法到达终点！")

                # C键：重置地图
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)
                    path = []

    pygame.quit()

if __name__ == "__main__":
    main(WIN, WIDTH)
