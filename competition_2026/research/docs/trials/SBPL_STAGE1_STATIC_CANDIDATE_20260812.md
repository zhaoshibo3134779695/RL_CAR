# Stage 1 SBPL + TEB 无单侧封锁静态候选（2026-08-12）

## 定位

本方案是与已跑通正式基线隔离的实验候选，不覆盖：

```text
GlobalPlanner + compact TEB + 单侧 passage closure
```

实验目标是让 SBPL 在全局层根据 `x/y/yaw`、矩形 footprint 和完整圆片几何自主选择绕行侧，替代人工把圆片某一侧通道封死的拓扑规则。

## 实验链

```text
RGB 黄线 + 两枚圆片
  → corridor / discs 冻结到 odom
  → 两条黄线内边缘写入 costmap
  → 两枚圆片按 0.05 m 半径填充为完整占用云
  → SBPL XYThetaLattice（1 cm、16 headings、全向 primitives）
  → compact-obstacle TEB（2 LineObstacle + 2 CircularObstacle）
```

明确关闭：

- passage closure；
- passage decision；
- DWA；
- SBPL/GlobalPlanner 双开；
- 底盘；
- 自动目标发送。

## 为什么增加完整圆片占用云

原 `disc_scan` 是从当前机器人位置向圆片投射的虚拟 LaserScan，只描述迎车侧表面。车辆从圆片侧面和后面经过时，单向扫描不能完整表达整个圆盘。SBPL候选因此使用冻结圆心重新生成两个完整的、半径 `0.05 m` 的填充点云，采样间隔 `0.005 m`，共 `634` 点。

这不是新的语义封路；点云只表达两个圆片真实物理面积。

## 为什么采用 1 cm 状态格

5 m × 4 m 地图若使用 5 mm 分辨率并离散 16 个姿态，约有 1280 万个 `x/y/yaw` 状态。完整圆片加入后，5秒和15秒搜索预算均不能稳定找到精确目标路径。

候选改用同尺寸 `1 cm` 空白地图和已有的 `yta_1cm.mprim`：

- 16 个离散朝向；
- 176 个运动基元；
- 每个朝向均有平移和原地旋转；
- heading 0 明确包含前后、左右横移及转向动作；
- 状态规模约为5 mm方案的四分之一；
- 矩形 footprint 和物理圆片尺寸不缩小。

## 最终锁车静态结果

当前真实双圆片场景下：

| 指标 | 结果 |
|---|---:|
| runner状态 | `DRY_RUN_READY` |
| 底盘连接 | 无 |
| goal发送 | `false` |
| passage closure | 0 |
| passage decision | 空 |
| 完整圆片云 | 634点 |
| 从启动到路径就绪 | 14.84 s |
| SBPL路径位姿数 | 792 |
| 路径长度 | 2.672 m |
| 路径横向范围 | -0.395～+0.215 m |
| 第一枚圆片物理最小净空 | +28.2 mm |
| 第二枚圆片物理最小净空 | +77.4 mm |
| 扫掠复核间距 | 5 mm |
| move_base稳定后RSS | 121 MB |

路径在两枚错列圆片之间自主形成S型横移，没有任何“封闭圆片某一侧”的附加几何。

## 当前结论

静态可行性已经成立：SBPL能够在完整圆盘、矩形车体和全向运动基元条件下，自主找到正净空的S型全局路线。

但它尚未经过实车闭环验证，因此当前关系仍为：

```text
已跑通回退基线：GlobalPlanner + TEB + 单侧封锁
新静态候选：    SBPL + TEB + 完整圆片 + 无单侧封锁
```

下一阶段若允许实车，只应单独启动候选profile进行A/B，不应覆盖已跑通基线。
