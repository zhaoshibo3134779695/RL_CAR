# 正式比赛规划器 A/B 基线

本轮把 DWA 冻结为历史基线，只比较一个主要变量：

```text
GlobalPlanner + DWA  →  GlobalPlanner + TEB
```

SBPL + TEB 已准备为第二级候选，但只有全局路径证据明确指向
GlobalPlanner 时才进入实车测试。

2026-08-10 Trial 3 锁止态结果见
[`COMPETITION_PLANNER_BASELINE_REPORT_20260810.md`](COMPETITION_PLANNER_BASELINE_REPORT_20260810.md)。

## 三个互斥入口

| 入口 | 全局规划器 | 局部规划器 | 用途 |
|---|---|---|---|
| `competition_global_teb_minimal.launch` | GlobalPlanner | TEB | 第一正式候选 |
| `competition_sbpl_teb_candidate.launch` | SBPL | TEB | 只做静态加载和路径对比 |
| `competition_dwa_baseline.launch` | GlobalPlanner | DWA | 冻结的历史 A/B 基线 |

三者只会启动一套全局规划器和一套局部规划器，并共享：

- 当前 640×480 标定感知；
- `odom` 中冻结的确认圆片；
- `/yellow_track_avoidance/disc_scan`；
- 视觉锁定的 0.91 m 黄线走廊；
- 0.34 m × 0.33 m 矩形 footprint；
- 相同 global/local costmap；
- 8 Hz move_base 控制频率；
- 禁用 move_base recovery。

最小核心中没有：

- `navigation_safety_gate`；
- `passage_supervisor`；
- 40 cm/gate controller；
- terminal controller；
- finish recovery；
- 自由区间 controller。

`move_base` 是唯一 `/cmd_vel` 发布者。所有入口默认
`start_chassis=false`，因此底盘串口命令消费者不会启动；走廊锁定也不会自动发送
目标。

## 可重复的 Trial 3 无底盘回放

使用从 2026-08-09 DWA Trial 3 恢复的左错列双圆片坐标，可以对两个 TEB
profile 做完全相同的硬件无关比较。以默认参数启动其中一个 profile 后运行：

```bash
rosrun yellow_track_avoidance locked_planner_ab_probe.py \
  --output /tmp/yta_locked_planner_report.json
```

探针会提供静止 odom 和记录场景。在发送目标前，它查询 ROS 节点图；只有
`/move_base` 是唯一 `/cmd_vel` 发布者、且没有底盘或未知速度订阅者时才继续。
探针记录全局/局部路径、TEB 选中轨迹、速度、圆片 costmap 单元和进程性能，随后
取消目标。它不打开底盘设备。

## 静态检查

```bash
source /opt/ros/noetic/setup.bash
source /home/mowen/Sebastian/devel/setup.bash

roslaunch --nodes yellow_track_avoidance \
  competition_global_teb_minimal.launch

roslaunch --dump-params yellow_track_avoidance \
  competition_global_teb_minimal.launch > /tmp/global_teb_params.yaml
```

## 锁止态现场检查

车辆保持物理急停或车轮架空，并且不要启动底盘命令桥：

```bash
roslaunch yellow_track_avoidance competition_global_teb_minimal.launch \
  start_camera:=true start_lidar:=true start_odometry:=true \
  start_chassis:=false start_rviz:=true
```

确认没有底盘订阅者：

```bash
rostopic info /cmd_vel
```

在发送目标前应先记录：

- `/yellow_track_disc_costmap_bridge/status`；
- `/move_base/global_costmap/costmap`；
- `/move_base/local_costmap/costmap`；
- `/move_base/GlobalPlanner/plan`；
- `/move_base/TebLocalPlannerROS/local_plan`；
- `/move_base/TebLocalPlannerROS/teb_feedback`；
- `/cmd_vel`；
- `/rosout` 中的控制周期、infeasible 和 diverged 信息；
- vision、move_base 和整机 CPU。

## 进入实车前

这个最小基线有意不提供软件安全门。只有完成锁止态报告、人工急停验证并获得明确
许可后，才能启动 `start_chassis=true`。第一轮只允许左错列低速测试，不自动继续
右镜像、中间布局或 SBPL 实车测试。
