# Stage 1 宽道完整实车报告（2026-08-10）

## 结论

本轮在 ROS 状态上完整运行并返回 `Goal reached`，但按现场物理结果判定为 **Stage 1 验收失败**，不能视为可比赛的成功方案。

现场确认：

1. 两枚圆片前车体都有转角，但未压到黄线；
2. 车体压到左侧圆片；
3. 最终车体探出过多，实际位置和姿态都没有到达所需终点。

因此，`move_base = SUCCEEDED` 只代表它到达了当前定义的 odom 目标，不代表车辆在真实赛道中合格到站。

## 实验基线

实际控制链：

```text
视觉 ACQUIRE
→ corridor + 两圆片冻结到 odom
→ 视觉 LOCKED
→ costmap
→ GlobalPlanner
→ TEB 6×4（4 个紧凑障碍）
→ /cmd_vel
→ 底盘
```

本轮未启用 DWA、SBPL、passage/gate、navigation safety gate、terminal visual controller 或 finish recovery。`move_base` 是唯一 `/cmd_vel` 发布者。

## 运行数据

| 项目 | 结果 |
|---|---:|
| ACTIVE 到 SUCCEEDED | 27.500 s |
| odom 路径长度 | 2.605 m |
| odom 横向范围 | -0.190 ~ +0.160 m |
| odom yaw 范围 | -29.19° ~ +34.31° |
| `/cmd_vel` 数量 | 221 |
| `/cmd_vel` 平均周期 | 125.00 ms（8.00 Hz） |
| `/cmd_vel` p95 / max | 135.58 / 163.39 ms |
| 周期 >187.5 ms | 0 |
| TEB feedback | 208，7.64 Hz |
| TEB feedback 周期 p95 | 131.90 ms |
| GlobalPlanner 路径发布 | 21 次 |
| TEB `trajectory is not feasible` | 10 次 |
| TEB reset | 10 次 |
| move_base control-loop missed 日志 | 0 |
| `/cmd_vel` `vx` | -0.0103 ~ +0.1000 m/s |
| `/cmd_vel` `vy` | -0.0800 ~ +0.0800 m/s |
| `/cmd_vel` `wz` | -0.3000 ~ +0.3000 rad/s |

10 次 infeasible/reset 相对任务开始的时刻为：

```text
5.509, 5.892, 6.386, 7.012, 7.511,
8.133, 8.757, 9.136, 9.507, 14.756 s
```

前 9 次集中在第一枚圆片前后；第 10 次与车体最贴近左侧圆片的时刻重合。本轮的控制周期和 CPU 已不是主要失败原因。

## 圆片通过结果

冻结 odom 坐标：

```text
disc 1: (0.565690, -0.116770) m
disc 2: (1.021111, +0.276936) m  # 左侧圆片
```

使用 0.34 × 0.33 m 矩形 footprint、0.05 m 圆片半径，对 20 Hz odom 轨迹做不大于 5 mm / 1° 的插值扫掠，得到：

| 障碍 | odom 模型最小净空 | 发生时刻 |
|---|---:|---:|
| disc 1 | +7.3 mm | 11.312 s |
| disc 2（左侧） | +2.8 mm | 14.761 s |

这是 odom/冻结模型估计，不是物理真值。现场观察是碰撞的最终依据。+2.8 mm 远小于感知、odom、车体尺寸和底盘跟踪误差，与“压到左侧圆片”的实车结果一致。

TEB 确实持续闭环重规划，但实际执行路径没有留出可吸收误差的安全裕量。完整 costmap feasibility 只会拒绝预测碰撞，不会保证实车跟踪后仍有 20 mm 净空。

## 车体转角

相机锁定的赛道方向为 -1.04°，但 odom 记录的运行 yaw 范围达 -29.19° ~ +34.31°，且 `/cmd_vel.wz` 多次达到 ±0.30 rad/s 限值。

因此，现场看到的“圆片前稍微拧转”在 odom 里并不是可忽略的小姿态变化。它改变了矩形车体的扫掠宽度，是圆片净空不稳定的直接相关因素。

## 终点失败的直接原因

本轮发送的 odom 目标：

```text
base center: (2.416990, -0.187340) m
yaw:        -1.043°
```

最终 odom：

```text
base center: (2.416836, -0.189319) m
yaw:        +0.440°
```

odom 中的位置误差只有 2.0 mm，yaw 误差为 1.48°，均满足当前 TEB 的 `xy_goal_tolerance=0.01 m` 和 `yaw_goal_tolerance=0.05 rad`。所以 `move_base` 返回 `SUCCEEDED` 符合它当前的判定逻辑。

但该目标不是由真实终点黄线生成的。当前代码按下列方式构造目标：

```text
人工生成长度为 1.85 m 的纵向 corridor
→ 取右黄线的人工远端点
→ 沿赛道方向再向前 0.57 m
→ 向左 0.24 m
→ 再减去 0.17 m 车头到 base_link 的距离
```

换算后，车头中点在 odom 模型中位于人工 corridor 远端前方 **0.570 m**。这与现场“车身探出很多”直接对应。

同时，视觉在世界模型冻结后进入 LOCKED，terminal visual controller 又被禁用，因此终点阶段没有任何对真实黄线位置或车体实际姿态的再校正。odom 自己认为到点，不能阻止真实车位偏差。

## 判定与下一主要工作

本轮证明了：

- compact-obstacle TEB 有能力在闭环中穿过双圆片区域；
- 计算周期已经基本可用；
- 但圆片安全裕量不足，且终点目标/定位闭环错误；
- 当前配置不建议再次实车，也不建议进入 Stage 2。

下一轮应分开处理两个问题，不能把它们归因为单一 TEB 参数：

1. 先重新定义真实终点停车平面，用实时黄线位置/姿态完成最后预对正和低速停车，不再相信单一 odom 点；
2. 再处理左侧圆片的执行净空和 TEB 姿态摆动，验收必须看实际执行扫掠净空，不只看某一帧优化轨迹。

本报告只记录和诊断，未自动调参，未进行第二次试车。

## 数据文件

```text
/home/mowen/Sebastian/stage1_global_teb_compact_20260810_trial1.bag
```

rosbag 为 LZ4 压缩，大小 6.6 MB，包含 7,494 条消息，包括 odom、global/local plan、TEB feedback、cmd_vel、move_base result/status、costmap、corridor 和 frozen-disc 状态。
