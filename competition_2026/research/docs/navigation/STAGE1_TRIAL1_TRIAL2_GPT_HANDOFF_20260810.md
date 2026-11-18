# Stage 1 两轮实车汇总与 GPT 交接材料（2026-08-10）

## 1. 用途与约束

本文汇总当前 `GlobalPlanner + TEB compact obstacles` 正式最小链从锁止态准备
到两轮 Stage 1 实车的关键证据，供 GPT 判断如何只解决终点目标和姿态收敛。

用户的最新边界条件：

- Trial 2 双圆片避障可以接受，两枚圆片均未接触；
- 不因为碰到围墙增加墙体检测、碰撞保护、安全门、watchdog 或 recovery；
  围墙只作为解释 odom 可能失真的现场背景；
- 冻结已成功的视觉、BEV、圆片冻结、costmap、GlobalPlanner、compact
  obstacles 和双圆片绕障逻辑；
- 只研究“像 RViz 粉色箭头一样，给一个位置和车头方向后完成目标”以及当前
  终点疯狂正反旋转的原因；
- 不引入复杂 terminal supervisor，优先寻找单一 `PoseStamped` 下的最小修正。

本轮只整理报告，没有修改代码或参数，也没有再次启动实车。

## 2. 当前正式链

```text
视觉 ACQUIRE
-> 左黄线 + 推算右黄线
-> 两枚圆片确认并冻结到 odom
-> corridor/discs 写入 local + global costmap
-> 视觉 LOCKED，停止重型逐帧识别
-> GlobalPlanner
-> TebLocalPlannerROS（2 条线 + 2 个圆，共 4 个紧凑障碍）
-> /cmd_vel
-> 底盘
```

正式 profile 中没有 DWA、SBPL、passage/gate、navigation safety gate、
terminal visual controller、finish recovery、自定义 recovery 或第二个
`/cmd_vel` 发布者。两轮实车中 `/move_base` 均为唯一速度发布者。

## 3. 实车前已完成的收敛

### 3.1 视觉 ACQUIRE -> LOCKED

视觉算法、640x480、标定、BEV、阈值、黄线截面和圆片验证均未削弱。

| 项目 | ACQUIRE | LOCKED |
|---|---:|---:|
| vision CPU mean | 283.4% | 8.87% |
| vision CPU p95 | 294.9% | 12.37% |
| 重型处理 | 完整运行 | image callback 在 cv_bridge 前返回 |
| disc scan | 正常 | 约 10 Hz 持续 |
| corridor cloud | 正常 | 约 5 Hz 持续 |

释放约 2.7 个 CPU 核后，原 1,078 点障碍 TEB 的 p95 和 infeasible/reset
几乎不变，证明视觉 CPU 不是当时 TEB 失稳的主因。完整报告：

`docs/navigation/PERCEPTION_ACQUIRE_LOCKED_REPORT_20260810.md`

### 3.2 1,078 点障碍 -> 4 个语义障碍

GlobalPlanner 和两个 costmap 仍保留原始 lethal 黄线条带与圆片栅格；只把
TEB 优化器的输入改成 2 个 `LineObstacle` 和 2 个半径 0.05 m 的
`CircularObstacle`。TEB 最终 polygon/costmap feasibility 检查仍保留。

| 项目 | 1,078 点 | 4 个紧凑障碍（重复轮） |
|---|---:|---:|
| move_base CPU | 70.88% | 15.40% |
| cmd period mean | 158.39 ms | 120.20 ms |
| cmd period p95 | 491.05 ms | 134.03 ms |
| >187.5 ms | 6 | 0 |
| infeasible/reset | 5 | 0 |

完整报告：`docs/navigation/TEB_COMPACT_OBSTACLE_AB_REPORT_20260810.md`

## 4. 实际生效的关键参数

```yaml
GlobalPlanner:
  orientation_mode: 1

move_base:
  controller_frequency: 8.0
  recovery_behavior_enabled: false

TebLocalPlannerROS:
  max_vel_x: 0.10
  max_vel_y: 0.08
  max_vel_theta: 0.30
  acc_lim_x: 0.20
  acc_lim_y: 0.20
  acc_lim_theta: 0.60
  xy_goal_tolerance: 0.01
  yaw_goal_tolerance: 0.05       # 2.86 deg
  free_goal_vel: false
  complete_global_plan: true
  global_plan_overwrite_orientation: false
  global_plan_viapoint_sep: 0.10
  weight_viapoint: 10.0
  weight_optimaltime: 1.0
  weight_max_vel_theta: 1.0
  weight_acc_lim_theta: 1.0
  weight_kinematics_nh: 0.0
  weight_kinematics_forward_drive: 0.0
  no_inner_iterations: 6
  no_outer_iterations: 4
  dt_ref: 0.20
  include_costmap_obstacles: false
  min_obstacle_dist: 0.020
  weight_obstacle: 50.0
  feasibility_check_no_poses: 50
  footprint_model: polygon       # 0.34 x 0.33 m
```

底盘/odom 角速度链存在一个必须让 GPT 注意、但尚不能直接判为错误的比例：

```text
car_bringup/pubv.py: serial raw z * 0.75 -> /vel_raw.angular.z
odometry_publisher: angular_scale = 2.22
合成比例：0.75 * 2.22 = 1.665
```

Trial 2 中 `/cmd_vel.wz = +/-0.30 rad/s` 时，odom 报告约 `+/-0.50 rad/s`，
与该比例吻合。该比例可能来自旧标定，也可能造成 TEB 模型与 odom 反馈不一致；
必须独立自由旋转标定后才能决定是否修改。

## 5. 当前目标不是“方框约束”

当前真正发送给 move_base 的消息已经是标准 `geometry_msgs/PoseStamped`：

```text
position.x / position.y = 目标 base_link 中心点
yaw                     = 锁定黄线的前进方向
```

它与 RViz `2D Nav Goal` 粉色箭头点击一个位置并拖出朝向，在控制语义上完全
相同。

RViz 中的绿色方框来自 `competition_corridor_map.py::_markers()` 的
`corridor_goal_vehicle`，只是按 0.34 x 0.33 m 画出的目标 footprint。TEB 不
读取这个 Marker，move_base 也不要求车辆“贴合方框”。

当前自动目标构造为：

```text
从起点车头平面构造 1.85 m 人工 corridor
-> 取右黄线人工远端点 reference
-> 沿黄线方向 forward +0.57 m
-> 向左 +0.24 m
-> 沿黄线方向回退 0.17 m，得到 base_link 中心
-> yaw 取黄线切向
-> 发布一个 PoseStamped
```

所以需要讨论的不是消息是“方框还是箭头”，而是：

1. 单一目标点和 yaw 应如何定义；
2. 如何让全向 TEB 在接近过程中逐步收敛 yaw，而不是把旋转留到最后几厘米；
3. 当前角速度模型、odom 比例、停止判据和 8 Hz 闭环是否允许稳定刹停。

## 6. Stage 1 Trial 1

现场结论：ROS 返回 `Goal reached`，但实车验收失败。

- 两枚圆片前均有转角；
- 未压黄线；
- 接触/压到左侧圆片；
- 最终车体探出过多，物理位置和姿态均不合格。

| 项目 | Trial 1 |
|---|---:|
| ACTIVE -> SUCCEEDED | 27.500 s |
| odom 路径长度 | 2.605 m |
| odom yaw 范围 | -29.19 deg ~ +34.31 deg |
| cmd period mean / p95 / max | 125.00 / 135.58 / 163.39 ms |
| TEB infeasible/reset | 10 / 10 |
| 冻结圆片 1 | (0.565690, -0.116770) m |
| 冻结圆片 2（左侧） | (1.021111, +0.276936) m |
| odom 模型圆片净空 | +7.3 mm / +2.8 mm |
| 目标 | (2.416990, -0.187340), yaw=-1.043 deg |
| 最终 odom 误差 | 2.0 mm / 1.48 deg |

虽然 odom 满足目标容差，但现场终点明显失败，说明 odom 成功不等于物理合格。

完整报告和 rosbag：

```text
docs/navigation/STAGE1_REAL_RUN_REPORT_20260810.md
/home/mowen/Sebastian/stage1_global_teb_compact_20260810_trial1.bag
```

## 7. Stage 1 Trial 2

现场结论：

- 双圆片避障效果可以；
- 两枚圆片均未接触；
- 终点附近疯狂正反旋转；
- 现场曾碰到围墙，之后轮式 odom 继续累计，实际车辆更难完成旋转；
- 用户明确要求不因围墙增加任何保护，围墙只作为定位失真背景。

### 7.1 总体数据

| 项目 | Trial 2 |
|---|---:|
| ACTIVE -> SUCCEEDED | 32.370 s |
| odom 路径长度 | 2.6055 m |
| odom x 范围 | 0.000 ~ 2.3986 m |
| odom y 范围 | -0.3483 ~ +0.1129 m |
| odom yaw 范围 | -46.36 deg ~ +27.74 deg |
| cmd 数量 | 259 |
| cmd period mean / p95 / max | 124.85 / 129.82 / 177.17 ms |
| TEB feedback 数量 / p95 | 251 / 129.83 ms |
| GlobalPlanner 发布 | 20 次 |
| TEB infeasible/reset | 6 / 6 |
| 目标 | (2.406824, -0.302198), yaw=-3.221 deg |
| 结果时 bag 的 odom 误差 | 9.51 mm / 4.31 deg |

六次 infeasible/reset 相对目标发送的时刻：

```text
5.50, 5.77, 6.38, 7.38, 8.63, 17.87 s
```

车辆首次进入目标 5 cm 范围是在 `24.22 s`，所以最后 `8.15 s` 的正反旋转
不是 feasibility reset 引起的。

### 7.2 圆片通过

冻结 odom 坐标：

```text
disc 1: (0.554428, -0.161850) m
disc 2: (1.084504, +0.258238) m
```

用 0.34 x 0.33 m 矩形、0.05 m 圆片半径并对 odom 轨迹做不大于
5 mm / 1 deg 插值扫掠：

| 圆片 | odom 模型最小净空 | 时刻 |
|---|---:|---:|
| disc 1 | +0.22 mm | 8.72 s |
| disc 2 | +48.55 mm | 10.82 s |

现场确认两枚均未接触，因此毫米级 odom 模型净空不能代替物理观察；不应仅凭
`+0.22 mm` 反推发生了碰撞。

### 7.3 终点旋转时间线

目标 yaw 为 `-3.22 deg`：

| 相对时刻 | 位置误差 | odom yaw | yaw 误差 | 现象 |
|---|---:|---:|---:|---|
| 23.0 s | 11.8 cm | -3.5 deg | +0.3 deg | 尚在前进，已穿过目标方向 |
| 24.2 s | 约 5 cm | 约 -30 deg | 约 +27 deg | 进入最终 5 cm 区域 |
| 25.0 s | 4.5 cm | -45.8 deg | +42.6 deg | 第一次大幅过冲 |
| 27.0 s | 1.8 cm | -0.1 deg | -3.1 deg | 再次接近目标方向，仍在转 |
| 28.0 s | 1.3 cm | +22.6 deg | -25.8 deg | 反方向再次过冲 |
| 30.0 s | 1.0 cm | -15.0 deg | +11.8 deg | 又一次反转 |
| 32.17 s | 约 1 cm | -3.08 deg | -0.14 deg | 瞬间满足 yaw，odom wz=-0.477 rad/s |
| 32.37 s | 0.95 cm | -7.53 deg | +4.31 deg | 报 Goal reached，odom wz=-0.038 |

终点 5 cm 范围内持续约 8.15 s，共 66 个 `/cmd_vel`，其中 42 个达到
`|wz| >= 0.29 rad/s`；出现六段明显的近零平移正反旋转。

规划器不是没有目标方向。它多次穿过正确方向，但角速度过大、制动/状态同步
不足或终点代价使其反复越过，无法平稳锁住。

Trial 2 rosbag：

```text
/home/mowen/Sebastian/stage1_global_teb_compact_20260810_trial2.bag
```

## 8. 两轮对比与当前结论

| 项目 | Trial 1 | Trial 2 |
|---|---:|---:|
| 双圆片完成 | 是 | 是 |
| 实物圆片接触 | 左侧圆片接触 | 两枚均未接触 |
| ROS 结果 | SUCCEEDED | SUCCEEDED |
| 物理终点合格 | 否 | 否 |
| ACTIVE 时间 | 27.50 s | 32.37 s |
| 路径长度 | 2.605 m | 2.6055 m |
| infeasible/reset | 10 | 6 |
| cmd p95 | 135.58 ms | 129.82 ms |
| yaw 极值跨度 | 63.5 deg | 74.1 deg |

compact-obstacle TEB 已有能力完成真实双圆片 S 型闭环避障。下一主要问题已
收敛到终点目标定义、接近阶段 yaw 策略及角速度/odom 闭环，而不是视觉 CPU、
GlobalPlanner 拓扑、障碍数量或圆片识别。

## 9. 请 GPT 重点回答

1. 当前 `PoseStamped(position + yaw)` 是否已经与 RViz `2D Nav Goal` 粉色箭头
   完全等价？绿色目标方框是否仅为 Marker、不会参与 TEB 优化？
2. 对全向 TEB 0.9.1，为什么车辆能沿 S 路径以较大 yaw 横移，并把姿态修正
   拖到最后几厘米？请结合当前 holonomic/viapoint/time-optimal 参数分析。
3. 不增加 terminal supervisor、不恢复视觉接管、仍只发送一个 PoseStamped，
   怎样让车辆在接近目标时逐渐对齐，而不是到点原地猛转？
4. `max_vel_theta=0.30`、`acc_lim_theta=0.60`，但 odom 报告约
   `0.50 rad/s`，且角速度链比例为 1.665。应先做什么自由旋转 A/B 标定，才能
   判断是 odom scale、TEB 速度/加速度，还是底盘执行制动问题？
5. 为什么 `32.17 s` yaw 瞬间满足时转速仍为 `-0.477 rad/s`，随后越过目标；
   而 `32.37 s` bag pose 已超出 0.05 rad 容差时仍报告 `Goal reached`？请检查
   TEB 0.9.1 goal-reached 判据、TF/odom 时序和速度阈值。
6. 自动目标虽是 PoseStamped，但由“1.85 m 人工 corridor 远端再前进 0.57 m”
   生成。若严格实现“直接给一个点和箭头”，最小代码结构应怎样改，使纵向点、
   横向点和 yaw 明确可审计？
7. 给出单变量、可回滚的最小修改顺序。不要同时改避障与终点参数；每步说明
   预期效果、副作用及锁止态/低速实车验收指标。

## 10. 可直接复制给 GPT 的请求

> 请审阅这份 ROS1 Noetic 麦轮底盘 Stage 1 两轮实车材料：
>
> `/home/mowen/Sebastian/src/yellow_track_avoidance/docs/navigation/STAGE1_TRIAL1_TRIAL2_GPT_HANDOFF_20260810.md`
>
> 第二轮双圆片避障已实物零接触，请冻结视觉、BEV、圆片冻结、costmap、
> GlobalPlanner、TEB 四个紧凑障碍及绕障参数。不要引入 SBPL、DWA、passage、
> navigation safety gate、terminal visual controller、finish recovery 或新
> supervisor。
>
> 围墙碰撞只是现场描述，不增加墙体检测、碰撞保护、watchdog 或安全逻辑。
> 真正要解决的是：当前标准 PoseStamped 目标虽等价于 RViz 粉色箭头，但全向
> TEB 把 yaw 修正留到最后，并在目标 5 cm 内以 +/-0.30 rad/s 多次正反越过
> 目标方向。请依据 rosbag、TEB 0.9.1 goal-reached 语义和当前 odom 角速度比例，
> 区分目标定义、TEB 终点优化、底盘角速度执行和 odom/TF 时序四类原因。
>
> 输出：证据支持的根因排序；绿色方框与箭头的准确解释；保持单一
> PoseStamped、无复杂状态机时的首选改法；第一轮只改 1～3 个参数或一小段
> 目标生成代码的方案；每项副作用；锁止态和一次低速实车的通过标准；如必须
> 标定角速度，给出不靠墙、可人工急停的最小标定步骤。

## 11. 相关文件

```text
src/yellow_track_avoidance/corridor_map.py
scripts/competition_corridor_map.py
config/competition_teb_baseline.yaml
config/click_nav_global_planner.yaml
launch/competition_global_teb_compact.launch
launch/competition_planner_core.launch

docs/navigation/COMPETITION_PLANNER_BASELINE_REPORT_20260810.md
docs/navigation/PERCEPTION_ACQUIRE_LOCKED_REPORT_20260810.md
docs/navigation/TEB_COMPACT_OBSTACLE_AB_REPORT_20260810.md
docs/navigation/STAGE1_REAL_RUN_REPORT_20260810.md
```
