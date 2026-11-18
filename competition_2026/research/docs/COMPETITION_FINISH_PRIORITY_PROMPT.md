# ROS1 比赛避障：完赛优先、禁止永久锁死的工程问题

请基于下面的真实 ROS1 Noetic 工程状态给出可落地方案。不要泛泛介绍
导航算法；请给出节点职责、状态机、参数、恢复时序、代价地图实现方式、
测试方法以及推荐的最终参数表。

## 1. 比赛目标与优先级

- 机器人为全向底盘，实际车体长 `0.34 m`、宽 `0.33 m`。
- 两个红白圆片直径约 `0.10 m`，圆片边缘最小间距实测约 `0.40 m`。
  车体理想居中时理论上每侧仅余约 `0.035 m`。
- 压到圆片只扣分；规划失败后永久停车会导致整场无分。因此策略必须是：
  **优先避开圆片，但任何视觉、几何或规划失败都不能形成永久锁死。**
- 实体墙、底盘通信故障、人工急停仍属于硬安全约束。
- 黄线正常情况下不能压；黄线视觉暂时丢失时，应先短暂停车/降速并使用
  最近可信走廊或静态地图继续，而不是永久锁死。

建议请明确区分：

1. 硬停止：人工急停、串口故障、真实激光墙体即将碰撞；
2. 可恢复暂停：短时视觉/TF/定位数据过期；
3. 软惩罚：圆片接触风险、圆片规划代价；
4. 最后兜底：避不开圆片时允许压片继续完成路线。

## 2. 当前系统

- ROS1 Noetic、Ubuntu 20.04、Intel N95 四核平台。
- 静态地图 + AMCL；全局规划器为 `SBPLLatticePlanner`，局部规划器为
  `teb_local_planner/TebLocalPlannerROS`，不希望无依据地更换算法。
- 摄像头原始帧约 `30 Hz`；当前视觉处理约 `153 ms/frame`，边界输出约
  `6.4 Hz`。
- 相机检测黄线和两个红白圆片。圆片有固定坐标跟踪、通行状态机和可视化。
- 只在地图右上角区域启用视觉通行监督：
  `[(0.00,1.20),(2.35,1.20),(2.35,1.95),(0.00,1.95)]`。
- 通行监督器检测单圆片、多圆片和窄门后，会取消 move_base，使用低速全向
  控制通过，再恢复原目标。
- 当前窄门速度 `0.04 m/s`，普通视觉绕行速度 `0.05 m/s`。

## 3. 本次真实锁死证据

导航目标发送时间约为 `1785781900.94`，故障链如下：

1. `1900.97`：识别到窄门并开始确认；
2. `1901.47`：通行监督器发送 cancel，move_base 被正常抢占；
3. `1901.82`：进入 `gate_align`；
4. `1903.97`：黄线边界丢失，进入 `degraded_hold`；
5. `1904.42`：超过 `degraded_hold_sec=0.40`，进入不可恢复
   `failed_stop`，原因 `critical_input_timeout`；
6. move_base 状态为 `PREEMPTED(2)`，不是 SBPL 自己 ABORT；最终速度门因
   `passage_stop:failed_stop` 永久输出零。

故障时的现场数据：

- 两圆片仍然新鲜，边缘间距 `0.40007 m`；
- 边界质量约 `0.483`；左侧只有 7/28 个有效段、支持率 `0.25`，低于配置的
  `minimum_valid_sections=12`、`minimum_support_ratio=0.38`；
- 相机画面中只剩左侧实体黄线位于画面边缘，另一侧主要是墙边；
- `boundary_fresh=false`、`route_valid=false`；
- AMCL 报告的位置标准差约 `0.418 m`，还需确认是否存在定位漂移或仅为初始
  协方差设置偏大。

所以本次直接根因是“视觉短时/持续丢线被升级成永久故障”，不是圆片太贵或
SBPL 主动判定无路。但原来的圆片 costmap 注入也可能让后续 move_base 无法
兜底。

## 4. 已实施的完赛优先修改

以下修改已经做了单元测试，但还需要评估是否为最终比赛方案：

- `competition_finish_priority=true`：
  - 接管后的视觉、几何、净空或控制失败不再进入永久 `failed_stop`；
  - 先输出零速完成交接，然后释放特殊控制权并重新发送保存的 move_base
    目标；
  - 视觉直接判断无可行通道时，不再占住控制权，允许 move_base 继续。
- 当前车入口默认 `disc_costmap_writes_enabled=false`：圆片仍被检测、跟踪、
  显示，并供视觉通行控制器绕行，但不再由标准 `obstacle_layer` 写成致命
  costmap 单元。对 move_base 而言圆片代价为最低值 `0`。
- `enforce_boundary_in_activation_zone=false`：视觉边界失效不会永久否决
  普通 TEB 指令；特殊视觉控制器仍在数据可靠时优先接管。
- `enforce_disc_emergency_clearance=false`：圆片接触视作比赛扣分而不是硬停；
  实体激光障碍仍由 costmap/TEB 处理。
- SBPL 的 `lethal_obstacle` 从 `20` 提高到 `253`，使膨胀区成为偏好代价，
  只有静态地图/真实激光产生的 inscribed/lethal 单元不可穿越。
- 车辆启动默认仍锁住，必须显式调用服务解锁；上述“完赛优先”不绕过人工
  急停、唯一 `/cmd_vel` 发布者检查和底盘订阅检查。

## 5. 当前关键参数

```yaml
# passage_supervisor.yaml
boundary_timeout_sec: 0.45
minimum_boundary_quality: 0.70
minimum_single_boundary_quality: 0.35
degraded_hold_sec: 0.40
competition_finish_priority: true
gate_takeover_distance_m: 1.05
minimum_predicted_gate_clearance_m: 0.013
emergency_direct_clearance_m: 0.005
gate_speed_mps: 0.040
bypass_speed_mps: 0.050

# current_car_navigation.launch effective policy
disc_costmap_writes_enabled: false
enforce_boundary_in_activation_zone: false
enforce_disc_emergency_clearance: false

# click_nav_sbpl.yaml
allocated_time: 5.0
initial_epsilon: 3.0
lethal_obstacle: 253

# click_nav_teb.yaml
max_vel_x: 0.10
max_vel_y: 0.08
max_vel_theta: 0.30
min_obstacle_dist: 0.025
inflation_dist: 0.080
weight_obstacle: 80.0
weight_viapoint: 20.0
enable_homotopy_class_planning: false
```

## 6. 希望重点回答的问题

1. ROS1 `costmap_2d` 标准 obstacle layer 会把每个 LaserScan 障碍直接写成
   lethal，无法为圆片单独设置低非零代价。是否应实现独立的圆片软代价层
   插件？请给出插件的 cost 写入公式、代价值范围、叠加方式和清除策略。
2. 在不把圆片写入 costmap 的情况下，用视觉通行控制器优先绕行、失败后让
   SBPL+TEB 忽略圆片继续，是否是本比赛更稳妥的方案？有哪些隐藏风险？
3. 黄线只剩一侧或移动时跑出画面，应如何组合：单边米制推断、最后可信边界
   随 odom 短时传播、置信度迟滞、静态地图走廊和最大盲行时间？请给具体
   状态机和建议秒数，要求最终不会永久锁死。
4. 对 `0.33 m` 宽车通过 `0.40 m` 圆片边缘间隙，SBPL、TEB 和视觉窄门控制
   的上述参数哪些还过于保守？最低合理净空、TEB obstacle distance、权重和
   速度应该设多少？
5. move_base/SBPL 或 TEB 本身 ABORT、振荡、连续无速度输出时，建议使用什么
   有界恢复序列？例如清 costmap、重规划、临时忽略圆片、沿最近可信中心线
   低速前进。请确保每个阶段都有超时和下一阶段，不能停在等待状态。
6. 是否需要增加“同一目标最多一次视觉特殊接管”的锁存，防止恢复目标后反复
   cancel/resume？如果需要，解除条件应是通过圆片、离开激活区还是新目标？
7. AMCL 位置标准差约 `0.418 m` 对这个局部任务是否应成为硬门槛？如何避免
   AMCL 漂移导致激活区和全局路径判断错误，同时又不因定位不收敛锁死？
8. 请设计可记录和统计的验收指标：完成率、压片次数、压黄线次数、最长零速
   时间、恢复次数、最小净空、视觉丢失持续时间和单次 CPU 峰值。

请最终给出三个可比较方案：

- A：当前“圆片 cost=0 + 视觉优先 + move_base 兜底”的最小改动方案；
- B：独立圆片软代价层方案；
- C：增加最后可信走廊/有限盲行控制器的比赛强化方案。

对每个方案说明成功条件、最坏行为、实施复杂度、CPU 开销和推荐顺序，并给出
你认为最适合比赛的最终方案及完整参数表。
