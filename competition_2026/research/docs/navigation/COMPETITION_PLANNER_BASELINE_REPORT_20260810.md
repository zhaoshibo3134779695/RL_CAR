# 2026-08-10 正式比赛规划器锁止态报告

## 结论

当前不能进入低速实车。

- `GlobalPlanner` 在 DWA Trial 3 左错列回放中仍能产生正确 S 路线；全局路线不是第一故障点。
- 原始最小 TEB 能输出连续 `vx + vy`，但未来 polygon 轨迹与近圆片重叠约 28 mm。
- 把障碍关联范围扩展到覆盖 0.237 m 车体外接半径、检查完整轨迹并把 `weight_obstacle` 提到 50 后，TEB 能产生正净空轨迹，但仍在“非零命令—trajectory not feasible—零速重置”之间循环。
- 当前 TEB feedback 含 1078 个点障碍；move_base 占约 74%～79% 的单核，控制循环常见 0.138～0.157 s、重置周期约 0.40 s，不能稳定满足 8 Hz。
- SBPL plugin 和运动基元能静态加载，0.08 m 容差预览能生成路线，但正式精确目标连续搜索失败；同时 RSS 增长到约 623 MiB。因此 SBPL 不进入第一主线。

本轮未连接底盘、未自动实车。所有目标都由无底盘探针发送并在采集后取消。

## 1. 最小 GlobalPlanner + TEB 节点图

```text
/yellow_track_competition_map_server
/robot_state_publisher
/yellow_track_map_to_odom
/yellow_track_avoidance                 (只感知，motion/planning=false)
/yellow_track_disc_costmap_bridge
/yellow_track_competition_corridor_map
/move_base                              (GlobalPlanner + TEB)
```

数据链为：

```text
视觉边界/圆片
  -> odom 固定圆片 + corridor cloud
  -> global/local costmap
  -> GlobalPlanner
  -> TebLocalPlannerROS
  -> /cmd_vel
```

## 2. `/cmd_vel` 控制权

锁止探针发送目标前从 ROS master 得到：

```text
publishers:  [/move_base]
subscribers: [/yellow_track_disc_costmap_bridge]
```

不存在 `/cmd_vel_listener`，所以没有底盘命令消费者。探针发现其他 publisher 或未知 subscriber 时会直接拒绝发送目标。

## 3. 正式 profile 中旁路的旧节点

- `navigation_safety_gate`；
- `passage_supervisor`；
- `gate_align/gate_cross/gate_exit` 和 40 cm controller；
- free-interval controller；
- terminal visual controller；
- finish recovery；
- 自定义 recovery；
- 第二套 global/local planner。

这些文件未删除，只从三个互斥 A/B profile 断开。

## 4. Trial 3 场景和 GlobalPlanner 路线

回放使用历史实车 Trial 3 坐标：

```text
近侧右圆片: (0.655319, -0.150049) m
远侧左圆片: (1.135951,  0.215365) m
终点:       (2.407789, -0.282988) m
终点 yaw:   -0.057 rad
```

两枚圆片在 global/local costmap 中的采样最大值均为 `100`（lethal）。GlobalPlanner 路线：

| 指标 | 结果 |
|---|---:|
| 点数 | 1006 |
| 长度 | 2.5118 m |
| 矩形 footprint 到圆片最小净空 | +0.0360 m |
| 矩形 footprint 到黄线内边最小净空 | +0.00466 m |
| 最大相邻 yaw 变化 | 0.2518 rad |
| 最大相邻路径方向变化（含末端短段） | 2.9616 rad |

路线净空为正，但终点附近黄线净空只有约 4.7 mm，不能把 GlobalPlanner 路线本身当作最终安全证明。

## 5. TEB local trajectory

### 初始基线

TEB 不是退化成 1～2 点：它生成 17～18 点、约 4.0 s、约 0.88 m 的未来轨迹，并给出连续正 `vx + vy`。但完整矩形扫掠复核得到近圆片净空 `-0.0281 m`。

根因是当 `min_obstacle_dist=0.005` 时，非 legacy 关联的强制/截止半径只有 0.015/0.040 m，远小于矩形车体约 0.237 m 的外接半径；且当时只检查前 8 个轨迹点，碰撞位于更后的点。

### 安全关联修正

最终留在候选配置中的相关值为：

```text
min_obstacle_dist: 0.020
obstacle_association_force_inclusion_factor: 12.0
obstacle_association_cutoff_factor: 15.0
feasibility_check_no_poses: 50
weight_obstacle: 50.0
```

修正后出现过 +0.0569～+0.0602 m 的安全轨迹，但也仍出现 -0.0153 m 和 +0.0051 m 的不可接受轨迹。完整 feasibility 检查会拒绝它们并发布零速，因此没有把已知碰撞轨迹直接放行；代价是锁止状态下约每 1.15 s 重置一次。

当前起点 local horizon 终止在约 `(0.856, 0.051)`，尚未覆盖远侧左圆片后的反向横移，所以本次只能确认第一段 `vy>0` 方向正确，不能声称第二段负 `vy` 已验证。

## 6. 实际生效 TEB 参数完整表

以下来自运行时参数服务器和 TEB dynamic reconfigure 展开值：

| 参数 | 实际值 |
|---|---|
| `acc_lim_theta` | `0.6` |
| `acc_lim_x` | `0.2` |
| `acc_lim_y` | `0.2` |
| `allow_init_with_backwards_motion` | `false` |
| `cmd_angle_instead_rotvel` | `false` |
| `complete_global_plan` | `true` |
| `costmap_converter_plugin` | `""` |
| `costmap_converter_spin_thread` | `false` |
| `costmap_obstacles_behind_robot_dist` | `0.8` |
| `dt_hysteresis` | `0.02` |
| `dt_ref` | `0.2` |
| `dynamic_obstacle_inflation_dist` | `0.005` |
| `enable_homotopy_class_planning` | `false` |
| `enable_multithreading` | `false` |
| `exact_arc_length` | `false` |
| `feasibility_check_no_poses` | `50` |
| `footprint_model` | `polygon: [[-0.17,-0.165],[-0.17,0.165],[0.17,0.165],[0.17,-0.165]]` |
| `force_reinit_new_goal_angular` | `0.78` |
| `force_reinit_new_goal_dist` | `1.0` |
| `free_goal_vel` | `false` |
| `global_plan_overwrite_orientation` | `false` |
| `global_plan_viapoint_sep` | `0.1` |
| `h_signature_prescaler` | `0.5` |
| `h_signature_threshold` | `0.1` |
| `include_costmap_obstacles` | `true` |
| `include_dynamic_obstacles` | `false` |
| `inflation_dist` | `0.005` |
| `is_footprint_dynamic` | `false` |
| `legacy_obstacle_association` | `false` |
| `map_frame` | `odom` |
| `max_global_plan_lookahead_dist` | `0.9` |
| `max_number_classes` | `1` |
| `max_samples` | `200` |
| `max_vel_theta` | `0.3` |
| `max_vel_x` | `0.1` |
| `max_vel_x_backwards` | `0.04` |
| `max_vel_y` | `0.08` |
| `min_obstacle_dist` | `0.02` |
| `min_turning_radius` | `0.0` |
| `no_inner_iterations` | `6` |
| `no_outer_iterations` | `4` |
| `obstacle_association_cutoff_factor` | `15.0` |
| `obstacle_association_force_inclusion_factor` | `12.0` |
| `obstacle_cost_exponent` | `1.0` |
| `obstacle_heading_threshold` | `0.45` |
| `obstacle_poses_affected` | `12` |
| `odom_topic` | `/odom` |
| `optimization_activate` | `true` |
| `optimization_verbose` | `false` |
| `oscillation_recovery` | `false` |
| `penalty_epsilon` | `0.0` |
| `publish_feedback` | `true` |
| `roadmap_graph_area_length_scale` | `1.0` |
| `roadmap_graph_area_width` | `1.2` |
| `roadmap_graph_no_samples` | `20` |
| `selection_alternative_time_cost` | `false` |
| `selection_cost_hysteresis` | `1.0` |
| `selection_obst_cost_scale` | `0.0` |
| `selection_prefer_initial_plan` | `0.9` |
| `selection_viapoint_cost_scale` | `0.0` |
| `shrink_horizon_backup` | `false` |
| `simple_exploration` | `false` |
| `switching_blocking_period` | `0.0` |
| `teb_autosize` | `true` |
| `via_points_ordered` | `true` |
| `viapoints_all_candidates` | `false` |
| `visualize_hc_graph` | `false` |
| `visualize_with_time_as_z_axis_scale` | `0.0` |
| `weight_acc_lim_theta` | `1.0` |
| `weight_acc_lim_x` | `1.0` |
| `weight_acc_lim_y` | `1.0` |
| `weight_adapt_factor` | `1.3` |
| `weight_dynamic_obstacle` | `0.0` |
| `weight_dynamic_obstacle_inflation` | `0.0` |
| `weight_inflation` | `0.0` |
| `weight_kinematics_forward_drive` | `0.0` |
| `weight_kinematics_nh` | `0.0` |
| `weight_kinematics_turning_radius` | `0.0` |
| `weight_max_vel_theta` | `1.0` |
| `weight_max_vel_x` | `2.0` |
| `weight_max_vel_y` | `2.0` |
| `weight_obstacle` | `50.0` |
| `weight_optimaltime` | `1.0` |
| `weight_shortest_path` | `0.0` |
| `weight_viapoint` | `10.0` |
| `wheelbase` | `1.0` |
| `xy_goal_tolerance` | `0.01` |
| `yaw_goal_tolerance` | `0.05` |

move_base controller frequency 为 8 Hz；recovery 和 clearing rotation 均关闭。

## 7. TEB 时间和 CPU

TEB 0.9.1 没有发布“纯 optimizer 用时”，因此不能把 controller 周期冒充 optimizer 单项时间。可观测结果为：

| 指标 | 初始关联 | 安全关联 + obstacle weight 50 |
|---|---:|---:|
| move_base 单核口径 CPU | 22.4% | 74%～79% |
| move_base RSS | 57.9 MiB | 65.8 MiB |
| `/cmd_vel` 平均周期 | 0.1248 s | 0.160～0.172 s |
| `/cmd_vel` p95 周期 | 0.1316 s | 约 0.49 s |
| rosout 控制周期 | 基本接近 8 Hz | 常见 0.138～0.157 s，重置约 0.40 s |
| TEB feedback 点障碍 | 未记录 | 1078 |

实景视觉只读测试中，vision 约占 245% CPU。视觉和修正后的 TEB 同时运行时，N95 四核的余量不足以容纳大抖动、costmap 和系统线程；算力问题主要来自视觉、多达 1078 个未转换点障碍、polygon 距离优化及 24 次 inner/outer 优化，而不是单独某个 ROS topic。

## 8. 左错列锁止态预测

- 全局 S 拓扑正确；
- 第一段横移方向正确，`vy` 全为正；
- TEB 轨迹长度足够，不是 pose 附近 1～2 点；
- 第二段反向 `vy` 尚不在 0.9 m local horizon 中，未验证；
- 当前会出现 infeasible 重置和周期超时；
- 因此预测是“有越过 DWA 原停滞区的驱动力，但尚不具备安全、稳定实车条件”。

## 9. 与 DWA Trial 3 的差异

| 指标 | DWA Trial 3 实车 | TEB 锁止态 |
|---|---|---|
| 初始横移 | 最大 `vy≈0.0218` | 可到 `vy=0.08` |
| 纵向 | 最大 `vx≈0.0827` | 可到 `vx=0.10` |
| 局部输出 | 约 0.2 m 后近零正反振荡 | 明显非零 `vx+vy`，但 infeasible 后归零重置 |
| 结果 | 22.71 s 后 `Robot is oscillating` | 未实车；静止回放重复 feasibility reset |
| 主要矛盾 | 采样执行振荡 | 点障碍优化安全性与 8 Hz 算力 |

TEB 已证明它和 DWA 的失败形态不同，但还没有证明能实车通过。

## 10. SBPL profile 审计和 A/B

静态审计通过：

- x86_64 SBPL plugin 和 package-local `libsbpl.so` 可加载；
- `XYThetaLattice + ARAPlanner`；
- primitive 分辨率 0.005 m，与实际静态 map 0.005 m 一致；
- 16 个角度、176 个 primitive；
- primitive 包含前进、后退、旋转和全向横移；
- footprint 与另外两个 profile 相同。

正式精确目标每次约 1.48 s 后返回 `Solution not found`，连续两次后 move_base abort。0.08 m 容差的 inactive `make_plan` 预览可成功：

| 指标 | GlobalPlanner | SBPL 容差预览 |
|---|---:|---:|
| 点数 | 1006 | 712 |
| 长度 | 2.5118 m | 2.7849 m |
| 圆片最小净空 | 0.0360 m | 0.0204 m |
| 黄线最小净空 | 0.0047 m | 0.0221 m |
| 最大相邻 yaw 变化 | 0.2518 rad | 0.0570 rad |
| 最大相邻方向变化 | 2.9616 rad | 1.5732 rad |
| active 搜索 | 成功 | 精确目标失败 |
| move_base CPU | 约 22%（原始 TEB 联合） | 约 48% |
| move_base RSS | 58 MiB | 481～623 MiB |

SBPL 的容差路线 yaw 更连续、离黄线更远，但更长、圆片净空更小，而且正式精确目标不可用。它没有显示出足以抵消失败率和内存成本的优势。

## 11. 下一轮只建议调的参数

不要一次改完，按顺序单变量/单组验证：

1. `no_inner_iterations: 6 -> 4`：降低每次图优化成本；副作用是约束收敛变差。
2. `no_outer_iterations: 4 -> 2`：与上一项作为同一“算力组”测试；副作用是 obstacle weight adaptation 次数减少。
3. `weight_viapoint: 10 -> 5`：降低路径点对障碍变形的竞争；副作用是局部轨迹可能偏离 GlobalPlanner 更多。
4. `dt_ref: 0.20 -> 0.25`：减少时间样本；副作用是轨迹离散更粗，必须继续做 5 mm 扫掠复核。

以下安全修正暂不回退：`feasibility_check_no_poses=50`、`min_obstacle_dist=0.02`、关联范围 12/15、`weight_obstacle=50`。

## 12. 是否建议低速实车

不建议。进入低速实车前至少要满足：

- 同一 Trial 3 锁止回放中连续采集不再出现 `trajectory is not feasible`；
- 5 mm 插值矩形扫掠的圆片、黄线净空持续为正，圆片建议至少 20 mm；
- 8 Hz 控制周期 p95 小于 0.125 s，或有证据支持把 controller frequency 降低且不影响底盘闭环；
- 第一段局部轨迹稳定，不再周期性重置；
- 再开一次真实相机只读联合负载，确认整机 CPU 余量。

满足后仍只允许左错列单次低速实车，不自动继续右镜像、中间布局或 SBPL。
