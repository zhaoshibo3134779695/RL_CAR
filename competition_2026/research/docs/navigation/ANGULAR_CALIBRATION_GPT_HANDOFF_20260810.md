# Stage 1 终点旋转标定与 GPT 交接（2026-08-10）

## 1. 当前目的

`GlobalPlanner + compact-obstacle TEB` 已在 Stage 1 Trial 2 中成功绕过两枚圆片，
但到终点附近出现长时间 `wz` 正反打满、车体反复旋转的问题。

本次只标定底盘角速度指令、物理转角和轮式 odom 的关系。测试期间：

- 不运行 move_base；
- 不运行 DWA、SBPL、passage、safety gate 或 terminal controller；
- `/cmd_vel` 只有专用标定发布者；
- 每次命令后连续发送零速度；
- 物理角度由地面基准线和量角器独立测量，不读取 RViz；
- 同时记录 `/cmd_vel`、`/vel_raw`、`/odom_raw` 和 `/odom`。

测试完成后已经发送零速度，并关闭底盘、轮速与 odom 标定节点。

## 2. 当前角速度链

命令链：

```text
/cmd_vel.angular.z
  -> car_bringup/scripts/newt.py
  -> cmd_twist_rotation * 1000
  -> 底盘串口
```

反馈链：

```text
底盘串口 z
  -> car_bringup/scripts/pubv.py
  -> /vel_raw.angular.z = z/1000 * 0.75
  -> car_bringup/src/base.cpp
  -> /odom_raw.angular.z = /vel_raw.angular.z * 2.22
  -> robot_localization
  -> /odom
```

反馈链当前合成比例为：

```text
0.75 * 2.22 = 1.665
```

`newt.py` 对正负角速度使用相同的数值转换和限幅；当前限幅是
`max_angular_z=0.50 rad/s`。

## 3. 有效标定结果

符号约定：左转为正，右转为负。

| 组别 | 命令 | 物理起点 | 物理终点 | 物理增量 | odom 增量 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| Group 1 | `+0.10`，首次定时 | `0.0°` | `+26.1°` | `+26.1°` | `+26.148°` | 首次 `rostopic` 启动延迟使有效命令时长不适合计算执行增益，但物理角与 odom 可比较 |
| Group 2 | `-0.10 × 3.0 s` | `+26.1°` | `+0.8°` | `-25.3°` | `-25.590°` | 精确定时；连续 odom 端点有效 |
| Group 3 | `+0.20 × 3.0 s` | `+0.8°` | `+47.2°` | `+46.4°` | `+56.160°` | 精确定时、完整 bag |
| Group 4 repeat | `-0.20 × 3.0 s` | `0.0°` | `-68.0°` | `-68.0°` | `-55.243°` | 移除车顶板后重做；精确定时、完整 bag |

有效 bag：

```text
/home/mowen/Sebastian/angular_calibration_20260810_group1.bag
/home/mowen/Sebastian/angular_calibration_20260810_group3.bag
/home/mowen/Sebastian/angular_calibration_20260810_group4_repeat.bag
```

Group 2 的 rosbag 在结束时未正常封存，不能作为可回放证据；表中 Group 2
odom 值来自未重启的连续 `/odom_raw` 端点读数。物理起点、终点和连续 odom
端点均已记录，因此本表仍可用于这次角度差比较，但后续正式结论应优先参考
可回放的 Group 1、3 和 Group 4 repeat。

另有一个原 Group 4：

```text
/home/mowen/Sebastian/angular_calibration_20260810_group4.bag
```

该次车顶遗留了一块板，物理条件不一致，明确作废，不得纳入计算。

## 4. 数据计算

### 4.1 低速 `|cmd wz|=0.10`

正转物理角与 odom：

```text
物理：+26.100°
odom：+26.148°
误差：+0.048°，约 +0.18%
```

反转物理角与 odom：

```text
物理：-25.300°
odom：-25.590°
误差：绝对值多报 0.290°，约 +1.15%
```

因此在 `0.10 rad/s` 量级，物理角与轮式 odom 基本一致。

Group 2 的理论命令积分为：

```text
0.10 rad/s * 3.0 s = 0.30 rad = 17.189°
```

而物理转角为 `25.3°`，本次短时动作的物理转角约为命令积分的
`1.47 倍`。

### 4.2 高速 `|cmd wz|=0.20`

正转：

```text
理论命令积分：+34.379°
物理真实增量：+46.400°
odom 增量：    +56.160°

物理 / 命令：1.35
odom / 物理：1.21
```

即正转时 odom 比物理角多报约：

```text
56.160 - 46.400 = 9.760°
9.760 / 46.400 = 21.0%
```

反转：

```text
理论命令积分：-34.379°
物理真实增量：-68.000°
odom 增量：    -55.243°

|物理 / 命令|：1.98
|odom / 物理|：0.81
```

即反转时 odom 比物理角少报约：

```text
68.000 - 55.243 = 12.757°
12.757 / 68.000 = 18.8%
```

同样的 `0.20 rad/s × 3 s`，实际左右转角分别为：

```text
正转：46.4°
反转：68.0°
```

反转物理转角是正转的约 `1.47 倍`。这是明显的方向不对称。

同时两方向 odom 峰值却都接近：

```text
Group 3 odom_raw peak |wz|: 0.4446 rad/s
Group 4 repeat peak |wz|:    0.4429 rad/s
```

也就是说，当前串口反馈/轮式 odom 没有反映出实车左右转角的巨大差异。
需要进一步确认串口 z 到底是实际编码器估计、控制器内部估计，还是接近命令回显。

## 5. 直接结论

1. 不能用一个新的全局 `angular_scale` 修复全部问题。
   `+0.20` 时 odom 需要向下修，而 `-0.20` 时 odom 反而需要向上修；修正方向相反。
2. 当前 `angular_scale=2.22` 在 `|cmd wz|=0.10` 量级与物理角吻合，不能因为高速数据直接删除或大幅降低。
3. 高角速度区域同时存在：
   - 物理执行大于 ROS 指令；
   - 正反方向执行严重不对称；
   - 轮式 odom 在高速下出现方向相关误差。
4. 这足以解释终点处 TEB 使用 `±0.30 rad/s` 时容易过冲、反向纠正并形成持续振荡。
5. 由于麦轮高速原地旋转可能打滑，但当前数据也不能排除电机/PID、固件映射、左右轮参数或反馈语义问题。

## 6. 当前 TEB 与版本事实

当前正式配置：

```yaml
max_vel_theta: 0.30
acc_lim_theta: 0.60
yaw_goal_tolerance: 0.05
```

Trial 2 终点附近大量命令打满 `wz=±0.30 rad/s`。

实际安装版本：

```text
ros-noetic-teb-local-planner 0.9.1-1focal.20250521.005922
```

已检查本机安装头文件和生成配置：当前 0.9.1 没有：

```text
theta_stopped_vel
trans_stopped_vel
```

精确 tag `0.9.1` 的 goal check 也没有调用
`base_local_planner::stopped()`。这些停止速度参数存在于更新的官方
`noetic-devel` 源码，但不是当前二进制的实际行为。

因此直接把这两个参数写进当前 YAML 会被忽略。若需要停止速度判据，应选择：

- 在工作空间对当前 0.9.1 做最小 stopped-condition backport；或
- 对完整 noetic-devel 做隔离 A/B 回归后再切换。

不应直接覆盖 `/opt/ros`，也不应把完整开发分支升级和本轮终点参数同时混为一个变量。

## 7. 终点位置是另一条独立问题

现场几何已经补充：

```text
物理黄线总长：1.85 m
车辆起跑时，车头已越过黄线起点：约 0.30 m
```

因此从当前车头到物理黄线终点的剩余距离是：

```text
1.85 - 0.30 = 1.55 m
```

若要求车头停在终点平面前 `0.02 m`，从当前起点计算的 base_link
纵向位移也是：

```text
1.55 - 0.02 = 1.53 m
```

车头前缘偏移在“当前和最终”两端相消，不需要再额外加一次 `0.17 m`。

当前正式 launch 仍使用：

```text
corridor_length_m = 1.85
goal_forward_from_right_endpoint_m = +0.57
```

这相当于纵向目标约 `2.42 m`，相对 `1.53 m` 的预期停车位多出约
`0.89 m`，与 Trial 1/2 终点明显探出相符。

后续目标仍应是单一 `PoseStamped(x,y,yaw)`：

- 纵向按上述物理长度计算；
- 横向沿用当前已经接受的偏置，不在同一轮改变；
- yaw 使用锁定黄线方向；
- 绿色车框只是 RViz Marker，不是第二个控制器。

## 8. 已经成功且不应回退的部分

- 视觉 ACQUIRE -> LOCKED：CPU 已从约 283% 降到约 8.9%；
- corridor 与两枚圆片冻结到 odom/costmap；
- GlobalPlanner S 型路线；
- TEB 使用 2 条 LineObstacle + 2 个 CircularObstacle；
- Trial 2 真实绕过两枚圆片且没有接触；
- 单一 `/cmd_vel` 权限链。

不要因为终点旋转问题恢复：

- DWA；
- SBPL；
- passage/gate；
- navigation_safety_gate；
- terminal visual controller；
- finish recovery；
- 墙体保护或新的 watchdog/supervisor。

用户明确要求墙体碰撞只作为现场现象描述，不因此增加任何保护逻辑。

## 9. 请 GPT 重点回答

请根据以上实测，而不是只根据 TEB 默认值，给出以下判断：

1. `0.10` 区域 odom 准确、`0.20` 区域严重且方向相关失真的首要原因排序是什么？
2. 当前串口 z 反馈更像真实编码器估计还是控制器内部回显？还需要查看底盘固件/电机 PID 的哪些证据？
3. 对比赛最小方案，是否应先把 TEB `max_vel_theta` 限制在已经验证可靠的约 `0.10 rad/s`，并把 `acc_lim_theta` 降到约 `0.20~0.25 rad/s^2`？
4. 是否需要正负方向分别做命令增益；如果需要，怎样设计不会把轮式 odom 与实际运动再次分离？
5. 在没有独立 IMU yaw 的情况下，还应补哪几组最少量的稳态/短时标定，才能区分轮胎打滑、底盘执行映射和反馈比例？
6. 对当前 TEB 0.9.1，是否建议只 backport stopped-condition，而不是完整升级 noetic-devel？
7. 是否同意先独立修正终点纵向目标为约 `1.53 m`，然后只改角速度/角加速度做一次 Stage 1 A/B？
8. 若低速限制后仍然在终点反复旋转，下一次只应改变哪个单一变量：GlobalPlanner `orientation_mode 1 -> 3`、TEB yaw tolerance、还是 stopped-condition？

要求给出严格的单变量实施顺序，并明确哪些结论来自实测、哪些只是待验证假设。
