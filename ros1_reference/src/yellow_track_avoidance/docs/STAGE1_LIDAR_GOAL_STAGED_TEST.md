# Stage 1 雷达入口终点分段测试

本流程只替换 Stage 1 的终点来源，不修改视觉识别、圆片代价地图、
GlobalPlanner、TEB 或底盘参数。Stage 2 尚未接入。

## 一条命令自动运行

正式 Stage 1 使用状态驱动串联，不依赖固定长延时：

```bash
roslaunch yellow_track_avoidance competition_stage1_auto.launch enable_motion:=true
```

程序依次等待视觉世界模型锁定、10帧雷达入口目标冻结和有效全局路径，
确认 `/move_base` 是唯一速度发布者且底盘消费者在线后自动发送目标。当前现场
实测冷启动至目标可发送约15秒，其中相机控制稳定约占10秒、雷达采集约1.35秒。
只做零运动全流程检查时省略运动开关：

```bash
roslaunch yellow_track_avoidance competition_stage1_auto.launch
```

此时状态到达 `DRY_RUN_READY`，不会启动底盘或发送目标。状态话题为：

```text
/yellow_track_stage1_auto_runner/status
```

## 阶段 1：视觉建模（底盘断开）

```bash
roslaunch yellow_track_avoidance competition_stage1_prepare.launch
```

该阶段启动 RGB、IMU/里程计、视觉世界模型、GlobalPlanner/TEB、RViz 与
识别结果窗口，但不启动底盘串口速度消费者。确认走廊和两枚圆片均已
冻结并写入 costmap 后，再进入阶段 2。

## 阶段 2：雷达冻结目标（仍不动车）

```bash
roslaunch yellow_track_avoidance competition_stage1_lidar_goal_capture.launch
```

节点等待以下条件稳定 0.5 s：走廊锁定、两枚圆片确认并冻结、costmap
写入两枚圆片。随后静止采集 60 帧 `/scan`，使用右侧连续长墙确定方向，
使用左入口稀疏点确定入口截面，并把目标锁存在 `map`：

```text
入口中心 E = (左入口点 P_L + 右墙垂足 P_R) / 2
Stage1 base_link 目标 = E - (0.17 m + 0.10 m) * 窄道方向
目标 yaw = 右墙方向
```

此阶段只发布目标 Marker 和 `/move_base/make_plan` 预览，不发送活动导航
目标，也不发布 `/cmd_vel`。状态可查看：

```bash
rostopic echo -n 1 /yellow_track_stage1_lidar_goal/status
```

## 阶段 3：人工确认后解锁并发送冻结目标

先启动唯一底盘速度消费者：

```bash
roslaunch yellow_track_avoidance competition_stage1_chassis_unlock.launch
```

随后立即发送阶段 2 已冻结的同一个目标：

```bash
rosservice call /yellow_track_stage1_lidar_goal/send_goal
```

之所以不在阶段 2 提前发送活动目标，是因为当前 `move_base` 静止振荡
超时为 6 s；提前发送会在人工观察 RViz 期间使目标失效。

## 重采集

只清除雷达终点：

```bash
rosservice call /yellow_track_stage1_lidar_goal/reset
```

正式新任务仍使用现有 scene reset，使走廊、圆片和雷达终点一起重新采集。
