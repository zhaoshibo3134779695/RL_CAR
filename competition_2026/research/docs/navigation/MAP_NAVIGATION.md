# 5 m × 4 m 地图导航说明

## 当前导航范围

地图源文件为桌面 `7.15.png`，原图 96×78 像素。它被最近邻转换为
500×400 的 ROS 占据栅格，分辨率 `0.01 m/px`，地图严格表示
`5.00 m × 4.00 m`。黑线为占用边界，白色为可通行区域。

当前只处理地图最上方通道：

```text
右上宽路 START
    → 从右向左行驶并进行红白圆片避障
    → x = 2.60 m 窄道交接点
    → 关闭圆片绕障，沿固定窄道中心线
    → FINISH
```

| 点 | 地图坐标 (m) | 含义 |
|---|---:|---|
| START | `(4.15, 3.49)` | 右上宽路起点，车头朝左 |
| 对齐开始 | `x=3.50` | 在宽路内提前向窄道中心平滑靠拢 |
| 对齐完成 | `x=2.85` | 在进入窄道前恢复水平姿态 |
| 交接点 | `(2.60, 3.66)` | 此处及之后关闭圆片绕障 |
| FINISH | `(1.00, 3.66)` | 窄道终点参考位置 |

地图坐标采用 ROS 约定：右为 `+x`，上为 `+y`。参考路线从右向左，
姿态约为 `π rad`。

## 车辆净空

当前车 `0.33 m × 0.30 m`，迁移车 `0.34 m × 0.33 m`。离线地图检查
对车身四周额外增加 `0.03 m`，两台车的完整矩形足迹均不得与地图
黑线相交。迁移前仍必须在真实窄道复测。

## 导航接入状态

`navigation_integration.launch` 已把感知、实时定位、地图融合路径和低速
路径控制器串起来。融合节点从 TF 读取 `map -> base_link`，不再使用固定
起点；控制器同时检查感知状态、融合路径新鲜度、TF 和速度发布者冲突。
任一条件失效都会发布零速度。

控制器有两道运动门：启动时必须显式设置 `enable_motion:=true`，运行后还
必须调用 `set_enabled` 服务解锁。默认启动参数只干跑并输出到隔离话题，
不会驱动车辆。

`map_navigation.launch` 仍只启动 `map_server` 和静态参考路线发布器，适合
单独检查地图。`map_fusion_navigation.launch` 仍使用固定起点，适合无定位
的相机干跑。

## 圆片与地图融合

融合模块使用当前帧圆片和局部候选路线：

1. 按车辆地图位姿把相机 `[前向, 左向]` 坐标转换到地图；
2. 从局部候选中减去中心候选，只保留绕障横向偏移；
3. 将横向偏移施加到静态地图路线的法向，避免局部路线与地图中心线
   在拼接处跳变；
4. 再次检查完整车辆矩形足迹与地图墙线；
5. 再次检查融合路线与所有圆片安全膨胀区；
6. 到 `x<=2.60 m` 后强制使用绿色固定路线，不再生成圆片绕障。

配置文件中的 `fusion.pose_source=fixed_start` 只作为离线默认值；导航接入
launch 会覆盖成 `tf`。相机局部前向零点相对车体参考点由
`fusion.local_forward_origin_offset_m` 表示，当前为 `0.0 m`，实车最终测试
前仍需确认相机地面投影原点与 `base_link` 的纵向偏移。

```bash
source /home/mo-bit/newznzc_ws1/devel/setup.bash
rosrun yellow_track_avoidance offline_map_navigation.py \
  --route /home/mo-bit/newznzc_ws1/src/yellow_track_avoidance/config/track_7_15_route.yaml \
  --output-dir /home/mo-bit/newznzc_ws1/src/yellow_track_avoidance/navigation_runs/current

roslaunch yellow_track_avoidance map_navigation.launch

# 相机感知 + 局部规划 + 地图融合，全程零运动输出
roslaunch yellow_track_avoidance map_fusion_navigation.launch
```

## 实时导航测试

先启动底盘、相机、雷达和里程计，确保已经存在 `odom -> base_link`。如果
系统已有可靠的 `map -> odom` 定位，保持 `start_amcl:=false`；否则可以让
本 launch 用当前 5 m × 4 m 地图启动 AMCL：

```bash
source /home/mo-bit/newznzc_ws1/devel/setup.bash
roslaunch yellow_track_avoidance navigation_integration.launch \
  start_amcl:=true
```

这一步是零运动干跑。重点检查：

```bash
rosrun tf tf_echo map base_link
rostopic echo /yellow_track_avoidance/map_fusion_status
rostopic echo /yellow_track_navigation/status
rostopic echo /yellow_track_avoidance/map_fused_path
```

融合状态必须持续为 `safe: true`，控制器必须为 `ready: true`，地图上的车体
位置需与实车一致。感知状态初始为锁止 STOP；在画面和融合路径稳定、现场
急停有人值守时复位：

```bash
rosservice call /yellow_track_avoidance/reset_stop
```

### 接现有 `/move_vision -> vel_ctrl -> /cmd_vel` 链

必须先停掉 `move_base`、`xy_path_follower`、`yaw_pid_controller`、健康检查
运动测试和其他 `/cmd_vel` 发布者，只保留 `vel_ctrl`：

```bash
roslaunch yellow_track_avoidance navigation_integration.launch \
  start_map_server:=true start_amcl:=true \
  enable_motion:=true output_cmd_topic:=/move_vision

rosservice call /yellow_track_avoidance/reset_stop
rosservice call /yellow_track_navigation/set_enabled "data: true"
```

控制器速度上限为 `0.08 m/s`、`0.30 rad/s`。随时软件撤销授权：

```bash
rosservice call /yellow_track_navigation/set_enabled "data: false"
```

若不使用 `vel_ctrl` 而直接连接底盘，可把 `output_cmd_topic` 设置为
`/cmd_vel`；此时 `/cmd_vel` 除本节点外不能有任何发布者。物理急停始终优先。
