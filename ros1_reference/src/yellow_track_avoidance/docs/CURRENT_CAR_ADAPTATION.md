# 当前车辆适配记录

## 已对齐的软件接口

- 主机：Ubuntu 20.04、ROS1 Noetic、`x86_64`；
- 底盘：全向，`car_bringup/newt.py` 订阅 `/cmd_vel`，串口为
  `/dev/carserial`，看门狗超时为 0.35 s；
- 里程计：`linear_scale_x=1.071`、`linear_scale_y=1.0`、
  `angular_scale=2.22`；
- TF：`map -> odom -> base_footprint -> base_link -> laser_link`；
- 激光：N10_P，话题 `/scan`，设备 `/dev/lslidar`；
- 摄像头：`usb_cam`，640×480@30，MJPEG，图像
  `/usb_cam/image_raw`，相机信息 `/usb_cam/camera_info`；
- 当前比赛宽道测试组合为 GlobalPlanner 全局规划、全向 DWA 局部规划和通道
  监督器。全场通用入口仍可使用原 SBPL+TEB 与 AMCL；当前车入口采用下述白图
  `odom` 模式，避免墙体地图偏移污染黄线和圆片的相对几何。

包内 `libsbpl.so` 已从原包锁定的 SBPL 1.3.1 commit 以 `-fsigned-char` 在
`x86_64` 上重新构建；只对本包启动的 `move_base` 生效，不替换系统库。
`current_car_navigation.launch` 默认使用 `global_planner/GlobalPlanner`、
`dwa_local_planner/DWAPlannerROS` 和 `/move_base/GlobalPlanner/plan`，并保持通道
监督逻辑。系统必须安装对应的 ROS 插件；原通用入口仍保留 SBPL+TEB：

```bash
sudo apt install ros-noetic-global-planner ros-noetic-dwa-local-planner \
  ros-noetic-sbpl-lattice-planner ros-noetic-teb-local-planner
```

## 比赛宽道白图模式

`current_car_navigation.launch` 默认使用
`maps/competition_blank_5x4.yaml`，不启动 AMCL，而由静态零变换保持
`map -> odom`。白图只是规划器所需的自由空间画布，不含旧场地墙线。

`competition_corridor_map.py` 连续确认 5 帧实时左黄线后：

1. 在 `odom` 中锁定左黄线直线；
2. 以 0.34 m 车长的前缘（`base_link` 前方 0.17 m）为起点，只延长 1.85 m；
3. 按黄色中心线间距 0.91 m 推算右黄线；
4. 将两条 0.03 m 宽线发布到
   `/yellow_track_avoidance/corridor_cloud`，供 GlobalPlanner/DWA 代价地图标记；
5. 发布 `/yellow_track_avoidance/corridor_markers` 和
   `/yellow_track_avoidance/corridor_centerline` 供 RViz 直接核对。

该点云只标记、不清除，黄线锁定后不会因逐帧抖动留下多条边界，也不会通过
清除射线擦掉真实激光障碍。比赛双圆片状态机继续使用 `odom` 中的实时圆片和
黄线中心线，且保持原控制权互斥与软件速度门控。

## 安全状态与待现场测量

`config/current_car.yaml` 明确保持以下锁定状态：

- USB 相机内参未标定；
- 地面单应性/BEV 未标定；
- `local_forward_origin_offset_m=0.0`，等待实测；
- `control.enable_motion=false`；
- 车辆包络暂按迁移说明预记录的 `0.34 m × 0.33 m`，启动车体前必须连同轮子、
  支架和线缆突出物复测。

因此当前配置只用于检测和零运动联调，不能作为实车解锁配置。此次适配时硬件
未连接、ROS master 未运行，尚未声称相机、激光、TF 或视觉效果已完成在线验收。

## 不编译工作空间的零运动检查

为遵守“只改本包”的边界，可先复用现有 `bit_ws` 下层工作空间，不运行
`catkin_make`：

```bash
source /home/mowen/bit_ws/devel/setup.bash
export ROS_PACKAGE_PATH=/home/mowen/Sebastian/src:$ROS_PACKAGE_PATH
export PYTHONPATH=/home/mowen/Sebastian/src/yellow_track_avoidance/src:$PYTHONPATH
cd /home/mowen/Sebastian/src/yellow_track_avoidance
./tools/migration_preflight.sh --profile current_car
```

预检仍会检查包内兼容入口的 SBPL 运行库；当前测试还必须能找到
`global_planner` 与 `dwa_local_planner`，缺少所选插件时不要启动主导航。

单独启动 USB 相机和检测（算法只能发布零速度）：

```bash
roslaunch yellow_track_avoidance detect_only.launch start_camera:=true
```

主链零运动联调；底盘桥不启动，两个运动锁都保持关闭：

```bash
roslaunch yellow_track_avoidance current_car_navigation.launch \
  start_camera:=true start_lidar:=true start_odometry:=true \
  start_chassis:=false \
  enable_motion:=false start_enabled:=false
```

只有完成 `VALIDATION_CHECKLIST.md` 中的现场测量、相机标定、在线视觉、架空轮和
急停验收后，才可以另存一份已标定配置并进入低速实车阶段。
