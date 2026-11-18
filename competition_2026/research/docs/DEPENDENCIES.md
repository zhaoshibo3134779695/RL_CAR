# 依赖说明

## 基础环境

- Ubuntu 20.04；
- ROS1 Noetic；
- Catkin 工作空间；
- Python 3、NumPy、OpenCV、PyYAML；
- 随包 `libsbpl.so` 必须与运行主机架构一致；当前迁移目标为 `x86_64`。

可由 Ubuntu/ROS 软件源安装的主要组件：

```bash
sudo apt install \
  ros-noetic-amcl ros-noetic-cv-bridge ros-noetic-map-server \
  ros-noetic-move-base ros-noetic-robot-localization \
  ros-noetic-robot-state-publisher ros-noetic-rqt-image-view \
  ros-noetic-rviz ros-noetic-sbpl-lattice-planner \
  ros-noetic-teb-local-planner python3-numpy python3-opencv python3-yaml
```

具体缺项以 `rosdep check yellow_track_avoidance` 和
`tools/migration_preflight.sh` 的输出为准。

## 不在本 ZIP 内的工作空间包

默认实车入口会调用以下外部包，迁移时必须从原工作空间一起复制，或在新车上
提供接口兼容的替代包：

| 包 | 用途 | 当前接口 |
|---|---|---|
| `orbbec_camera` | DaBai DCW2 RGB 驱动 | `/camera/color/image_raw`、相机服务 |
| `lslidar_driver` | N10_P 激光雷达 | `/scan`、`laser_link` |
| `mowen` | 机器人模型和静态 TF | `base_link`、`base_footprint`、`laser_link` |
| `car_bringup` | 底盘反馈、里程计、串口速度桥 | `/odom_raw`、`/cmd_vel`、`/dev/carserial` |
| `wit_ros_imu` | 可选 IMU | `/wit/imu`；默认不开启 |

仅复制本 ZIP 而不复制这些包，检测和离线测试仍可使用，但
`real_navigation*.launch` 不能完整启动。

## 硬件/系统约定

- 激光设备别名：`/dev/lslidar`；
- 底盘串口别名：`/dev/carserial`；
- 默认相机：Orbbec DaBai DCW2，640×480@30；
- 默认底盘模型：全向运动，AMCL 使用 `odom_model_type=omni`；
- TF 主链：`map -> odom -> base_footprint/base_link -> laser_link/camera`。

新车上的 udev 规则、串口波特率、轮向、里程计比例、TF 和话题名必须重新核对，
不能因为设备节点恰好同名就认为接口正确。

## SBPL 平台注意事项

SBPL 1.3.1 用普通 `char` 保存 XYTheta 动作偏移；为在各架构上保持完全一致的
有符号语义，包内运行库统一以 `-fsigned-char` 编译，并仅对本包的 `move_base`
设置 `LD_LIBRARY_PATH`。

迁移到任何不同架构后运行：

```bash
./tools/build_sbpl_runtime.sh
```

该脚本会按已锁定的 SBPL 1.3.1 commit 在当前主机重新编译；随后重新运行预检
确认 ELF 架构和动态依赖。历史入口 `build_arm64_sbpl.sh` 仍保留并转调该脚本。
