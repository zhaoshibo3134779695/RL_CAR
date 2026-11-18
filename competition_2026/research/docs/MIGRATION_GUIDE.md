# 换车迁移指南

这份流程假定目标仍是 ROS1 Noetic 全向小车。若底盘变为差速/阿克曼结构，
TEB 运动学、AMCL 里程计模型和动作原语都需要重新设计，不能只改车身尺寸。

## 1. 解压与编译

```bash
cd ~/new_robot_ws/src
unzip /path/to/yellow_track_avoidance_migration_20260731.zip
cd ..
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
roscd yellow_track_avoidance
./tools/migration_preflight.sh
```

若预检提示缺少 `orbbec_camera`、`lslidar_driver`、`mowen` 或
`car_bringup`，先迁移这些兄弟包。它们不包含在本 ZIP 中。

## 2. 先确认新车几何

当前车为长 `0.33 m`、宽 `0.30 m`。档案中预记录的目标车为长 `0.34 m`、
宽 `0.33 m`，但必须连同轮子、护板、相机架和线缆突出部分重新实测。

主实车入口已统一尺寸参数。目标车若确认是 `0.34 × 0.33 m`，启动时使用：

```bash
roslaunch yellow_track_avoidance real_navigation.launch \
  vehicle_length_m:=0.34 vehicle_width_m:=0.33
```

这两个参数会同步覆盖感知规划、全局/局部代价地图 footprint、TEB、圆片记忆、
导航安全门和通道监督器。不要再分别手改六套半长/半宽。

`config/vehicle_profiles.yaml` 用于离线路线对所有已记录车辆做碰撞验证；它不是
实车主 launch 的替代品。确认尺寸后再更新其中的 `migration_vehicle` 记录。

## 3. 相机与安装位姿必须重新验收

只有相机型号、分辨率、镜头、焦距、安装高度、俯仰角、横向位置全部未变时，
旧 `config/calibrated.yaml` 才可能复用。只要相机或支架移动，就至少需要：

1. 检查/重做 RGB 内参；
2. 重做地面单应性与 BEV；
3. 测量 `base_link` 到最近可见 BEV 地面的前向偏移；
4. 在新光照下复核黄色线和红白圆片；
5. 保持配置中的 `control.enable_motion: false` 完成离线和在线零运动验收。

当前偏移 `0.39 m` 是旧车旧支架的测量值。新值通过
`local_forward_origin_offset_m:=...` 传入主 launch，并会同步给感知、圆片桥和
安全门。详细标定步骤见
[`calibration/BEV_CALIBRATION.md`](calibration/BEV_CALIBRATION.md)。

## 4. 底盘、里程计与 TF

在接入导航前单独验收：

- 正 `linear.x` 是否前进，正 `linear.y` 是否向左，正 `angular.z` 是否逆时针；
- `/odom` 的米制比例、角度方向和静止漂移；
- `map -> odom -> base_link -> laser_link` 是否连续且时间戳新鲜；
- 激光正前方是否对应 `base_link +x`；
- `/cmd_vel` 是否只有安全门一个上游发布者，底盘桥是否有失联看门狗。

如果新车不是全向底盘，停止迁移实车控制；当前配置允许横移，不能直接用于
差速或阿克曼底盘。

## 5. 分阶段启动

先进行不接底盘的检测：

```bash
roslaunch yellow_track_avoidance detect_only.launch \
  config:=/path/to/new_vehicle_calibrated.yaml
```

再进行主链零运动联调：

```bash
roslaunch yellow_track_avoidance click_navigation.launch \
  config:=/path/to/new_vehicle_calibrated.yaml \
  vehicle_length_m:=0.34 vehicle_width_m:=0.33 \
  start_chassis:=false enable_motion:=false start_enabled:=false
```

确认相机、激光、TF、AMCL、地图、代价地图和规划均正确后，才允许连接底盘。
所有实车入口默认 `start_enabled=false`。

## 6. 定位行为

主 launch 默认执行静态场地匹配：读取当前激光和右侧黄线，发布
`/initialpose`，不会旋转或平移。状态：

```bash
rostopic echo /yellow_track_field_localization/status
```

兼容入口 `/yellow_track_navigation_gate/start_auto_localization` 现在也只调用
同一套静止的“相机黄线＋激光地图”融合定位。旋转式 AMCL 全局定位链路已删除；
该服务不会解锁车辆，也不会向 `/cmd_vel` 发布定位速度。

## 7. 解锁前验收

按 [VALIDATION_CHECKLIST.md](VALIDATION_CHECKLIST.md) 完成全部项目。在有人控制
物理急停、场地清空、定位成功、传感器新鲜且 `/cmd_vel` 为零时才调用：

```bash
rosservice call /yellow_track_navigation_gate/set_enabled "data: true"
```

先架空车轮验证，再做低速直线/横移/旋转，最后测试地图导航和圆片场景。
