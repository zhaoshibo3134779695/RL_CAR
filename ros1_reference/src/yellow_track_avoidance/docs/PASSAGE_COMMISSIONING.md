# 右上角黄线圆片避障：分阶段验收

比赛规则中两个红白圆片的随机摆放示例见
[`assets/competition_disc_placement_reference.jpg`](assets/competition_disc_placement_reference.jpg)。
该截图用于理解赛题，不提供可直接使用的尺度或坐标。

## 已实现的控制边界

当前车入口使用实测 `0.34 m × 0.33 m` 矩形包络。只有 727 地图右上方
多边形 `[(0.00,1.20), (2.35,1.20), (2.35,1.95), (0.00,1.95)]`
启用视觉走廊安全层；区域外继续使用静态地图、AMCL、SBPL 和 TEB。
2026-08-07 的一次摆放中，实测固定起始位的车头到左圆片近侧边缘为
`0.640 m`。加上圆片半径 `0.050 m` 和车体中心到车头的 `0.170 m` 后，
该次摆放在控制坐标中的参考前向距离为 `0.860 m`。实时确认轨迹给出的值约为
`0.8595 m`，与参考相差约 `0.0005 m`，因此继续保留
`local_forward_origin_offset_m=0.5463`，不按单次尺量噪声微调。

上述 `0.640 m` 只用于本次摆放的传感器前向偏移校验，不是圆片固定位置，也不
参与比赛目标点计算。正式比赛中圆片位置随机，仍完全使用实时检测坐标。窄门
接管距离 `1.05 m` 是提前进入低速控制的范围，不等同于圆片位置。

区域内的两类速度都必须通过相同的硬检查：

- 黄线边界与圆片输入新鲜；
- 当前场地只有一侧实体黄线时，以标定的 `0.88 + 0.03 = 0.91 m`
  中心距在米制坐标中重建另一侧；实体黄线本身丢失仍然停车；
- 圆片已进入局部代价地图；
- 对未来 `0.35 s` 的速度逐段采样，检查完整车体矩形；
- 车体到黄线实体内边缘至少保留 `0.010 m`；
- 车体不能与任何新鲜、已确认圆片相交；
- 输入过期、TF 丢失或检查失败时发布零速度。

## 阶段 1：离线与静态检查

```bash
cd /home/mowen/Sebastian
source /opt/ros/noetic/setup.bash
catkin_make --pkg yellow_track_avoidance
PYTHONPATH=/home/mowen/Sebastian/src/yellow_track_avoidance/src \
  python3 -m unittest discover \
  -s src/yellow_track_avoidance/test -p 'test_*.py'
```

## 阶段 2：零运动联调

保持 `start_chassis:=false enable_motion:=false start_enabled:=false`。启动后在
RViz 中同时检查黄线、圆片、代价地图和
`/yellow_track_avoidance/passage/markers`。激活区轮廓为绿色表示车辆在区内，
橙色表示区外，灰色表示没有可用地图位姿。

```bash
roslaunch yellow_track_avoidance current_car_navigation.launch \
  start_camera:=true start_lidar:=true start_odometry:=true \
  start_chassis:=false start_rviz:=true start_perception_views:=false \
  enable_motion:=false start_enabled:=false auto_field_localization:=false
```

关键状态：

```bash
rostopic echo -n 1 /yellow_track_avoidance/passage/status
rostopic echo -n 1 /yellow_track_navigation_gate/status
rostopic info /cmd_vel
```

零运动阶段必须看到 `motion_capable: false`、`allowed: false`，并确认
`/cmd_vel` 没有底盘订阅者。

## 阶段 3：录包复现

默认不录图像以降低 CPU 与磁盘带宽；需要复查视觉原帧时才加
`--with-images`。

```bash
"$(rospack find yellow_track_avoidance)/tools/record_passage_validation.sh" \
  /tmp/passage_trial
# 或："$(rospack find yellow_track_avoidance)/tools/record_passage_validation.sh" \
#       --with-images /tmp/passage_trial_images
```

回放时不要启动真实底盘：

```bash
rosbag play --clock /tmp/passage_trial/passage_validation.bag
```

## 阶段 4：现场低速验收

这一步只能在车辆架空或空旷场地、人员持有急停、相机和雷达状态稳定后执行。
先保持 `start_enabled:=false` 启动底盘，再通过服务显式解锁。依次验证：无圆片、
单圆片、双圆片、黄线短暂丢失、圆片输入中断。任一输入中断必须立即输出零速度。

```bash
rosservice call /yellow_track_navigation_gate/set_enabled "data: true"
# 随时停止：
rosservice call /yellow_track_navigation_gate/set_enabled "data: false"
```

现场运行时关闭 RViz 和两个图像窗口。当前机器上的视觉节点已经接近占满一个 CPU
核心，而 move_base 和通道监督器负载明显更低；算法选择不是当前主要算力瓶颈。
