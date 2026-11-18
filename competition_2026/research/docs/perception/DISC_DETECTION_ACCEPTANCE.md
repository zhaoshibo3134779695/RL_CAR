# 红白圆片识别与验收说明

## 当前状态

圆片模块已经独立完成以下能力：

```text
当前帧红色候选
  → 红环与白心结构检查
  → 透视椭圆拟合
  → 红环扇区覆盖和缺口检查
  → 红白中心对齐检查
  → 白心亮度、色度和红心排斥检查
  → 双黄线可行驶区域检查
  → 当前帧输出与逐候选拒绝原因
```

黄线核心算法没有修改。`bev.calibrated` 和 `control.enable_motion` 仍为 `false`，圆片检测不使用历史补检、历史显示或卡尔曼预测。

## 检测约束

一个候选必须同时满足：

1. 红色外轮廓面积和形状合理；
2. 可拟合为透视椭圆，长短轴和轴比合理；
3. 椭圆外环中有足够红色像素；
4. 20 个环形扇区中有足够连续红色覆盖；
5. 中心区域为低饱和、Lab 接近中性的白色；
6. 白心亮度高于红环；
7. 中心区域不能被红色填满；
8. 红环像素中心与白心证据中心必须接近；
9. 圆片中心必须位于双黄线围成的赛道区域内，允许靠黄线方向有 20 px 余量；
10. 与更高置信度候选重复时只保留一个。

检测对象使用 `cv2.findContours` 和 `cv2.fitEllipse`，没有强制使用霍夫圆，因此允许远端圆片因透视变成细长椭圆。

## 当前帧规则

每一帧均从零开始生成候选：

```text
当前帧证据不足 → disc_detected=false
```

圆片输出中的 `used_history` 固定为 `false`。跟踪器不会参与当前未标定模式下的圆片验收，也不会拿上一帧圆片代替当前测量。

## ROS 调试输出

```text
/yellow_track_avoidance/debug/disc_overlay
/yellow_track_avoidance/debug/disc_red_mask
/yellow_track_avoidance/debug/disc_white_mask
/yellow_track_avoidance/debug/disc_candidates
/yellow_track_avoidance/debug/disc_status
```

`disc_status` 每帧包含：

```text
timestamp / frame_id
left_line_detected / right_line_detected
disc_detected / disc_count
圆心、椭圆长短轴、角度、轮廓面积、置信度
红色面积、白色面积、红白中心距离
候选数量、拒绝数量和拒绝原因
是否位于赛道内、到左右黄线的像素距离
processing_time_ms
used_history / motion_enabled
```

## 启动验收模块

该命令同时启动当前帧检测和验收记录器：

```bash
source /opt/ros/noetic/setup.bash
source /home/mo-bit/newznzc_ws1/devel/setup.bash
roslaunch yellow_track_avoidance disc_acceptance.launch
```

默认输出目录：

```text
/home/mo-bit/newznzc_ws1/src/yellow_track_avoidance/disc_acceptance_runs/current
```

如果检测节点已经单独运行，只启动记录器：

```bash
roslaunch yellow_track_avoidance disc_acceptance.launch start_detector:=false
```

## 一次场景采集

开始一个单圆片静态场景：

```bash
rosrun yellow_track_avoidance disc_capture_control.py start \
  --scene-name P1_left_near \
  --disc-present true \
  --expected-disc-count 1 \
  --position left \
  --distance near \
  --occlusion 0 \
  --reflection normal \
  --near-yellow-line false
```

采满后停止并自动生成报告：

```bash
rosrun yellow_track_avoidance disc_capture_control.py stop
```

无圆片场景：

```bash
rosrun yellow_track_avoidance disc_capture_control.py start \
  --scene-name no_disc \
  --disc-present false \
  --expected-disc-count 0
```

重新生成报告：

```bash
rosrun yellow_track_avoidance disc_capture_control.py report
```

## 输出文件

```text
disc_frames.csv
disc_summary.json
```

CSV 保存逐帧原始检测值、场景标签、全部圆片 JSON 和候选拒绝信息。JSON 按总体和每个场景分别统计：

- 召回率和误检率；
- 最长连续漏检；
- 每个静止圆片的圆心标准差、P95 跳变和最大跳变；
- 椭圆长短轴标准差；
- 置信度分位数和直方图；
- 处理耗时 P50、P95 和换算帧率；
- 使用历史的帧数；
- 运动开启的帧数；
- 每项验收条件以及最终通过状态。

多圆片静止场景按每帧圆心横坐标从左到右对应，只用于离线统计，不影响检测输出，也不构成历史补检。

## 当前现场结果

当前现场先采集两个静止圆片，再将圆片全部拿走采集无圆片负样本。最终结果：

| 指标 | 结果 |
|---|---:|
| 采集帧数 | 184 |
| 双圆片同时检出 | 183/184 |
| 双圆片召回率 | 99.46% |
| 最长连续漏检 | 1 帧 |
| 左圆片中心标准差 X/Y | 0.60 / 0.32 px |
| 右圆片中心标准差 X/Y | 0.41 / 0.20 px |
| 左圆片最大单帧跳变 | 3.77 px |
| 右圆片最大单帧跳变 | 1.62 px |
| 置信度中位数 | 0.973 |
| 处理耗时 P50/P95 | 87.6 / 98.1 ms |
| 实际 ROS 输出 | 约 10.05 Hz |
| 黄线双线检出率 | 100% |
| 使用历史帧数 | 0 |
| 运动开启帧数 | 0 |
| 无圆片负样本 | 336 帧 |
| 无圆片误检 | 0/336 |
| 无圆片误检率 | 0% |
| 合并总帧数 | 520 |
| 总体验收状态 | 通过 |

双圆片场景已通过召回、连续漏检、中心稳定和处理速度门槛；无圆片场景连续 336 帧没有误检。合并报告中的召回率为 99.46%、误检率为 0%，`used_history` 和 `motion_enabled` 全程为 `false`，最终 `acceptance.passed=true`。

这说明当前画面中圆片消失后，模块会在当前帧立即报告未检出，不会继续显示或输出上一帧圆片。该结果完成了进入相机去畸变和米制 BEV 标定前的圆片模块验收门槛。

## 性能优化

圆片模块曾因对白色地面产生的每个白心假候选执行完整多尺度椭圆计算，实时频率下降到约 7.8 Hz。当前增加了两个不影响真实红环的前置条件：

1. 白心候选附近必须先存在最低数量的红色像素；
2. 已经被有效红环候选覆盖的白心候选直接判为重复。

优化后保存帧完整管线约 12 FPS，实际相机与 ROS 调试输出恢复到约 10 Hz。

## 主要代码

| 功能 | 文件 |
|---|---|
| 圆片候选和结构约束 | `src/yellow_track_avoidance/vision.py` |
| 赛道内检查和调试图 | `src/yellow_track_avoidance/pipeline.py` |
| ROS 圆片状态和调试话题 | `src/yellow_track_avoidance/ros_node.py` |
| 验收记录器 | `scripts/disc_acceptance_recorder.py` |
| 采集控制命令 | `scripts/disc_capture_control.py` |
| 验收启动文件 | `launch/disc_acceptance.launch` |
| 阈值 | `config/calibrated.yaml` |

圆片专项完成后，下一阶段才允许设置地面单应性和米制 BEV；当前模块不输出或使用未经标定的距离值。
