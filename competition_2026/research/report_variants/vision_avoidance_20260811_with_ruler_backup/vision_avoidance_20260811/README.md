# 2.3.4 视觉避障技术报告图片包

生成日期：2026-08-11  
项目：`/home/mowen/Sebastian/src/yellow_track_avoidance`

本目录包含《2026 睿抗机器人开发者大赛（RAICOM）智能侦查技术报告》2.3.4“视觉避障”可用的 10 张正式图片。所有检测结果、轨迹和统计数字均来自当前源码、当日静止实景采集、既有实车 rosbag 或已有验收报告，没有补画不存在的检测结果或实验数据。

当日重新采集时只启动了 RGB 相机和视觉检测节点，`enable_motion=false`、`enable_planning=false`，没有启动底盘运动，也没有发布正式 `/cmd_vel`。采集结束后相关 ROS 节点已经关闭。

## 图片清单

| 文件 | 推荐放置位置 | 建议图题 | 该图证明什么 | 真实数据来源 |
|---|---|---|---|---|
| `fig01_visual_avoidance_architecture.png` | 2.3.4.1 视觉避障总体架构 | 视觉避障系统总体架构 | 系统以 RGB 图像为输入，经过黄线/圆片检测、米制 BEV、odom 世界建模后，为 costmap、GlobalPlanner 和 TEB 提供环境约束，最终驱动麦克纳姆底盘；同时展示 ACQUIRE→LOCKED 的资源调度关系。 | 按当前正式源码和 competition 链路绘制的结构图；不是实验截图。 |
| `fig02_yellow_line_detection.png` | 2.3.4.2 黄色道路边界检测 | 黄色道路边界检测过程 | 同一真实帧内，从原始 RGB、连续黄色置信度、ROI 内 28 条法向截面，到 DP 支持点和加权拟合线的完整过程；该帧有效截面 26/28、质量 0.933、拟合 RMS 0.89 px。 | 2026-08-11 当日静止实景同步采集，并用当前检测源码重新计算。 |
| `fig03_disc_detection_geometry.png` | 2.3.4.3 红白反坦克雷检测 | 红白圆片的几何与颜色联合检测 | 真实圆片候选轮廓、椭圆拟合、20 扇区红环覆盖、白色中心和最终中心位置；所选帧两枚圆片置信度分别约 0.918、0.963。 | 2026-08-11 当日静止实景同步采集，并用当前 `DiskDetector` 重新计算。 |
| `fig04_perspective_to_metric_bev.png` | 2.3.4.4 相机标定与米制 BEV | 原始透视图到米制鸟瞰图的转换 | 图像中的圆片和黄线能够通过地面投影变换为以米为单位的车体坐标。本次摆放为错列场景，两圆片中心坐标约为 `(1.077, 0.308) m`、`(0.556, -0.118) m`，中心距约 0.673 m。 | 2026-08-11 当日同步 RGB、BEV 和检测 JSON。 |
| `fig05_scale_validation_500mm.png` | 2.3.4.4 相机标定与米制 BEV / 实验验证 | 米制 BEV 的 500 mm 尺度验证 | 对实际 500.000 mm 圆心距进行 60 帧测量，均值 501.828 mm、标准差 1.010 mm，测量范围 499.933～503.852 mm，说明重新标定后的尺度恢复能力。 | `calibration_runs/current_car_640x480_20260809_repair/calibration_report.json`。 |
| `fig06_navigation_world_model.png` | 2.3.4.5 世界模型与导航融合 | 视觉障碍物在导航世界模型中的表达 | 左图展示两条边界、两枚圆片、全局 costmap、GlobalPlanner 路径和实车 odom 轨迹；右图展示局部 costmap、机器人矩形 footprint 与 TEB 局部轨迹。 | 成功实车 `stage1_global_teb_compact_standard_ekf_20260810_trial7.bag` 离线重建。 |
| `fig07_stage1_s_curve_five_moments.png` | 2.3.4.5 世界模型与导航融合 / 实车验证 | 双圆片区域 S 形避障的五个关键时刻 | 用五个真实 rosbag 时刻展示进入、第一段横移、两圆片之间、第二段横移和离开，说明全局路径、TEB 局部轨迹和实车轨迹在闭环运动中的变化。 | 成功实车 Trial 7 的真实消息离线重建；不是相机录像。 |
| `fig08_acquire_locked_cpu_ab.png` | 2.3.4.6 ACQUIRE—LOCKED 计算优化 | ACQUIRE 与 LOCKED 状态下视觉 CPU 占用对比 | 视觉 CPU 均值从 283.4% 降至 8.87%，下降约 96.9%；图中同时标出 p95 和最大值，说明世界模型锁存后跳过重复重型感知的资源收益。 | `docs/navigation/PERCEPTION_ACQUIRE_LOCKED_REPORT_20260810.md`。 |
| `fig09_disc_520frame_acceptance.png` | 2.3.4.7 实验结果与性能分析 | 红白圆片检测 520 帧专项验收结果 | 有圆片正样本 184 帧中，183 帧同时检出两枚圆片；无圆片负样本 336 帧中误检 0 帧。该结果只代表报告所述专项场景。 | `docs/perception/DISC_DETECTION_ACCEPTANCE.md`。 |
| `fig10_visual_navigation_closed_loop.png` | 2.3.4.1 章节开头或 2.3.4.5 | 从视觉感知到导航规划的完整闭环 | 用真实检测画面、真实米制 BEV 和真实实车 bag 轨迹串联“感知→建模→规划”，适合作为本节总览图。 | 左/中为 2026-08-11 实景；右为成功实车 Trial 7。 |

## 素材可用性与替代说明

- 10 张计划图片均已生成。
- 图 2、图 3、图 4 使用了当日重新录制的同一组真实同步素材，因此它们可以彼此对应。
- 图 6、图 7 使用成功 Stage 1 Trial 7 的真实 rosbag 离线重建，显示的是 bag 中实际记录的 costmap、障碍物、全局/局部路径、footprint 和 odom，而不是人为绘制的假轨迹。
- 现有 Stage 1 rosbag 没有保存同步相机视频，因此图 7 不能制作成“五张实车相机画面”。本包采用五个真实 RViz 数据时刻替代，没有合成相机画面。
- ACQUIRE→LOCKED 现有报告保存了 mean、p95、max 和跳过帧数，但没有原始逐时刻 CPU 序列，因此图 8采用真实汇总值柱状图，没有伪造切换时间曲线。
- 520 帧验收报告保留了汇总统计，但当前未找到完整逐帧 CSV/图像序列，因此图 9只表达报告中的真实汇总数据，没有生成虚假的逐帧曲线。
- 图 4 中 0.673 m 是 2026-08-11 当前错列摆放的实测中心距；图 5 的 0.500 m 是 2026-08-09 独立尺度标定实验。两者是不同实验，正文中不可混写。

## 推荐选图方案

若报告篇幅充足，建议十张图全部保留。若后续需要压缩篇幅，可优先保留：

1. `fig10_visual_navigation_closed_loop.png`：章节总览；
2. `fig02_yellow_line_detection.png`：黄线算法；
3. `fig03_disc_detection_geometry.png`：圆片算法；
4. `fig04_perspective_to_metric_bev.png`：米制建模；
5. `fig06_navigation_world_model.png`：导航融合；
6. `fig08_acquire_locked_cpu_ab.png`：工程优化；
7. `fig09_disc_520frame_acceptance.png`：检测验收。

图 1 与图 10 的功能有一定重合：图 1 更适合解释软件结构，图 10 更适合展示真实效果。图 5 与图 9适合放在实验验证部分。图 7横向较长，适合跨页宽排版。

## 数据和复核文件

- `contact_sheet.png`：全部图片缩略预览。
- `source_manifest.json`：数据文件路径、哈希、当前帧测量值及真实性约束。
- `sources/`：当日选定帧的原始图、各调试图和同步 JSON。
- `generate_report_figures.py`：报告图片生成脚本，只读取数据并绘图，不启动 ROS、不发布运动指令。

正式插入 Word/PDF 时，建议直接使用 PNG 原图，不要从 `contact_sheet.png` 裁剪。图片标题应使用 Word 的统一“题注”功能添加，本包没有把完整图题画入图片内部。
