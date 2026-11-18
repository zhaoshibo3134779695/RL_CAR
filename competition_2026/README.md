# RAICOM 2026 · 睿抗智能侦查小车

实车采用麦克纳姆轮全向底盘。本目录是 Ubuntu 上用于论文、数据分析和离线复现的 ROS 2 Lyrical 工作空间。导航地图、试车录像、黄线与圆片避障算法和相机标定取自车载包。三维显示采用用户选定的公开 Mowen 参考模型。水弹机构和比赛动作编排已排除。当前有两个可构建包，分别是车辆描述 `mowen` 和研究工具 `raicom_research_assets`。

## 打开地图和小车

在 Ubuntu 终端运行以下命令。在仓库根目录按 README 完成模型准备与构建后执行。

```bash
cd /path/to/raicom-2026-robot-assets/competition_2026
source /opt/ros/lyrical/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=77
ros2 launch raicom_research_assets research.launch.py
```

RViz 默认显示车模、`727best_map`、灰色墙面、坐标系、导航轮廓，以及车载启动文件中的起点箭头。默认比赛地图将车放在左上角起点通道内，车头朝图下方。这个展示位姿按用户指定的位置设置。换图时在启动命令末尾添加 `map:="$PWD/src/raicom_research_assets/maps/track/track_7_15_5x4.yaml"`。重新构建可运行 `colcon build --symlink-install`。 [车模预览](research/ros2_examples/mowen_reference.png) · [模型来源与尺寸](src/mowen/SOURCE.md) · [地图目录](catalog/maps.tsv)

在同一终端环境中运行 `ros2 launch raicom_research_assets gazebo.launch.py` 可同时打开 Gazebo 和 RViz。如果 RViz 已在运行，添加 `with_research_view:=false` 仅打开 Gazebo。Gazebo 直接读取车载地图的黑色占用格生成三维墙面与碰撞网格。每格为 0.005 m，墙的位置、厚度和通道宽度取自原图。原车资料未给出实测墙高，当前统一按 0.20 m 展示。Gazebo 的墙面为中灰色。两边的小车均从地图左上角 (−2.21 m, 1.72 m) 展示，偏航角 −90°，即车头沿地图 −Y 方向朝下。这一展示模式下车辆保持静止。 [手绘底图](src/raicom_research_assets/maps/vehicle/727best_source.png) · [车载导航原图](src/raicom_research_assets/maps/vehicle/727best_map.png)

## 麦克纳姆轮底盘

车载底盘接口和里程计保留车体系前进速度 `vx`、横移速度 `vy` 和偏航角速度 `wz`。ROS 2 AMCL 使用 `nav2_amcl::OmniMotionModel`。按车载启动覆盖顺序，第四区后台 TEB 的横向上限 `max_vel_y` 为 0.10 m/s，横向加速度上限为 0.20 m/s²，非完整运动学约束权重为 0。前三区和第四区的两个直控脚本在输出时均明确写了 `command.linear.y = 0.0`，这个局部策略与底盘的横移能力可以同时成立。

8 月 10 日 Trial 7 的 210 条速度指令中，195 条的 `|vy| > 0.0001 m/s`，指令范围为 −0.08 至 +0.08 m/s。同期融合里程计上报的横向速度范围为 −0.0864 至 +0.0646 m/s。这些是一次试车记录，不能代替比赛日的数据。车身外观使用公开单体网格。历史回放中的位姿来自实录，独立的平面运动仿真使用 Gazebo 的反馈位姿。四轮独立转速与滚子接触动力学尚未建模。 [原车避障试验总结](research/docs/黄线赛道视觉避障与规划算法实验总结_20260825.md) · [原始话题索引](catalog/recording_topics.tsv)

## 原车导航和速度参数

`815final.launch` 调用的两个直控入口有各自的限速。下表按脚本默认值、启动文件覆盖值、路点局部覆盖值核对。数值是配置上限，论文中描述实测车速请使用里程计记录。默认展示模式保持静止，研究节点提供参数查询。平面仿真中的跟踪器沿规划路径使用前三区的通用速度与转向上限。

| 启动范围 | 线速度上限 | 角速度上限 | 调参细节 |
| --- | --- | --- | --- |
| 前三区，`725first51_direct_all_tasks.launch` | 0.40 m/s | 0.45 rad/s | 30 Hz，线加速度上限 0.45 m/s²，进入目标前 0.35 m 减速。最终入口关闭 `move_base`。 |
| 前三区局部覆盖，`7.25_first51.yaml` | 第 10、13 项圆弧设为 0.23 m/s | 第 18 项设为 0.15 rad/s | 第 18 项同时把前视距离改为 0.30 m、航向增益改为 1.0。项号按原始数组从 1 起算。 |
| 第四区，`stage1_direct_test.launch` | 常规 0.324 m/s，末段 0.42 m/s | 0.18 rad/s | 30 Hz，线加速度上限 0.18 m/s²，角加速度上限 0.35 rad/s²。第四区的视觉配置和后台 TEB 各有自己的参数。 |

[ROS 2 导航参数](src/raicom_research_assets/config/competition/navigation.yaml)保存两套直控和路点覆盖值。[原包参数索引](catalog/competition_parameters.tsv)列出从两条入口解析的 本次归档的导航参数与调参记录，每行含比赛阶段、原始键名、值和 ZIP 成员路径。前三区的三项直控启动覆盖来自 `725first51_direct_all_tasks.launch`，由 `725first51_all_tasks.launch` 传入节点。原车激光、IMU 驱动和模型的部分被引用启动文件未恢复有效正文，设备内部默认值未经核实。11 月的 [DWA 试验参数](research/navigation/nav_demo/navigation_dwa_test.yaml)独立保留。

两阶段的轮速里程计均使用纵向系数 1.071、横向系数 1.0、角速度系数 2.22。前三区先按旋转方向补偿轮速反馈，再用 IMU 更新航向。第四区的轮速节点关掉 IMU 航向，另由 20 Hz EKF 融合轮速和 IMU。赛道展示位置来自用户指定的左上角，橙色箭头标出车载配置的初始位姿 (−2.206142，1.518535，−1.578953 rad)。RViz 还可查看 `nav05_path_dwa_scan.launch` 选用的 0.295 × 0.310 m 代价地图轮廓。第四区避障模型的车身尺寸为 0.34 × 0.33 m，公开 URDF 的质量、惯量和网格另见[模型来源](src/mowen/SOURCE.md)。

在已加载工作空间、连接到展示窗口的终端，可直接查询参数。轮速重算入口接受 `phase:=regions_1_3` 或 `phase:=region_4`，输入为对应原始轮速和 IMU 数据；前三区的旋转补偿还使用原始导航指令输入。独立输出为 `/wheel_odometry/odom`。

```bash
ros2 param get /competition_reference regions_1_3.direct_max_linear
ros2 param get /competition_reference region_4.terminal_max_linear_mps
ros2 param get /competition_reference waypoint_overrides.entry_18.direct_max_angular
ros2 launch raicom_research_assets wheel_odometry.launch.py phase:=regions_1_3
```

## RViz 图层与真实试车回放

[主窗口配置](src/raicom_research_assets/rviz/research.rviz)合并了原车导航、建图、激光检查、IMU 检查和两阶段避障的 9 份 RViz 配置。共 54 个显示项，包含原始话题、Nav2、交互规划目标与路径，以及视觉调试图。按左侧分组展开并勾选即可查看。相机调试图需有对应图像输入；试车数据中出现的圆片扫描源于视觉投影。

从本工作空间根目录运行下面的命令，可打开 [8 月 10 日 Trial 7 回放窗口](src/raicom_research_assets/rviz/trial7.rviz)。它在本机独立 ROS domain 78 中使用录制时间，显示真实局部代价地图、TEB 局部路径、里程计和走廊目标，小车位置跟随记录中的 `odom → base_footprint`。约 49 秒的记录播完后，窗口停在末帧。当前比赛地图窗口继续使用 domain 77。两个窗口的坐标和日期各自独立。

```bash
ros2 launch raicom_research_assets trial_view.launch.py
```

## 论文资产入口

| 素材 | 可以直接取用的内容 |
| --- | --- |
| [原车地图](src/raicom_research_assets/maps/vehicle)和[第四赛区地图](src/raicom_research_assets/maps/track) | 33 份 Nav2 配置，保留分辨率、原点、占用阈值和配套图像。默认图为 978 × 799 像素、0.005 m/像素，由原车导航配置选用。 |
| [8 月 19 日相机标定](research/calibration_runs/extrinsic_restore_20260819) | 原始标定帧、角点、跨分辨率验证、BEV 对照和关闭报告。整套标定目录内有 908 张可解码图像，另有 60 份 JSON。 |
| [第四赛区相机配置](src/raicom_research_assets/config/vision/region4_competition_20260819.yaml) | 8 月 20 日车载启动文件实际引用的 640 × 480 配置。保留原始参数和注释。配置内的历史绝对路径用于溯源。 |
| [避障报告及原图](research/docs/report_assets) | 赛道边界、红白圆片、投影、代价地图和调试图。8 月 25 日的总结位于 [research/docs](research/docs)，属于赛后整理材料。 |
| [ROS 2 回放](recordings)和[话题清单](catalog/recording_topics.tsv) | 12 段有有效消息的 8 月 10 日标定和试车记录。121,383 条标准消息已转成 MCAP，6,737 条旧式类型消息保留为[带时间戳的 JSONL](research/legacy_bag_messages)。[每段来源、起点、时长和恢复说明](catalog/recordings.tsv) |
| [原车避障算法](src/raicom_research_assets/yellow_track_avoidance) | ROS 无关的 Python 核心保留原始实现。ROS 2 图像接口、离线单帧分析、轮速里程计及轨迹导出位于 [scripts](src/raicom_research_assets/scripts)。 |
| [ROS 2 实际输出样例](research/ros2_examples) | 模型预览、同一真实帧生成的 BEV 和圆片叠图，以及 Trial 7 导出的 980 行里程计。里程计是车载估计值。 |

## 离线调用

以下命令沿用前面终端中的工作目录和 ROS 环境。单帧输入保留标定所需的原始分辨率。该示例检出两个圆片，单帧的跟踪确认数为零。输出的 JSON 保留了输入和所用配置路径。 [输入图](research/docs/report_assets/vision_avoidance_20260811/sources/raw_rgb.png) · [检测叠图](research/ros2_examples/track_frame/disc_overlay.png)

```bash
ros2 run raicom_research_assets analyze_frame \
  research/docs/report_assets/vision_avoidance_20260811/sources/raw_rgb.png \
  /tmp/raicom_frame
ros2 bag info recordings/stage1_global_teb_compact_standard_ekf_20260810_trial7
ros2 run raicom_research_assets export_trajectory \
  recordings/stage1_global_teb_compact_standard_ekf_20260810_trial7 \
  /tmp/raicom_trial7_odom.csv
```

轨迹默认取 `/odom`，可以用 `--topic /odom_raw` 导出原始轮速里程计。CSV 包含记录时间、消息时间、坐标系、平面位姿和速度。导出器要求新的 CSV 文件名，以保护已有分析结果。

要检查融合里程计，在两个已加载相同工作空间、使用相同 ROS domain 的终端中分别运行下面两行。实时滤波输出为 `/research/odometry/filtered`。旧数据的速度控制记录保存为 `/recorded/cmd_vel`。图像节点沿用 `/usb_cam/image_raw`，发布 `/track_perception/measurements` 和调试图。传感器外参由实际实验设置提供。

```bash
ros2 launch raicom_research_assets ekf.launch.py
ros2 bag play recordings/stage1_global_teb_compact_standard_ekf_20260810_trial7 --clock 100 --topics /odom_raw /imu0
```

单独查看地图可运行 `ros2 launch raicom_research_assets map.launch.py`。增加 `amcl:=true` 可加载全向底盘的 AMCL 配置，定位需要匹配的激光、TF 与初始位姿。轮速重算入口为 `ros2 run raicom_research_assets wheel_odometry --ros-args -p use_sim_time:=true`，输入为 `/vel_raw`。

交互式路径规划使用 `ros2 launch raicom_research_assets plan_on_map.launch.py interactive:=true`。默认同时打开比赛地图的 Gazebo 和 RViz。在 RViz 工具栏选 **2D Goal Pose**，在可通行区域按下鼠标左键，拖动设定终点朝向后松开。青色线路从画面中的车位开始，再点一个目标就会替换上一条路线。规划失败会清掉旧路线并在目标处显示 **No route**。此入口只计算静态地图上的 NavFn/Dijkstra 全局路径。

在默认比赛地图启动交互规划时，**Recorded waypoints** 显示车载文件 `7.25_first51.yaml` 中的 13 个纯导航点，数字沿用原编号，箭头表示原车航向。原表共有 22 条，另有 9 条带比赛任务。**Recorded waypoint plan** 用紫线显示 Nav2 按原顺序计算的全局路径。此次从左上角车位经 13 个点到 `goal_24` 共 17.54 m，逐点核对了规划结果。青色单目标路线可以另行点击比较，两条路线可独立勾选。按 0.295 × 0.310 m 导航轮廓与地图占用格对照，`09`、`14`、`15` 的轮廓覆盖墙格，图中以橙色标记。这条路线用于规划预览，未执行底盘运动。 [部署路点与来源校验值](src/raicom_research_assets/config/competition/waypoints.yaml)

运行记录点时，在工作空间用 `ros2 launch raicom_research_assets plan_on_map.launch.py interactive:=true motion:=true` 启动平面运动仿真。在相同 ROS 域的另一终端运行 `ros2 service call /route_simulation/start std_srvs/srv/Trigger '{}'` 开始一次。两个窗口从 Gazebo 读取同一车位。RViz 中紫线是原始导航点的规划路径，绿色 **Driven trail** 是本次仿真行驶轨迹。每个点需要同时达到原车 0.03 m 位置容差和仿真设置的 0.025 rad 航向容差。平面车体使用 0.295 × 0.310 m 的车载导航轮廓，沿途检查车身即将扫过的地图格。预计压墙时立即置零速度并保留停车画面。终态和已到编号发布在 `/route_simulation/status`。 本次 Gazebo 反馈确认依次到达 01、02、05、06，驶向 09 时停车。停车后连续 75 个位姿样本稳定在 (−0.259 m, 1.040 m)，距离 09 号点 0.763 m。绿色轨迹采样长度约 7.98 m。 [本次停车画面](research/ros2_examples/recorded_route_stop.png)

静态地图规划使用 `ros2 launch raicom_research_assets plan_on_map.launch.py`。在这个独立场景中向 Nav2 的 `/compute_path_to_pose` 发送带显式起点的请求。已在默认地图测通从 (0.5, 1.5) 到 (1.5, 1.5) 的请求。该 NavFn/Dijkstra 结果属于 ROS 2 对比实验，历史 ROS 1 规划轨迹保存在回放里。

## 8 月 20 日版本依据

PIOS 将赛项记为“睿抗智能侦查”，记载 8 月 19 日车上同时使用 `bit_ws/src/nav_demo` 和 `Sebastian/src/yellow_track_avoidance`，9 月 5 日由用户确认国一及结项。工作区身份由以下文件内容交叉核对。比赛当天实际执行的命令、完整运行日志和官方赛期没有随三份包提供。

| 材料 | 核验结果 |
| --- | --- |
| bit_ws.zip | 整车任务入口 `815final.launch` 在这里，第四赛区引用 Sebastian 的避障包。7 个文件的 ZIP 内时间为 2026-08-20，含主流程、备用参数和当日试用权重。另有 10 至 11 月的试验文件。 |
| Sebastian.zip | 保存第四赛区算法、原图、标定、录像和 8 月 19 日原车交接包。内嵌交接 ZIP 的 SHA-256 与 PIOS 历史收件一致，75 条内部校验通过。73 个交接文件在本轮两个车载 ZIP 中均找到，70 个逐字节相同，另 3 个有后续修改。 |
| 智能侦察总任务包.zip | 730 个文件条目的未压缩内容合计 0 字节。按路径回查两个车载包，623 项有内容、95 项也是空文件、12 项缺失。 |
| 车体外形 | 两个车载模型目录里的 STL、URDF 和 Xacro 为空。工作空间使用已选的公开 Mowen。物理参数引用比赛配置时采用长 0.34 m、宽 0.33 m；展示用网格尺寸另见模型来源。 |

ZIP 内时间按其原值记录，无时区字段。8 月 10 日的录像用于还原调试实验。8 月 20 日的版本判断依靠交接包、文件引用和当日改动。 交接包逐项对比（本机原始归档索引） · 总任务包目录对照（本机原始归档索引）

## 来源、去重与验证


原车算法的 123 项独立单元测试通过。ROS 2 逐条解码了全部 MCAP 消息，Nav2 map server 逐份读取了 33 份地图；另实测了真实图像的 ROS 发布订阅、回放时的轮速里程计及 EKF 输出、静态规划 action，以及 RViz 对车模的订阅和 TF 链。实车传感器驱动、底盘运动和比赛全程没有实测。 [验收数据](catalog/verification.json) · [ROS 1 和 ROS 2 数据转换工具](https://ternaris.gitlab.io/rosbags/topics/convert.html)

发布文件与复现步骤见[仓库首页](../README.md)，完整文件校验见 [SHA-256](../archive-files.sha256)。源 ZIP 私有保留，导航与避障数据直接包含在本目录。
