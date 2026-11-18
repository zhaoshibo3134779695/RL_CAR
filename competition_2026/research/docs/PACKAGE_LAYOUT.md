# 包结构说明

本包保持 ROS1 Catkin 的标准目录，不随意移动 `launch/`、`scripts/`、
`src/<package>/`、`config/` 和 `maps/`，这样 `roslaunch`、`rosrun` 与
`catkin_python_setup()` 的接口不会因整理目录而失效。

```text
yellow_track_avoidance/
├── config/       实车参数、标定、地图规划器、车辆档案
├── docs/         迁移、依赖、结构和验收说明
│   ├── assets/      文档引用的赛场照片和 TF 图
│   ├── calibration/ 标定专题
│   ├── migration/   迁移包范围
│   ├── navigation/  地图导航专题
│   ├── perception/  黄线与圆片感知专题
│   └── vehicle/     车辆尺寸专题
├── launch/       ROS 启动入口
├── maps/         map_server 地图与源图
├── scripts/      ROS 节点和可执行标定/离线工具
├── src/yellow_track_avoidance/
│                 不依赖 ROS 的算法与状态机
├── test/         单元测试
├── tools/        迁移预检、打包和工程辅助工具
├── vendor/       特定平台第三方运行库及许可证
├── CMakeLists.txt
├── package.xml
└── setup.py
```

右上角黄线圆片避障的区域配置、零运动联调、录包和现场低速步骤见
`PASSAGE_COMMISSIONING.md`；复现工具是
`tools/record_passage_validation.sh`。
本次锁死链路、完赛优先参数和待研究软代价层问题见
`COMPETITION_FINISH_PRIORITY_PROMPT.md`。

以下目录属于现场产物，不是运行时源码：

- `calibration_runs/current_car_640x480/`：当前车辆有效标定证据；
- `calibration_runs/archive/`：标定原图、过程图和未采用的尝试；
- `debug_captures/`：离线检测调试图；
- `disc_acceptance_runs/`：圆片验收记录；
- `navigation_runs/`、`offline_navigation_runs/`：路线和融合实验输出；
- `Log/`：设备日志。

迁移 ZIP 只保留 `calibration_runs/current/` 中四个精简验收文件，其余现场
产物留在原工作区，不删除。详见
[`migration/ARCHIVE_CONTENTS.md`](migration/ARCHIVE_CONTENTS.md)。

比赛场地中两个红白圆片的随机摆放参考图保存在
`docs/assets/competition_disc_placement_reference.jpg`。该图只是赛题参考，
不是相机标定或定位地图。
