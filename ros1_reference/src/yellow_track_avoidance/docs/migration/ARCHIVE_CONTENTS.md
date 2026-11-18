# 迁移压缩包内容

迁移 ZIP 是可运行源码包，不是现场数据的完整历史备份。解压后顶层目录保持为
`yellow_track_avoidance/`，可直接放进 Catkin 工作空间的 `src/`。

## 包含

- `src/`：ROS 无关的核心算法库；
- `scripts/`：ROS 节点和标定/离线工具；
- `launch/`：检测、规划、实车导航入口；
- `config/`：当前相机配置、车辆尺寸、move_base/TEB/SBPL 参数；
- `maps/`：当前 5 m × 4 m 场地图；
- `test/`：单元测试；
- `tools/`：迁移预检、压缩包生成、SBPL 重编译和通道验收录包工具；
- `vendor/`：与目标主机同架构的 signed-char SBPL 运行库、头文件和许可证；
- `docs/` 及根目录说明文件；
- `calibration_runs/current/` 中四个精简验收文件：配置候选、JSON 报告、
  在线验证 JSON 和 640×480 原图。

## 排除

- `__pycache__`、`.pytest_cache`、`*.pyc`；
- `Log/`；
- `debug_captures/`、`disc_acceptance_runs/`、`navigation_runs/`、
  `offline_navigation_runs/`；
- `calibration_runs/archive/` 和
  `calibration_runs/current_car_640x480/` 中的完整现场标定历史；
- 大体积棋盘格帧、离线中间图、旧压缩包与 `*.tar.gz`。

这些排除项只是不进入迁移 ZIP，原工作区内的数据不会被删除。若需要保存完整
实验历史，应另行备份原包目录或上述数据目录。
