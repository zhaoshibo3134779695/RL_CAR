# 标定数据目录

本目录按“当前证据”和“历史过程”分层，运行配置只应引用当前证据。

- `current/`：迁移包使用的精简验收材料；
- `current_car_640x480/`：当前车辆 640×480 标定的有效报告、配置引用图和最终验证结果；
- `archive/`：棋盘格全集、冷启动逐帧图片、调试中间图和未采用的分辨率尝试。

当前车辆的主要结论见：

- `current_car_640x480/calibration_report.json`；
- `current_car_640x480/live_validation.json`；
- `current_car_640x480/cold_start_validation.json`；
- `current_car_640x480/archive_parameter_audit/report.json`。

`archive/` 中的文件用于追溯，不是运行时输入。不要把其中的旧场景或旧参数
重新接入当前 launch；需要重新标定时，应创建新的运行目录并保留旧报告。
