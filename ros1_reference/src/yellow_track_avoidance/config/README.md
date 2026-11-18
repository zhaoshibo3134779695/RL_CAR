# 配置目录

- `calibrated.yaml`：迁移 ZIP 中旧车实测验收的 640×480 检测参数基线；
- `current_car.yaml`：继承上述黄线/圆片参数的安全配置，内参/BEV 仍无效且运动关闭；
- `current_car_hd.yaml`：当前 Orbbec 的 1280×720 视觉测试配置，仍未标定且运动关闭；
- `current_car_intrinsic_640x480.yaml`：当前 Orbbec 的 640×480 内参结果；
- `current_car_calibrated_640x480.yaml`：当前默认检测配置；保留本车内参、BEV、ROI，黄线/圆片参数与迁移 ZIP 基线逐项一致，运动仍关闭；
- `calibrated_1920x1080_checkerboard.yaml`：1080p 棋盘格版本；
- `default.yaml`、`intrinsic_calibrated*.yaml`：标定阶段配置，不应直接解锁实车；
- `click_nav_*.yaml`：move_base、SBPL、TEB、全局/局部代价地图参数；
- `passage_supervisor.yaml`：单圆片、多圆片和窄门控制；
- `vehicle_profiles.yaml`：离线路线验证的车辆档案；
- `track_7_15_route.yaml`、`yta_1cm*.mprim`：场地路线与 1 cm 动作原语；
- `yta_5mm_metric.mprim`：当前 727 地图使用的 5 mm 动作原语，保持 1 cm 版本的实际运动距离；
- `navigation.rviz`：显示配置。

主 launch 的 `vehicle_length_m`、`vehicle_width_m` 会覆盖所有实车导航 footprint
和感知中的车体尺寸；YAML 内旧车尺寸继续作为离线工具与后向兼容默认值。
本车相机内参、BEV 和颜色检测已验收；`base_link` 到最近可见 BEV
截面的安装偏移仍需实测，因此不能据此开启运动。
