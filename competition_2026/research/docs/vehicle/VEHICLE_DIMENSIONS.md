# 车辆尺寸记录

## 当前车辆

- 长：`0.33 m`
- 宽：`0.30 m`
- 当前 `config/calibrated.yaml`、`default.yaml` 和
  `intrinsic_calibrated.yaml` 均使用这组尺寸。

## 后续迁移车辆

- 长：`0.34 m`
- 宽：`0.33 m`
- 目前只记录在 `config/vehicle_profiles.yaml`，没有参与当前规划。

迁移时必须先把活动配置的 `physical.robot_length_m` 和
`physical.robot_width_m` 切换为迁移车辆尺寸，重新检查地图窄道净空、
圆片膨胀半径、最大曲率和低速停车距离，不能直接沿用当前车验收结果。
