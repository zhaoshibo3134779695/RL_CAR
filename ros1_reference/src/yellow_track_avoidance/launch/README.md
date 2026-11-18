# Launch 入口

| 文件 | 用途 | 默认运动状态 |
|---|---|---|
| `current_car_camera.launch` | 当前 USB 摄像头驱动 | 无运动接口 |
| `detect_only.launch` | 相机检测与调参 | 禁止运动 |
| `current_car_navigation.launch` | 当前 x86_64 车的 GlobalPlanner+DWA 测试主链 | 不启动底盘、双重锁定 |
| `competition_global_teb_minimal.launch` | 单一控制权的 GlobalPlanner+TEB 第一候选 | 不启动相机、里程计或底盘 |
| `competition_sbpl_teb_candidate.launch` | 第二级 SBPL+TEB 静态候选 | 不启动相机、里程计或底盘 |
| `competition_dwa_baseline.launch` | 冻结的 GlobalPlanner+DWA A/B 基线 | 不启动相机、里程计或底盘 |
| `planning_only.launch` | 地图/SBPL/TEB 规划检查 | 无底盘 |
| `click_navigation.launch` | 可组合的完整主入口 | `enable_motion=false`、锁定 |
| `real_navigation.launch` | 640×480 实车入口 | 可运动但启动锁定 |
| `real_navigation_640_verified.launch` | 当前 640×480 固定配置入口 | 可运动但启动锁定 |
| `motion_low_speed.launch` | 独立低速感知控制试验 | 必须人工明确启用 |
| `offline_navigation.launch` | 离线图片/路线检查 | 禁止运动 |

这台车优先从 `current_car_navigation.launch` 做零运动联调；迁移来源车仍使用
`real_navigation.launch`。明确传入最终标定配置和相机安装偏移后才能进入实车
验收。所有实车入口仍需调用安全门服务解锁，不要把 `start_enabled` 改为默认
`true`。

三个 `competition_*` 规划器入口共用
`competition_planner_core.launch`，但每次只加载一个 global plugin 和一个 local
plugin。其第一阶段流程和锁止检查见
`docs/navigation/COMPETITION_PLANNER_AB.md`。这些入口特意不启动软件安全门，实车
前必须保留物理急停，并在获得明确许可后才将 `start_chassis` 设为 `true`。
