# yellow_track_avoidance

> 迁移到另一台小车请先阅读 [迁移指南](docs/MIGRATION_GUIDE.md)，然后运行
> `tools/migration_preflight.sh`。目录职责和迁移压缩包范围分别见
> [包结构说明](docs/PACKAGE_LAYOUT.md) 与
> [迁移包内容说明](docs/migration/ARCHIVE_CONTENTS.md)。

ROS1 Noetic/OpenCV implementation for following the safe interior of a yellow
track while detecting and avoiding every visible red-ring/white-center disk.
The perception, planning, state machine, and controller are ROS-independent;
only `ros_node.py` contains ROS communication.

黄线识别改造与现场验证说明：
[黄线检测方案](docs/perception/YELLOW_DETECTION_SOLUTION.md)。

红白圆片当前帧检测与专项验收说明：
[圆片检测验收](docs/perception/DISC_DETECTION_ACCEPTANCE.md)。

米制地面与 BEV 标定说明：
[BEV 标定](docs/calibration/BEV_CALIBRATION.md)。

比赛“完赛优先、禁止永久锁死”的故障复盘与外部方案提示词：
[COMPETITION_FINISH_PRIORITY_PROMPT.md](docs/COMPETITION_FINISH_PRIORITY_PROMPT.md)。

5 m × 4 m 静态地图与右上通道路线：
[地图导航](docs/navigation/MAP_NAVIGATION.md)。

当前车的比赛宽道入口不再依赖该全场静态地图：它使用全白自由地图、启动时
`odom` 原点、实时左黄线及 0.91 m 推算右边界。实现与话题见
[当前车辆适配记录](docs/CURRENT_CAR_ADAPTATION.md)。

带实时 TF 定位、融合路径和安全门控低速控制器的导航接入入口：
`launch/navigation_integration.launch`。默认仍为零运动干跑，实车步骤见上面的
地图导航说明。

## Safety status

The checked-in configuration is deliberately **not motion-ready** even though
metric BEV calibration is now valid:

- `control.enable_motion: false`;
- `bev.calibrated: true`, with the accepted depth/track-width ground calibration;
- the mapped BEV is 1.01 m wide by 1.569 m forward;
- output defaults to the isolated `/yellow_track_avoidance/cmd_vel`, not the
  chassis `/cmd_vel`;
- STOP is latched and requires `~reset_stop` after consecutive safe frames;
- direct output refuses non-zero motion if another node publishes the same
  topic.

Known measured values are already represented:

- red-white disk outer diameter: 0.10 m;
- grey road clear width between the yellow lines: 0.88 m;
- each yellow line width: 0.03 m;
- expected yellow centerline separation: 0.91 m.

The current-car entrypoint uses the measured `0.33 m` width and `0.34 m`
length. The generic donor entrypoint retains its legacy `0.30 m × 0.33 m`
defaults. Both values are propagated to every footprint consumer.
Confirm that no wheel, camera bracket, guard, or cable protrudes beyond those
measurements before enabling motion.

The live navigation launch now accepts `vehicle_length_m` and
`vehicle_width_m` and propagates them to perception, both costmaps, TEB, the
disc bridge, the safety gate, and the passage supervisor. Camera geometry and
`local_forward_origin_offset_m` are separate measurements and must still be
recalibrated after changing the camera mount.

## Audited platform

- Ubuntu 20.04.6 LTS, aarch64, Tegra kernel;
- ROS1 Noetic (`/opt/ros/noetic`);
- Catkin workspace `/home/mo-bit/newznzc_ws1`;
- Python 3.8.10;
- OpenCV Python 4.12.0 and system OpenCV 4.2.0;
- `cv_bridge` and `image_transport` installed;
- `move_base` and `robot_localization` installed;
- no `twist_mux` or `yocs_velocity_smoother` installed;
- live Orbbec DaBai DCW2 RGB topic `/camera/color/image_raw`, `rgb8`, 640x480,
  approximately 30 Hz;
- live camera-info topic `/camera/color/camera_info` remains uncalibrated
  (`K[0] == 0`), while the node uses the checked-in calibrated RGB intrinsics;
- bottom half of the RGB image is the current commissioning ROI;
- OrbbecSDK_ROS exposure services were exercised successfully; white-balance
  services are advertised but this device returns an unsupported-property
  error.

## Existing and new control chains

Existing task route:

```text
xy_path_follower -> /xy_vel -> yaw_pid_controller -> /move_vision
  -> task_schedule/vel_ctrl -> /cmd_vel -> car_bringup/newt.py
  -> /dev/carserial
```

`move_base`, `xy_path_follower`, `yaw_pid_controller`, `vel_ctrl`, and
`device_health_check.py` can all publish velocity commands in some launch
configurations. They must not run as competing publishers during direct tests.

Default new chain (safe, not connected to chassis):

```text
yellow_track_avoidance -> /yellow_track_avoidance/cmd_vel
```

Explicit direct test chain, only after all other `/cmd_vel` publishers stop:

```text
yellow_track_avoidance -> /cmd_vel -> car_bringup/newt.py -> /dev/carserial
```

There is no mux in the current installation. The package therefore does not
silently splice itself into any existing package or launch file.

## Processing

1. Crop the configurable six-point ground ROI before dense colour processing.
2. Build a continuous yellow confidence from YCrCb `Cr-Cb`, green-excess and
   neutral-highlight penalties, with HSV hue used only as a saturation-gated
   bonus. The binary mask is retained for display and is not the line fit input.
3. Sample 28 one-dimensional profiles normal to each nominal perspective side,
   combining absolute confidence, road-relative colour, multi-scale ridge
   prominence, and the track-inner edge.
4. Select a continuous left/right path with dynamic programming and fit a
   weighted general line. Strict current-frame support, coverage, confidence,
   prominence, fit-error, gap, and direction gates reject weak paths. Drawing
   is clipped to the observed support and never uses a previous-frame line.
5. Find every red contour, check area, circularity, ellipse axis ratio, red
   annulus occupancy, white center occupancy, and solid-red rejection.
6. Confirm disks over multiple frames, retain short misses, and expire stale
   tracks.
7. Shrink yellow centerlines by half the line width, half the measured robot
   width, and the boundary margin.
8. Expand every disk by its radius, half the robot width, and obstacle margin.
9. Generate center/left/right smoothstep candidates, collision-check every
   sample against both safe boundaries and all disks, reject excess curvature,
   and score clearance, offset, length, curvature, smoothness, previous-path
   consistency, and return-to-center cost.
10. Run `NORMAL`, `APPROACH`, `AVOID`, `RETURN`, and latched `STOP` states.
11. Track the chosen metric path with limited low-speed pure pursuit.

The checked-in BEV and planner code remains safety-locked until intrinsic and
ground calibration are complete. It must not be treated as metric output yet.

## Fixed camera photometry

On the current DaBai DCW2, the verified manual RGB settings are exposure 480,
gain 0, and automatic exposure disabled. The value was selected by a sweep at
the fixed camera pose; it is not a vendor example value. Apply and verify it
after starting the camera driver:

```bash
roslaunch yellow_track_avoidance lock_orbbec_photometry.launch
```

Read current values without changing the camera:

```bash
roslaunch yellow_track_avoidance lock_orbbec_photometry.launch audit_only:=true
```

The main perception node never changes camera state itself. Re-audit the
service names and exposure after migrating to another camera.

## Calibration

The current accepted intrinsic and metric ground results are documented in
`docs/calibration/BEV_CALIBRATION.md`; the machine-readable report is
`calibration_runs/current/calibration_report.json`. The live `CameraInfo`
still has zero intrinsics, so the node uses the checked-in calibration.

The commands below are the manual recalibration fallback. Calibrate at the real
640x480 operating resolution with the camera mount, focus, exposure, and gain
fixed. `--board-cols` and `--board-rows` are **inner-corner counts** and
`--square-size-m` must be physically measured:

```bash
rosrun yellow_track_avoidance intrinsic_calibrator.py \
  --config /home/mo-bit/newznzc_ws1/src/yellow_track_avoidance/config/default.yaml \
  --output-config /home/mo-bit/newznzc_ws1/src/yellow_track_avoidance/config/intrinsic_calibrated.yaml \
  --output-camera-info /tmp/yellow_track_camera_info.yaml \
  --output-preview /tmp/yellow_track_undistorted.png \
  --ros-topic /camera/color/image_raw \
  --board-cols MEASURED_INNER_COLUMNS \
  --board-rows MEASURED_INNER_ROWS \
  --square-size-m MEASURED_SQUARE_SIZE
```

Move the board through the center and all four corners, change tilt and
distance, press Space for each accepted view, and press Enter after at least 15
diverse views. The tool rejects poor coverage and high-error views. It writes a
new file and does not overwrite `default.yaml`.

After intrinsic calibration, capture a representative raw track image. The
homography tool undistorts it from the supplied configuration before asking
for clicks. Click
points in exactly this order:

1. near-left yellow line center;
2. near-right yellow line center;
3. far-right yellow line center;
4. far-left yellow line center.

Because points are on yellow-line centers, the measured width is 0.91 m. The
forward length must be measured on the real ground.

```bash
source /home/mo-bit/newznzc_ws1/devel/setup.bash
rosrun yellow_track_avoidance homography_calibrator.py \
  --image /path/to/real_camera_image.jpg \
  --config /home/mo-bit/newznzc_ws1/src/yellow_track_avoidance/config/intrinsic_calibrated.yaml \
  --output-config /home/mo-bit/newznzc_ws1/src/yellow_track_avoidance/config/calibrated.yaml \
  --output-bev /tmp/yellow_track_bev.png \
  --real-width-m 0.91 \
  --real-length-m MEASURED_LENGTH \
  --lateral-margin-m 0.05
```

The 0.91 m value is the distance between clicked yellow centrelines. The
default 0.05 m margin on each side keeps both yellow lines inside the BEV image
instead of clipping their centres at its edges. The tool never overwrites the
base configuration unless the same path is explicitly supplied.

## Offline image command

This command does not import ROS and forcibly disables motion even if a supplied
configuration says otherwise:

```bash
rosrun yellow_track_avoidance offline_image_test.py \
  --image /home/mo-bit/newznzc_ws1/src/photos/1.jpg \
  --config /path/to/calibrated.yaml \
  --output-dir /tmp/yellow_track_photo1
```

It writes `raw_overlay.png`, `yellow_score.png`, `yellow_mask.png`,
`red_mask.png`, `bev.png`, `plan.png`, and `result.json`.

## Live detect-only command

```bash
roslaunch yellow_track_avoidance detect_only.launch \
  image_topic:=/verified/image/topic \
  camera_info_topic:=/verified/camera_info/topic
```

`detect_only.launch` defaults to the checked-in `config/calibrated.yaml`, which
contains the current camera intrinsics, metric BEV and profile-DP detector.
Vehicle motion remains disabled. Use `config:=...` only to test another
configuration deliberately.

Debug outputs are under `/yellow_track_avoidance/debug`:

- `raw_overlay`, `yellow_score`, `yellow_mask`, `red_mask`, `bev`, `plan` images;
- `status` JSON with state, stop reason, spaces, and command;
- `disks` JSON with centers, contours, ellipses, estimated sizes, confidence,
  track IDs, and timestamps;
- `candidates` JSON with every candidate and rejection/cost details;
- `selected_path` as `nav_msgs/Path` in the `yellow_track_bev` frame.

## Explicit low-speed motion command

Only after camera-only validation and raised-wheel checks:

1. stop `move_base`, `xy_path_follower`, `yaw_pid_controller`, `vel_ctrl`, and
   every other `/cmd_vel` publisher;
2. keep only the chassis subscriber `car_bringup/newt.py`;
3. verify `rostopic info /cmd_vel` shows no other publisher;
4. launch the direct node;
5. call reset only while an operator controls the emergency stop.

```bash
roslaunch yellow_track_avoidance motion_low_speed.launch \
  config:=/path/to/calibrated.yaml \
  image_topic:=/verified/image/topic

rosservice call /yellow_track_avoidance/reset_stop
```

Configured speeds remain below the currently observed navigation limits of
0.18 m/s translation and 0.5 rad/s rotation.

## Single-package live RViz navigation

The live vehicle navigation entrypoint starts the camera, lidar, chassis
feedback and command bridge, AMCL, map server, `move_base`, yellow-track and
red/white-disc perception, disc costmap bridge, RViz, and both live perception
views from this package:

```bash
roslaunch yellow_track_avoidance real_navigation.launch
```

The launch is motion-capable but starts unarmed. With a staffed physical
emergency stop, a clear test area, valid `map -> base_link`, stable perception,
and no competing `/cmd_vel` publisher, release the two runtime gates:

```bash
rosservice call /yellow_track_avoidance/reset_stop
rosservice call /yellow_track_navigation_gate/set_enabled "data: true"
```

Disable vehicle motion immediately without stopping visualization:

```bash
rosservice call /yellow_track_navigation_gate/set_enabled "data: false"
```

Every live launch now waits for fresh lidar and right-yellow-wall data, locks
the navigation gate, matches the current field geometry to the static map,
publishes `/initialpose`, and clears both costmaps. It retries for up to 90
seconds and never unlocks vehicle motion. Progress is reported on
`/yellow_track_field_localization/status`.

Camera disc detections and RViz markers remain active during this startup
localization, but the disc bridge suppresses costmap marking until the latched
localization state becomes `localized`. This prevents the provisional AMCL
transform from leaving duplicate disc obstacles at their pre-correction map
coordinates.

During navigation, confirmed discs are remembered in the fixed `odom` frame
instead of being repeatedly stamped at an old `base_link` coordinate. A disc
therefore remains at its physical field position through short camera
occlusion and is removed as soon as the vehicle rear has completely passed it.
A confirmed disc cannot expire merely because the vehicle waits beside it;
its 30-second timeout only removes stale memories already outside the active
scan range. One-frame unconfirmed detections expire after 0.8 seconds. The
virtual disc scan runs at 10 Hz, matching the effective obstacle input rate;
costmap observation persistence remains zero so the clearing scan can remove
old cells immediately.

The default live profile is the verified 640x480@30 calibration. The camera
still runs at its 30 FPS hardware maximum, while this computer processes about
7 avoidance frames per second instead of roughly 2.7 at 1920x1080.

To disable this startup behavior for an offline diagnostic launch:

```bash
roslaunch yellow_track_avoidance real_navigation.launch \
  auto_field_localization:=false
```

The same one-shot correction can also be requested manually:

```bash
rosservice call /yellow_track_field_localization/localize "{}"
```

This locks navigation and performs a zero-motion fusion. In the current-car
profile the search is limited to the known far upper-right segment; the
detected right yellow edge is the primary lateral/yaw reference, with its
measured offset from the outer wall represented explicitly, while live lidar
endpoints correct the pose against the static walls. The result is accepted
only when the best camera/lidar pose has a low residual and a clear score
margin over the next geometrically different pose. It then publishes
`/initialpose` and clears both costmaps. The map overlay is published at
`/yellow_track_field_localization/debug_overlay`.

The compatibility trigger now calls exactly the same stationary fusion path:

```bash
rosservice call /yellow_track_navigation_gate/start_auto_localization "{}"
```

It cancels the current `move_base` goal, locks the navigation gate, and invokes
camera-yellow plus lidar-map matching. It never calls `/global_localization`,
never enables motion, and never publishes a localization velocity. Progress is
reported under `auto_localization` in
`/yellow_track_navigation_gate/status` with mode
`stationary_camera_lidar_fusion`.

Cancel a pending stationary localization request while keeping the gate locked:

```bash
rosservice call /yellow_track_navigation_gate/cancel_auto_localization "{}"
```

The current-car calibrated forward offset from `base_link` to the visible
BEV reference is `0.5463 m` (`0.39 m` remains the donor default). Disc JSON exposes both the unchanged local
`center_m` and vehicle-referenced `center_base_m`; the costmap and navigation
safety gate use the latter.

### Unified local-passage supervisor

The formal current-car launch uses geometry and free-space topology rather
than the raw number of detections:

```text
no fresh confirmed disc in road  -> ordinary move_base navigation
one blocking disc                -> deterministic left/right smooth bypass
one side disc or multiple discs  -> sampled free-interval channel graph
no executable channel            -> cancel move_base + recoverable SAFE_HOLD
stable safe channel returns      -> passage controller takes it directly
confirmed road becomes clear     -> resume saved move_base goal
```

A visible disc outside the physical yellow road does not trigger special
control. Every fresh confirmed disc inside the road and local decision horizon
does, even when it does not intersect the original centerline. This closes the
otherwise unsafe gap created by keeping discs out of DWA's costmap.

Single and multi-obstacle passages use the detected yellow-line inner edges,
the full current-car `0.33 m` vehicle width, disc radius, line/disc margins and geometry
uncertainty. Direction is confirmed and locked with hysteresis. The reference
is a smooth holonomic path; a small lateral velocity remains available for
centering correction.

Formal competition mode does not classify a pair as a gate and never forces a
route between two discs. The perpendicular-bisector gate controller remains
available only through `40cm_precision_test.launch`. That regression targets a
`0.40 m` clear gap and reserves `0.020 m` modeled/direct clearance. Forward
crossing pauses whenever the `0.005 m` lateral or `1.5 deg` yaw alignment
tolerance is lost; only lateral/yaw recentering remains active until alignment
recovers.

Every special mode follows the same handoff:

```text
save final goal -> cancel move_base -> verify measured stop
-> freeze route/channel -> low-speed special control
-> stop after rear clearance -> republish final goal
```

If planning fails after a disc has entered the road context, the handoff stops
at supervisor-owned zero velocity and keeps replanning. It never releases to
the disc-blind DWA first.

The latest valid SBPL path is cached so an SBPL failure beside a disc does not
erase the special reference. When no SBPL path has yet arrived, two fresh
yellow boundaries provide a local centerline fallback.

Only `yellow_track_navigation_gate` publishes chassis `/cmd_vel`. Its priority
is localization, passage supervisor, then move_base, with common emergency
disc clearance and stale-command checks. Once passage authority is requested,
lost status fails closed instead of falling back to a still-fresh move_base
command.

For the 727 map, the current-car launch enables this visual safety layer only
in the upper-right polygon configured in `config/passage_supervisor.yaml`.
Inside it, every ordinary TEB command and every special-controller command is
checked by sampling the full vehicle rectangle over the next `0.35 s` against
fresh yellow inner edges and confirmed discs. Outside it, ordinary SBPL + TEB
continues without making the whole map dependent on camera boundaries. The
complete commissioning sequence is in `docs/PASSAGE_COMMISSIONING.md`.
The current segment exposes one physical yellow edge in the camera view, so
the opposite corridor edge is reconstructed in metric space from the accepted
`0.91 m` yellow-center separation; the observed edge must still be fresh.

Runtime diagnostics:

```text
/yellow_track_avoidance/passage/status
/yellow_track_avoidance/passage/markers
/yellow_track_navigation_gate/status
```

The passage status reports scene class, active track IDs, route source,
longitudinal/lateral disc coordinates, selected side/channel, residual
clearances, control error, command source and stop reason.

Reset a latched passage failure only after correcting the cause and stopping
the robot:

```bash
rosservice call /yellow_track_passage_supervisor/reset "{}"
```

Disable all special passage modes while retaining normal navigation:

```bash
roslaunch yellow_track_avoidance real_navigation.launch \
  enable_passage_supervisor:=false
```

All final competition parameters are in
`config/passage_supervisor.yaml`.

## Emergency stop

Preferred: use the robot's physical emergency stop. Software fallback:

```bash
rostopic pub -r 20 /cmd_vel geometry_msgs/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'
```

Then stop the avoidance launch. `newt.py` also has a 0.3 s command watchdog,
but it is not a substitute for a physical emergency stop.

## Required commissioning values

- verified live image and camera_info topics;
- image encoding, width, height, and frame rate;
- fixed camera mounting pose and ROI;
- four homography image points;
- measured forward ground length represented by the BEV;
- confirmation that 0.33 m by 0.34 m covers every physical protrusion;
- HSV and morphology values under the final lighting;
- safe margins validated at low speed;
- whether the destination robot uses `/cmd_vel`, an intermediate topic, or a
  proper command mux.

## Test order

Do not skip stages: build/unit/offline image; live camera with motion disabled;
raised wheels at low speed; then ground testing with a physical emergency stop.
Test left, right, center, staggered two-disk, no-safe-gap, camera disconnect,
and boundary-loss cases separately.
