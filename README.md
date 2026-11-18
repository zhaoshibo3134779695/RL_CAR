# RAICOM 2026 competition robot assets

This repository archives the Ubuntu reconstruction of the mecanum competition robot, maps, vehicle tuning, obstacle-avoidance software and recordings alongside the surviving ROS 1 package sources. The original competition source archives are identified by byte count and SHA-256 in [the input register](competition_2026/catalog/inputs.tsv). The original report dates and August recordings distinguish competition-day configuration from earlier trials.

| Entry | Contents |
|---|---|
| [ROS 2 competition workspace](competition_2026/README.md) | Two ROS 2 packages for the reference model, mapping, RViz, Gazebo, localization, route replay and yellow-track obstacle avoidance. |
| [Onboard ROS 1 sources](ros1_reference/README.md) | Seven surviving navigation/vehicle packages and three mapping/test launches; a separately pinned, optional vendor submodule. |
| [Maps](competition_2026/catalog/maps.tsv) and [onboard navigation parameters](competition_2026/catalog/competition_parameters.tsv) | 33 configurations with their map images, stage-specific tuning, control and odometry parameters. |
| [Recordings](competition_2026/recordings) and [topic index](competition_2026/catalog/recording_topics.tsv) | 12 real ROS 2 MCAP recordings converted from race-season calibration and test data. Historic custom messages are in [compressed JSONL](competition_2026/research/legacy_bag_messages). |
| [Calibration](competition_2026/research/calibration_runs) and [research documentation](competition_2026/research/docs) | Vehicle camera measurements, valid images, maps, measured plans, trial documents and reproduction evidence. |
| [Published inventory](competition_2026/catalog/publication.json) and [SHA-256 manifest](archive-files.sha256) | Coverage, nonempty source file counts and file integrity for the archive payload. |

## Build the ROS 2 workspace

The local integration was tested under native Ubuntu with ROS 2 Lyrical. Use the workspace root for launch commands. ROS 1 sources remain isolated.

```bash
cd competition_2026
python3 src/mowen/fetch_reference.py
source /opt/ros/lyrical/setup.bash
colcon build --symlink-install --base-paths src
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch raicom_research_assets research.launch.py
```

The original onboard model files were empty. The selected public visual reference comes from an upstream repository without a declared redistribution license. The preparation command fetches the exact URDF and STL from that repository and checks both SHA-256 values. The mesh serves as a visual reference; the vehicle navigation footprint and the fourth-zone profile retain their distinct original dimensions. [Visual-model origin](competition_2026/src/mowen/SOURCE.md)

To verify the archived files, run `sha256sum --check --quiet archive-files.sha256` from the repository root after checkout. The manifest covers the published workspace and recovered ROS 1 files; the optional upstream submodule and visual model are independently pinned. Each original license and attribution remains with its source. The repository omits private device/chat records, mixed actuation/target-recognition materials, broken archive members and generated build outputs.
