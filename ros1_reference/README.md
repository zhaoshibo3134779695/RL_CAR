# Original vehicle ROS 1 package material

This directory holds surviving catkin package files directly recovered from the vehicle archives. A COLCON_IGNORE marker isolates these packages from the ROS 2 workspace. It is a source archive. The ROS 2 stack in the neighboring competition_2026 directory is the tested entry point.

| Package | Material available here |
|---|---|
| car_bringup, mbot_bringup, mbot_teleop | Chassis communications, teleoperation, sensor configuration, localization and odometry. |
| nav_demo | Mapping, localization, navigation launch/configuration, pure motion and map generation scripts. Mixed competition task runners are outside the published source tree. |
| rf2o_laser_odometry, robot_pose_ekf | Original surviving odometry and EKF package files, including license notices in original sources. |
| yellow_track_avoidance | Vehicle-side Python algorithms, scripts, tests, profiles, SBPL files, valid calibration images, reports, maps and plotting inputs. |
| mycobot_ros | General-purpose vendor library pinned to the commit from the vehicle's own checkout as a Git submodule. Optional for use of the archive. |

To retrieve the optional vendor package from its owner:

```bash
git submodule update --init ros1_reference/src/mycobot_ros
```

The vendor checkout recorded noetic commit `25fdeb5754cfee2cbbe8603bb14c855aec7a8010`. We compared the archived package tree to the upstream Git blob identities. 1,622 nonempty files matched exactly, 318 archived entries were empty, and seven differed (four malformed CAD assets and three frozen model exports). Initializing this submodule retrieves the upstream version. These seven divergent blobs have not been substituted for verified racing data.

Several other onboard package trees contain only zero-byte members. Examples include the originally named base model, IMU and LiDAR driver trees, mapping packages and the camera driver. Their names alone cannot reconstruct the drivers or establish licenses. Driver-specific operating defaults that did not survive are unknown. The vehicle's copies of the three independent mapping/test launches are in `vehicle_launch/`. This archive omits caches, notebooks' generated checkpoints, embedded Git object stores, private handoffs/chats, empty placeholders, invalid files and mixed non-navigation actuation/recognition code.
