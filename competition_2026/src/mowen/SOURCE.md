# Mowen vehicle model

The user selected this public reference on 2026-11-17 to fill the onboard ZIP's empty URDF and STL. The original package name, mesh URI, link names, scale and file contents are preserved. The ROS 2 packaging and visualization launch were added locally. The model has a single visual mesh and a fixed joint between base_footprint and base_link.

The competition vehicle has a mecanum, holonomic chassis. The onboard odometry represents lateral velocity independently. The public mesh supplies the body appearance only. The default Gazebo view holds it at a fixed pose. The optional planar motion world uses Gazebo body velocity control, a rectangular map-footprint collision shape and feedback from the simulator. Wheel joints, roller contact and wheel-speed kinematics are unmodeled.

| File | Git blob | SHA-256 |
| --- | --- | --- |
| urdf/mowen.urdf | b39735765614d31f09b11d1e2df1d9c2128d603e | 66b5f29c3eff416f4aa6fc06f178b531122d947e1aaeed6accb183609e5186ea |
| meshes/base_link.STL | 32f7455e35bfafeb33c0a56413fb2b98e9803cfa | eca4584c04c6995f0e2dc1cc4cfc5caeca50209f2cca2648004bad1f97299e4f |

The STL contains 151,588 triangles. In the unscaled URDF mesh frame its bounding box measures 0.3334 × 0.3025 × 0.2220 m along x, y and z. The obstacle-avoidance profile from the car uses 0.34 m length and 0.33 m width. Use the latter values when describing that algorithm. The empty onboard STL provides no byte identity for the geometry installed on race day.

The URDF and STL come from the pinned [RAICOM_edu model package](https://github.com/EnochxynL/RAICOM_edu/tree/62253df73839c9212f6f60932cdd7b65946e4413/1_robot_description/src/robot_description/mowen). A second repository carries the same Git blobs for both files. [Matching model package](https://github.com/562862/altas-26-33/tree/b0739cee58ec328b599d67b8c270733b4c3a2051/src/robot_description/mowen)

GitHub reports no declared repository license for the selected source. Preserve attribution and check publication rights when reproducing its geometry. Historical vehicle sensor extrinsics cannot be inferred from this single-link mesh.
