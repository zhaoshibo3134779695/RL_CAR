# TEB compact semantic-obstacle A/B — 2026-08-10

## Outcome

The compact representation is a major performance improvement, but it is not
yet safe for a moving-vehicle trial.

The TEB optimization container now receives exactly:

- left yellow physical inner edge: one line obstacle;
- right yellow physical inner edge: one line obstacle;
- frozen disc 1: one circular obstacle, radius 0.05 m;
- frozen disc 2: one circular obstacle, radius 0.05 m.

GlobalPlanner and both costmaps still receive the original lethal corridor
strips and disc scan.  The original polygon footprint and final costmap
feasibility call remain enabled.

## Scope control

The runtime TEB parameter dictionaries before and after the change were
compared.  The only changed value was:

```text
include_costmap_obstacles: true -> false
```

The compact profile still used:

```text
inner/outer:       6 / 4
dt_ref:            0.20
weight_viapoint:   10
weight_obstacle:   50
min_obstacle_dist: 0.020
feasibility poses: 50
controller:        8 Hz
```

No vision, GlobalPlanner, costmap, footprint, inflation, velocity, association,
goal, or recovery parameter was changed.

## Complete-scene gate

The compact publisher emits four obstacles only when all conditions are true
for the same `scene_generation`:

- locked and valid corridor pair;
- exactly two memory, active, scan and costmap discs;
- both active tracks confirmed with finite fixed-frame coordinates;
- mission position freezing enabled;
- costmap writes enabled;
- fixed-frame transform available;
- corridor and discs use the same fixed frame.

Before that point, after a reset generation change, or for any partial scene,
it publishes an empty obstacle array so TEB cannot retain an old custom scene.

## Trial 3 locked replay results

No chassis command consumer was running.  `/move_base` remained the sole
`/cmd_vel` publisher.  Both disc cells were lethal in local and global
costmaps.  The perception node was `LOCKED`; in the repeated camera run it had
received/skipped 5,841 frames and processed zero heavy frames.

| Metric | 1,078 costmap points | Compact run | Compact repeat |
|---|---:|---:|---:|
| TEB feedback obstacle count | 1,078 | 4 | 4 |
| move_base CPU | 70.88% | 15.62% | 15.40% |
| move_base RSS | 65.34 MiB | 56.81 MiB | 56.81 MiB |
| whole-machine busy | 41.73% | 32.63% | 32.43% |
| cmd period mean | 158.39 ms | 122.46 ms | 120.20 ms |
| cmd period p95 | 491.05 ms | 139.78 ms | 134.03 ms |
| periods over 187.5 ms | 6 | 0 | 0 |
| infeasible/reset | 5 | 1 | 0 |
| control-loop-missed logs | not captured | 0 | 0 |
| feedback samples | not captured | 47 | 49 |
| swept disc clearance range | unstable | -39.17 to +60.16 mm | -24.61 to +14.11 mm |
| node-only minimum disc clearance | not captured | not captured | +7.69 mm |
| swept yellow clearance minimum | positive | +92.13 mm | +92.18 mm |

The oscillation abort at approximately six seconds is expected in this locked
replay because odometry is intentionally held at the origin while nonzero
planner commands are not executed.  It is not counted as a physical trial.

Raw reports:

- `/tmp/yta_teb_compact_trial3_locked_camera_exact_7s_20260810.json`
- `/tmp/yta_teb_compact_trial3_locked_camera_exact_repeat_7s_20260810.json`

## Why final feasibility did not guarantee the required margin

TEB's costmap feasibility step is still active and uses the complete polygon
footprint.  It is, however, a collision test, not a 20 mm clearance test.

The Noetic implementation checks TEB poses and adds intermediate checks only
when adjacent translation exceeds the robot inscribed radius or adjacent yaw
exceeds `min_resolution_collision_check_angular`.  For this configuration the
thresholds are 0.165 m and the default pi radians.  That is much coarser than
the report's 5 mm swept-rectangle audit around a 0.10 m disc.  The repeated
trajectory had adjacent pose gaps up to 0.269 m; sparse internal interpolation
could therefore remain collision-free at its tested points while the 5 mm
sweep between them intersected the disc.

Separately, `min_obstacle_dist` is represented in the optimizer as a weighted
penalty, not a hard 20 mm invariant.  The repeated run's TEB nodes themselves
reached only +7.69 mm clearance even when no costmap collision was reported.

## Decision

Keep the compact profile as an A/B candidate because it removes the main
computational waste.  Do not use it on the moving chassis yet.

This round rules out obstacle count and whole-machine CPU as the remaining
primary safety problem.  The single next bottleneck is TEB geometric safety:
the optimizer does not consistently maintain the requested 20 mm disc margin,
and the built-in costmap feasibility sampling is too coarse to certify the
continuous rectangular sweep.  Do not reduce 6x4 iterations before that issue
is addressed.

## Verification

- package tests: 166/166 passed;
- catkin build: passed;
- launch XML/runtime parameter load: passed;
- runtime TEB parameter diff: only `include_costmap_obstacles` changed;
- hardware command consumer: absent;
- all test nodes stopped after capture.
