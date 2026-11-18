# Competition perception ACQUIRE → LOCKED report (2026-08-10)

## Outcome

The competition minimal profiles now run perception in exactly two mission
states:

```text
ACQUIRE -> LOCKED
```

No detector, calibration, BEV, image size, threshold, tracking confirmation or
geometry parameter was changed.  Once the downstream odom/costmap world model
has been continuously complete for 0.5 s, the image callback returns before
format validation and `cv_bridge`, while the ROS node and USB camera remain
alive.

The optimization is successful.  It is not yet safe to run the current TEB
profile on the chassis: the remaining dominant failure is TEB feasibility and
obstacle optimization, not vision CPU.

## Runtime lock contract

All of the following must be true in the same scene generation for 0.5 s:

- corridor status is `locked`;
- at least one odom-fixed `corridor_cloud` has been published;
- bridge mission position freezing is enabled;
- memory contains exactly two confirmed tracks;
- both tracks have distinct IDs and finite odom coordinates;
- costmap disc count is exactly two;
- costmap writes are enabled;
- the fixed-frame transform is available.

`LOCKED` is a mission latch.  Line loss, disc loss, stale visual input and
planner failure do not unlock it.  The only return to `ACQUIRE` is a successful
explicit call to:

```text
/yellow_track_disc_costmap_bridge/reset_scene
```

The bridge publishes one generation-tagged reset event.  The bridge memory,
corridor lock and perception latch all reset from that single event.  The
minimal profile does not require the absent navigation safety gate for this
service, but it still requires fresh odometry, a fresh zero output command, no
active navigation goal and a stationary robot.

## ACQUIRE regression (300 new real-camera frames)

The latch was deliberately withheld from downstream status during this test so
that the unchanged full pipeline could be sampled for 300 additional frames.

| Metric | Result |
|---|---:|
| Processed / received | 300 / 300 |
| `2 raw + 2 confirmed` status frames | 300 / 300 |
| `2 raw + 2 confirmed` disc messages | 300 / 300 |
| Left boundary detected | 300 / 300 |
| Boundary quality mean / minimum | 0.9096 / 0.9036 |
| Processing time mean / p95 / max | 76.45 / 90.49 / 103.59 ms |
| Vision CPU mean / p95 / max | 283.4% / 294.9% / 301.0% |
| Vision RSS mean / max | 154.2 / 159.7 MiB |
| Threads | 17 |

The two track-coordinate spans over the 300 frames were:

| Track | Mean base coordinate (m) | Forward span | Lateral span |
|---|---:|---:|---:|
| 1 | (1.0299, 0.2683) | 14.6 mm | 6.6 mm |
| 2 | (0.5650, -0.1172) | 9.3 mm | 5.6 mm |

These results show no recognition regression.  The higher CPU sample than the
earlier 263.2% baseline reflects the test load/run conditions; the code path and
vision parameters are unchanged.

## LOCKED performance (31.13 s, real camera still at about 30 Hz)

| Metric | Result |
|---|---:|
| Vision CPU mean / p95 / max | 8.87% / 12.37% / 12.77% |
| CPU reduction vs 283.4% ACQUIRE mean | 96.9% |
| Whole-machine busy mean / p95 | 30.14% / 34.34% |
| Vision RSS mean / max | 145.1 / 146.0 MiB |
| Vision threads | 20 |
| Raw image callbacks | 919 (29.52 Hz) |
| Frames skipped before `cv_bridge` | 919 |
| Heavy frames processed | 0 |
| `disc_scan` | 311 (9.99 Hz) |
| `corridor_cloud` | 156 (5.01 Hz) |
| Bridge status remained exactly two discs | all samples |
| Corridor status remained ready | all samples |
| GlobalPlanner preview | 977 poses |
| Global costmap disc cell maxima | 100, 100 |
| Local costmap disc cell maxima | 100, 100 |

An explicit reset integration test succeeded at scene generation 1.  The
camera pipeline resumed, reprocessed 18 frames, rebuilt both downstream models,
and latched again with two discs and a ready corridor cloud.

## Original TEB 6 x 4 replay after vision lock

The exact Trial 3 left-staggered world was frozen before the camera was started,
then the camera continued to feed the already-LOCKED perception node.  No TEB,
GlobalPlanner, footprint, costmap or obstacle parameter was changed.

Confirmed parameters included:

```text
inner=6, outer=4, dt_ref=0.20, weight_viapoint=10,
weight_obstacle=50, min_obstacle_dist=0.020,
association force/cutoff=12/15,
feasibility_check_no_poses=50, controller_frequency=8 Hz
```

The same-duration comparison against the prior weight-50 Trial 3 baseline is:

| Metric | Before lock optimization | After lock optimization |
|---|---:|---:|
| Vision CPU | about 263% | about 9% |
| move_base CPU (7 s window) | 74.21% | 70.88% |
| `/cmd_vel` mean period | 164.85 ms | 158.39 ms |
| `/cmd_vel` p95 period | 484.63 ms | 491.05 ms |
| Periods above 187.5 ms | 6 / 39 | 6 / 39 |
| Feedback obstacles | 1078 | 1078 |
| `trajectory is not feasible` / reset | present | 5 / 5 |
| Final outcome | oscillation abort | oscillation abort |

The first exact post-lock run did produce a 60.2 mm disc-clearance trajectory
and later a 56.9 mm trajectory, proving that TEB can find a safe solution.
However, it reset every roughly 1.15 s and aborted after five infeasible cycles.
A repeated 7 s run ended with only 9.7 mm disc clearance, below the required
20 mm.  Therefore the trajectory is not consistently feasible or safe even
though vision has ample CPU margin.

## Verification

- 162 package regression tests passed.
- Catkin configure/build passed.
- `/cmd_vel` had exactly one publisher (`/move_base`).
- No chassis command consumer was present.
- No real-motion test was started.

## Decision

Do not proceed to chassis testing with this TEB profile.  Do not weaken vision
or undo the ACQUIRE/LOCKED optimization.  The single next main bottleneck is
TEB's 1078-point obstacle representation and resulting feasibility/optimization
stability.  Any next experiment should change only that side of the system.
