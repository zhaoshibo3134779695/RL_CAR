# Low-rate disc-landmark x/y correction design

## Scope

This is a design only.  The first implementation step is the single-layer
wheel/IMU EKF in `competition_robot_localization_wheel_xy_imu_yaw.yaml`.
No visual pose publisher or EKF `pose0` input is enabled yet.

The intended competition sensor ownership is:

```text
wheel odometry       -> body-frame vx, vy
Yesense IMU          -> relative yaw, yaw rate
frozen disc landmarks -> occasional odom-frame x, y correction
```

The correction reuses the existing RGB disc detector.  It does not run the
yellow-boundary profile/DP pipeline, create a second controller, or change the
frozen world model.

## Frames and measurement equation

For frozen disc `i`, let:

- `P_i^O` be its fixed two-dimensional position in `odom`;
- `p_i^B` be the same disc measured from the current image and transformed to
  `base_link`, at the image timestamp;
- `theta` be the Yesense-relative robot yaw at that timestamp;
- `R(theta)` be the planar rotation from `base_link` to `odom`.

The base position implied by that landmark is:

```text
t_i^O = P_i^O - R(theta) p_i^B
```

The result is already the position of `base_link`, not the camera.  Camera
extrinsics must therefore be applied exactly once while producing `p_i^B`.

With two visible, correctly associated discs, calculate one `t_i^O` per disc
and combine them using inverse-covariance weighting.  Their disagreement is a
measurement-consistency test.  With one visible disc, x/y remains observable
because yaw comes from the IMU, but the published covariance must be larger.

The two-disc bearing difference may provide a diagnostic yaw residual, but the
first version must not publish or fuse visual yaw.

## Timing

- Keep the existing `ACQUIRE -> LOCKED` mission latch.
- In `LOCKED`, inspect at most one image every 0.3--0.5 seconds (2--3 Hz).
- Run only the existing disc proposal, validation, ground projection and
  frozen-track association needed for the pose observation.
- Do not run yellow normal profiles, boundary DP, route planning, overlays or
  image publications for these samples.
- Use the camera message timestamp everywhere.  Look up yaw/TF at that stamp;
  never combine a past image with the latest pose.
- Skip all intermediate image callbacks before `cv_bridge`, as the current
  LOCKED implementation does.

## Proposed ROS output

```text
topic:    /yellow_track_avoidance/visual_pose_correction
type:     geometry_msgs/PoseWithCovarianceStamped
frame_id: odom
stamp:    source image timestamp
pose:     estimated base_link pose in odom
```

The quaternion must be valid and may contain the timestamp-matched IMU yaw,
but EKF configuration must ignore its orientation fields.  Only x and y are
measurement variables.

Proposed `robot_localization` addition after visual validation:

```yaml
pose0: /yellow_track_avoidance/visual_pose_correction
pose0_config: [true,  true,  false,
               false, false, false,
               false, false, false,
               false, false, false,
               false, false, false]
pose0_queue_size: 1
pose0_nodelay: true
pose0_differential: false
pose0_relative: false
```

This block is intentionally absent from the live EKF configuration today.

## Covariance

For a single landmark, propagate all three relevant uncertainties instead of
publishing an arbitrarily small fixed covariance:

```text
Sigma_t_i = Sigma_P_i
          + R Sigma_p_i R^T
          + J_theta sigma_theta^2 J_theta^T
```

where `Sigma_P_i` is frozen-landmark uncertainty, `Sigma_p_i` is current BEV
measurement uncertainty and `J_theta` is the derivative of
`-R(theta) p_i^B` with respect to yaw.

For two discs, use inverse-covariance weighting, but retain a nonzero floor
because both estimates share camera calibration, frozen-map and IMU-yaw error.
Initial floors must be chosen from stationary repeatability data, not guessed
from pixel resolution.  Until that data exists, use conservative centimetre-
scale covariance and make the one-disc covariance larger than the two-disc
covariance.  Inflate it further during high yaw rate, long range, partial
occlusion or poor ellipse/ring confidence.

## Association and rejection

A correction is publishable only when all of the following hold:

1. the corridor and two official frozen disc tracks belong to the current
   scene generation;
2. the current observation is confirmed, finite and timestamp-transformable;
3. it matches a frozen disc by identity or gated predicted position;
4. its detector confidence and projected diameter remain valid;
5. for two observations, the two independent base-position estimates agree
   within their combined covariance;
6. the innovation relative to the current fused pose passes a covariance-aware
   outlier gate.

Do not publish an averaged correction when the two landmark estimates disagree;
report the residual and skip that cycle.  Do not move frozen landmark positions
to make a bad observation fit.

## Limits of the method

The correction keeps the robot-to-disc geometry coherent through the dual-disc
area.  It does not create an independent absolute survey of the field: frozen
landmarks inherit the initial camera/BEV/odom uncertainty.

Once both discs leave the field of view, this correction cannot constrain
longitudinal drift.  A later terminal implementation therefore needs a real
longitudinal landmark (for example a visible finish plane/end marker).  A
straight yellow line alone constrains lateral offset and heading, not position
along the line.

## Validation order

1. Record stationary single- and dual-disc repeatability without EKF fusion.
2. Replay recorded observations through the pose calculation and inspect
   residuals/covariance offline.
3. Enable the topic but leave `pose0` disabled; compare it with wheel+IMU odom.
4. Enable EKF x/y fusion with the chassis disconnected and replay a bag.
5. Run short, independently measured forward and lateral motions.
6. Only after those pass, repeat Stage 1 with all TEB parameters unchanged.

