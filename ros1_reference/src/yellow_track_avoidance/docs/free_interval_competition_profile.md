# Formal DWA/costmap competition profile

The formal current-car profile treats yellow lines as corridor boundaries and
confirmed discs as odom-fixed costmap obstacles. It does not require two discs
to form a gate or to straddle the original route. GlobalPlanner and holonomic
DWA own ordinary planning and control.

Formal control ownership is:

```text
camera tracker confirmation
  -> freeze disc centre in odom
  -> publish a 10 Hz virtual LaserScan
  -> global/local costmap obstacle layers
  -> GlobalPlanner + holonomic DWA
  -> navigation safety gate
  -> /cmd_vel
```

The formal launch flags are all false:

```text
competition_two_disc_mode
enable_gate_special_mode
force_between_disc_pair
competition_finish_priority
```

Only already-confirmed tracks enter the formal costmaps. Their odom positions
do not move with later camera jitter and do not expire after visual loss. They
remain in the mission world model until the process restarts or the guarded
`/yellow_track_disc_costmap_bridge/reset_scene` service clears the scene while
the navigation gate is locked, no goal is active, commands are zero, and odom
is stationary.

`enable_costmap_writes` is true and the passage supervisor is not launched in
the formal current-car profile. The existing free-interval and gate code is
therefore unable to cancel move_base or compete for velocity authority.

The gate states (`gate_armed`, `gate_align`, `gate_cross`, `gate_exit`) remain
implemented only for rollback. The separate `40cm_precision_test.launch`
explicitly enables the passage supervisor and disables disc costmap writes;
it is frozen as an experimental metric pressure test, not a formal competition
control chain.

Run the locked/offline acceptance report with:

```bash
PYTHONPATH=src python3 scripts/offline_passage_scenarios.py
```

The competition reference photograph has no metric scale.  The three official
layout fixtures therefore preserve its qualitative placement and mirror
relationships but are not claimed as measurements of the photograph.
