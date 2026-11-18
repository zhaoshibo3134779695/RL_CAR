from pathlib import Path
import xml.etree.ElementTree as ET


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _launch_args(path):
    root = ET.parse(str(path)).getroot()
    return {
        element.attrib["name"]: element.attrib.get("default")
        for element in root.findall("arg")
    }


def test_all_launch_files_are_well_formed_xml():
    launch_files = sorted((PACKAGE_DIR / "launch").glob("*.launch"))
    assert launch_files
    for launch_file in launch_files:
        assert ET.parse(str(launch_file)).getroot().tag == "launch"


def test_live_launch_exposes_single_vehicle_geometry_interface():
    args = _launch_args(PACKAGE_DIR / "launch" / "real_navigation.launch")
    assert args["vehicle_length_m"] == "0.33"
    assert args["vehicle_width_m"] == "0.30"

    click_text = (
        PACKAGE_DIR / "launch" / "click_navigation.launch"
    ).read_text(encoding="utf-8")
    assert click_text.count("$(arg vehicle_half_length_m)") >= 10
    assert click_text.count("$(arg vehicle_half_width_m)") >= 10
    assert 'name="robot_length_m" value="$(arg vehicle_length_m)"' in click_text
    assert 'name="robot_width_m" value="$(arg vehicle_width_m)"' in click_text

    current_args = _launch_args(
        PACKAGE_DIR / "launch" / "current_car_navigation.launch"
    )
    assert current_args["start_odometry"] == "false"
    assert current_args["start_chassis"] == "false"
    assert current_args["enable_motion"] == "false"
    assert current_args["start_enabled"] == "false"
    assert current_args["use_amcl"] == "false"
    assert current_args["start_competition_corridor_map"] == "true"
    assert current_args["enable_passage_supervisor"] == "false"
    assert current_args["passage_activation_zone_enabled"] == "false"
    assert current_args["disc_costmap_writes_enabled"] == "true"
    assert current_args["disc_mission_lock_enabled"] == "true"
    assert current_args["enforce_boundary_in_activation_zone"] == "false"
    assert current_args["enforce_disc_emergency_clearance"] == "true"
    assert current_args["disc_emergency_clearance_m"] == "0.020"
    assert current_args["competition_corridor_length_m"] == "1.85"
    assert current_args["publish_corridor_goal_on_lock"] == "false"
    assert current_args["corridor_goal_left_offset_m"] == "0.24"
    assert current_args["corridor_goal_forward_offset_m"] == "0.57"
    current_text = (
        PACKAGE_DIR / "launch" / "current_car_navigation.launch"
    ).read_text(encoding="utf-8")
    assert 'name="vehicle_length_m" value="0.34"' in current_text
    assert 'name="vehicle_width_m" value="0.33"' in current_text
    assert (
        'name="passage_activation_zone_enabled"\n'
        '         value="$(arg passage_activation_zone_enabled)"'
        in current_text
    )

    assert 'name="enforce_boundary_in_activation_zone"' in click_text
    assert 'name="enable_costmap_writes"' in click_text
    assert 'name="enforce_disc_emergency_clearance"' in click_text
    assert 'name="activation_zone_enabled"' in click_text
    assert 'type="competition_corridor_map.py"' in click_text
    assert 'name="use_amcl" default="true"' in click_text


def test_current_car_uses_blank_odom_corridor_map():
    current_text = (
        PACKAGE_DIR / "launch" / "current_car_navigation.launch"
    ).read_text(encoding="utf-8")
    click_text = (
        PACKAGE_DIR / "launch" / "click_navigation.launch"
    ).read_text(encoding="utf-8")
    costmap_text = (
        PACKAGE_DIR / "config" / "click_nav_costmap_common.yaml"
    ).read_text(encoding="utf-8")
    rviz_text = (
        PACKAGE_DIR / "config" / "navigation.rviz"
    ).read_text(encoding="utf-8")

    assert "competition_blank_5x4.yaml" in current_text
    assert 'unless="$(arg use_amcl)" pkg="tf2_ros"' in click_text
    assert 'if="$(arg use_amcl)" pkg="amcl"' in click_text
    assert "/yellow_track_avoidance/corridor_cloud" in costmap_text
    assert "clearing: false" in costmap_text
    assert "/yellow_track_avoidance/corridor_markers" in rviz_text
    assert "/yellow_track_avoidance/corridor_centerline" in rviz_text
    assert (
        'name="navigation_boundary_topic"\n'
        '         value="/yellow_track_avoidance/corridor_boundaries"'
        in current_text
    )
    assert 'name="disc_confirmed_memory_timeout_sec" value="120.0"' not in current_text
    assert 'name="passage_track_fresh_sec" value="120.0"' not in current_text
    assert (
        'name="disc_mission_lock_enabled"\n'
        '         value="$(arg disc_mission_lock_enabled)"'
        in current_text
    )
    assert 'value="$(arg passage_track_fresh_sec)"' in click_text
    assert click_text.count('value="$(arg navigation_boundary_topic)"') == 1
    assert "/yellow_track_avoidance/corridor_boundaries" in click_text
    assert (
        'if self.config.competition_two_disc_mode and boundary_fresh:'
        in (PACKAGE_DIR / "scripts" / "passage_supervisor.py").read_text(
            encoding="utf-8"
        )
    )

    blank_map = (
        PACKAGE_DIR / "maps" / "competition_blank_5x4.pgm"
    ).read_bytes()
    magic, dimensions, maximum, pixels = blank_map.split(b"\n", 3)
    assert magic == b"P5"
    assert dimensions == b"1000 800"
    assert maximum == b"255"
    assert len(pixels) == 1000 * 800
    assert set(pixels) == {255}


def test_current_car_selects_global_planner_and_dwa_test_stack():
    move_base_config = (
        PACKAGE_DIR / "config" / "click_nav_move_base.yaml"
    ).read_text(encoding="utf-8")
    click_text = (
        PACKAGE_DIR / "launch" / "click_navigation.launch"
    ).read_text(encoding="utf-8")
    current_text = (
        PACKAGE_DIR / "launch" / "current_car_navigation.launch"
    ).read_text(encoding="utf-8")

    assert "base_global_planner: SBPLLatticePlanner" in move_base_config
    assert (
        "base_local_planner: teb_local_planner/TebLocalPlannerROS"
        in move_base_config
    )
    assert "controller_frequency: 8.0" in move_base_config
    assert "oscillation_timeout: 6.0" in move_base_config
    assert (
        'name="global_plan_topic"\n'
        '       default="/move_base/SBPLLatticePlanner/plan"'
        in click_text
    )
    assert (
        'name="enable_passage_supervisor"\n'
        '         value="$(arg enable_passage_supervisor)"'
        in current_text
    )
    assert 'name="competition_two_disc_mode" default="false"' in current_text
    assert 'name="enable_gate_special_mode" default="false"' in current_text
    assert 'name="claim_all_road_discs" default="true"' in current_text
    assert 'name="force_between_disc_pair" default="false"' in current_text
    assert 'name="competition_finish_priority" default="false"' in current_text
    precision_text = (
        PACKAGE_DIR / "launch" / "40cm_precision_test.launch"
    ).read_text(encoding="utf-8")
    assert 'name="competition_two_disc_mode" value="true"' in precision_text
    assert 'name="enable_gate_special_mode" value="true"' in precision_text
    assert 'name="claim_all_road_discs" value="true"' in precision_text
    assert 'name="force_between_disc_pair" value="true"' in precision_text
    assert 'name="competition_finish_priority" value="false"' in precision_text
    assert 'name="enable_passage_supervisor" value="true"' in precision_text
    assert 'name="disc_costmap_writes_enabled" value="false"' in precision_text
    assert 'name="disc_mission_lock_enabled" value="false"' in precision_text
    assert (
        'name="disc_confirmed_memory_timeout_sec" value="120.0"'
        in precision_text
    )
    assert 'name="passage_track_fresh_sec" value="120.0"' in precision_text
    assert '<param name="enable_planning" value="false"/>' in click_text
    assert (
        'name="global_planner_plugin" default="global_planner/GlobalPlanner"'
        in current_text
    )
    assert (
        'name="local_planner_plugin"\n'
        '       default="dwa_local_planner/DWAPlannerROS"'
        in current_text
    )
    assert "/move_base/GlobalPlanner/plan" in current_text
    assert (
        'name="terminal_boundary_topic"\n'
        '         value="/yellow_track_avoidance/debug/boundaries"'
        in current_text
    )
    assert (
        'name="finish_recovery_enabled" value="false"'
        in current_text
    )
    assert (
        'name="passage_minimum_predicted_gate_clearance_m"\n'
        '         value="0.020"'
        in current_text
    )

    dwa_config = (
        PACKAGE_DIR / "config" / "click_nav_dwa.yaml"
    ).read_text(encoding="utf-8")
    assert "yaw_goal_tolerance: 0.05" in dwa_config
    assert "latch_xy_goal_tolerance: false" in dwa_config
    assert "trans_stopped_vel: 0.01" in dwa_config
    assert "theta_stopped_vel: 0.025" in dwa_config

    sbpl_config = (
        PACKAGE_DIR / "config" / "click_nav_sbpl.yaml"
    ).read_text(encoding="utf-8")
    assert "lethal_obstacle: 253" in sbpl_config


def test_navigation_planners_can_be_selected_at_launch_time():
    click_text = (
        PACKAGE_DIR / "launch" / "click_navigation.launch"
    ).read_text(encoding="utf-8")
    current_text = (
        PACKAGE_DIR / "launch" / "current_car_navigation.launch"
    ).read_text(encoding="utf-8")

    assert 'name="global_planner_plugin" default="SBPLLatticePlanner"' in click_text
    assert (
        'name="local_planner_plugin"\n'
        '       default="teb_local_planner/TebLocalPlannerROS"'
        in click_text
    )
    assert 'name="base_global_planner" value="$(arg global_planner_plugin)"' in click_text
    assert 'name="base_local_planner" value="$(arg local_planner_plugin)"' in click_text
    assert "config/click_nav_dwa.yaml" in click_text
    assert click_text.count('value="$(arg global_plan_topic)"') == 2
    assert 'name="global_planner_plugin" value="$(arg global_planner_plugin)"' in current_text
    assert 'name="local_planner_plugin" value="$(arg local_planner_plugin)"' in current_text
    assert 'name="global_plan_topic" value="$(arg global_plan_topic)"' in current_text


def test_competition_planner_ab_profiles_are_minimal_and_locked_by_default():
    launch_dir = PACKAGE_DIR / "launch"
    profiles = {
        "competition_global_teb_minimal.launch": (
            "global_planner/GlobalPlanner",
            "teb_local_planner/TebLocalPlannerROS",
        ),
        "competition_sbpl_teb_candidate.launch": (
            "SBPLLatticePlanner",
            "teb_local_planner/TebLocalPlannerROS",
        ),
        "competition_dwa_baseline.launch": (
            "global_planner/GlobalPlanner",
            "dwa_local_planner/DWAPlannerROS",
        ),
        "competition_global_teb_compact.launch": (
            "global_planner/GlobalPlanner",
            "teb_local_planner/TebLocalPlannerROS",
        ),
    }
    for filename, (global_plugin, local_plugin) in profiles.items():
        path = launch_dir / filename
        assert path.is_file()
        args = _launch_args(path)
        assert args["start_camera"] == "false"
        assert args["start_lidar"] == "false"
        assert args["start_odometry"] == "false"
        assert args["start_chassis"] == "false"
        if filename == "competition_global_teb_compact.launch":
            assert args["start_imu"] == "true"
        text = path.read_text(encoding="utf-8")
        assert 'launch/competition_planner_core.launch' in text
        assert 'name="global_planner_plugin" value="{}"'.format(
            global_plugin
        ) in text
        assert 'name="local_planner_plugin"\n         value="{}"'.format(
            local_plugin
        ) in text

    core_text = (
        launch_dir / "competition_planner_core.launch"
    ).read_text(encoding="utf-8")
    assert 'type="move_base" name="move_base"' in core_text
    assert "navigation_safety_gate.py" not in core_text
    assert "passage_supervisor.py" not in core_text
    assert "narrow_gate_controller.py" not in core_text
    assert "map_path_controller.py" not in core_text
    assert 'name="enable_planning" value="false"' in core_text
    assert 'name="enable_motion" value="false"' in core_text
    assert 'name="publish_goal_on_lock" value="false"' in core_text
    # The commissioned start pose puts the vehicle front at the near yellow
    # endpoint (0 cm offset), so the complete 1.85 m corridor is modeled.
    assert 'name="corridor_length_m" value="1.85"' in core_text
    assert (
        'name="goal_left_from_right_endpoint_m" value="0.24"'
        in core_text
    )
    assert (
        'name="goal_forward_from_right_endpoint_m" value="0.57"'
        in core_text
    )
    assert 'name="enable_mission_perception_latch" default="true"' in core_text
    assert 'value="$(arg enable_mission_perception_latch)"' in core_text
    assert 'name="perception_lock_stable_sec" value="3.0"' in core_text
    assert 'name="expected_frozen_disc_count" value="2"' in core_text
    assert core_text.count(
        'value="/yellow_track_avoidance/reset_scene"'
    ) == 3
    assert (
        'name="require_navigation_gate_for_scene_reset" value="false"'
        in core_text
    )
    assert 'name="output_cmd_topic"\n           value="/yellow_track_avoidance/perception_cmd_vel"' in core_text
    assert '<remap from="cmd_vel"' not in core_text
    assert 'name="start_imu" default="false"' in core_text
    assert 'type="yesense_imu_node"' in core_text
    assert 'name="use_imu_yaw" value="false"' in core_text
    assert 'name="imu_topic" value="$(arg imu_topic)"' in core_text
    assert 'name="imu_timeout" value="$(arg imu_timeout)"' in core_text
    assert "competition_robot_localization_wheel_xy_imu_yaw.yaml" in core_text
    assert '<node if="$(arg start_imu)"' in core_text
    assert '<node unless="$(arg start_imu)"' in core_text
    assert core_text.count('name="ekf_localization"') == 2

    imu_config = (
        PACKAGE_DIR / "config" / "competition_yesense_imu.yaml"
    ).read_text(encoding="utf-8")
    assert "yesense_port: /dev/imu" in imu_config
    assert "yesense_baudrate: 460800" in imu_config
    assert "frame_id: base_link" in imu_config
    assert "imu_topic: /imu0" in imu_config
    assert "angular_velocity_stddev: 0.0025" in imu_config
    assert "orientation_stddev: 0.0025" in imu_config

    ekf_config = (
        PACKAGE_DIR
        / "config"
        / "competition_robot_localization_wheel_xy_imu_yaw.yaml"
    ).read_text(encoding="utf-8")
    assert "odom0: /odom_raw" in ekf_config
    assert "imu0: /imu0" in ekf_config
    assert "imu0_relative: true" in ekf_config
    assert "imu0_differential: false" in ekf_config
    assert (
        "true,  true,  false,\n"
        "               false, false, false"
        in ekf_config
    )
    assert (
        "false, false, true,\n"
        "              false, false, false,\n"
        "              false, false, true"
        in ekf_config
    )

    move_base_text = (
        PACKAGE_DIR / "config" / "competition_move_base_no_recovery.yaml"
    ).read_text(encoding="utf-8")
    assert "controller_frequency: 8.0" in move_base_text
    assert "recovery_behavior_enabled: false" in move_base_text
    assert "clearing_rotation_allowed: false" in move_base_text

    teb_text = (
        PACKAGE_DIR / "config" / "competition_teb_baseline.yaml"
    ).read_text(encoding="utf-8")
    assert "max_vel_y: 0.08" in teb_text
    assert "acc_lim_y: 0.20" in teb_text
    assert "include_dynamic_obstacles: false" in teb_text
    assert 'costmap_converter_plugin: ""' in teb_text
    assert "enable_homotopy_class_planning: false" in teb_text
    assert "enable_multithreading: false" in teb_text
    assert "shrink_horizon_backup: false" in teb_text
    assert "oscillation_recovery: false" in teb_text

    compact_launch = (
        launch_dir / "competition_global_teb_compact.launch"
    ).read_text(encoding="utf-8")
    compact_override = (
        PACKAGE_DIR
        / "config"
        / "competition_teb_compact_obstacles_override.yaml"
    ).read_text(encoding="utf-8")
    assert 'name="use_compact_teb_obstacles" value="true"' in compact_launch
    assert "competition_teb_baseline.yaml" in compact_launch
    assert "include_costmap_obstacles: false" in compact_override
    assert "no_inner_iterations" not in compact_override
    assert "no_outer_iterations" not in compact_override
    assert 'if="$(arg use_compact_teb_obstacles)"' in core_text
    assert 'type="teb_compact_obstacles.py"' in core_text
    assert "/move_base/TebLocalPlannerROS/obstacles" in core_text

    probe = PACKAGE_DIR / "tools" / "locked_planner_ab_probe.py"
    probe_text = probe.read_text(encoding="utf-8")
    assert "assert_motion_disconnected" in probe_text
    assert '"/cmd_vel_listener"' in probe_text
    assert "TRIAL3_DISCS" in probe_text


def test_stage1_sbpl_profile_is_isolated_locked_and_has_no_gap_closures():
    launch_dir = PACKAGE_DIR / "launch"
    sbpl_profile = launch_dir / "competition_sbpl_teb_compact.launch"
    sbpl_prepare = launch_dir / "competition_stage1_sbpl_prepare.launch"
    sbpl_auto = launch_dir / "competition_stage1_sbpl_auto.launch"

    for path in (sbpl_profile, sbpl_prepare, sbpl_auto):
        assert path.is_file()
        assert ET.parse(str(path)).getroot().tag == "launch"

    auto_args = _launch_args(sbpl_auto)
    assert auto_args["enable_motion"] == "false"
    assert auto_args["costmap_inflation_radius"] == "0.24"

    profile_text = sbpl_profile.read_text(encoding="utf-8")
    assert 'name="global_planner_plugin" value="SBPLLatticePlanner"' in profile_text
    assert 'name="use_sbpl" value="true"' in profile_text
    assert 'name="use_teb" value="true"' in profile_text
    assert 'name="use_compact_teb_obstacles" value="true"' in profile_text
    assert 'name="enable_stage1_passage_closures" value="false"' in profile_text
    assert 'name="publish_sbpl_full_disc_cloud" value="true"' in profile_text
    assert 'name="costmap_footprint_padding_m" default="0.005"' in profile_text
    assert "competition_stage1_sbpl.yaml" in profile_text
    assert "competition_move_base_sbpl_no_recovery.yaml" in profile_text
    assert "click_nav_costmap_common_no_passage_closures.yaml" in profile_text
    assert "yta_1cm.mprim" in profile_text
    assert "competition_blank_5x4_1cm.yaml" in profile_text

    no_closure_costmap = (
        PACKAGE_DIR
        / "config"
        / "click_nav_costmap_common_no_passage_closures.yaml"
    ).read_text(encoding="utf-8")
    assert "observation_sources: scan sbpl_discs yellow_corridor" in no_closure_costmap
    assert "/yellow_track_avoidance/sbpl_disc_cloud" in no_closure_costmap
    assert "topic: /yellow_track_avoidance/disc_scan" not in no_closure_costmap
    assert "passage_closures:" not in no_closure_costmap
    assert "passage_closure_cloud" not in no_closure_costmap

    auto_text = sbpl_auto.read_text(encoding="utf-8")
    assert 'name="start_chassis" value="$(arg enable_motion)"' in auto_text
    assert 'name="require_passage_closures" value="false"' in auto_text

    # The proven GlobalPlanner profile remains unchanged and closure-gated.
    baseline_prepare = (
        launch_dir / "competition_stage1_prepare.launch"
    ).read_text(encoding="utf-8")
    baseline_args = _launch_args(
        launch_dir / "competition_stage1_prepare.launch"
    )
    baseline_goal = (
        launch_dir / "competition_stage1_lidar_goal_capture.launch"
    ).read_text(encoding="utf-8")
    assert baseline_args["costmap_inflation_radius"] == "0.24"
    assert baseline_args["global_costmap_inflation_radius"] == "0.24"
    assert baseline_args["local_costmap_inflation_radius"] == "0.0"
    assert baseline_args["global_costmap_cost_scaling_factor"] == "20.0"
    assert baseline_args["local_costmap_cost_scaling_factor"] == "70.0"
    assert baseline_args["disc_collision_radius_m"] == "0.08"
    assert baseline_args["teb_disc_collision_radius_m"] == "0.09"
    assert baseline_args["local_disc_collision_radius_m"] == "0.09"
    assert baseline_args["passage_total_clearance_m"] == "0.05"
    assert 'name="enable_stage1_passage_closures" value="true"' in baseline_prepare
    assert 'name="include_passage_closures_in_teb" value="false"' in baseline_prepare
    assert 'value="frozen_discs yellow_corridor passage_closures"' in baseline_prepare
    assert 'value="local_frozen_discs yellow_corridor"' in baseline_prepare
    assert 'name="require_passage_closures" default="true"' in baseline_goal

    guard_override = (
        PACKAGE_DIR / "config" / "competition_stage1_teb_guard_override.yaml"
    ).read_text(encoding="utf-8")
    assert "controller_patience: 2.0" in guard_override
    assert "orientation_mode: 2" in guard_override
    assert "max_vel_y: 0.10" in guard_override
    assert "max_vel_theta: 0.05" in guard_override
    assert "acc_lim_theta: 0.10" in guard_override
    assert "penalty_epsilon: 0.005" in guard_override


def test_stage1_sbpl_motion_lattice_matches_map_and_is_holonomic():
    primitive_path = PACKAGE_DIR / "config" / "yta_1cm.mprim"
    lines = primitive_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "resolution_m: 0.010000"
    assert lines[1] == "numberofangles: 16"
    assert lines[2] == "totalnumberofprimitives: 176"

    by_angle = {angle: [] for angle in range(16)}
    for index, line in enumerate(lines):
        if not line.startswith("startangle_c:"):
            continue
        angle = int(line.split(":", 1)[1])
        end_pose = tuple(
            int(value)
            for value in lines[index + 1].split(":", 1)[1].split()
        )
        by_angle[angle].append(end_pose)

    assert all(len(actions) == 11 for actions in by_angle.values())
    for angle, actions in by_angle.items():
        # Every heading has two in-place rotations and translations that keep
        # yaw unchanged. At heading zero, the latter explicitly include long
        # positive and negative lateral motion (mecanum strafe).
        assert sum((dx, dy) == (0, 0) for dx, dy, _ in actions) == 2
        assert sum(end_angle == angle for _, _, end_angle in actions) >= 7
    assert (0, 8, 0) in by_angle[0]
    assert (0, -8, 0) in by_angle[0]

    map_yaml = (
        PACKAGE_DIR / "maps" / "competition_blank_5x4_1cm.yaml"
    ).read_text(encoding="utf-8")
    assert "resolution: 0.01" in map_yaml
    map_image = PACKAGE_DIR / "maps" / "competition_blank_5x4_1cm.pgm"
    assert map_image.is_file()


def test_localization_path_has_no_spin_or_global_particle_reset():
    gate_text = (
        PACKAGE_DIR / "scripts" / "navigation_safety_gate.py"
    ).read_text(encoding="utf-8")
    launch_text = (
        PACKAGE_DIR / "launch" / "click_navigation.launch"
    ).read_text(encoding="utf-8")

    assert "/global_localization" not in gate_text
    assert "auto_localization_angular_speed" not in gate_text
    assert "auto_localization_required_rotation" not in gate_text
    assert "stationary_camera_lidar_fusion" in gate_text
    assert "auto_localization_angular_speed_radps" not in launch_text
    assert "auto_localization_required_rotation_rad" not in launch_text


def test_migration_documentation_and_tools_are_present():
    required = [
        "docs/migration/ARCHIVE_CONTENTS.md",
        "docs/calibration/BEV_CALIBRATION.md",
        "docs/navigation/MAP_NAVIGATION.md",
        "docs/perception/DISC_DETECTION_ACCEPTANCE.md",
        "docs/perception/YELLOW_DETECTION_SOLUTION.md",
        "docs/vehicle/VEHICLE_DIMENSIONS.md",
        "LICENSE",
        "docs/MIGRATION_GUIDE.md",
        "docs/DEPENDENCIES.md",
        "docs/CURRENT_CAR_ADAPTATION.md",
        "docs/PACKAGE_LAYOUT.md",
        "docs/VALIDATION_CHECKLIST.md",
        "docs/PASSAGE_COMMISSIONING.md",
        "docs/navigation/VISUAL_LANDMARK_XY_CORRECTION.md",
        "docs/COMPETITION_FINISH_PRIORITY_PROMPT.md",
        "tools/migration_preflight.sh",
        "tools/record_passage_validation.sh",
        "tools/create_migration_archive.sh",
        "tools/build_sbpl_runtime.sh",
        "config/current_car.yaml",
        "launch/current_car_camera.launch",
        "launch/current_car_navigation.launch",
        "vendor/sbpl_signed_char/LICENSE",
    ]
    for relative_path in required:
        assert (PACKAGE_DIR / relative_path).is_file(), relative_path

    archive_script = (
        PACKAGE_DIR / "tools" / "create_migration_archive.sh"
    ).read_text(encoding="utf-8")
    assert 'calibration_runs/archive/*' in archive_script
    assert 'calibration_runs/current_car_640x480/*' in archive_script


def test_runtime_sources_do_not_embed_old_workspace_path():
    candidates = [
        PACKAGE_DIR / "launch",
        PACKAGE_DIR / "scripts",
        PACKAGE_DIR / "src" / "yellow_track_avoidance",
    ]
    for directory in candidates:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".launch", ".py"}:
                text = path.read_text(encoding="utf-8")
                assert "/home/mo-bit/" not in text, str(path)
