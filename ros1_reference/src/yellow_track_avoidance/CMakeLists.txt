cmake_minimum_required(VERSION 3.0.2)
project(yellow_track_avoidance)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  costmap_converter
  cv_bridge
  geometry_msgs
  image_transport
  map_msgs
  nav_msgs
  rosgraph
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  visualization_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS actionlib_msgs costmap_converter cv_bridge geometry_msgs image_transport map_msgs nav_msgs rosgraph rospy sensor_msgs std_msgs std_srvs tf2_ros visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/yellow_track_avoidance_node.py
  scripts/offline_image_test.py
  scripts/homography_calibrator.py
  scripts/intrinsic_calibrator.py
  scripts/orbbec_photometry.py
  scripts/v4l2_camera_controls.py
  scripts/disc_acceptance_recorder.py
  scripts/disc_capture_control.py
  scripts/offline_map_navigation.py
  scripts/map_route_publisher.py
  scripts/map_avoidance_fusion.py
  scripts/map_path_controller.py
  scripts/disc_laserscan_bridge.py
  scripts/competition_corridor_map.py
  scripts/stage1_lidar_goal.py
  scripts/stage1_auto_runner.py
  scripts/stage1_shadow_odom.py
  scripts/stage1_shadow_observer.py
  scripts/stage2_lidar_goal.py
  scripts/stage2_frozen_scene.py
  scripts/stage2_auto_runner.py
  scripts/competition_stage12_auto.py
  scripts/stage2_lidar_pose_adapter.py
  scripts/teb_compact_obstacles.py
  scripts/narrow_gate_controller.py
  scripts/passage_supervisor.py
  scripts/navigation_safety_gate.py
  scripts/field_pose_localizer.py
  scripts/offline_map_fusion.py
  scripts/offline_passage_scenarios.py
  tools/locked_planner_ab_probe.py
  tools/stage1_shadow_parameter_search.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config docs launch maps
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES
  README.md
  LICENSE
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  tools/migration_preflight.sh
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/tools
)

install(PROGRAMS tools/record_passage_validation.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES vendor/sbpl_signed_char/lib/libsbpl.so
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/vendor/sbpl_signed_char/lib
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_core.py)
  catkin_add_nosetests(test/test_disc_passage.py)
  catkin_add_nosetests(test/test_disc_scan.py)
  catkin_add_nosetests(test/test_corridor_map.py)
  catkin_add_nosetests(test/test_narrow_gate.py)
  catkin_add_nosetests(test/test_passage_navigation.py)
  catkin_add_nosetests(test/test_navigation_gate.py)
  catkin_add_nosetests(test/test_terminal_control.py)
  catkin_add_nosetests(test/test_finish_recovery.py)
  catkin_add_nosetests(test/test_field_localization.py)
  catkin_add_nosetests(test/test_package_layout.py)
  catkin_add_nosetests(test/test_offline_passage_scenarios.py)
  catkin_add_nosetests(test/test_perception_latch.py)
  catkin_add_nosetests(test/test_compact_obstacles.py)
  catkin_add_nosetests(test/test_stage1_lidar_goal.py)
  catkin_add_nosetests(test/test_stage1_auto.py)
  catkin_add_nosetests(test/test_shadow_odom.py)
  catkin_add_nosetests(test/test_stage2_lidar_goal.py)
  catkin_add_nosetests(test/test_stage2_frozen_scene.py)
  catkin_add_nosetests(test/test_stage2_auto.py)
  catkin_add_nosetests(test/test_stage12_auto.py)
  catkin_add_nosetests(test/test_camera_photometry.py)
endif()
