cmake_minimum_required(VERSION 3.20)
project(raicom_research_assets)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(course_map "${CMAKE_CURRENT_SOURCE_DIR}/maps/vehicle/727best_map")
set(walls_mesh "${CMAKE_CURRENT_BINARY_DIR}/727best_walls.stl")
set(course_world "${CMAKE_CURRENT_BINARY_DIR}/competition_map.sdf")
set(motion_world "${CMAKE_CURRENT_BINARY_DIR}/competition_motion.sdf")
set(navigation_config "${CMAKE_CURRENT_SOURCE_DIR}/config/competition/navigation.yaml")
set(world_template "${CMAKE_CURRENT_SOURCE_DIR}/worlds/competition_map.sdf.in")
add_custom_command(OUTPUT "${walls_mesh}" "${course_world}" "${motion_world}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/map_walls.py"
          "${course_map}.yaml" "${walls_mesh}" "${world_template}" "${course_world}" --height 0.20
          --motion-world "${motion_world}" --navigation-config "${navigation_config}"
  DEPENDS "${navigation_config}" tools/map_walls.py "${world_template}" "${course_map}.yaml" "${course_map}.png"
  VERBATIM)
add_custom_target(competition_walls ALL DEPENDS "${walls_mesh}" "${course_world}" "${motion_world}")
install(FILES "${walls_mesh}" "${course_world}" "${motion_world}" DESTINATION share/${PROJECT_NAME}/worlds)
ament_python_install_package(yellow_track_avoidance)
ament_python_install_package(raicom_research_assets)
install(DIRECTORY config maps launch rviz DESTINATION share/${PROJECT_NAME})
install(FILES LICENSE DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/simulate_route scripts/recorded_waypoints scripts/rviz_plan_goal scripts/competition_reference scripts/recorded_odom_tf scripts/velocity_center scripts/track_perception scripts/analyze_frame scripts/wheel_odometry scripts/export_trajectory DESTINATION lib/${PROJECT_NAME})
ament_package()
