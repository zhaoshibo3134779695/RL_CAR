cmake_minimum_required(VERSION 3.0.2)
project(xy_path_follower)

## 使用 C++14
add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2
  std_msgs
  roslib
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    geometry_msgs
    nav_msgs
    tf2_ros
    tf2_geometry_msgs
    tf2
    std_msgs
    roslib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(xy_path_follower src/xy_path_follower.cpp)
target_link_libraries(xy_path_follower ${catkin_LIBRARIES})
add_dependencies(xy_path_follower ${catkin_EXPORTED_TARGETS})

add_executable(yaw_pid_controller
  src/yaw_pid_controller.cpp
)

target_link_libraries(yaw_pid_controller
  ${catkin_LIBRARIES}
)

add_dependencies(yaw_pid_controller
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

