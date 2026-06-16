cmake_minimum_required(VERSION 3.0.2)
project(lidar_loc)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    std_srvs
    nav_msgs
    sensor_msgs
    geometry_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    cv_bridge
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(lidar_loc
  src/lidar_loc.cpp
)
add_dependencies(lidar_loc ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(lidar_loc
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

#############
## Install ##
#############

# install(TARGETS lidar_loc
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

