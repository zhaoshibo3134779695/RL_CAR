cmake_minimum_required(VERSION 3.0.2)
project(yolo_detector)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs cv_bridge
)

include_directories(
  include/yolo_detector
  ${catkin_INCLUDE_DIRS}
)


add_executable(yolo_json_parser_node
  src/yolo_json_parser_node.cpp
)

target_link_libraries(yolo_json_parser_node
  ${catkin_LIBRARIES}
)

install(TARGETS yolo_json_parser_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/runtime_paths.py
  scripts/onnx_yolo_detector_node.py
  scripts/camera_send.py
  scripts/rA.py
  scripts/rB.py
  scripts/weitingshibieA.py
  scripts/weitingshibieB.py
  scripts/bzshibie.py
  scripts/chepaishibie1.py
  scripts/chepaishibie2.py
  scripts/chepaishibie3.py
  scripts/ddczuizhong.py
  scripts/summary_rAB.py
  scripts/summary_ddcAB.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
