cmake_minimum_required(VERSION 3.0.2)
project(fire_detection_tool)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  capture_images_simple.py
  fire_detection_pipeline1.py
  fire_detection_pipeline2.py
  fire_detection_pipeline3.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
