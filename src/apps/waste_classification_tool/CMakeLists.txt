cmake_minimum_required(VERSION 3.0.2)
project(waste_classification_tool)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  capture_images_simple.py
  capture_and_infer.py
  onnx_infer_output.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
