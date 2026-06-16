cmake_minimum_required(VERSION 3.0.2)
project(voice_broadcast_tool)

find_package(catkin REQUIRED)

catkin_package()

catkin_install_python(PROGRAMS
  piper_monitor.py
  speak.py
  speak_cp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
