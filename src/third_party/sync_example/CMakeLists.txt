cmake_minimum_required(VERSION 3.0.2)
# 包名必须和package.xml中的name一致
project(sync_example)

# 查找catkin和必要依赖（rospy是Python节点必需的）
find_package(catkin REQUIRED COMPONENTS
  rospy
)

# 声明catkin包
catkin_package(
  CATKIN_DEPENDS rospy
)

# 包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 关键：安装Python脚本到ROS可执行路径
# 确保scripts/sync_example.py是你的节点文件路径
catkin_install_python(PROGRAMS
  scripts/sync_example.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
