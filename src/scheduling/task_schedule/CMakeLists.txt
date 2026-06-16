cmake_minimum_required(VERSION 3.0.2)
project(task_schedule)

## 使用 C++11（或更新版本）
#add_compile_options(-std=c++11)

## 找到需要的 catkin 包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
  geometry_msgs
  std_msgs
  roslib
)

## catkin 包配置
catkin_package(
  CATKIN_DEPENDS roscpp tf geometry_msgs std_msgs roslib
)

## 头文件搜索路径
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 生成可执行文件
add_executable(task_schedule src/task_schedule.cpp)

## 链接 ROS 库
target_link_libraries(task_schedule
  ${catkin_LIBRARIES}
)

add_executable(tf_info src/tf_info.cpp)
target_link_libraries(tf_info
  ${catkin_LIBRARIES}
)

add_executable(vel_ctrl src/vel_ctrl.cpp)
target_link_libraries(vel_ctrl
  ${catkin_LIBRARIES}
)
