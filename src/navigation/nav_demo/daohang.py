#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler

class AutoNavigation:
    def __init__(self):
        # 初始化节点
        rospy.init_node('auto_navigation_node', anonymous=False)
        
        # 连接到 move_base 的 action 服务器
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("等待 move_base 动作服务器...")
        server_timeout = rospy.Duration(rospy.get_param("~server_timeout", 15.0))
        if not self.client.wait_for_server(server_timeout):
            raise RuntimeError("等待 move_base 动作服务器超时")
        rospy.loginfo("连接成功，开始自主导航任务。")
        self.goal_timeout = rospy.Duration(rospy.get_param("~goal_timeout", 120.0))

        # 定义目标点列表 (可以根据需要修改)
        # 每个元素是一个字典，包含位置(x, y)和偏航角(yaw)
        self.goals = [
            # 目标点 1
            {'position': Point(0.8114241112468027, 0.32689150073395473, 0.0), 'yaw': 0.9833520883043403},
            # 目标点 2
            {'position': Point(1.2850652674702252, 1.0319850964810269, 0.0), 'yaw':  0.9774427096358757},  # 90度
            # 目标点 3
            {'position': Point(2.0775240264458494,  0.10301134348342367, 0.0), 'yaw': 0.5538309571294644},
            # 可以继续添加更多目标点...
        ]

    def create_goal(self, position, yaw):
        """创建一个 MoveBaseGoal 消息"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"  # 通常导航的目标点在 map 坐标系下
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # 设置位置
        goal.target_pose.pose.position = position
        
        # 设置姿态 (从偏航角转换)
        q = quaternion_from_euler(0, 0, yaw)  # roll, pitch, yaw
        goal.target_pose.pose.orientation = Quaternion(*q)
        
        return goal

    def run(self):
        """依次发送所有目标点"""
        for i, goal_params in enumerate(self.goals):
            rospy.loginfo(f"正在前往目标点 {i+1}: 位置({goal_params['position'].x}, {goal_params['position'].y})")
            
            goal = self.create_goal(goal_params['position'], goal_params['yaw'])
            self.client.send_goal(goal)
            
            # 等待当前目标点完成
            finished = self.client.wait_for_result(self.goal_timeout)
            if not finished:
                self.client.cancel_goal()
                rospy.logwarn(f"目标点 {i+1} 执行失败或超时")
                continue
            
            state = self.client.get_state()
            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo(f"目标点 {i+1} 到达成功！")
            else:
                rospy.logwarn(f"目标点 {i+1} 到达失败，状态码: {state}")
            
            # 可选：两个目标点之间稍作停顿
            rospy.sleep(1.0)
        
        rospy.loginfo("所有目标点执行完毕。")

if __name__ == '__main__':
    try:
        navigator = AutoNavigation()
        navigator.run()
    except RuntimeError as e:
        rospy.logerr(str(e))
    except rospy.ROSInterruptException:
        rospy.loginfo("导航任务被中断。")
