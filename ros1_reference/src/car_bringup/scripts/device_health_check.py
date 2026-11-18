#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import Twist
import time
import threading

# 打印颜色定义
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class DeviceHealthCheck:
    def __init__(self):
        # 初始化节点
        rospy.init_node('quality_inspection_checker', anonymous=True)
        
        # 创建一个字典来保存每个组件的状态
        self.status = {
            "相机": False,
            "IMU": False,
            "激光雷达": False,
            "底盘": False  # 这个状态现在由 /vel_raw 的回调函数更新
        }

        # 从参数服务器获取话题名称，如果未设置则使用默认值
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/color/image_raw')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu0')
        self.lidar_topic = rospy.get_param('~lidar_topic', '/scan')
        self.chassis_command_topic = rospy.get_param('~chassis_command_topic', '/cmd_vel') # 指令话题
        self.chassis_feedback_topic = rospy.get_param('~chassis_feedback_topic', '/vel_raw')   # 反馈话题

        # 订阅传感器和底盘反馈话题
        rospy.Subscriber(self.camera_topic, Image, self.callback_factory("相机"))
        rospy.Subscriber(self.imu_topic, Imu, self.callback_factory("IMU"))
        rospy.Subscriber(self.lidar_topic, LaserScan, self.callback_factory("激光雷达"))
        rospy.Subscriber(self.chassis_feedback_topic, Twist, self.callback_factory("底盘"))

        # 创建一个发布者来向底盘发送指令
        self.chassis_pub = rospy.Publisher(self.chassis_command_topic, Twist, queue_size=1)

    def callback_factory(self, component_name):
        """ 使用工厂模式为每个回调函数创建独立的闭包，避免变量覆盖 """
        def callback(msg):
            # 只有在状态为False时才更新并打印信息，避免刷屏
            if not self.status[component_name]:
                rospy.loginfo(bcolors.OKGREEN + "✅ %s 数据接收成功!" % component_name + bcolors.ENDC)
                self.status[component_name] = True
        return callback

    def send_chassis_command(self):
        """ 
        主动等待/cmd_vel的订阅者连接，然后发布一条前进指令来触发反馈。
        """
        try:
            rospy.loginfo(bcolors.OKBLUE + "正在等待底盘控制节点订阅 (%s)..." % self.chassis_command_topic + bcolors.ENDC)
            
            # 等待最多5秒，直到找到一个订阅者
            wait_for_subscriber_start_time = rospy.Time.now()
            subscriber_found = False
            while not rospy.is_shutdown() and (rospy.Time.now() - wait_for_subscriber_start_time < rospy.Duration(5.0)):
                if self.chassis_pub.get_num_connections() > 0:
                    subscriber_found = True
                    break
                rospy.sleep(0.1)

            if subscriber_found:
                rospy.loginfo(bcolors.OKBLUE + "已连接到底盘控制节点。正在发送前进指令持续1秒..." + bcolors.ENDC)
                
                move_cmd = Twist()
                move_cmd.linear.x = 0.2
                stop_cmd = Twist()
                
                start_time = rospy.Time.now()
                rate = rospy.Rate(10) 
                while not rospy.is_shutdown() and rospy.Time.now() - start_time < rospy.Duration(1.0):
                    self.chassis_pub.publish(move_cmd)
                    rate.sleep()
                
                self.chassis_pub.publish(stop_cmd)
                
                rospy.loginfo(bcolors.OKGREEN + "✅ 底盘移动指令已发送至 %s。" % self.chassis_command_topic + bcolors.ENDC)
            else:
                rospy.logwarn(bcolors.WARNING + "⚠️  超时！在5秒内未找到 (%s) 的订阅者。无法测试底盘移动。" % self.chassis_command_topic + bcolors.ENDC)
        except Exception as e:
            rospy.logerr(bcolors.FAIL + "❌ 发送底盘指令时出错: %s" % e + bcolors.ENDC)


    def run_check(self):
        """ 运行检查并打印最终报告 """
        # 增加安全提示和确认环节
        print("\n" + bcolors.WARNING + "="*60 + bcolors.ENDC)
        print(bcolors.BOLD + "警告: 测试程序将驱动底盘电机转动。" + bcolors.ENDC)
        print(bcolors.BOLD + "请在测试时确认车辆已经离地，或处于绝对安全的环境中！" + bcolors.ENDC)
        print(bcolors.WARNING + "="*60 + bcolors.ENDC)
        
        # 在新的一行提示用户按回车
        print(bcolors.OKBLUE + "\n准备就绪后，请按【回车键】开始底盘控制检查..." + bcolors.ENDC)

        # raw_input 在 Python 2.x (ROS常用) 中使用
        try:
            raw_input() # 等待用户输入
        except NameError:
            # input 在 Python 3.x 中使用, 作为备用
            input() # 等待用户输入

        rospy.loginfo(bcolors.HEADER + "出厂前质量检查已开始..." + bcolors.ENDC)

        # 在一个单独的线程中发送底盘移动指令，以防阻塞主线程
        chassis_thread = threading.Thread(target=self.send_chassis_command)
        chassis_thread.start()

        rospy.loginfo(bcolors.OKBLUE + "正在监听所有传感器和底盘反馈话题，持续10秒..." + bcolors.ENDC)
        
        # 等待所有数据或超时
        timeout = rospy.Time.now() + rospy.Duration(10)
        while not rospy.is_shutdown() and rospy.Time.now() < timeout:
            if all(self.status.values()):
                rospy.loginfo(bcolors.OKGREEN + "所有项目提前检查完毕!" + bcolors.ENDC)
                break
            rospy.sleep(0.5)
        
        chassis_thread.join() # 等待指令发送线程结束

        # 打印最终报告
        print("\n" + "="*50)
        print(bcolors.BOLD + "          出厂前质量检查报告" + bcolors.ENDC)
        print("="*50)
        
        all_ok = True
        for component, is_ok in self.status.items():
            status_text = bcolors.OKGREEN + "✅ 通过" if is_ok else bcolors.FAIL + "❌ 失败"
            print("%-25s: %s" % (component + " 状态", status_text + bcolors.ENDC))
            if not is_ok:
                all_ok = False
        
        print("="*50)

        if all_ok:
            print(bcolors.OKGREEN + bcolors.BOLD + "所有项目检查通过，车辆符合出厂标准！" + bcolors.ENDC)
        else:
            print(bcolors.FAIL + bcolors.BOLD + "一个或多个项目未能通过检查，请检修后重新测试。" + bcolors.ENDC)
        
        # 完成测试后关闭节点
        rospy.signal_shutdown("质量检查完成。")


if __name__ == '__main__':
    try:
        checker = DeviceHealthCheck()
        checker.run_check()
    except rospy.ROSInterruptException:
        print("ROS 节点被中断。")

