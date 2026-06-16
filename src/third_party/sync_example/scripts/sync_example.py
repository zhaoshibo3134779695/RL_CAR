#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import message_filters
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry

# 全局变量，用于保存同步后的数据
laser_scan_data = None
imu_data = None
odom_data = None

def callback(laser_scan, imu, odom):
    global laser_scan_data, imu_data, odom_data
    # 更新全局变量
    laser_scan_data = laser_scan
    imu_data = imu
    odom_data = odom
    # rospy.loginfo("Received synchronized data")

def listener():
    rospy.init_node('sync_example', anonymous=True)

    # 创建消息订阅者
    laser_scan_sub = message_filters.Subscriber('scan', LaserScan)
    imu_sub = message_filters.Subscriber('/wit/imu', Imu)
    odom_sub = message_filters.Subscriber('odom_raw', Odometry)

    # 创建同步器
    ts = message_filters.ApproximateTimeSynchronizer(
        [laser_scan_sub, imu_sub, odom_sub], 
        queue_size=10, 
        slop=0.1
    )
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
