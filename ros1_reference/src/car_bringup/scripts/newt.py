#!/usr/bin/env python3
#coding=utf-8

import serial
import roslib
import rospy
from geometry_msgs.msg import Twist, Quaternion, TransformStamped
from math import sqrt, atan2, pow
from threading import Lock, Thread
from nav_msgs.msg import Odometry
import time

last_cmd_time = None
stop_sent = True
serial_lock = Lock()
allow_rotation = True
max_angular_z = 0.50
stop_timeout = 0.35


def int16_hex(value):
    value = max(-32768, min(32767, int(value)))
    if value < 0:
        value = value & 0xffff
    data = "{:04x}".format(value)
    return bytes.fromhex(data[:2]), bytes.fromhex(data[2:])


def send_cmd(cmd_twist_x, cmd_twist_y, cmd_twist_rotation):
    th_1 = cmd_twist_rotation * 1000
    x_1 = cmd_twist_x * 1000
    y_1 = cmd_twist_y * 1000

    x_hi, x_lo = int16_hex(x_1)
    y_hi, y_lo = int16_hex(y_1)
    th_hi, th_lo = int16_hex(th_1)

    with serial_lock:
        ser.write(b'\xAA')
        time.sleep(0.0004)
        ser.write(b'\xBB')
        time.sleep(0.0004)
        ser.write(b'\x0A')
        time.sleep(0.0004)
        ser.write(b'\x12')
        time.sleep(0.0004)
        ser.write(b'\x02')
        time.sleep(0.0004)
        ser.write(x_lo)
        time.sleep(0.0004)
        ser.write(x_hi)
        time.sleep(0.0004)
        ser.write(y_lo)
        time.sleep(0.0004)
        ser.write(y_hi)
        time.sleep(0.0004)
        ser.write(th_lo)
        time.sleep(0.0004)
        ser.write(th_hi)
        time.sleep(0.0004)
        ser.write(b'\x00')
        time.sleep(0.0004)


def callback(msg):
    global last_cmd_time, stop_sent
    cmd_twist_rotation = msg.angular.z
    cmd_twist_x = msg.linear.x
    cmd_twist_y = msg.linear.y
    if not allow_rotation:
        cmd_twist_rotation = 0.0
    elif max_angular_z > 0.0:
        cmd_twist_rotation = max(-max_angular_z, min(max_angular_z, cmd_twist_rotation))
    last_cmd_time = rospy.Time.now()
    stop_sent = False
    send_cmd(cmd_twist_x, cmd_twist_y, cmd_twist_rotation)


def watchdog(event):
    global stop_sent
    if last_cmd_time is None or stop_sent:
        return
    if (rospy.Time.now() - last_cmd_time).to_sec() > stop_timeout:
        send_cmd(0.0, 0.0, 0.0)
        stop_sent = True


def shutdown_stop():
    try:
        send_cmd(0.0, 0.0, 0.0)
    except Exception:
        pass

def listener():
    global allow_rotation, max_angular_z, stop_timeout
    rospy.init_node('cmd_vel_listener')
    allow_rotation = rospy.get_param("~allow_rotation", True)
    max_angular_z = rospy.get_param("~max_angular_z", 0.50)
    stop_timeout = rospy.get_param("~stop_timeout", 0.35)
    rospy.Subscriber("/cmd_vel", Twist, callback)
    rospy.Timer(rospy.Duration(0.1), watchdog)
    rospy.on_shutdown(shutdown_stop)
    # rospy.Rate(10)
    rospy.spin()

if __name__ == '__main__':
    print('serial')
    ser = serial.Serial("/dev/carserial",115200,timeout = 5)
    if ser.is_open:
        print('open_success')
    ser.write(b'\x11')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    ser.write(b'\x00')
    time.sleep(0.001)
    print('init_success')
    listener()
