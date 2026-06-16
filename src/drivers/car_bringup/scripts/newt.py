#!/usr/bin/env python3
# coding=utf-8

import atexit
import threading
import time
import rospy
import serial
from geometry_msgs.msg import Twist

SERIAL_PORT = "/dev/carserial"
BAUDRATE = 115200
CMD_TIMEOUT = 0.3
WATCHDOG_HZ = 20.0

ser = None
ser_lock = threading.Lock()
last_cmd_time = 0.0
stopped = True


def _int16_bytes(value):
    value = int(value) & 0xFFFF
    return bytes([(value >> 8) & 0xFF]), bytes([value & 0xFF])


def send_init_frame():
    if ser is None or not ser.is_open:
        return
    with ser_lock:
        ser.write(b"\x11\x00\x00\x00\x00\x00\x00\x00\x00")
        ser.flush()


def send_motion_frame(linear_x, linear_y, angular_z):
    x_hi, x_lo = _int16_bytes(linear_x * 1000.0)
    y_hi, y_lo = _int16_bytes(linear_y * 1000.0)
    th_hi, th_lo = _int16_bytes(angular_z * 1000.0)

    frame = b"".join([
        b"\xAA",
        b"\xBB",
        b"\x0A",
        b"\x12",
        b"\x02",
        x_lo,
        x_hi,
        y_lo,
        y_hi,
        th_lo,
        th_hi,
        b"\x00",
    ])

    with ser_lock:
        ser.write(frame)
        ser.flush()


def stop_car(force=False):
    global stopped
    if stopped and not force:
        return
    send_motion_frame(0.0, 0.0, 0.0)
    stopped = True


def cmd_callback(msg):
    global last_cmd_time, stopped
    send_motion_frame(msg.linear.x, msg.linear.y, msg.angular.z)
    last_cmd_time = time.monotonic()
    stopped = (
        abs(msg.linear.x) < 1e-6
        and abs(msg.linear.y) < 1e-6
        and abs(msg.angular.z) < 1e-6
    )


def watchdog_loop():
    rate = rospy.Rate(WATCHDOG_HZ)
    while not rospy.is_shutdown():
        if time.monotonic() - last_cmd_time > CMD_TIMEOUT:
            stop_car()
        rate.sleep()


def shutdown_hook():
    try:
        stop_car(force=True)
        send_init_frame()
    except Exception:
        pass
    if ser is not None and ser.is_open:
        try:
            ser.close()
        except Exception:
            pass


def main():
    global ser, last_cmd_time

    rospy.init_node("cmd_vel_listener")
    rospy.loginfo("opening chassis serial: %s", SERIAL_PORT)

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    send_init_frame()
    stop_car(force=True)

    last_cmd_time = time.monotonic()

    atexit.register(shutdown_hook)
    rospy.on_shutdown(shutdown_hook)
    rospy.Subscriber("/cmd_vel", Twist, cmd_callback, queue_size=10)

    watchdog = threading.Thread(target=watchdog_loop, daemon=True)
    watchdog.start()

    rospy.loginfo("chassis cmd bridge ready with watchdog timeout %.2fs", CMD_TIMEOUT)
    rospy.spin()


if __name__ == "__main__":
    main()
