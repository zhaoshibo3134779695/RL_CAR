#!/usr/bin/env python3
#coding=utf-8
import serial
import rospy
from geometry_msgs.msg import Twist


def _decode_signed_int16(low_byte, high_byte):
    value = (high_byte << 8) | low_byte
    if high_byte & 0x80:
        value -= 0x10000
    return value


def main():
    rospy.init_node('vel_raw_pub')
    vel_publisher = rospy.Publisher("/vel_raw", Twist, queue_size=100)

    rospy.loginfo("opening chassis feedback serial /dev/carserial")
    ser = serial.Serial("/dev/carserial", 115200, timeout=5)

    try:
        while not rospy.is_shutdown():
            raw_data = ser.read(12)
            if len(raw_data) < 12:
                continue

            for i in range(len(raw_data) - 1):
                if raw_data[i] != 0xAA or raw_data[i + 1] != 0xBB:
                    continue
                if i != 0:
                    break

                a1 = raw_data[i + 2]
                a2 = raw_data[i + 3]
                xl = raw_data[i + 4]
                xh = raw_data[i + 5]
                yl = raw_data[i + 6]
                yh = raw_data[i + 7]
                zl = raw_data[i + 8]
                zh = raw_data[i + 9]
                checksum = raw_data[i + 11]

                if ((a1 + a2 + xl + xh + yl + yh + zl + zh) & 0xFF) != checksum:
                    break

                twist = Twist()
                twist.linear.x = _decode_signed_int16(xl, xh) / 1000.0
                twist.linear.y = _decode_signed_int16(yl, yh) / 1000.0
                twist.angular.z = _decode_signed_int16(zl, zh) / 1000.0 * 0.95
                vel_publisher.publish(twist)
                break
    finally:
        ser.close()


if __name__ == '__main__':
    main()
