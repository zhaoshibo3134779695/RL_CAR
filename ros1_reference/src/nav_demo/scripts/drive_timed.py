#!/usr/bin/env python3
"""Publish a linear velocity for an exact wall-clock duration."""

import time

import rospy
from geometry_msgs.msg import Twist


def stop(publisher):
    command = Twist()
    for _ in range(10):
        publisher.publish(command)
        rospy.sleep(0.02)


def main():
    rospy.init_node("drive_timed")
    distance = rospy.get_param("~distance", 1.0)
    speed = rospy.get_param("~speed", 0.20)
    axis = rospy.get_param("~axis", "x").lower()
    rate_hz = rospy.get_param("~rate", 30.0)

    if abs(distance) < 0.001 or speed <= 0.0:
        rospy.logerr("distance must be non-zero and speed must be positive")
        return
    if axis not in ("x", "y"):
        rospy.logerr("axis must be either 'x' or 'y'")
        return

    publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.on_shutdown(lambda: stop(publisher))
    wait_started = time.monotonic()
    while publisher.get_num_connections() == 0 and not rospy.is_shutdown():
        if time.monotonic() - wait_started > 5.0:
            rospy.logerr("No /cmd_vel subscriber connected")
            return
        rospy.sleep(0.02)

    duration = abs(distance) / speed
    signed_speed = speed if distance > 0.0 else -speed
    command = Twist()
    if axis == "x":
        command.linear.x = signed_speed
    else:
        command.linear.y = signed_speed
    rate = rospy.Rate(rate_hz)
    started = time.monotonic()
    rospy.loginfo(
        "Driving along %s at %.3f m/s for %.3f s (nominal %.3f m)",
        axis, signed_speed, duration, distance)
    while not rospy.is_shutdown() and time.monotonic() - started < duration:
        publisher.publish(command)
        rate.sleep()

    stop(publisher)
    rospy.loginfo("Straight test command complete")


if __name__ == "__main__":
    main()
