#!/usr/bin/env python3
"""Publish the validated static reference route; this node sends no velocity."""

import json
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String

from yellow_track_avoidance.map_navigation import validate_route


def main():
    rospy.init_node("yellow_track_map_route")
    route_file = rospy.get_param("~route_file")
    topic = rospy.get_param(
        "~path_topic", "/yellow_track_avoidance/map_reference_path"
    )
    status_topic = rospy.get_param(
        "~status_topic", "/yellow_track_avoidance/map_route_status"
    )
    route, _, _, _, _, points, headings, report = validate_route(route_file)
    if not report["route_valid_for_all_recorded_vehicles"]:
        raise RuntimeError("reference route footprint validation failed")
    publisher = rospy.Publisher(topic, Path, queue_size=1, latch=True)
    status_publisher = rospy.Publisher(status_topic, String, queue_size=1, latch=True)
    message = Path()
    message.header.frame_id = str(route["frame_id"])
    message.header.stamp = rospy.Time.now()
    for (x, y), yaw in zip(points, headings):
        pose = PoseStamped()
        pose.header = message.header
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(0.5 * float(yaw))
        pose.pose.orientation.w = math.cos(0.5 * float(yaw))
        message.poses.append(pose)
    publisher.publish(message)
    status_publisher.publish(
        String(data=json.dumps(report, ensure_ascii=False, allow_nan=False))
    )
    rospy.loginfo(
        "loaded 5x4 m map route: points=%d avoidance_until_x=%.2f motion_output=none",
        len(points), float(route["avoidance"]["handoff_x_m"]),
    )
    rospy.spin()


if __name__ == "__main__":
    main()
