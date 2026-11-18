#!/usr/bin/env python3
"""Start, stop, or report a labelled disc-acceptance capture."""

import argparse

import rospy
from std_srvs.srv import Trigger


PARAMETER_ROOT = "/disc_acceptance_recorder"


def _boolean(text):
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _set(name, value):
    rospy.set_param(PARAMETER_ROOT + "/" + name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="start one labelled static scene")
    start.add_argument("--scene-name", required=True)
    start.add_argument("--disc-present", required=True, type=_boolean)
    start.add_argument("--expected-disc-count", type=int, default=1)
    start.add_argument(
        "--position",
        choices=["left", "center", "right", "unspecified"],
        default="unspecified",
    )
    start.add_argument(
        "--distance",
        choices=["near", "middle", "far", "unspecified"],
        default="unspecified",
    )
    start.add_argument(
        "--occlusion",
        default="0",
        help="fraction such as 0.2 or percentage such as 20",
    )
    start.add_argument(
        "--reflection", default="normal", help="normal/low/high or a custom label"
    )
    start.add_argument("--near-yellow-line", type=_boolean, default=False)
    subparsers.add_parser("stop", help="stop capture and regenerate report")
    subparsers.add_parser("report", help="regenerate report from existing CSV")
    args = parser.parse_args()

    rospy.init_node("disc_capture_control", anonymous=True, disable_signals=True)
    if args.command == "start":
        _set("scene_name", args.scene_name)
        _set("disc_present", args.disc_present)
        _set("expected_disc_count", max(0, args.expected_disc_count))
        _set("position_left_center_right", args.position)
        _set("distance_near_middle_far", args.distance)
        _set("occlusion_level", str(args.occlusion))
        _set("reflection_level", args.reflection)
        _set("near_yellow_line", args.near_yellow_line)
    service_name = PARAMETER_ROOT + "/" + args.command
    rospy.wait_for_service(service_name, timeout=5.0)
    response = rospy.ServiceProxy(service_name, Trigger)()
    print(response.message)
    if not response.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
