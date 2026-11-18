#!/usr/bin/env python3
"""Audit or apply fixed RGB photometry through configurable ROS services.

The service classes are resolved at runtime so this package does not depend on
one camera vendor's message package.  The camera driver must already be
running.  This tool exits after one audit/apply/verify transaction.
"""

import argparse
import sys

import rospy
import rosservice
import yaml


def _service_call(name, timeout_sec, data=None):
    rospy.wait_for_service(name, timeout=timeout_sec)
    service_class = rosservice.get_service_class_by_name(name)
    if service_class is None:
        raise RuntimeError("cannot resolve service type for %s" % name)
    request = service_class._request_class()
    if data is not None:
        if not hasattr(request, "data"):
            raise RuntimeError("service %s request has no data field" % name)
        request.data = data
    response = rospy.ServiceProxy(name, service_class)(request)
    if hasattr(response, "success") and not response.success:
        raise RuntimeError(
            "%s rejected request: %s" % (name, getattr(response, "message", ""))
        )
    return response


def _read_value(name, timeout_sec):
    response = _service_call(name, timeout_sec)
    if not hasattr(response, "data"):
        raise RuntimeError("service %s response has no data field" % name)
    return response.data


def _load(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    section = config.get("camera_photometry")
    if not isinstance(section, dict):
        raise RuntimeError("camera_photometry section is missing from %s" % path)
    return section


def main():
    parser = argparse.ArgumentParser(
        description="Audit/apply fixed camera exposure and gain using ROS services"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="read current values without changing camera state",
    )
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("yellow_track_orbbec_photometry", anonymous=False)
    audit_only = bool(args.audit_only or rospy.get_param("~audit_only", False))

    cfg = _load(args.config)
    timeout = float(cfg.get("service_timeout_sec", 5.0))
    service_keys = (
        "set_auto_exposure_service",
        "get_auto_exposure_service",
        "set_exposure_service",
        "get_exposure_service",
        "set_gain_service",
        "get_gain_service",
    )
    missing = [key for key in service_keys if not cfg.get(key)]
    if missing:
        raise RuntimeError("missing camera_photometry keys: %s" % ", ".join(missing))

    before = {
        "auto_exposure": bool(_read_value(cfg["get_auto_exposure_service"], timeout)),
        "exposure": int(_read_value(cfg["get_exposure_service"], timeout)),
        "gain": int(_read_value(cfg["get_gain_service"], timeout)),
    }
    rospy.loginfo("RGB photometry before: %s", before)
    if audit_only:
        return 0

    requested = {
        "auto_exposure": bool(cfg.get("auto_exposure", False)),
        "exposure": int(cfg["exposure"]),
        "gain": int(cfg["gain"]),
    }
    _service_call(cfg["set_auto_exposure_service"], timeout, requested["auto_exposure"])
    _service_call(cfg["set_exposure_service"], timeout, requested["exposure"])
    _service_call(cfg["set_gain_service"], timeout, requested["gain"])

    after = {
        "auto_exposure": bool(_read_value(cfg["get_auto_exposure_service"], timeout)),
        "exposure": int(_read_value(cfg["get_exposure_service"], timeout)),
        "gain": int(_read_value(cfg["get_gain_service"], timeout)),
    }
    rospy.loginfo("RGB photometry after: %s", after)
    if after != requested:
        raise RuntimeError("camera verification mismatch: requested=%s actual=%s" % (requested, after))
    if not bool(cfg.get("white_balance_supported", False)):
        rospy.logwarn("RGB white-balance service is marked unsupported; no write attempted")
    rospy.loginfo("fixed RGB photometry verified")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (rospy.ROSException, rospy.ServiceException, RuntimeError, KeyError, ValueError) as exc:
        rospy.logerr("camera photometry failed: %s", exc)
        sys.exit(1)
