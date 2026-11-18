#!/usr/bin/env python3

import math
import threading
import unittest
from unittest import mock

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from yellow_track_avoidance.perception_latch import (
    ACQUIRE,
    LOCKED,
    MissionPerceptionLatch,
)
from yellow_track_avoidance.ros_node import YellowTrackAvoidanceNode


def corridor_status(generation=0, locked=True, cloud_ready=True):
    return {
        "scene_generation": generation,
        "state": "locked" if locked else "collecting",
        "corridor_cloud_ready": cloud_ready,
    }


def bridge_status(
    generation=0,
    disc_count=2,
    frozen=True,
    writes=True,
    transform=True,
):
    return {
        "scene_generation": generation,
        "confirmed_positions_frozen": frozen,
        "memory_count": disc_count,
        "costmap_disc_count": disc_count,
        "costmap_writes_enabled": writes,
        "fixed_transform_available": transform,
        "active_tracks": [
            {
                "memory_id": index + 1,
                "confirmed": True,
                "fixed_xy": [1.0 + index * 0.3, (-1.0) ** index * 0.2],
            }
            for index in range(disc_count)
        ],
    }


class MissionPerceptionLatchTest(unittest.TestCase):
    def setUp(self):
        self.latch = MissionPerceptionLatch(
            expected_disc_count=2, stable_sec=0.5
        )

    def ready_at(self, stamp):
        self.latch.update_corridor(corridor_status(), stamp)
        return self.latch.update_bridge(bridge_status(), stamp)

    def test_requires_continuous_half_second_downstream_confirmation(self):
        self.assertFalse(self.ready_at(10.0))
        self.assertEqual(self.latch.mode, ACQUIRE)
        self.assertTrue(self.latch.lock_ready)

        self.assertFalse(
            self.latch.update_corridor(corridor_status(), 10.49)
        )
        self.assertEqual(self.latch.mode, ACQUIRE)
        self.assertTrue(
            self.latch.update_bridge(bridge_status(), 10.50)
        )
        self.assertEqual(self.latch.mode, LOCKED)
        self.assertEqual(self.latch.locked_since, 10.50)

    def test_collects_additional_confirmed_discs_without_restarting_window(self):
        self.assertFalse(self.ready_at(10.0))
        self.assertFalse(
            self.latch.update_bridge(bridge_status(disc_count=3), 10.25)
        )
        self.assertTrue(
            self.latch.update_bridge(bridge_status(disc_count=3), 10.50)
        )
        self.assertEqual(self.latch.mode, LOCKED)
        self.assertEqual(self.latch.frozen_disc_count, 3)

    def test_one_disc_or_costmap_mismatch_never_locks(self):
        self.latch.update_corridor(corridor_status(), 1.0)
        self.latch.update_bridge(bridge_status(disc_count=1), 1.0)
        self.latch.update_bridge(bridge_status(disc_count=1), 2.0)
        self.assertEqual(self.latch.mode, ACQUIRE)

        payload = bridge_status()
        payload["costmap_disc_count"] = 1
        self.latch.update_bridge(payload, 3.0)
        self.latch.update_bridge(payload, 4.0)
        self.assertEqual(self.latch.mode, ACQUIRE)

    def test_invalid_or_duplicate_fixed_coordinates_never_locks(self):
        self.latch.update_corridor(corridor_status(), 1.0)
        payload = bridge_status()
        payload["active_tracks"][1]["fixed_xy"] = [math.nan, 0.0]
        self.latch.update_bridge(payload, 1.0)
        self.latch.update_bridge(payload, 2.0)
        self.assertEqual(self.latch.mode, ACQUIRE)

        payload = bridge_status()
        payload["active_tracks"][1]["memory_id"] = 1
        self.latch.update_bridge(payload, 3.0)
        self.latch.update_bridge(payload, 4.0)
        self.assertEqual(self.latch.mode, ACQUIRE)

    def test_status_glitch_restarts_stability_window(self):
        self.assertFalse(self.ready_at(1.0))
        self.assertFalse(
            self.latch.update_corridor(
                corridor_status(cloud_ready=False), 1.4
            )
        )
        self.assertIsNone(self.latch.lock_ready_since)
        self.assertFalse(
            self.latch.update_corridor(corridor_status(), 1.5)
        )
        self.assertFalse(self.latch.update_bridge(bridge_status(), 1.99))
        self.assertTrue(self.latch.update_bridge(bridge_status(), 2.0))

    def test_locked_is_mission_latch_until_higher_generation_reset(self):
        self.ready_at(1.0)
        self.assertTrue(self.latch.update_bridge(bridge_status(), 1.5))
        self.assertEqual(self.latch.mode, LOCKED)

        self.assertFalse(
            self.latch.update_corridor(
                corridor_status(locked=False, cloud_ready=False), 5.0
            )
        )
        self.assertFalse(
            self.latch.update_bridge(
                bridge_status(disc_count=0, writes=False), 5.0
            )
        )
        self.assertEqual(self.latch.mode, LOCKED)

        self.latch.reset(scene_generation=1)
        self.assertEqual(self.latch.mode, ACQUIRE)
        self.assertFalse(
            self.latch.update_corridor(corridor_status(generation=0), 6.0)
        )
        self.assertFalse(
            self.latch.update_bridge(bridge_status(generation=0), 7.0)
        )
        self.assertFalse(self.latch.lock_ready)

    def test_frame_counters_distinguish_processed_and_skipped(self):
        self.latch.note_received_frame()
        self.latch.note_processed_frame()
        self.latch.note_received_frame()
        self.latch.note_skipped_frame()
        snapshot = self.latch.snapshot()
        self.assertEqual(snapshot["received_frames"], 2)
        self.assertEqual(snapshot["processed_frames"], 1)
        self.assertEqual(snapshot["skipped_frames"], 1)

    @mock.patch("yellow_track_avoidance.ros_node.rospy.Time.now")
    def test_locked_image_callback_returns_before_format_and_cv_bridge(
        self, mocked_now
    ):
        mocked_now.return_value.to_sec.return_value = 20.0
        node = YellowTrackAvoidanceNode.__new__(YellowTrackAvoidanceNode)
        node.lock = threading.RLock()
        node.last_image_received_sec = None
        node.mission_latch_enabled = True
        node.perception_latch = MissionPerceptionLatch()
        node.perception_latch.mode = LOCKED
        node._resolution_valid = mock.Mock(
            side_effect=AssertionError("format check must be skipped")
        )
        node.bridge = mock.Mock()

        node._image_callback(Image())

        node._resolution_valid.assert_not_called()
        node.bridge.imgmsg_to_cv2.assert_not_called()
        self.assertEqual(node.last_image_received_sec, 20.0)
        self.assertEqual(node.perception_latch.received_frames, 1)
        self.assertEqual(node.perception_latch.skipped_frames, 1)

    @mock.patch("yellow_track_avoidance.ros_node.rospy.Time.now")
    def test_photometry_wait_returns_before_cv_bridge(self, mocked_now):
        mocked_now.return_value.to_sec.return_value = 21.0
        node = YellowTrackAvoidanceNode.__new__(YellowTrackAvoidanceNode)
        node.lock = threading.RLock()
        node.last_image_received_sec = None
        node.mission_latch_enabled = False
        node.require_photometry_ready = True
        node.photometry_ready = False
        node.photometry_wait_frames = 0
        node._resolution_valid = mock.Mock(
            side_effect=AssertionError("format check must be skipped")
        )
        node.bridge = mock.Mock()

        node._image_callback(Image())

        node._resolution_valid.assert_not_called()
        node.bridge.imgmsg_to_cv2.assert_not_called()
        self.assertEqual(node.photometry_wait_frames, 1)

    def test_photometry_ready_keeps_final_exposure_far_disc_evidence(self):
        node = YellowTrackAvoidanceNode.__new__(YellowTrackAvoidanceNode)
        node.lock = threading.RLock()
        node.photometry_ready = False
        node.pipeline = mock.Mock()

        node._photometry_ready_callback(Bool(data=True))

        node.pipeline.reset_perception_history.assert_called_once_with(
            preserve_far_red_evidence=True
        )

if __name__ == "__main__":
    unittest.main()
