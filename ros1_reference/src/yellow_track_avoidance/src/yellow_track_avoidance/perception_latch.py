"""Two-state mission latch for the competition perception pipeline.

The latch deliberately owns no ROS resources.  ROS callbacks feed it the
already-existing corridor and disc-bridge status payloads while the node's
existing lock provides thread safety.
"""

import math


ACQUIRE = "ACQUIRE"
LOCKED = "LOCKED"


def _nonnegative_int(value):
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if numeric >= 0 else None


def _finite_xy(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        return math.isfinite(float(value[0])) and math.isfinite(float(value[1]))
    except (TypeError, ValueError, OverflowError):
        return False


class MissionPerceptionLatch:
    """Collect confirmed discs, then latch the complete downstream world model."""

    def __init__(self, expected_disc_count=2, stable_sec=0.5):
        # Keep the public name for launch/API compatibility. It is the minimum
        # count needed to start collection, not an exact final count.
        self.expected_disc_count = max(1, int(expected_disc_count))
        self.stable_sec = max(0.0, float(stable_sec))
        self.scene_generation = 0
        self.reset(scene_generation=0)

    def reset(self, scene_generation=None):
        if scene_generation is not None:
            self.scene_generation = max(0, int(scene_generation))
        self.mode = ACQUIRE
        self.lock_ready = False
        self.lock_ready_since = None
        self.locked_since = None
        self.corridor_locked = False
        self.corridor_cloud_ready = False
        self.corridor_scene_generation = None
        self.bridge_scene_generation = None
        self.confirmed_positions_frozen = False
        self.memory_count = 0
        self.frozen_disc_count = 0
        self.costmap_disc_count = 0
        self.costmap_writes_enabled = False
        self.fixed_transform_available = False
        self.frozen_disc_coordinates_valid = False
        self.received_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0

    def note_received_frame(self):
        self.received_frames += 1

    def note_processed_frame(self):
        self.processed_frames += 1

    def note_skipped_frame(self):
        self.skipped_frames += 1

    def update_corridor(self, payload, now_sec):
        if self.mode == LOCKED:
            return False
        self.corridor_locked = str(payload.get("state", "")) == "locked"
        self.corridor_cloud_ready = bool(
            payload.get("corridor_cloud_ready", False)
        )
        self.corridor_scene_generation = _nonnegative_int(
            payload.get("scene_generation")
        )
        return self._evaluate(float(now_sec))

    def update_bridge(self, payload, now_sec):
        if self.mode == LOCKED:
            return False
        self.confirmed_positions_frozen = bool(
            payload.get("confirmed_positions_frozen", False)
        )
        self.memory_count = _nonnegative_int(payload.get("memory_count")) or 0
        self.costmap_disc_count = (
            _nonnegative_int(payload.get("costmap_disc_count")) or 0
        )
        self.costmap_writes_enabled = bool(
            payload.get("costmap_writes_enabled", False)
        )
        self.fixed_transform_available = bool(
            payload.get("fixed_transform_available", False)
        )
        self.bridge_scene_generation = _nonnegative_int(
            payload.get("scene_generation")
        )

        tracks = payload.get("active_tracks", [])
        if not isinstance(tracks, list):
            tracks = []
        valid_ids = []
        coordinates_valid = True
        for track in tracks:
            if not isinstance(track, dict) or not bool(track.get("confirmed", False)):
                coordinates_valid = False
                continue
            memory_id = _nonnegative_int(track.get("memory_id"))
            if memory_id is None or not _finite_xy(track.get("fixed_xy")):
                coordinates_valid = False
                continue
            valid_ids.append(memory_id)
        self.frozen_disc_count = len(valid_ids)
        self.frozen_disc_coordinates_valid = bool(
            coordinates_valid
            and len(valid_ids) >= self.expected_disc_count
            and len(set(valid_ids)) == len(valid_ids)
        )
        return self._evaluate(float(now_sec))

    def _conditions_ready(self):
        generation_matches = bool(
            self.corridor_scene_generation is not None
            and self.bridge_scene_generation is not None
            and self.corridor_scene_generation == self.scene_generation
            and self.bridge_scene_generation == self.scene_generation
        )
        return bool(
            generation_matches
            and self.corridor_locked
            and self.corridor_cloud_ready
            and self.confirmed_positions_frozen
            and self.memory_count >= self.frozen_disc_count
            and self.frozen_disc_count >= self.expected_disc_count
            and self.costmap_disc_count == self.frozen_disc_count
            and self.costmap_writes_enabled
            and self.fixed_transform_available
            and self.frozen_disc_coordinates_valid
        )

    def _evaluate(self, now_sec):
        if self.mode == LOCKED:
            return False
        ready = self._conditions_ready()
        self.lock_ready = ready
        if not ready:
            self.lock_ready_since = None
            return False
        if self.lock_ready_since is None:
            self.lock_ready_since = now_sec
            if self.stable_sec > 0.0:
                return False
        if now_sec - self.lock_ready_since < self.stable_sec:
            return False
        self.mode = LOCKED
        self.locked_since = now_sec
        return True

    def snapshot(self, now_sec=None):
        skipped_hz = None
        if (
            now_sec is not None
            and self.locked_since is not None
            and float(now_sec) > self.locked_since
        ):
            skipped_hz = self.skipped_frames / (
                float(now_sec) - self.locked_since
            )
        return {
            "perception_mode": self.mode,
            "lock_ready": self.lock_ready,
            "minimum_disc_count": self.expected_disc_count,
            "collection_window_sec": self.stable_sec,
            "corridor_locked": self.corridor_locked,
            "corridor_cloud_ready": self.corridor_cloud_ready,
            "confirmed_positions_frozen": self.confirmed_positions_frozen,
            "memory_count": self.memory_count,
            "frozen_disc_count": self.frozen_disc_count,
            "costmap_disc_count": self.costmap_disc_count,
            "costmap_writes_enabled": self.costmap_writes_enabled,
            "fixed_transform_available": self.fixed_transform_available,
            "frozen_disc_coordinates_valid": (
                self.frozen_disc_coordinates_valid
            ),
            "scene_generation": self.scene_generation,
            "corridor_scene_generation": self.corridor_scene_generation,
            "bridge_scene_generation": self.bridge_scene_generation,
            "lock_ready_since": self.lock_ready_since,
            "locked_since": self.locked_since,
            "received_frames": self.received_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "skipped_frame_rate_hz": skipped_hz,
        }
