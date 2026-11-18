"""Short-lived multi-object tracking for red-white disks."""

import math
from typing import Dict, List

from .model import DiskDetection, TrackedDisk


class DiskTracker:
    def __init__(self, config):
        self.cfg = config["disk_detection"]
        self.tracks: Dict[int, TrackedDisk] = {}
        self.next_id = 1

    @staticmethod
    def _distance(a, b) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def update(self, detections: List[DiskDetection], stamp_sec: float) -> List[TrackedDisk]:
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        pairs = []
        for track_id, track in self.tracks.items():
            for detection_id, detection in enumerate(detections):
                distance = self._distance(track.center_m, detection.center_m)
                if distance <= float(self.cfg["tracking_match_distance_m"]):
                    pairs.append((distance, track_id, detection_id))

        for _, track_id, detection_id in sorted(pairs):
            if track_id not in unmatched_tracks or detection_id not in unmatched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[detection_id]
            alpha = float(self.cfg["position_smoothing_alpha"])
            track.center_m = (
                (1.0 - alpha) * track.center_m[0] + alpha * detection.center_m[0],
                (1.0 - alpha) * track.center_m[1] + alpha * detection.center_m[1],
            )
            track.center_px = (
                (1.0 - alpha) * track.center_px[0] + alpha * detection.center_px[0],
                (1.0 - alpha) * track.center_px[1] + alpha * detection.center_px[1],
            )
            track.estimated_diameter_m = (
                (1.0 - alpha) * track.estimated_diameter_m + alpha * detection.estimated_diameter_m
            )
            track.confidence = max(track.confidence * (1.0 - alpha), detection.confidence)
            track.stamp_sec = stamp_sec
            track.hits += 1
            track.misses = 0
            detection.track_id = track_id
            detection.confirmed = track.hits >= int(self.cfg["confirmation_frames"])
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_id)

        for detection_id in unmatched_detections:
            detection = detections[detection_id]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = TrackedDisk(
                track_id=track_id,
                center_m=detection.center_m,
                center_px=detection.center_px,
                estimated_diameter_m=detection.estimated_diameter_m,
                confidence=detection.confidence,
                stamp_sec=stamp_sec,
                hits=1,
            )
            detection.track_id = track_id
            detection.confirmed = int(self.cfg["confirmation_frames"]) <= 1

        expired = []
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            if (
                track.misses > int(self.cfg["allowed_missed_frames"])
                or stamp_sec - track.stamp_sec > float(self.cfg["track_timeout_sec"])
            ):
                expired.append(track_id)
        for track_id in expired:
            del self.tracks[track_id]

        confirmation = int(self.cfg["confirmation_frames"])
        return sorted(
            [track for track in self.tracks.values() if track.hits >= confirmation],
            key=lambda item: item.center_m[0],
        )

    def clear(self) -> None:
        self.tracks.clear()
