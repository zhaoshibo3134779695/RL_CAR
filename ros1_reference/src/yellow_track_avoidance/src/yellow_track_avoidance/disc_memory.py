"""World-fixed short-term memory for camera-detected field discs."""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Set, Tuple

from .disc_passage import (
    PHASE_TOMBSTONE,
    PHASE_UNCONFIRMED,
    PassageState,
)


@dataclass
class RememberedDisc:
    memory_id: int
    fixed_xy: Tuple[float, float]
    first_seen_sec: float
    last_seen_sec: float
    confirmed: bool = False
    estimated_diameter_m: Optional[float] = None
    confidence: Optional[float] = None
    track_ids: Set[int] = field(default_factory=set)
    passage: PassageState = field(default_factory=PassageState)


class StableDiscMemory:
    """Merge observations in a fixed frame and expire only stale entries."""

    def __init__(
        self,
        confirmed_timeout_sec: float,
        unconfirmed_timeout_sec: float,
        merge_distance_m: float,
        smoothing_alpha: float,
        tombstone_match_distance_m: float = 0.15,
        freeze_confirmed_position: bool = False,
    ):
        self.confirmed_timeout_sec = max(0.0, float(confirmed_timeout_sec))
        self.unconfirmed_timeout_sec = max(
            0.0, float(unconfirmed_timeout_sec)
        )
        self.merge_distance_m = max(0.0, float(merge_distance_m))
        self.smoothing_alpha = min(
            1.0, max(0.0, float(smoothing_alpha))
        )
        self.tombstone_match_distance_m = max(
            0.0, float(tombstone_match_distance_m)
        )
        self.freeze_confirmed_position = bool(
            freeze_confirmed_position
        )
        self._entries: Dict[int, RememberedDisc] = {}
        self._track_to_memory: Dict[int, int] = {}
        self._next_memory_id = 1
        self.navigation_epoch = 0

    @staticmethod
    def _distance(first, second) -> float:
        return math.hypot(
            float(first[0]) - float(second[0]),
            float(first[1]) - float(second[1]),
        )

    def _nearest(
        self,
        fixed_xy,
        tombstone: bool,
        maximum_distance_m: float,
    ) -> Optional[RememberedDisc]:
        entries = [
            entry
            for entry in self._entries.values()
            if (
                entry.passage.phase == PHASE_TOMBSTONE
            ) == bool(tombstone)
        ]
        if not entries:
            return None
        distance, entry = min(
            (
                (self._distance(item.fixed_xy, fixed_xy), item)
                for item in entries
            ),
            key=lambda pair: pair[0],
        )
        return entry if distance <= float(maximum_distance_m) else None

    def observe(
        self,
        track_id: Optional[int],
        fixed_xy,
        stamp_sec: float,
        confirmed: bool,
        estimated_diameter_m: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> RememberedDisc:
        x_value = float(fixed_xy[0])
        y_value = float(fixed_xy[1])
        stamp_value = float(stamp_sec)
        if not (
            math.isfinite(x_value)
            and math.isfinite(y_value)
            and math.isfinite(stamp_value)
        ):
            raise ValueError("disc observation must be finite")

        numeric_track_id = (
            None if track_id is None else int(track_id)
        )
        diameter_value = (
            None
            if estimated_diameter_m is None
            or not math.isfinite(float(estimated_diameter_m))
            else float(estimated_diameter_m)
        )
        confidence_value = (
            None
            if confidence is None or not math.isfinite(float(confidence))
            else float(confidence)
        )
        memory_id = (
            None
            if numeric_track_id is None
            else self._track_to_memory.get(numeric_track_id)
        )
        entry = self._entries.get(memory_id) if memory_id is not None else None
        if (
            entry is not None
            and entry.passage.phase == PHASE_TOMBSTONE
            and self._distance(entry.fixed_xy, (x_value, y_value))
            > self.tombstone_match_distance_m
        ):
            # Perception track IDs can be reused after a tracker restart. A
            # far-away observation must not be suppressed merely because its
            # numeric ID once belonged to a passed disc.
            if numeric_track_id is not None:
                self._track_to_memory.pop(numeric_track_id, None)
            entry = None
        if entry is None:
            entry = self._nearest(
                (x_value, y_value),
                tombstone=False,
                maximum_distance_m=self.merge_distance_m,
            )
        if entry is None:
            entry = self._nearest(
                (x_value, y_value),
                tombstone=True,
                maximum_distance_m=self.tombstone_match_distance_m,
            )

        if entry is None:
            entry = RememberedDisc(
                memory_id=self._next_memory_id,
                fixed_xy=(x_value, y_value),
                first_seen_sec=stamp_value,
                last_seen_sec=stamp_value,
                confirmed=bool(confirmed),
                estimated_diameter_m=diameter_value,
                confidence=confidence_value,
            )
            self._entries[entry.memory_id] = entry
            self._next_memory_id += 1
        elif entry.passage.phase == PHASE_TOMBSTONE:
            # Keep the fixed passed position and suppress a new active ID.
            entry.last_seen_sec = max(entry.last_seen_sec, stamp_value)
        else:
            alpha = self.smoothing_alpha
            # In a one-pass mission, a confirmed physical disc is a fixed
            # world obstacle.  Continue accepting observations for freshness
            # and diagnostics, but do not let camera/TF jitter drag its odom
            # position after the confirmation frame has locked it.
            if not (
                self.freeze_confirmed_position and entry.confirmed
            ):
                entry.fixed_xy = (
                    (1.0 - alpha) * entry.fixed_xy[0] + alpha * x_value,
                    (1.0 - alpha) * entry.fixed_xy[1] + alpha * y_value,
                )
            entry.last_seen_sec = max(entry.last_seen_sec, stamp_value)
            entry.confirmed = entry.confirmed or bool(confirmed)
            if diameter_value is not None:
                entry.estimated_diameter_m = (
                    diameter_value
                    if entry.estimated_diameter_m is None
                    else (
                        (1.0 - alpha) * entry.estimated_diameter_m
                        + alpha * diameter_value
                    )
                )
            if confidence_value is not None:
                entry.confidence = confidence_value

        was_confirmed = entry.confirmed
        entry.confirmed = entry.confirmed or bool(confirmed)
        if (
            entry.confirmed
            and (
                not was_confirmed
                or entry.passage.phase == PHASE_UNCONFIRMED
            )
        ):
            entry.passage.activate(self.navigation_epoch)

        if numeric_track_id is not None:
            entry.track_ids.add(numeric_track_id)
            self._track_to_memory[numeric_track_id] = entry.memory_id
        return entry

    def _timeout(self, entry: RememberedDisc) -> float:
        return (
            self.confirmed_timeout_sec
            if entry.confirmed
            else self.unconfirmed_timeout_sec
        )

    def is_stale(
        self, entry: RememberedDisc, now_sec: float
    ) -> bool:
        return (
            float(now_sec) - entry.last_seen_sec > self._timeout(entry)
        )

    def stale_removal_reason(
        self,
        entry: RememberedDisc,
        now_sec: float,
        distance_m: float,
        active_range_m: float,
    ) -> Optional[str]:
        """Return only timeout-based reasons, never a pose-heading shortcut."""
        if entry.passage.phase == PHASE_TOMBSTONE:
            return None
        if not self.is_stale(entry, now_sec):
            return None
        if not entry.confirmed:
            return "unconfirmed_timeout"
        if float(distance_m) > float(active_range_m):
            return "distant_timeout"
        return None

    def begin_navigation_epoch(self) -> int:
        self.navigation_epoch += 1
        for entry in list(self._entries.values()):
            if entry.passage.phase == PHASE_TOMBSTONE:
                self.remove(entry.memory_id)
            elif entry.confirmed:
                entry.passage.activate(self.navigation_epoch)
        return self.navigation_epoch

    def reactivate_tombstone(self, memory_id: int) -> bool:
        entry = self._entries.get(int(memory_id))
        if entry is None or entry.passage.phase != PHASE_TOMBSTONE:
            return False
        entry.passage.activate(self.navigation_epoch)
        return True

    def garbage_collect_tombstones(
        self,
        now_sec: float,
        navigation_active: bool,
        timeout_sec: float,
    ) -> List[int]:
        if navigation_active:
            return []
        expired = [
            entry.memory_id
            for entry in self._entries.values()
            if (
                entry.passage.phase == PHASE_TOMBSTONE
                and entry.passage.passed_sec is not None
                and float(now_sec) - entry.passage.passed_sec
                > float(timeout_sec)
            )
        ]
        for memory_id in expired:
            self.remove(memory_id)
        return expired

    def expire(self, now_sec: float) -> List[int]:
        now_value = float(now_sec)
        expired = [
            entry.memory_id
            for entry in self._entries.values()
            if self.is_stale(entry, now_value)
        ]
        for memory_id in expired:
            self.remove(memory_id)
        return expired

    def remove(self, memory_id: int) -> None:
        entry = self._entries.pop(int(memory_id), None)
        if entry is None:
            return
        for track_id in entry.track_ids:
            if self._track_to_memory.get(track_id) == entry.memory_id:
                del self._track_to_memory[track_id]

    def clear(self, reset_identity: bool = False) -> None:
        self._entries.clear()
        self._track_to_memory.clear()
        if reset_identity:
            self._next_memory_id = 1
            self.navigation_epoch = 0

    def entries(self) -> List[RememberedDisc]:
        return sorted(
            self._entries.values(), key=lambda item: item.memory_id
        )

    def active_entries(self) -> List[RememberedDisc]:
        return [
            entry
            for entry in self.entries()
            if entry.passage.phase != PHASE_TOMBSTONE
        ]

    def tombstone_entries(self) -> List[RememberedDisc]:
        return [
            entry
            for entry in self.entries()
            if entry.passage.phase == PHASE_TOMBSTONE
        ]

    def __len__(self) -> int:
        return len(self._entries)
