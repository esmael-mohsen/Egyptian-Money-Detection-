from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, Dict, Iterable

from config import CounterConfig
from scripts.tracker import TrackObject


MONEY_MAP: Dict[str, int] = {
    
    "5 Pounds": 5,
    "10 Pounds": 10,
    "20 Pounds": 20,
    "50 Pounds": 50,
    "100 Pounds": 100,
    "200 Pounds": 200,

}


@dataclass(frozen=True)
class CountEvent:
    track_id: int
    class_name: str
    value: int


class CounterEngine:
    """Counts only valid stable tracks once and exposes live stats."""

    def __init__(self, config: CounterConfig) -> None:
        self._config = config
        self._counted_track_ids: set[int] = set()
        self._denomination_counts: Counter[str] = Counter()
        self._total_amount: int = 0
        self._last_count_event: CountEvent | None = None
        self._recent_counted_signatures: Deque[tuple[str, float, float, int]] = deque(
            maxlen=config.recent_memory_limit
        )

    def update(self, tracks: Iterable[TrackObject]) -> list[CountEvent]:
        events: list[CountEvent] = []
        for track in tracks:
            if not self._is_track_countable(track):
                continue
            if track.track_id in self._counted_track_ids:
                continue
            if self._looks_recently_counted(track):
                continue

            value = MONEY_MAP.get(track.class_name)
            if value is None:
                continue

            self._counted_track_ids.add(track.track_id)
            self._denomination_counts[track.class_name] += 1
            self._total_amount += value
            event = CountEvent(track_id=track.track_id, class_name=track.class_name, value=value)
            self._last_count_event = event
            self._recent_counted_signatures.append(_track_signature(track))
            events.append(event)

        return events

    def add_manual_count(self, class_name: str, value: int, track_id: int = -1) -> CountEvent:
        self._denomination_counts[class_name] += 1
        self._total_amount += value
        event = CountEvent(track_id=track_id, class_name=class_name, value=value)
        self._last_count_event = event
        return event

    def _is_track_countable(self, track: TrackObject) -> bool:
        return (
            track.is_stable
            and track.age_frames >= self._config.stable_required_frames
            and track.confidence >= self._config.confidence_threshold
            and not track.counted
        )

    def reset(self) -> None:
        self._counted_track_ids.clear()
        self._denomination_counts.clear()
        self._total_amount = 0
        self._last_count_event = None
        self._recent_counted_signatures.clear()

    def reset_tracking_state(self) -> None:
        self._counted_track_ids.clear()
        self._recent_counted_signatures.clear()

    def get_total(self) -> int:
        return self._total_amount

    def get_statistics(self) -> dict[str, int]:
        return {
            "total_amount": self._total_amount,
            "counted_notes": sum(self._denomination_counts.values()),
        }

    def get_denomination_breakdown(self) -> dict[str, int]:
        return dict(self._denomination_counts)

    def is_counted(self, track_id: int) -> bool:
        return track_id in self._counted_track_ids

    def get_last_count_event(self) -> CountEvent | None:
        return self._last_count_event

    def _looks_recently_counted(self, track: TrackObject) -> bool:
        class_name, cx, cy, area = _track_signature(track)
        threshold = max(80.0, sqrt(max(area, 1)) * 0.35)

        for recent_class, recent_cx, recent_cy, recent_area in self._recent_counted_signatures:
            if recent_class != class_name:
                continue
            distance = sqrt((cx - recent_cx) ** 2 + (cy - recent_cy) ** 2)
            area_delta = abs(area - recent_area) / max(recent_area, 1)
            if distance <= threshold and area_delta <= 0.35:
                return True

        return False


def _track_signature(track: TrackObject) -> tuple[str, float, float, int]:
    x1, y1, x2, y2 = track.bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return (
        track.class_name,
        x1 + width / 2,
        y1 + height / 2,
        width * height,
    )
