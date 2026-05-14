from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from config import TrackerConfig

if TYPE_CHECKING:
    from scripts.detector import Detection


@dataclass
class TrackObject:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    age_frames: int = 1
    missing_frames: int = 0
    is_stable: bool = False
    counted: bool = False
    history: list[tuple[int, int, int, int]] = field(default_factory=list)
    class_votes: Counter[str] = field(default_factory=Counter)

    def update(self, class_name: str, bbox: tuple[int, int, int, int], confidence: float, stable_after: int, alpha: float) -> None:
        smoothed = _smooth_bbox(self.bbox, bbox, alpha)
        self.bbox = smoothed
        self.class_votes[class_name] += 1
        self.class_name = self.class_votes.most_common(1)[0][0]
        self.confidence = confidence
        self.age_frames += 1
        self.missing_frames = 0
        self.history.append(smoothed)
        if len(self.history) > 32:
            self.history.pop(0)
        self.is_stable = self.age_frames >= stable_after


@dataclass(frozen=True)
class TrackerResult:
    active_tracks: List[TrackObject]
    removed_track_ids: List[int]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(union, 1)


def _smooth_bbox(old_box: tuple[int, int, int, int], new_box: tuple[int, int, int, int], alpha: float) -> tuple[int, int, int, int]:
    return tuple(int(alpha * o + (1.0 - alpha) * n) for o, n in zip(old_box, new_box))


class Tracker:
    """Lightweight IOU tracker with stable ID assignment and stale cleanup."""

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._tracks: Dict[int, TrackObject] = {}
        self._next_track_id: int = 1

    def update(self, detections: List["Detection"]) -> TrackerResult:
        filtered = [d for d in detections if _bbox_area(d.bbox) >= self._config.min_box_area]

        unmatched_track_ids = set(self._tracks.keys())
        unmatched_detection_indices = set(range(len(filtered)))

        matches: list[tuple[int, int, float]] = []
        for track_id, track in self._tracks.items():
            best_det_idx: Optional[int] = None
            best_iou = 0.0
            for det_idx in unmatched_detection_indices:
                det = filtered[det_idx]
                iou_score = _iou(track.bbox, det.bbox)
                if iou_score > best_iou and iou_score >= self._config.min_iou_match:
                    best_iou = iou_score
                    best_det_idx = det_idx

            if best_det_idx is not None:
                matches.append((track_id, best_det_idx, best_iou))

        for track_id, det_idx, _ in matches:
            if track_id not in unmatched_track_ids or det_idx not in unmatched_detection_indices:
                continue
            det = filtered[det_idx]
            self._tracks[track_id].update(
                det.class_name,
                det.bbox,
                det.confidence,
                self._config.stable_after_frames,
                self._config.smoothing_alpha,
            )
            unmatched_track_ids.discard(track_id)
            unmatched_detection_indices.discard(det_idx)

        for det_idx in unmatched_detection_indices:
            det = filtered[det_idx]
            self._create_track(det)

        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.missing_frames += 1

        removed_track_ids: List[int] = []
        for track_id in list(self._tracks.keys()):
            if self._tracks[track_id].missing_frames > self._config.max_missing_frames:
                removed_track_ids.append(track_id)
                del self._tracks[track_id]

        return TrackerResult(active_tracks=list(self._tracks.values()), removed_track_ids=removed_track_ids)

    def _create_track(self, det: "Detection") -> None:
        track = TrackObject(
            track_id=self._next_track_id,
            class_name=det.class_name,
            bbox=det.bbox,
            confidence=det.confidence,
            is_stable=self._config.stable_after_frames <= 1,
            history=[det.bbox],
            class_votes=Counter({det.class_name: 1}),
        )
        self._tracks[self._next_track_id] = track
        self._next_track_id += 1

    def mark_counted(self, track_id: int) -> None:
        if track_id in self._tracks:
            self._tracks[track_id].counted = True

    def get_track(self, track_id: int) -> Optional[TrackObject]:
        return self._tracks.get(track_id)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
