from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2

from config import FlipConfig
from scripts.counter import MONEY_MAP


FLIP_STATES = {
    "WAITING",
    "ENTERING_GATE",
    "CONFIRMING",
    "COUNTED_WAIT_EXIT",
    "EXITED",
}


@dataclass(frozen=True)
class FlipDebugState:
    state: str
    reason: str
    best_class: str | None
    confidence: float
    vote_ratio: float
    direction: str | None
    bbox: tuple[int, int, int, int] | None
    gate_bounds: tuple[str, int, int]
    enter_zone: tuple[str, int, int]
    exit_zone: tuple[str, int, int]
    count_line: int
    motion_pixels: float
    confirmed_frames: int


@dataclass(frozen=True)
class FlipCountEvent:
    value: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    reason: str
    frame_index: int
    state: str
    vote_ratio: float
    direction: str | None
    metadata_path: Path | None = None


@dataclass(frozen=True)
class FlipRejectEvent:
    reason: str
    best_class: str | None
    confidence: float
    bbox: tuple[int, int, int, int] | None
    frame_index: int
    state: str
    vote_ratio: float
    reason_detail: str = ""


@dataclass(frozen=True)
class FlipCounterResult:
    count_events: list[FlipCountEvent]
    reject_events: list[FlipRejectEvent]
    debug_state: FlipDebugState


@dataclass(frozen=True)
class _Zones:
    orientation: str
    count_line: int
    gate: tuple[str, int, int]
    enter: tuple[str, int, int]
    exit: tuple[str, int, int]


@dataclass(frozen=True)
class _Observation:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: int
    frame_index: int
    timestamp: float


@dataclass(frozen=True)
class _VoteSummary:
    best_class: str | None
    confidence: float
    vote_ratio: float
    margin: float
    confirmed_frames: int
    bbox: tuple[int, int, int, int] | None
    dwell_seconds: float


class FlipCounterEngine:
    """Motion-based stack counter for partial, hand-flipped banknotes."""

    def __init__(self, config: FlipConfig) -> None:
        self._config = config
        self._buffer: deque[_Observation] = deque(maxlen=max(1, config.prediction_buffer_frames))
        self._state = "WAITING"
        self._clear_frames = 0
        self._last_count_time = -10_000.0
        self._uncertain_capture_times: deque[float] = deque()
        self._last_debug = self._empty_debug(("vertical", 0, 0), ("vertical", 0, 0), ("vertical", 0, 0), 0)

    def update(
        self,
        detections: Iterable[object],
        frame,
        frame_index: int,
        mode: str,
        now: float | None = None,
    ) -> FlipCounterResult:
        now = time.monotonic() if now is None else now
        zones = self.zone_bounds(frame.shape[:2])
        in_enter_zone = [d for d in detections if _intersects_zone(d.bbox, zones.enter)]

        if not in_enter_zone:
            return self._handle_empty_gate(zones, frame_index)

        self._clear_frames = 0
        if self._state == "COUNTED_WAIT_EXIT":
            debug = self._debug(
                zones=zones,
                reason="waiting_exit",
                best_class=None,
                confidence=0.0,
                vote_ratio=0.0,
                direction=None,
                bbox=None,
                motion_pixels=0.0,
                confirmed_frames=0,
            )
            event = FlipRejectEvent(
                reason="waiting_exit",
                best_class=None,
                confidence=0.0,
                bbox=None,
                frame_index=frame_index,
                state=self._state,
                vote_ratio=0.0,
                reason_detail="current note must leave enter zone before another count",
            )
            return FlipCounterResult([], [event], self._set_debug(debug))

        if self._state in {"WAITING", "EXITED"}:
            self._state = "ENTERING_GATE"

        observation, quality_reason = self._select_observation(in_enter_zone, frame.shape[:2], frame_index, now)
        if observation is None:
            return self._reject(
                frame=frame,
                frame_index=frame_index,
                mode=mode,
                zones=zones,
                reason=quality_reason,
                best_class=None,
                confidence=0.0,
                bbox=None,
                vote_ratio=0.0,
                reason_detail="no usable known detection in enter zone",
                capture=True,
            )

        self._buffer.append(observation)
        summary = self._vote_summary()
        direction, motion_pixels = self._motion_for(summary.best_class, zones.orientation)
        debug = self._debug(
            zones=zones,
            reason="collecting_evidence",
            best_class=summary.best_class,
            confidence=summary.confidence,
            vote_ratio=summary.vote_ratio,
            direction=direction,
            bbox=summary.bbox,
            motion_pixels=motion_pixels,
            confirmed_frames=summary.confirmed_frames,
        )

        if summary.confidence < self._config.confidence_threshold:
            return self._reject_from_debug(
                frame,
                frame_index,
                mode,
                debug,
                "low_confidence",
                "best detection is below flip confidence threshold",
                capture=True,
            )

        self._state = "CONFIRMING"
        if summary.confirmed_frames < self._config.min_confirmed_frames:
            return self._reject_from_debug(
                frame,
                frame_index,
                mode,
                debug,
                "collecting_evidence",
                "waiting for more confirmed frames",
            )

        if summary.vote_ratio < self._config.vote_ratio_threshold:
            return self._reject_from_debug(
                frame,
                frame_index,
                mode,
                debug,
                "mixed_votes",
                "best class vote ratio is below threshold",
                capture=True,
            )

        if summary.margin < self._config.class_margin_threshold:
            return self._reject_from_debug(
                frame,
                frame_index,
                mode,
                debug,
                "mixed_votes",
                "best class confidence margin is too close to second class",
                capture=True,
            )

        crossed_count_line = self._has_crossed_count_line(
            summary.best_class,
            zones,
        )
        dwell_ready = self._is_dwell_count_ready(summary, zones)
        if not dwell_ready and (motion_pixels < self._config.min_motion_pixels or not crossed_count_line):
            return self._reject_from_debug(
                frame,
                frame_index,
                mode,
                debug,
                "waiting_crossing",
                "waiting for clear motion across count line or stable dwell",
            )

        if now - self._last_count_time < self._config.cooldown_seconds:
            return self._reject_from_debug(
                frame,
                frame_index,
                mode,
                debug,
                "cooldown",
                "minimum time between counted notes has not elapsed",
            )

        assert summary.best_class is not None
        assert summary.bbox is not None
        value = MONEY_MAP[summary.best_class]
        event_reason = "stable_dwell" if dwell_ready and not crossed_count_line else "motion_crossing"
        self._state = "COUNTED_WAIT_EXIT"
        debug = self._debug(
            zones=zones,
            reason="counted",
            best_class=summary.best_class,
            confidence=summary.confidence,
            vote_ratio=summary.vote_ratio,
            direction=direction,
            bbox=summary.bbox,
            motion_pixels=motion_pixels,
            confirmed_frames=summary.confirmed_frames,
        )
        metadata_path = self._capture(
            frame=frame,
            mode=mode,
            bucket="counted",
            reason=event_reason,
            class_name=summary.best_class,
            confidence=summary.confidence,
            frame_index=frame_index,
            debug=debug,
            counted=True,
        )
        self._last_count_time = now
        event = FlipCountEvent(
            value=value,
            class_name=summary.best_class,
            confidence=summary.confidence,
            bbox=summary.bbox,
            reason=event_reason,
            frame_index=frame_index,
            state=self._state,
            vote_ratio=summary.vote_ratio,
            direction=direction,
            metadata_path=metadata_path,
        )
        return FlipCounterResult([event], [], self._set_debug(debug))

    def reset(self) -> None:
        self._buffer.clear()
        self._state = "WAITING"
        self._clear_frames = 0
        self._last_count_time = -10_000.0
        self._uncertain_capture_times.clear()

    def gate_bounds(self, frame_shape: tuple[int, int]) -> tuple[str, int, int]:
        return self.zone_bounds(frame_shape).gate

    def zone_bounds(self, frame_shape: tuple[int, int]) -> _Zones:
        height, width = frame_shape
        if self._config.gate_orientation == "horizontal":
            count_line = int(height * self._config.count_line_ratio)
            gate_half = max(1, int(height * self._config.gate_width_ratio / 2))
            enter_half = max(gate_half, int(height * self._config.enter_zone_width_ratio / 2))
            exit_half = max(gate_half, int(height * self._config.exit_zone_width_ratio / 2))
            return _Zones(
                orientation="horizontal",
                count_line=count_line,
                gate=("horizontal", max(0, count_line - gate_half), min(height, count_line + gate_half)),
                enter=("horizontal", max(0, count_line - enter_half), min(height, count_line + enter_half)),
                exit=("horizontal", max(0, count_line - exit_half), min(height, count_line + exit_half)),
            )

        count_line = int(width * self._config.count_line_ratio)
        gate_half = max(1, int(width * self._config.gate_width_ratio / 2))
        enter_half = max(gate_half, int(width * self._config.enter_zone_width_ratio / 2))
        exit_half = max(gate_half, int(width * self._config.exit_zone_width_ratio / 2))
        return _Zones(
            orientation="vertical",
            count_line=count_line,
            gate=("vertical", max(0, count_line - gate_half), min(width, count_line + gate_half)),
            enter=("vertical", max(0, count_line - enter_half), min(width, count_line + enter_half)),
            exit=("vertical", max(0, count_line - exit_half), min(width, count_line + exit_half)),
        )

    def get_debug_state(self) -> FlipDebugState:
        return self._last_debug

    def _handle_empty_gate(self, zones: _Zones, frame_index: int) -> FlipCounterResult:
        if self._state == "COUNTED_WAIT_EXIT":
            self._clear_frames += 1
            if self._clear_frames >= self._config.exit_clear_frames:
                self._state = "EXITED"
                self._buffer.clear()
                reason = "exited"
            else:
                reason = "waiting_exit"
        else:
            self._clear_frames += 1
            self._buffer.clear()
            if self._state == "EXITED":
                self._state = "WAITING"
            reason = "waiting"

        debug = self._debug(
            zones=zones,
            reason=reason,
            best_class=None,
            confidence=0.0,
            vote_ratio=0.0,
            direction=None,
            bbox=None,
            motion_pixels=0.0,
            confirmed_frames=0,
        )
        return FlipCounterResult([], [], self._set_debug(debug))

    def _select_observation(
        self,
        detections: Iterable[object],
        frame_shape: tuple[int, int],
        frame_index: int,
        now: float,
    ) -> tuple[_Observation | None, str]:
        candidates = []
        for detection in detections:
            class_name = getattr(detection, "class_name", None)
            if class_name not in MONEY_MAP:
                continue
            bbox = detection.bbox
            if not self._bbox_quality_ok(bbox, frame_shape):
                continue
            candidates.append(detection)

        if not candidates:
            return None, "low_quality"

        selected = max(candidates, key=lambda d: (float(d.confidence), _bbox_area(d.bbox)))
        bbox = selected.bbox
        return (
            _Observation(
                class_name=selected.class_name,
                confidence=float(selected.confidence),
                bbox=bbox,
                center=_center(bbox),
                area=_bbox_area(bbox),
                frame_index=frame_index,
                timestamp=now,
            ),
            "",
        )

    def _bbox_quality_ok(self, bbox: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> bool:
        height, width = frame_shape
        area = _bbox_area(bbox)
        if area < self._config.min_bbox_area:
            return False
        x1, y1, x2, y2 = bbox
        visible_x1, visible_y1 = max(0, x1), max(0, y1)
        visible_x2, visible_y2 = min(width, x2), min(height, y2)
        visible_area = max(0, visible_x2 - visible_x1) * max(0, visible_y2 - visible_y1)
        return visible_area / max(area, 1) >= 0.65

    def _vote_summary(self) -> _VoteSummary:
        if not self._buffer:
            return _VoteSummary(None, 0.0, 0.0, 0.0, 0, None, 0.0)

        counts: Counter[str] = Counter()
        confidence_by_class: dict[str, list[float]] = {}
        best_bbox_by_class: dict[str, tuple[float, tuple[int, int, int, int]]] = {}
        for observation in self._buffer:
            counts[observation.class_name] += 1
            confidence_by_class.setdefault(observation.class_name, []).append(observation.confidence)
            previous = best_bbox_by_class.get(observation.class_name)
            if previous is None or observation.confidence > previous[0]:
                best_bbox_by_class[observation.class_name] = (observation.confidence, observation.bbox)

        best_class, best_count = counts.most_common(1)[0]
        total = sum(counts.values())
        vote_ratio = best_count / max(total, 1)
        avg_confidences = {
            class_name: sum(values) / len(values)
            for class_name, values in confidence_by_class.items()
        }
        best_confidence = max(confidence_by_class[best_class])
        other_confidences = [
            confidence
            for class_name, confidence in avg_confidences.items()
            if class_name != best_class
        ]
        margin = avg_confidences[best_class] - max(other_confidences, default=0.0)
        confirmed_frames = sum(
            1
            for observation in self._buffer
            if observation.class_name == best_class
            and observation.confidence >= self._config.confidence_threshold
        )
        class_observations = [item for item in self._buffer if item.class_name == best_class]
        dwell_seconds = 0.0
        if len(class_observations) >= 2:
            dwell_seconds = class_observations[-1].timestamp - class_observations[0].timestamp

        return _VoteSummary(
            best_class=best_class,
            confidence=best_confidence,
            vote_ratio=vote_ratio,
            margin=margin,
            confirmed_frames=confirmed_frames,
            bbox=best_bbox_by_class[best_class][1],
            dwell_seconds=dwell_seconds,
        )

    def _motion_for(self, class_name: str | None, orientation: str) -> tuple[str | None, float]:
        observations = [item for item in self._buffer if item.class_name == class_name]
        if len(observations) < 2:
            return None, 0.0
        first, last = observations[0], observations[-1]
        axis = 1 if orientation == "horizontal" else 0
        delta = last.center[axis] - first.center[axis]
        if abs(delta) < 1:
            return None, 0.0
        if orientation == "horizontal":
            direction = "down" if delta > 0 else "up"
        else:
            direction = "right" if delta > 0 else "left"
        return direction, abs(delta)

    def _has_crossed_count_line(self, class_name: str | None, zones: _Zones) -> bool:
        observations = [item for item in self._buffer if item.class_name == class_name]
        if len(observations) < 2:
            return False
        axis = 1 if zones.orientation == "horizontal" else 0
        centers = [item.center[axis] for item in observations]
        return min(centers) <= zones.count_line <= max(centers)

    def _is_dwell_count_ready(self, summary: _VoteSummary, zones: _Zones) -> bool:
        if not self._config.dwell_count_enabled:
            return False
        if summary.best_class is None or summary.bbox is None:
            return False
        if summary.confirmed_frames < self._config.dwell_required_frames:
            return False
        if summary.dwell_seconds < self._config.dwell_seconds:
            return False
        return _intersects_zone(summary.bbox, zones.gate)

    def _reject_from_debug(
        self,
        frame,
        frame_index: int,
        mode: str,
        debug: FlipDebugState,
        reason: str,
        reason_detail: str,
        capture: bool = False,
    ) -> FlipCounterResult:
        return self._reject(
            frame=frame,
            frame_index=frame_index,
            mode=mode,
            zones=None,
            reason=reason,
            best_class=debug.best_class,
            confidence=debug.confidence,
            bbox=debug.bbox,
            vote_ratio=debug.vote_ratio,
            reason_detail=reason_detail,
            capture=capture,
            debug=debug,
        )

    def _reject(
        self,
        frame,
        frame_index: int,
        mode: str,
        zones: _Zones | None,
        reason: str,
        best_class: str | None,
        confidence: float,
        bbox: tuple[int, int, int, int] | None,
        vote_ratio: float,
        reason_detail: str,
        capture: bool = False,
        debug: FlipDebugState | None = None,
    ) -> FlipCounterResult:
        if debug is None:
            assert zones is not None
            debug = self._debug(
                zones=zones,
                reason=reason,
                best_class=best_class,
                confidence=confidence,
                vote_ratio=vote_ratio,
                direction=None,
                bbox=bbox,
                motion_pixels=0.0,
                confirmed_frames=0,
            )
        else:
            debug = self._debug_replace_reason(debug, reason)

        if capture and self._can_capture_uncertain():
            self._capture(
                frame=frame,
                mode=mode,
                bucket="uncertain",
                reason=reason,
                class_name=best_class,
                confidence=confidence,
                frame_index=frame_index,
                debug=debug,
                counted=False,
            )

        event = FlipRejectEvent(
            reason=reason,
            best_class=best_class,
            confidence=confidence,
            bbox=bbox,
            frame_index=frame_index,
            state=self._state,
            vote_ratio=vote_ratio,
            reason_detail=reason_detail,
        )
        return FlipCounterResult([], [event], self._set_debug(debug))

    def _can_capture_uncertain(self) -> bool:
        now = time.monotonic()
        while self._uncertain_capture_times and now - self._uncertain_capture_times[0] > 60:
            self._uncertain_capture_times.popleft()
        if len(self._uncertain_capture_times) >= self._config.max_uncertain_captures_per_minute:
            return False
        self._uncertain_capture_times.append(now)
        return True

    def _capture(
        self,
        frame,
        mode: str,
        bucket: str,
        reason: str,
        class_name: str | None,
        confidence: float,
        frame_index: int,
        debug: FlipDebugState,
        counted: bool,
    ) -> Path | None:
        if not self._config.capture_enabled:
            return None
        if bucket == "counted" and not self._config.save_counted:
            return None
        if bucket == "uncertain" and not self._config.save_uncertain:
            return None

        target_dir = self._config.capture_dir / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_class = _safe_name(class_name or "unknown")
        safe_reason = _safe_name(reason)
        stem = f"{timestamp}_frame{frame_index}_{mode}_{safe_class}_{confidence:.2f}_{safe_reason}"
        image_path = target_dir / f"{stem}.jpg"
        cv2.imwrite(str(image_path), frame)

        metadata_path = target_dir / f"{stem}.json"
        if self._config.capture_metadata:
            metadata = {
                "timestamp": timestamp,
                "mode": mode,
                "state": debug.state,
                "reason": reason,
                "predicted_class": class_name,
                "confidence": confidence,
                "vote_ratio": debug.vote_ratio,
                "bbox": debug.bbox,
                "gate_bounds": debug.gate_bounds,
                "enter_zone": debug.enter_zone,
                "exit_zone": debug.exit_zone,
                "count_line": debug.count_line,
                "crossing_direction": debug.direction,
                "frame_index": frame_index,
                "counted": counted,
                "motion_pixels": debug.motion_pixels,
                "confirmed_frames": debug.confirmed_frames,
                "profile": self._config.profile,
            }
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            return metadata_path
        return image_path

    def _debug(
        self,
        zones: _Zones,
        reason: str,
        best_class: str | None,
        confidence: float,
        vote_ratio: float,
        direction: str | None,
        bbox: tuple[int, int, int, int] | None,
        motion_pixels: float,
        confirmed_frames: int,
    ) -> FlipDebugState:
        return FlipDebugState(
            state=self._state,
            reason=reason,
            best_class=best_class,
            confidence=confidence,
            vote_ratio=vote_ratio,
            direction=direction,
            bbox=bbox,
            gate_bounds=zones.gate,
            enter_zone=zones.enter,
            exit_zone=zones.exit,
            count_line=zones.count_line,
            motion_pixels=motion_pixels,
            confirmed_frames=confirmed_frames,
        )

    def _debug_replace_reason(self, debug: FlipDebugState, reason: str) -> FlipDebugState:
        return FlipDebugState(**{**asdict(debug), "state": self._state, "reason": reason})

    def _set_debug(self, debug: FlipDebugState) -> FlipDebugState:
        self._last_debug = debug
        return debug

    def _empty_debug(
        self,
        gate: tuple[str, int, int],
        enter: tuple[str, int, int],
        exit_zone: tuple[str, int, int],
        count_line: int,
    ) -> FlipDebugState:
        return FlipDebugState(
            state=self._state,
            reason="waiting",
            best_class=None,
            confidence=0.0,
            vote_ratio=0.0,
            direction=None,
            bbox=None,
            gate_bounds=gate,
            enter_zone=enter,
            exit_zone=exit_zone,
            count_line=count_line,
            motion_pixels=0.0,
            confirmed_frames=0,
        )


def _intersects_zone(bbox: tuple[int, int, int, int], zone: tuple[str, int, int]) -> bool:
    orientation, start, end = zone
    x1, y1, x2, y2 = bbox
    if orientation == "horizontal":
        return y2 >= start and y1 <= end
    return x2 >= start and x1 <= end


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "unknown"
