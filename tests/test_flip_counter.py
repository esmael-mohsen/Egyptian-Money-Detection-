from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import FlipConfig
from scripts.flip_counter import FlipCounterEngine
from scripts.wallet import TransactionType, WalletStore


@dataclass(frozen=True)
class FakeDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


def make_engine(**overrides) -> FlipCounterEngine:
    values = {
        "capture_enabled": False,
        "cooldown_seconds": 0.65,
        "min_confirmed_frames": 1,
        "dwell_required_frames": 3,
        "dwell_seconds": 0.30,
        "vote_ratio_threshold": 0.70,
        "class_margin_threshold": 0.12,
        "min_motion_pixels": 18,
        "exit_clear_frames": 3,
    }
    values.update(overrides)
    return FlipCounterEngine(FlipConfig(**values))


def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def note_at(cx: int, class_name: str = "50 Pounds", confidence: float = 0.90, w: int = 70) -> FakeDetection:
    return FakeDetection(
        class_name=class_name,
        confidence=confidence,
        bbox=(cx - w // 2, 120, cx + w // 2, 240),
    )


def cross_count_line(engine: FlipCounterEngine, image, start: int = 280, end: int = 350, now: float = 10.0):
    first = engine.update([note_at(start)], image, frame_index=1, mode="FLIP_SCAN", now=now)
    second = engine.update([note_at(end)], image, frame_index=2, mode="FLIP_SCAN", now=now + 0.2)
    return first, second


class FlipCounterEngineTests(unittest.TestCase):
    def test_note_entering_then_crossing_count_line_counts_once(self) -> None:
        engine = make_engine()
        image = frame()

        first, second = cross_count_line(engine, image)

        self.assertEqual(first.count_events, [])
        self.assertEqual(len(second.count_events), 1)
        self.assertEqual(second.count_events[0].value, 50)
        self.assertEqual(second.count_events[0].state, "COUNTED_WAIT_EXIT")
        self.assertGreaterEqual(second.count_events[0].vote_ratio, 0.70)

    def test_same_note_staying_in_gate_is_not_counted_twice(self) -> None:
        engine = make_engine()
        image = frame()
        _, counted = cross_count_line(engine, image)

        repeated = engine.update([note_at(350)], image, frame_index=3, mode="FLIP_SCAN", now=11.0)

        self.assertEqual(len(counted.count_events), 1)
        self.assertEqual(repeated.count_events, [])
        self.assertEqual(repeated.reject_events[0].reason, "waiting_exit")

    def test_new_same_value_counts_after_exit_clear_frames(self) -> None:
        engine = make_engine()
        image = frame()

        _, first_count = cross_count_line(engine, image, now=10.0)
        for idx in range(3, 6):
            engine.update([], image, frame_index=idx, mode="FLIP_SCAN", now=10.0 + idx)
        first_new = engine.update([note_at(280)], image, frame_index=6, mode="FLIP_SCAN", now=14.0)
        second_new = engine.update([note_at(350)], image, frame_index=7, mode="FLIP_SCAN", now=14.2)

        self.assertEqual(len(first_count.count_events), 1)
        self.assertEqual(first_new.count_events, [])
        self.assertEqual(len(second_new.count_events), 1)

    def test_note_moving_back_before_crossing_does_not_count(self) -> None:
        engine = make_engine(dwell_count_enabled=False)
        image = frame()

        engine.update([note_at(280)], image, frame_index=1, mode="FLIP_SCAN", now=10.0)
        result = engine.update([note_at(270)], image, frame_index=2, mode="FLIP_SCAN", now=10.2)

        self.assertEqual(result.count_events, [])
        self.assertEqual(result.reject_events[0].reason, "waiting_crossing")

    def test_stable_dwell_counts_slow_held_note_without_crossing(self) -> None:
        engine = make_engine(dwell_required_frames=3, dwell_seconds=0.30)
        image = frame()

        first = engine.update([note_at(330)], image, frame_index=1, mode="FLIP_SCAN", now=10.0)
        second = engine.update([note_at(331)], image, frame_index=2, mode="FLIP_SCAN", now=10.2)
        third = engine.update([note_at(329)], image, frame_index=3, mode="FLIP_SCAN", now=10.4)

        self.assertEqual(first.count_events, [])
        self.assertEqual(second.count_events, [])
        self.assertEqual(len(third.count_events), 1)
        self.assertEqual(third.count_events[0].reason, "stable_dwell")

    def test_low_confidence_inside_gate_is_rejected(self) -> None:
        engine = make_engine()
        image = frame()

        result = engine.update([note_at(280, confidence=0.20)], image, frame_index=1, mode="FLIP_SCAN", now=10.0)

        self.assertEqual(result.count_events, [])
        self.assertEqual(result.reject_events[0].reason, "low_confidence")

    def test_mixed_vote_ratio_rejects_count(self) -> None:
        engine = make_engine(vote_ratio_threshold=0.70)
        image = frame()

        engine.update([note_at(280, class_name="50 Pounds")], image, frame_index=1, mode="FLIP_SCAN", now=10.0)
        result = engine.update([note_at(350, class_name="200 Pounds")], image, frame_index=2, mode="FLIP_SCAN", now=10.2)

        self.assertEqual(result.count_events, [])
        self.assertEqual(result.reject_events[0].reason, "mixed_votes")

    def test_close_class_margin_rejects_count(self) -> None:
        engine = make_engine(vote_ratio_threshold=0.50, class_margin_threshold=0.12)
        image = frame()

        engine.update([note_at(280, class_name="50 Pounds", confidence=0.72)], image, frame_index=1, mode="FLIP_SCAN", now=10.0)
        engine.update([note_at(300, class_name="50 Pounds", confidence=0.72)], image, frame_index=2, mode="FLIP_SCAN", now=10.1)
        result = engine.update([note_at(350, class_name="200 Pounds", confidence=0.68)], image, frame_index=3, mode="FLIP_SCAN", now=10.2)

        self.assertEqual(result.count_events, [])
        self.assertEqual(result.reject_events[0].reason, "mixed_votes")

    def test_small_bbox_is_rejected_as_low_quality(self) -> None:
        engine = make_engine()
        image = frame()

        result = engine.update([note_at(320, w=8)], image, frame_index=1, mode="FLIP_SCAN", now=10.0)

        self.assertEqual(result.count_events, [])
        self.assertEqual(result.reject_events[0].reason, "low_quality")

    def test_counted_capture_writes_image_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(capture_enabled=True, capture_dir=Path(tmp), capture_metadata=True)
            image = frame()

            _, result = cross_count_line(engine, image)

            self.assertEqual(len(result.count_events), 1)
            metadata_path = result.count_events[0].metadata_path
            self.assertIsNotNone(metadata_path)
            assert metadata_path is not None
            self.assertTrue(metadata_path.exists())
            image_path = metadata_path.with_suffix(".jpg")
            self.assertTrue(image_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["counted"])
            self.assertEqual(metadata["reason"], "motion_crossing")

    def test_uncertain_capture_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            engine = make_engine(
                capture_enabled=True,
                capture_dir=Path(tmp),
                capture_metadata=True,
                max_uncertain_captures_per_minute=1,
            )
            image = frame()

            engine.update([note_at(320, w=8)], image, frame_index=1, mode="FLIP_SCAN", now=10.0)
            engine.update([note_at(320, w=8)], image, frame_index=2, mode="FLIP_SCAN", now=10.2)

            uncertain_dir = Path(tmp) / "uncertain"
            self.assertEqual(len(list(uncertain_dir.glob("*.jpg"))), 1)
            self.assertEqual(len(list(uncertain_dir.glob("*.json"))), 1)


class FlipWalletFlowTests(unittest.TestCase):
    def test_flip_deposit_pending_notes_commit_only_after_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wallet = WalletStore(Path(tmp) / "wallet.db")
            wallet.set_balance(100)
            wallet.commit_transaction()

            wallet.begin_transaction(TransactionType.DEPOSIT)
            wallet.add_scanned_note(50)

            self.assertEqual(wallet.get_balance(), 100)
            self.assertEqual(wallet.preview_transaction_total(), 50)
            snapshot = wallet.commit_transaction()
            self.assertEqual(snapshot.balance, 150)

    def test_flip_payment_rejects_amount_greater_than_balance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wallet = WalletStore(Path(tmp) / "wallet.db")
            wallet.set_balance(50)
            wallet.commit_transaction()

            wallet.begin_transaction(TransactionType.PAYMENT)
            wallet.add_scanned_note(200)

            with self.assertRaises(ValueError):
                wallet.commit_transaction()
            self.assertEqual(wallet.get_balance(), 50)


if __name__ == "__main__":
    unittest.main()
