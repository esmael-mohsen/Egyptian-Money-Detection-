from __future__ import annotations

import unittest

from config import CounterConfig
from scripts.counter import CounterEngine
from scripts.tracker import TrackObject


def make_track(track_id: int, bbox: tuple[int, int, int, int] = (10, 10, 110, 80)) -> TrackObject:
    return TrackObject(
        track_id=track_id,
        class_name="50 Pounds",
        bbox=bbox,
        confidence=0.95,
        age_frames=5,
        is_stable=True,
    )


class CounterEngineTests(unittest.TestCase):
    def test_same_track_is_counted_once(self) -> None:
        counter = CounterEngine(CounterConfig(stable_required_frames=3))
        track = make_track(1)

        self.assertEqual(len(counter.update([track])), 1)
        self.assertEqual(len(counter.update([track])), 0)
        self.assertEqual(counter.get_total(), 50)

    def test_reset_clears_session_total(self) -> None:
        counter = CounterEngine(CounterConfig(stable_required_frames=3))
        counter.update([make_track(1)])
        counter.reset()

        self.assertEqual(counter.get_total(), 0)
        self.assertIsNone(counter.get_last_count_event())

    def test_recent_memory_blocks_near_duplicate_track(self) -> None:
        counter = CounterEngine(CounterConfig(stable_required_frames=3))
        first = make_track(1)
        duplicate = make_track(2, bbox=(12, 12, 112, 82))

        self.assertEqual(len(counter.update([first])), 1)
        self.assertEqual(len(counter.update([duplicate])), 0)
        self.assertEqual(counter.get_total(), 50)


if __name__ == "__main__":
    unittest.main()
