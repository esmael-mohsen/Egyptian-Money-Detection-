from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import cv2

from config import LOGGING_CONFIG


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else getattr(logging, LOGGING_CONFIG.level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(level)

    for logger_name in (LOGGING_CONFIG.logger_name, "voice", "audio", "wallet"):
        named_logger = logging.getLogger(logger_name)
        named_logger.setLevel(level)
        named_logger.propagate = True

    logger = logging.getLogger(LOGGING_CONFIG.logger_name)
    return logger


@dataclass
class FPSMeter:
    window: int = 30
    _samples: Deque[float] = field(default_factory=deque)
    _last_time: float = field(default_factory=time.perf_counter)

    def tick(self) -> float:
        now = time.perf_counter()
        delta = max(now - self._last_time, 1e-6)
        self._last_time = now
        fps = 1.0 / delta
        self._samples.append(fps)
        if len(self._samples) > self.window:
            self._samples.popleft()
        return self.current

    @property
    def current(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)


def draw_text_block(frame, lines: list[str], x: int = 15, y: int = 24, line_step: int = 24) -> None:
    for idx, line in enumerate(lines):
        py = y + idx * line_step
        cv2.putText(frame, line, (x, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(frame, line, (x, py), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 1)
