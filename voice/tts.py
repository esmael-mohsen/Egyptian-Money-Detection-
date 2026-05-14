from __future__ import annotations

import itertools
import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from voice.config import VoiceConfig
from voice.tts_providers import TTSProvider, build_tts_provider


class SpeechPriority(IntEnum):
    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass(order=True, frozen=True)
class SpeechItem:
    priority: int
    sequence: int
    text: str = field(compare=False)
    language: str = field(compare=False)


class TTS:
    """Priority TTS queue with primary/fallback providers and provider-level cache."""

    def __init__(
        self,
        config: VoiceConfig,
        primary: TTSProvider | None = None,
        fallback: TTSProvider | None = None,
    ) -> None:
        self._config = config
        self._logger = logging.getLogger("voice")
        self._primary = primary or build_tts_provider(config.tts.primary, config)
        self._fallback = fallback or build_tts_provider(config.tts.fallback, config)
        self._queue: "queue.PriorityQueue[SpeechItem]" = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = itertools.count()
        self.on_start: Callable[[str], None] | None = None
        self.on_finish: Callable[[str], None] | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="voice-tts", daemon=True)
        self._thread.start()

    def speak(
        self,
        text: str,
        language: str = "en",
        priority: SpeechPriority = SpeechPriority.NORMAL,
    ) -> None:
        if text.strip():
            self._logger.info(
                "TTS_QUEUE text=%r language=%s priority=%s queue_size=%s",
                text,
                language,
                priority.name,
                self._queue.qsize(),
            )
            self._queue.put(
                SpeechItem(
                    priority=int(priority),
                    sequence=next(self._sequence),
                    text=text,
                    language=language,
                )
            )

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(SpeechItem(-1, next(self._sequence), "", "en"))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not item.text:
                continue

            try:
                if self.on_start:
                    self.on_start(item.text)
                self._logger.info(
                    "TTS_START text=%r language=%s priority=%s remaining_queue=%s",
                    item.text,
                    item.language,
                    item.priority,
                    self._queue.qsize(),
                )
                self._speak_with_fallback(item)
            except Exception:
                self._logger.exception("TTS failed for text")
            finally:
                self._logger.info("TTS_FINISH text=%r", item.text)
                if self.on_finish:
                    self.on_finish(item.text)

    def _speak_with_fallback(self, item: SpeechItem) -> None:
        try:
            self._logger.info("TTS_PROVIDER_START provider=%s", self._primary.name)
            self._primary.speak(item.text, item.language)
            self._logger.info("TTS_PROVIDER_SUCCESS provider=%s", self._primary.name)
            return
        except Exception as exc:
            self._logger.warning("Primary TTS failed: %s", exc)

        try:
            self._logger.info("TTS_FALLBACK_START provider=%s", self._fallback.name)
            self._fallback.speak(item.text, item.language)
            self._logger.info("TTS_FALLBACK_SUCCESS provider=%s", self._fallback.name)
        except Exception as exc:
            self._logger.warning("Fallback TTS failed: %s", exc)
