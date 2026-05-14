from __future__ import annotations

import logging
import sys

from config import AudioConfig


class AudioFeedback:
    """Small cross-platform count beep that is independent from TTS."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("audio")

    def count_beep(self) -> None:
        if not self._config.count_beep_enabled:
            self._logger.info("AUDIO_BEEP_SKIPPED enabled=False")
            return

        try:
            if sys.platform.startswith("win"):
                import winsound

                self._logger.info(
                    "AUDIO_BEEP provider=winsound frequency=%s duration_ms=%s",
                    self._config.beep_frequency,
                    self._config.beep_duration_ms,
                )
                winsound.Beep(
                    self._config.beep_frequency,
                    self._config.beep_duration_ms,
                )
                return

            self._logger.info("AUDIO_BEEP provider=terminal_bell")
            print("\a", end="", flush=True)
        except Exception:
            self._logger.warning("Count beep failed", exc_info=True)
