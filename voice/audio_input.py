from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - environment dependent
    sr = None

from voice.config import VoiceConfig


@dataclass(frozen=True)
class AudioPacket:
    audio_data: object
    raw_data: bytes
    sample_rate: int
    sample_width: int

    @property
    def is_empty(self) -> bool:
        return not self.raw_data


class AudioInput:
    """Microphone capture with one-time ambient calibration."""

    def __init__(self, config: VoiceConfig) -> None:
        if sr is None:
            raise RuntimeError("SpeechRecognition is missing.")
        self._config = config
        self._logger = logging.getLogger("voice")
        self._recognizer = sr.Recognizer()
        self._calibrated = False

    @property
    def recognizer(self):
        return self._recognizer

    def listen(self) -> AudioPacket:
        with sr.Microphone(sample_rate=self._config.sample_rate) as source:
            if not self._calibrated:
                self._recognizer.adjust_for_ambient_noise(
                    source,
                    duration=self._config.ambient_noise_duration,
                )
                self._calibrated = True

            audio_data = self._recognizer.listen(
                source,
                timeout=self._config.speech_timeout,
                phrase_time_limit=self._config.phrase_time_limit,
            )

        return AudioPacket(
            audio_data=audio_data,
            raw_data=audio_data.get_raw_data(
                convert_rate=self._config.sample_rate,
                convert_width=2,
            ),
            sample_rate=self._config.sample_rate,
            sample_width=2,
        )
