from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    from vosk import KaldiRecognizer, Model
except ImportError:  # pragma: no cover - environment dependent
    KaldiRecognizer = None
    Model = None

from voice.audio_input import AudioPacket
from voice.config import VoiceConfig


@dataclass(frozen=True)
class STTResult:
    text: str
    language: str
    confidence: float
    provider: str
    latency_ms: int


class STTProvider(Protocol):
    name: str

    def transcribe(self, audio: AudioPacket) -> STTResult:
        ...


class GoogleSTTProvider:
    name = "google"

    def __init__(self, config: VoiceConfig, recognizer) -> None:
        self._config = config
        self._recognizer = recognizer
        self._language = config.stt.language

    def transcribe(self, audio: AudioPacket) -> STTResult:
        started = time.perf_counter()
        text = self._recognizer.recognize_google(
            audio.audio_data,
            language=self._language,
        )
        return STTResult(
            text=text,
            language=self._language,
            confidence=0.9 if text.strip() else 0.0,
            provider=self.name,
            latency_ms=_elapsed_ms(started),
        )

    def set_language(self, language: str) -> None:
        self._language = self._config.stt.fallback_language if language == "ar" else self._config.stt.language


class VoskSTTProvider:
    name = "vosk"

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("voice")
        self._model = None

    def transcribe(self, audio: AudioPacket) -> STTResult:
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("Vosk is missing.")

        started = time.perf_counter()
        recognizer = KaldiRecognizer(self._get_model(), audio.sample_rate)
        recognizer.AcceptWaveform(audio.raw_data)
        result = json.loads(recognizer.FinalResult() or "{}")
        text = result.get("text", "")
        return STTResult(
            text=text,
            language="offline",
            confidence=float(result.get("confidence", 0.75 if text else 0.0)),
            provider=self.name,
            latency_ms=_elapsed_ms(started),
        )

    def _get_model(self):
        if self._model is not None:
            return self._model

        candidates = [
            Path(self._config.vosk_model_path_en),
            Path(self._config.vosk_model_path_ar),
        ]
        for path in candidates:
            if path.exists():
                self._model = Model(str(path))
                return self._model

        raise RuntimeError("No Vosk model found.")


class OpenAISTTProvider:
    name = "openai"

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._language = config.stt.language

    def transcribe(self, audio: AudioPacket) -> STTResult:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is not installed.") from exc

        started = time.perf_counter()
        client = OpenAI()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
        try:
            _write_wav(temp_path, audio)
            with open(temp_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=self._config.stt.optional_openai_model,
                    file=audio_file,
                )
            text = getattr(response, "text", "")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        return STTResult(
            text=text,
            language=self._language,
            confidence=0.95 if text.strip() else 0.0,
            provider=self.name,
            latency_ms=_elapsed_ms(started),
        )

    def set_language(self, language: str) -> None:
        self._language = self._config.stt.fallback_language if language == "ar" else self._config.stt.language


class STTService:
    def __init__(self, primary: STTProvider, fallback: STTProvider | None = None) -> None:
        self._primary = primary
        self._fallback = fallback
        self._logger = logging.getLogger("voice")

    def transcribe(self, audio: AudioPacket) -> STTResult:
        self._logger.info("STT_PROVIDER_START provider=%s", self._primary.name)
        try:
            result = self._primary.transcribe(audio)
            if result.text.strip():
                self._logger.info(
                    "STT_PROVIDER_SUCCESS provider=%s latency_ms=%s text=%r",
                    result.provider,
                    result.latency_ms,
                    result.text,
                )
                return result
            self._logger.info("STT_PROVIDER_EMPTY provider=%s", result.provider)
        except Exception as exc:
            self._logger.warning("Primary STT failed: %s", exc)

        if self._fallback is None:
            return STTResult("", "", 0.0, "none", 0)

        self._logger.info("STT_FALLBACK_START provider=%s", self._fallback.name)
        try:
            result = self._fallback.transcribe(audio)
            self._logger.info(
                "STT_FALLBACK_RESULT provider=%s latency_ms=%s text=%r",
                result.provider,
                result.latency_ms,
                result.text,
            )
            return result
        except Exception as exc:
            self._logger.warning("Fallback STT failed: %s", exc)
            return STTResult("", "", 0.0, "none", 0)

    def set_language(self, language: str) -> None:
        for provider in (self._primary, self._fallback):
            setter = getattr(provider, "set_language", None)
            if setter:
                setter(language)


def build_stt_service(config: VoiceConfig, recognizer) -> STTService:
    return STTService(
        primary=_build_provider(config.stt.primary, config, recognizer),
        fallback=_build_provider(config.stt.fallback, config, recognizer),
    )


def _build_provider(name: str, config: VoiceConfig, recognizer) -> STTProvider:
    normalized = name.lower()
    if normalized == "google":
        return GoogleSTTProvider(config, recognizer)
    if normalized == "vosk":
        return VoskSTTProvider(config)
    if normalized == "openai":
        return OpenAISTTProvider(config)
    raise ValueError(f"Unsupported STT provider: {name}")


def _write_wav(path: str, audio: AudioPacket) -> None:
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(audio.sample_width)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(audio.raw_data)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
