from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - environment dependent
    sr = None

from voice.audio_input import AudioInput
from voice.command_catalog import CommandCatalog
from voice.commands import CommandId, ParsedCommand, parse_command
from voice.config import VoiceConfig
from voice.stt import STTResult, STTService, build_stt_service


@dataclass(frozen=True)
class ListenerResult:
    command_id: CommandId | None
    raw_text: str
    normalized_text: str
    amount: int | None = None
    confidence: float = 0.0
    stt_provider: str = ""
    stt_latency_ms: int = 0
    matched_alias: str = ""
    requires_amount: bool = False
    confirmation_required: bool = False

    @classmethod
    def from_parsed(cls, parsed: ParsedCommand, stt_result: STTResult) -> "ListenerResult":
        return cls(
            command_id=parsed.command_id,
            raw_text=parsed.raw_text,
            normalized_text=parsed.normalized_text,
            amount=parsed.amount,
            confidence=parsed.confidence,
            stt_provider=stt_result.provider,
            stt_latency_ms=stt_result.latency_ms,
            matched_alias=parsed.matched_alias,
            requires_amount=parsed.requires_amount,
            confirmation_required=parsed.confirmation_required,
        )


class VoiceListener:
    """Audio capture + STT provider fallback + command parsing."""

    def __init__(
        self,
        config: VoiceConfig,
        audio_input: AudioInput | None = None,
        stt_service: STTService | None = None,
        catalog: CommandCatalog | None = None,
    ) -> None:
        self._config = config
        self._logger = logging.getLogger("voice")
        self._audio_input = audio_input or AudioInput(config)
        self._stt_service = stt_service or build_stt_service(config, self._audio_input.recognizer)
        self._catalog = catalog or CommandCatalog.load(config.command_catalog)

    def listen_for_command(self) -> ListenerResult:
        self._logger.info("VOICE_LISTEN_START")
        stt_result = self._listen_once()
        if not stt_result.text.strip():
            self._logger.info("VOICE_LISTEN_RETRY_EMPTY_STT provider=%s", stt_result.provider)
            stt_result = self._listen_once()

        self._logger.info(
            "STT_RESULT provider=%s latency_ms=%s confidence=%.2f text=%r",
            stt_result.provider,
            stt_result.latency_ms,
            stt_result.confidence,
            stt_result.text,
        )
        parsed = parse_command(
            stt_result.text,
            catalog=self._catalog,
            min_confidence=self._config.min_command_confidence,
        )
        self._logger.info(
            "COMMAND_PARSE command=%s confidence=%.2f amount=%s alias=%r normalized=%r",
            parsed.command_id.value if parsed.command_id else None,
            parsed.confidence,
            parsed.amount,
            parsed.matched_alias,
            parsed.normalized_text,
        )
        return ListenerResult.from_parsed(parsed, stt_result)

    def detect_wake_word(self) -> bool:
        return False

    def set_language(self, language: str) -> None:
        setter = getattr(self._stt_service, "set_language", None)
        if setter:
            setter(language)

    def shutdown(self) -> None:
        return

    def _listen_once(self) -> STTResult:
        try:
            audio = self._audio_input.listen()
            if audio.is_empty:
                self._logger.info("AUDIO_CAPTURE_EMPTY")
                return STTResult("", "", 0.0, "none", 0)
            self._logger.info(
                "AUDIO_CAPTURED bytes=%s sample_rate=%s sample_width=%s",
                len(audio.raw_data),
                audio.sample_rate,
                audio.sample_width,
            )
            return self._stt_service.transcribe(audio)
        except Exception as exc:
            if sr is not None and isinstance(exc, sr.WaitTimeoutError):
                self._logger.info("AUDIO_TIMEOUT no speech before timeout")
                return STTResult("", "", 0.0, "none", 0)
            self._logger.warning("Voice listening failed: %s", exc)
            return STTResult("", "", 0.0, "none", 0)
