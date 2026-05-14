from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Callable

from voice.commands import CommandId
from voice.config import VoiceConfig
from voice.listener import ListenerResult, VoiceListener
from voice.responses import response
from voice.tts import SpeechPriority, TTS


class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    SPEAKING = "SPEAKING"
    ACTIVE_SESSION = "ACTIVE_SESSION"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


VoiceHandler = Callable[[ListenerResult], str]


class VoiceManager:
    """Always-on command session with pro STT/TTS providers and no wake word."""

    def __init__(
        self,
        config: VoiceConfig | None = None,
        listener: VoiceListener | None = None,
        tts: TTS | None = None,
    ) -> None:
        self._config = config or VoiceConfig()
        self._logger = logging.getLogger("voice")
        self._state_lock = threading.Lock()
        self._state = VoiceState.IDLE
        self._language = self._config.default_language
        self._listener = listener or VoiceListener(self._config)
        self._tts = tts or TTS(self._config)
        self._tts.on_start = lambda _text: self._mark_tts_started()
        self._tts.on_finish = lambda _text: self._mark_tts_finished()
        self._handlers: dict[CommandId, VoiceHandler] = {}
        self._running = threading.Event()
        self._command_thread: threading.Thread | None = None
        self._tts_busy = threading.Event()
        self._last_response = ""
        self._last_state = self._state
        self._logger.info(
            "VOICE_MANAGER_INIT language=%s no_barge_in=%s min_confidence=%.2f",
            self._language,
            self._config.no_barge_in,
            self._config.min_command_confidence,
        )

    @property
    def state(self) -> VoiceState:
        with self._state_lock:
            return self._state

    @property
    def language(self) -> str:
        return self._language

    def register_handler(self, command_id: CommandId | str, handler: VoiceHandler) -> None:
        self._handlers[CommandId(command_id)] = handler
        self._logger.info("VOICE_HANDLER_REGISTERED command=%s", CommandId(command_id).value)

    def start(self) -> None:
        if self._running.is_set():
            return

        self._running.set()
        self._tts.start()
        self._logger.info("VOICE_SESSION_START welcome=%r", self._config.welcome_text_en)
        self._speak(self._config.welcome_text_en, SpeechPriority.HIGH)
        self._command_thread = threading.Thread(
            target=self._command_loop,
            name="voice-session",
            daemon=True,
        )
        self._command_thread.start()

    def stop(self) -> None:
        self._set_state(VoiceState.SHUTDOWN)
        self._running.clear()
        self._logger.info("VOICE_SESSION_STOP")
        self._listener.shutdown()
        self._tts.stop()
        if self._command_thread and self._command_thread.is_alive():
            self._command_thread.join(timeout=1.5)

    def set_language(self, language: str) -> None:
        if language not in {"en", "ar"}:
            raise ValueError("language must be 'en' or 'ar'")
        self._language = language
        self._logger.info("VOICE_LANGUAGE_SET language=%s", language)
        setter = getattr(self._listener, "set_language", None)
        if setter:
            setter(language)

    def _command_loop(self) -> None:
        while self._running.is_set():
            if self._config.no_barge_in and self._tts_busy.is_set():
                self._logger.debug("VOICE_WAIT_TTS_BUSY")
                time.sleep(0.05)
                continue

            self._set_state(VoiceState.LISTENING)
            result = self._listener.listen_for_command()
            self._set_state(VoiceState.PROCESSING)
            self._logger.info(
                "VOICE_COMMAND_RESULT command=%s raw=%r normalized=%r confidence=%.2f amount=%s provider=%s latency_ms=%s",
                result.command_id.value if result.command_id else None,
                result.raw_text,
                result.normalized_text,
                result.confidence,
                result.amount,
                result.stt_provider,
                result.stt_latency_ms,
            )

            if not result.command_id:
                self._handle_unparsed_result(result)
                continue

            self._set_state(VoiceState.EXECUTING)
            reply = self._execute(result)
            if reply:
                self._last_response = reply
                self._speak(reply, self._priority_for(result))
            else:
                self._set_state(VoiceState.ACTIVE_SESSION)

    def _handle_unparsed_result(self, result: ListenerResult) -> None:
        self._logger.info(
            "VOICE_UNPARSED raw=%r confidence=%.2f amount=%s alias=%r",
            result.raw_text,
            result.confidence,
            result.amount,
            result.matched_alias,
        )
        if result.amount is not None and CommandId.SET_BALANCE in self._handlers:
            try:
                reply = self._handlers[CommandId.SET_BALANCE](result)
            except Exception:
                self._logger.exception("Amount fallback command failed")
                reply = response("voice_error", self._language)

            if reply:
                self._last_response = reply
                self._logger.info("VOICE_AMOUNT_FALLBACK_REPLY reply=%r", reply)
                self._speak(reply, SpeechPriority.NORMAL)
            return

        if 0.0 < result.confidence < self._config.min_command_confidence:
            self._logger.info("VOICE_LOW_CONFIDENCE confidence=%.2f", result.confidence)
            self._speak(response("low_confidence", self._language), SpeechPriority.HIGH)
        self._set_state(VoiceState.ACTIVE_SESSION)

    def _execute(self, result: ListenerResult) -> str:
        self._logger.info("VOICE_EXECUTE command=%s", result.command_id.value if result.command_id else None)
        if result.command_id == CommandId.SWITCH_ARABIC:
            self.set_language("ar")
            return response("arabic_on", "ar")

        if result.command_id == CommandId.SWITCH_ENGLISH:
            self.set_language("en")
            return response("english_on", "en")

        if result.command_id == CommandId.REPEAT:
            return self._last_response or response("unknown", self._language)

        handler = self._handlers.get(result.command_id)
        if handler is None:
            return response("unknown", self._language)

        try:
            reply = handler(result)
            self._logger.info("VOICE_EXECUTE_REPLY command=%s reply=%r", result.command_id.value, reply)
            return reply
        except Exception:
            self._logger.exception("Voice command failed: %s", result.command_id)
            self._set_state(VoiceState.ERROR)
            return response("voice_error", self._language)

    def _speak(self, text: str, priority: SpeechPriority = SpeechPriority.NORMAL) -> None:
        self._logger.info("VOICE_SPEAK_REQUEST text=%r language=%s priority=%s", text, self._language, priority.name)
        self._tts_busy.set()
        self._tts.speak(text, self._language, priority)

    def _mark_tts_started(self) -> None:
        self._tts_busy.set()
        self._set_state(VoiceState.SPEAKING)

    def _mark_tts_finished(self) -> None:
        self._tts_busy.clear()
        if self._running.is_set():
            self._set_state(VoiceState.ACTIVE_SESSION)

    def _set_state(self, state: VoiceState) -> None:
        with self._state_lock:
            if self._state != state:
                self._logger.info("VOICE_STATE %s -> %s", self._state.value, state.value)
            self._state = state

    @staticmethod
    def _priority_for(result: ListenerResult) -> SpeechPriority:
        if result.command_id in {CommandId.CONFIRM, CommandId.CANCEL, CommandId.EXIT_APP}:
            return SpeechPriority.HIGH
        return SpeechPriority.NORMAL
