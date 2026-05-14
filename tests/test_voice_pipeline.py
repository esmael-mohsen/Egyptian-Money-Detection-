from __future__ import annotations

import time
import unittest
from dataclasses import dataclass

from voice.audio_input import AudioPacket
from voice.command_catalog import CommandCatalog
from voice.config import VoiceConfig
from voice.listener import VoiceListener
from voice.responses import response
from voice.stt import STTResult, STTService
from voice.tts import SpeechPriority, TTS


@dataclass
class FakeProvider:
    name: str
    text: str = ""
    fail: bool = False
    spoken: list[str] | None = None

    def transcribe(self, audio: AudioPacket) -> STTResult:
        if self.fail:
            raise RuntimeError("boom")
        return STTResult(self.text, "en-US", 0.9 if self.text else 0.0, self.name, 1)

    def speak(self, text: str, language: str) -> None:
        if self.fail:
            raise RuntimeError("boom")
        assert self.spoken is not None
        self.spoken.append(text)


class FakeAudioInput:
    @property
    def recognizer(self):
        return object()

    def listen(self) -> AudioPacket:
        return AudioPacket(audio_data=object(), raw_data=b"1234", sample_rate=16000, sample_width=2)


class VoicePipelineTests(unittest.TestCase):
    def test_stt_fallback_after_primary_failure(self) -> None:
        service = STTService(
            primary=FakeProvider("primary", fail=True),
            fallback=FakeProvider("fallback", text="wallet balance"),
        )
        listener = VoiceListener(
            VoiceConfig(),
            audio_input=FakeAudioInput(),
            stt_service=service,
            catalog=CommandCatalog.default(),
        )

        result = listener.listen_for_command()
        self.assertEqual(result.stt_provider, "fallback")
        self.assertEqual(result.raw_text, "wallet balance")

    def test_empty_stt_result_produces_no_command(self) -> None:
        service = STTService(
            primary=FakeProvider("primary", text=""),
            fallback=FakeProvider("fallback", text=""),
        )
        listener = VoiceListener(
            VoiceConfig(),
            audio_input=FakeAudioInput(),
            stt_service=service,
            catalog=CommandCatalog.default(),
        )

        result = listener.listen_for_command()
        self.assertIsNone(result.command_id)

    def test_tts_priority_queue(self) -> None:
        spoken: list[str] = []
        tts = TTS(
            VoiceConfig(),
            primary=FakeProvider("primary", spoken=spoken),
            fallback=FakeProvider("fallback", spoken=spoken),
        )
        tts.speak("normal", priority=SpeechPriority.NORMAL)
        tts.speak("high", priority=SpeechPriority.HIGH)
        tts.start()
        time.sleep(0.2)
        tts.stop()

        self.assertGreaterEqual(len(spoken), 2)
        self.assertEqual(spoken[0], "high")

    def test_language_response_templates(self) -> None:
        self.assertEqual(response("ready", "en"), "Money detection is running.")
        self.assertEqual(response("ready", "ar"), "كاشف الفلوس يعمل الآن.")


if __name__ == "__main__":
    unittest.main()
