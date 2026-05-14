from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

try:
    import edge_tts
except ImportError:  # pragma: no cover - environment dependent
    edge_tts = None

try:
    from playsound import playsound
except ImportError:  # pragma: no cover - environment dependent
    playsound = None

from voice.config import VoiceConfig


class TTSProvider(Protocol):
    name: str

    def speak(self, text: str, language: str) -> None:
        ...


class EdgeTTSProvider:
    name = "edge"

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("voice")
        self._cache_dir = config.tts.cache_dir / "edge"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def speak(self, text: str, language: str) -> None:
        if edge_tts is None or playsound is None:
            raise RuntimeError("edge-tts or playsound is missing.")
        audio_path = self._cache_path(text, language)
        if not audio_path.exists():
            self._logger.info("TTS_CACHE_MISS provider=edge path=%s", audio_path)
            asyncio.run(self._save(text, language, audio_path))
        else:
            self._logger.info("TTS_CACHE_HIT provider=edge path=%s", audio_path)
        self._logger.info("TTS_PLAY provider=edge path=%s", audio_path)
        playsound(str(audio_path), block=True)

    async def _save(self, text: str, language: str, audio_path: Path) -> None:
        communicator = edge_tts.Communicate(
            text=text,
            voice=self._voice_for_language(language),
            rate=self._config.tts.rate,
            volume=self._config.tts.volume,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            temp_path = Path(temp_audio.name)
        try:
            await communicator.save(str(temp_path))
            shutil.move(str(temp_path), str(audio_path))
            self._logger.info("TTS_CACHE_WRITE provider=edge path=%s", audio_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _voice_for_language(self, language: str) -> str:
        return self._config.tts.voice_ar if language == "ar" else self._config.tts.voice_en

    def _cache_path(self, text: str, language: str) -> Path:
        key = hashlib.sha1(f"{language}:{text}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.mp3"


class PiperTTSProvider:
    name = "piper"

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("voice")

    def speak(self, text: str, language: str) -> None:
        piper = shutil.which("piper")
        if not piper:
            raise RuntimeError("piper executable was not found.")

        voice_path = self._voice_for_language(language)
        if not voice_path:
            raise RuntimeError("Piper voice model path is not configured.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = Path(temp_audio.name)
        try:
            subprocess.run(
                [piper, "--model", str(voice_path), "--output_file", str(temp_path)],
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _play_audio_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _voice_for_language(self, language: str) -> Path | None:
        return self._config.tts.piper_voice_ar if language == "ar" else self._config.tts.piper_voice_en


class OpenAITTSProvider:
    name = "openai"

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        self._logger = logging.getLogger("voice")
        self._cache_dir = config.tts.cache_dir / "openai"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def speak(self, text: str, language: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai package is not installed.") from exc

        audio_path = self._cache_path(text, language)
        if not audio_path.exists():
            self._logger.info("TTS_CACHE_MISS provider=openai path=%s", audio_path)
            client = OpenAI()
            response = client.audio.speech.create(
                model=self._config.tts.optional_openai_model,
                voice="alloy",
                input=text,
            )
            response.write_to_file(audio_path)
        else:
            self._logger.info("TTS_CACHE_HIT provider=openai path=%s", audio_path)
        _play_audio_file(audio_path)

    def _cache_path(self, text: str, language: str) -> Path:
        key = hashlib.sha1(f"{language}:{text}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.mp3"


class SystemTTSProvider:
    name = "system"

    def speak(self, text: str, language: str) -> None:
        if os.name == "nt":
            escaped = text.replace("'", "''")
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$s.Speak('{escaped}')"
                    ),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if not espeak:
            raise RuntimeError("No system TTS backend available.")
        subprocess.run(
            [espeak, text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_tts_provider(name: str, config: VoiceConfig) -> TTSProvider:
    normalized = name.lower()
    if normalized == "edge":
        return EdgeTTSProvider(config)
    if normalized == "piper":
        return PiperTTSProvider(config)
    if normalized == "openai":
        return OpenAITTSProvider(config)
    if normalized == "system":
        return SystemTTSProvider()
    raise ValueError(f"Unsupported TTS provider: {name}")


def _play_audio_file(path: Path) -> None:
    if playsound is not None:
        playsound(str(path), block=True)
        return

    for player in ("aplay", "ffplay", "mpg123"):
        executable = shutil.which(player)
        if executable:
            subprocess.run(
                [executable, str(path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

    raise RuntimeError("No audio playback backend available.")
