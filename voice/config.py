from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class STTConfig:
    primary: str = "google"
    fallback: str = "vosk"
    optional_openai_model: str = "gpt-4o-mini-transcribe"
    language: str = "en-US"
    fallback_language: str = "ar-EG"


@dataclass(frozen=True)
class TTSConfig:
    primary: str = "edge"
    fallback: str = "system" if os.name == "nt" else "piper"
    optional_openai_model: str = "gpt-4o-mini-tts"
    cache_dir: Path = Path("data/tts_cache")
    voice_en: str = "en-US-JennyNeural"
    voice_ar: str = "ar-EG-SalmaNeural"
    piper_voice_en: Path | None = None
    piper_voice_ar: Path | None = None
    rate: str = "+0%"
    volume: str = "+0%"


@dataclass(frozen=True)
class VoiceConfig:
    default_language: str = "en"
    enable_wake_word: bool = False
    online_first: bool = True
    command_catalog: Path = Path("voice/command_catalog.yaml")
    min_command_confidence: float = 0.72
    no_barge_in: bool = True
    welcome_text_en: str = "Money detection is running"
    vosk_model_path_en: str = "models/vosk-en"
    vosk_model_path_ar: str = "models/vosk-ar"
    sample_rate: int = 16000
    chunk_size: int = 4096
    speech_timeout: float = 6.0
    phrase_time_limit: float = 4.0
    ambient_noise_duration: float = 0.5
    log_level: str = "INFO"
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


VOICE_DEFAULT_RESPONSES = {
    "UNKNOWN_EN": "Say that again.",
    "UNKNOWN_AR": "كرر الأمر.",
    "ERROR_EN": "Voice error.",
    "ERROR_AR": "خطأ في الصوت.",
    "LOW_CONFIDENCE_EN": "I am not sure. Repeat.",
    "LOW_CONFIDENCE_AR": "لست متأكدا. كرر.",
}
