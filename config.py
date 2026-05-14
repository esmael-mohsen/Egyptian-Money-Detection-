from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path = Path("models/best.pt")
    conf_threshold: float = 0.68
    iou_threshold: float = 0.45
    max_det: int = 50
    use_half: bool = False


@dataclass(frozen=True)
class TrackerConfig:
    min_iou_match: float = 0.25
    max_missing_frames: int = 20
    stable_after_frames: int = 8
    smoothing_alpha: float = 0.6
    min_box_area: int = 1500


@dataclass(frozen=True)
class CounterConfig:
    confidence_threshold: float = 0.68
    stable_required_frames: int = 8
    recent_memory_limit: int = 64


@dataclass(frozen=True)
class FlipConfig:
    enabled: bool = True
    profile: str = "slow_safe"
    gate_orientation: str = "vertical"
    gate_center_ratio: float = 0.50
    gate_width_ratio: float = 0.12
    confidence_threshold: float = 0.62
    cooldown_seconds: float = 0.80
    prediction_buffer_frames: int = 12
    min_confirmed_frames: int = 3
    vote_ratio_threshold: float = 0.70
    class_margin_threshold: float = 0.12
    min_motion_pixels: int = 12
    dwell_count_enabled: bool = True
    dwell_required_frames: int = 5
    dwell_seconds: float = 0.45
    exit_clear_frames: int = 3
    enter_zone_width_ratio: float = 0.18
    count_line_ratio: float = 0.50
    exit_zone_width_ratio: float = 0.18
    min_bbox_area: int = 1200
    max_uncertain_captures_per_minute: int = 20
    capture_enabled: bool = True
    capture_dir: Path = Path("data/captures/flip_count")
    capture_metadata: bool = True
    save_counted: bool = True
    save_uncertain: bool = True


@dataclass(frozen=True)
class VoiceSTTConfig:
    primary: str = "google"
    fallback: str = "vosk"
    optional_openai_model: str = "gpt-4o-mini-transcribe"


@dataclass(frozen=True)
class VoiceTTSConfig:
    primary: str = "edge"
    fallback: str = "system" if os.name == "nt" else "piper"
    optional_openai_model: str = "gpt-4o-mini-tts"
    cache_dir: Path = Path("data/tts_cache")


@dataclass(frozen=True)
class VoiceRuntimeConfig:
    enabled: bool = True
    enable_wake_word: bool = False
    default_language: str = "en"
    online_first: bool = True
    welcome_text_en: str = "Money detection is running"
    command_catalog: Path = Path("voice/command_catalog.yaml")
    min_command_confidence: float = 0.72
    no_barge_in: bool = True
    stt: VoiceSTTConfig = field(default_factory=VoiceSTTConfig)
    tts: VoiceTTSConfig = field(default_factory=VoiceTTSConfig)


@dataclass(frozen=True)
class AudioConfig:
    count_beep_enabled: bool = True
    beep_frequency: int = 880
    beep_duration_ms: int = 120


@dataclass(frozen=True)
class WalletConfig:
    db_path: Path = Path("data/wallet.db")


@dataclass(frozen=True)
class RuntimeConfig:
    camera_source: int | str = os.getenv("EGY_MONEY_CAMERA_SOURCE", "0")
    camera_width: int = 640
    camera_height: int = 480
    target_fps: int = 15
    queue_size: int = 64
    show_debug_overlay: bool = True
    headless: bool = False
    debug_window: bool = True
    default_mode: str = "IDLE"
    inference_every_n_frames: int = 2
    verbose_logging: bool = True
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    counter: CounterConfig = field(default_factory=CounterConfig)
    flip: FlipConfig = field(default_factory=FlipConfig)
    voice: VoiceRuntimeConfig = field(default_factory=VoiceRuntimeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)


@dataclass(frozen=True)
class LoggingConfig:
    logger_name: str = "currency_runtime"
    level: str = "INFO"


APP_CONFIG = RuntimeConfig()
LOGGING_CONFIG = LoggingConfig()


def resolve_camera_source(source: int | str) -> int | str:
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source
