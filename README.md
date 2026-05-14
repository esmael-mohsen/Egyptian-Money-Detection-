# Egyptian Money Detection

Assistive Egyptian banknote detection and wallet tracking system built for blind and visually impaired users. The project combines YOLO-based banknote detection, voice commands, bilingual responses, persistent wallet logic, and a Raspberry Pi friendly runtime.

The runtime is designed to be launched by an external voice assistant. It does not use an internal wake word by default. When the session starts, it announces:

```text
Money detection is running.
```

## Highlights

- Detects Egyptian banknotes: 5, 10, 20, 50, 100, and 200 EGP.
- Supports English-first voice commands with Arabic aliases.
- Uses cloud-first STT/TTS with offline fallbacks.
- Plays a short bell when a note is counted successfully.
- Tracks a persistent local wallet balance using SQLite.
- Supports confirmed deposit and payment flows.
- Includes flip/stack counting for natural hand-counting motion.
- Saves optional flip-count captures with JSON metadata for future model improvement.
- Runs with a headless-friendly design for Raspberry Pi 4 Model B.

## Core Flows

### Session Scan

Use this when you only want to count the currently visible notes.

```text
start scan
total
stop scan
```

### Wallet Deposit

Deposit mode scans money that will be added to the wallet only after confirmation.

```text
start deposit
finish deposit
confirm
```

### Wallet Payment

Payment mode scans money that will be deducted from the wallet only after confirmation.

```text
start payment
finish payment
confirm
```

If the payment total is greater than the wallet balance, the transaction is rejected.

### Flip / Stack Counting

Flip mode is designed for counting a bundle naturally by showing one note at a time. It uses enter/count/exit zones, vote confidence, and motion or stable dwell logic before counting.

```text
flip count
total
```

Wallet-aware stack flows:

```text
deposit stack
finish stack deposit
confirm
```

```text
pay from stack
finish stack payment
confirm
```

## Voice Commands

Common English commands:

- `start scan`, `scan money`, `count notes`
- `stop scan`, `end scan`
- `total`, `session total`, `how much do I have`
- `last note`, `repeat`
- `wallet balance`, `my balance`
- `set balance to 500`
- `start deposit`, `finish deposit`
- `start payment`, `finish payment`
- `flip count`, `count stack`, `count bundle`
- `deposit stack`, `pay from stack`
- `confirm`, `cancel`
- `switch to Arabic`, `switch to English`
- `exit`

Arabic aliases are supported in `voice/command_catalog.yaml`.

## Voice Pipeline

STT providers:

- Primary: Google STT through `SpeechRecognition`
- Fallback: Vosk using local models
- Optional: OpenAI STT if `OPENAI_API_KEY` is configured

TTS providers:

- Primary: `edge-tts`
- Fallback: system voice on Windows or Piper on Linux/Raspberry Pi
- Optional: OpenAI TTS if `OPENAI_API_KEY` is configured

The system uses short assistive responses and does not listen while speaking.

## Runtime Modes

- `IDLE`: no detection
- `SCAN`: count visible notes for the current session
- `DEPOSIT`: scan notes into a pending wallet deposit
- `PAYMENT`: scan notes into a pending wallet payment
- `FLIP_SCAN`: count a stack/bundle using motion-based flip counting
- `FLIP_DEPOSIT`: count a stack into a pending deposit
- `FLIP_PAYMENT`: count a stack into a pending payment

Wallet balance changes only after `confirm`.

## Flip Counting Logic

Flip mode uses a safer motion-based engine:

- State machine: `WAITING`, `ENTERING_GATE`, `CONFIRMING`, `COUNTED_WAIT_EXIT`, `EXITED`
- Enter/count/exit zones instead of one simple gate
- Class voting over a prediction buffer
- Confidence threshold and class margin checks
- Motion crossing check for natural flipping
- Stable dwell fallback for slow or held notes
- Rate-limited uncertain captures
- Image + JSON metadata for counted and uncertain events

Captured data is stored under:

```text
data/captures/flip_count/counted/
data/captures/flip_count/uncertain/
```

These folders are ignored by Git because they may contain personal camera data.

## Project Structure

```text
.
|-- main.py
|-- config.py
|-- requirements.txt
|-- scripts/
|   |-- audio_feedback.py
|   |-- counter.py
|   |-- detector.py
|   |-- flip_counter.py
|   |-- runtime.py
|   |-- tracker.py
|   `-- wallet.py
|-- voice/
|   |-- audio_input.py
|   |-- command_catalog.py
|   |-- command_catalog.yaml
|   |-- commands.py
|   |-- config.py
|   |-- listener.py
|   |-- responses.py
|   |-- stt.py
|   |-- tts.py
|   |-- tts_providers.py
|   `-- voice_manager.py
`-- tests/
```

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/Raspberry Pi:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyAudio may require system audio headers on Linux/Raspberry Pi:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

## Required Local Models

Large model files are not committed to GitHub. Create the following paths locally:

```text
models/best.pt
models/vosk-en/
models/vosk-ar/
```

Minimum required:

- `models/best.pt` for YOLO banknote detection
- `models/vosk-en/` for offline English STT fallback
- `models/vosk-ar/` for optional offline Arabic STT fallback

## Run

```powershell
python main.py
```

The default camera source is `0`. You can override it with an environment variable:

```powershell
$env:EGY_MONEY_CAMERA_SOURCE="1"
python main.py
```

## Use a Phone Camera

The runtime accepts OpenCV-compatible camera sources:

- Local webcam index: `0`, `1`, `2`
- MJPEG stream: `http://PHONE_IP:PORT/video`
- RTSP stream: `rtsp://...`

Example:

```powershell
$env:EGY_MONEY_CAMERA_SOURCE="http://192.168.1.4:4747/video"
python main.py
```

Make sure the phone and computer are connected to the same Wi-Fi network.

## Important Configuration

Main runtime config lives in `config.py`.

Useful defaults:

- `voice.default_language="en"`
- `voice.enable_wake_word=False`
- `voice.welcome_text_en="Money detection is running"`
- `voice.min_command_confidence=0.72`
- `runtime.default_mode="IDLE"`
- `runtime.inference_every_n_frames=2`
- `flip.profile="slow_safe"`
- `flip.confidence_threshold=0.62`
- `flip.cooldown_seconds=0.80`
- `flip.dwell_count_enabled=True`
- `wallet.db_path="data/wallet.db"`

## Tests

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

Current coverage includes:

- Command parsing and Arabic/English aliases
- STT/TTS provider fallback behavior
- Wallet persistence and confirmation logic
- Counter/tracker duplicate prevention
- Flip counting state machine, voting, quality checks, and capture metadata

## Raspberry Pi Notes

Recommended starting point:

- Raspberry Pi 4 Model B
- USB camera or Raspberry Pi compatible camera
- 640x480 camera profile
- 15 FPS target
- Headless runtime unless debugging the GUI
- Offline Vosk and Piper models installed locally for fully offline use

For production use, keep captured data and wallet database private.

## Repository Hygiene

The following are intentionally ignored:

- `.venv/`
- `__pycache__/`
- `data/`
- `models/`
- `.env`
- generated audio/cache/capture files

This keeps the GitHub repository source-only and avoids committing private or large local files.

## License

No license has been selected yet. Add a license before distributing or accepting external contributions.
