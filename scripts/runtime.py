from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum

import cv2

from config import APP_CONFIG, RuntimeConfig, resolve_camera_source
from scripts.audio_feedback import AudioFeedback
from scripts.counter import CounterEngine
from scripts.detector import Detector
from scripts.flip_counter import FlipCounterEngine
from scripts.tracker import Tracker
from scripts.wallet import TransactionType, WalletStore
from utils import FPSMeter, setup_logging
from voice.commands import CommandId
from voice.config import STTConfig, TTSConfig, VoiceConfig
from voice.listener import ListenerResult
from voice.voice_manager import VoiceManager


BOX_COLORS = {
    "5 Pounds": (0, 200, 0),
    "10 Pounds": (200, 100, 0),
    "20 Pounds": (0, 100, 255),
    "50 Pounds": (0, 200, 255),
    "100 Pounds": (255, 0, 150),
    "200 Pounds": (255, 0, 0),
}

MONEY_STOP_FILE_ENV = "EGY_MONEY_STOP_FILE"


class SessionMode(str, Enum):
    IDLE = "IDLE"
    SCAN = "SCAN"
    DEPOSIT = "DEPOSIT"
    PAYMENT = "PAYMENT"
    FLIP_SCAN = "FLIP_SCAN"
    FLIP_DEPOSIT = "FLIP_DEPOSIT"
    FLIP_PAYMENT = "FLIP_PAYMENT"


@dataclass
class RuntimeState:
    mode: SessionMode = SessionMode.IDLE
    shutdown_requested: bool = False
    waiting_for_balance_amount: bool = False

    @property
    def detection_enabled(self) -> bool:
        return self.mode in {
            SessionMode.SCAN,
            SessionMode.DEPOSIT,
            SessionMode.PAYMENT,
            SessionMode.FLIP_SCAN,
            SessionMode.FLIP_DEPOSIT,
            SessionMode.FLIP_PAYMENT,
        }


class Runtime:
    """Main integration orchestrator for detector, tracker, counter, voice, and wallet."""

    def __init__(self, config: RuntimeConfig = APP_CONFIG) -> None:
        self._config = config
        self._logger = setup_logging(config.verbose_logging)
        self._state = RuntimeState(mode=SessionMode(config.default_mode))
        self._stop_file_path = os.environ.get(MONEY_STOP_FILE_ENV, "").strip()
        self._frame_index = 0
        self._last_tracks = []
        self._last_detections = []
        self._last_count_flash: tuple[str, tuple[int, int, int], float] | None = None
        self._last_flip_debug = None
        self._last_runtime_diag = 0.0

        self._detector = Detector(config.detector)
        self._tracker = Tracker(config.tracker)
        self._counter = CounterEngine(config.counter)
        self._flip_counter = FlipCounterEngine(config.flip)
        self._last_flip_debug = self._flip_counter.get_debug_state()
        self._wallet = WalletStore(config.wallet.db_path)
        self._audio = AudioFeedback(config.audio)
        self._fps = FPSMeter(window=30)

        self._voice = None
        self._logger.info(
            "RUNTIME_INIT mode=%s camera=%sx%s headless=%s debug_window=%s inference_every_n_frames=%s voice_enabled=%s",
            self._state.mode.value,
            config.camera_width,
            config.camera_height,
            config.headless,
            config.debug_window,
            config.inference_every_n_frames,
            config.voice.enabled,
        )
        if config.voice.enabled:
            try:
                self._voice = VoiceManager(
                    VoiceConfig(
                        default_language=config.voice.default_language,
                        enable_wake_word=config.voice.enable_wake_word,
                        online_first=config.voice.online_first,
                        welcome_text_en=config.voice.welcome_text_en,
                        command_catalog=config.voice.command_catalog,
                        min_command_confidence=config.voice.min_command_confidence,
                        no_barge_in=config.voice.no_barge_in,
                        stt=STTConfig(
                            primary=config.voice.stt.primary,
                            fallback=config.voice.stt.fallback,
                            optional_openai_model=config.voice.stt.optional_openai_model,
                        ),
                        tts=TTSConfig(
                            primary=config.voice.tts.primary,
                            fallback=config.voice.tts.fallback,
                            optional_openai_model=config.voice.tts.optional_openai_model,
                            cache_dir=config.voice.tts.cache_dir,
                        ),
                    )
                )
                self._wire_voice_commands()
                self._logger.info("VOICE_INIT_OK")
            except Exception as exc:
                self._logger.warning("Voice disabled due to initialization failure: %s", exc)

    def run(self) -> None:
        camera_source = resolve_camera_source(self._config.camera_source)
        self._logger.info("CAMERA_SOURCE_RESOLVED raw=%r resolved=%r", self._config.camera_source, camera_source)

        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError("Camera unavailable")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)

        self._logger.info("Runtime started")
        self._logger.info(
            "CAMERA_OPENED source=%s requested_width=%s requested_height=%s actual_width=%s actual_height=%s",
            camera_source,
            self._config.camera_width,
            self._config.camera_height,
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        if self._voice:
            self._voice.start()

        try:
            while not self._state.shutdown_requested:
                if self._external_stop_requested():
                    self._logger.info("RUNTIME_EXTERNAL_STOP_REQUESTED")
                    self._state.shutdown_requested = True
                    break
                ok, frame = cap.read()
                if not ok:
                    self._logger.error("Camera frame read failed")
                    break

                self._frame_index += 1
                if self._state.detection_enabled:
                    self._process_detection_frame(frame)
                else:
                    self._tracker.reset()

                fps = self._fps.tick()
                self._log_runtime_heartbeat(fps)
                if self._should_show_window():
                    self._draw_overlay(frame, fps)
                    cv2.imshow("Egyptian Currency Runtime", frame)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)

            self._logger.info("Runtime stopping")

        finally:
            cap.release()
            if self._should_show_window():
                cv2.destroyAllWindows()

            if self._voice:
                self._voice.stop()

    def _external_stop_requested(self) -> bool:
        return bool(self._stop_file_path) and os.path.exists(self._stop_file_path)

    def _process_detection_frame(self, frame) -> None:
        should_infer = (
            self._frame_index % max(1, self._config.inference_every_n_frames) == 0
        )
        if should_infer:
            detections = self._detector.infer(frame)
            self._last_detections = detections
            self._logger.info(
                "DETECTOR_INFER frame=%s detections=%s mode=%s",
                self._frame_index,
                len(detections),
                self._state.mode.value,
            )
            if self._is_flip_mode():
                self._process_flip_detections(frame, detections)
                if self._should_show_window():
                    self._draw_detections(frame, detections)
                    self._draw_flip_gate(frame)
                return

            tracker_result = self._tracker.update(detections)
            self._last_tracks = tracker_result.active_tracks
        else:
            if self._is_flip_mode():
                if self._should_show_window():
                    self._draw_detections(frame, self._last_detections)
                    self._draw_flip_gate(frame)
                return
            tracker_result = self._tracker.update([])
            self._last_tracks = tracker_result.active_tracks

        if should_infer or tracker_result.removed_track_ids:
            self._logger.info(
                "TRACKER_UPDATE frame=%s active_tracks=%s removed=%s",
                self._frame_index,
                len(tracker_result.active_tracks),
                tracker_result.removed_track_ids,
            )

        count_events = self._counter.update(tracker_result.active_tracks)

        for event in count_events:
            self._audio.count_beep()
            self._tracker.mark_counted(event.track_id)
            if self._state.mode in {SessionMode.DEPOSIT, SessionMode.PAYMENT}:
                self._wallet.add_scanned_note(event.value)
                self._logger.info(
                    "WALLET_PENDING_ADD mode=%s value=%s pending_total=%s",
                    self._state.mode.value,
                    event.value,
                    self._wallet.preview_transaction_total(),
                )
            self._logger.info(
                "Counted %s (%s EGP) track_id=%s mode=%s",
                event.class_name,
                event.value,
                event.track_id,
                self._state.mode.value,
            )
            self._set_count_flash(event.class_name, event.value)

        if self._should_show_window():
            if should_infer:
                self._draw_detections(frame, self._last_detections)
            self._draw_tracks(frame, tracker_result.active_tracks)

    def _process_flip_detections(self, frame, detections) -> None:
        result = self._flip_counter.update(
            detections=detections,
            frame=frame,
            frame_index=self._frame_index,
            mode=self._state.mode.value,
        )
        self._last_flip_debug = result.debug_state

        for reject in result.reject_events:
            self._logger.info(
                "FLIP_REJECT frame=%s state=%s reason=%s detail=%s class=%s confidence=%.2f vote_ratio=%.2f bbox=%s mode=%s",
                reject.frame_index,
                reject.state,
                reject.reason,
                reject.reason_detail,
                reject.best_class,
                reject.confidence,
                reject.vote_ratio,
                reject.bbox,
                self._state.mode.value,
            )

        for event in result.count_events:
            self._audio.count_beep()
            self._counter.add_manual_count(event.class_name, event.value, track_id=-1)
            if self._state.mode in {SessionMode.FLIP_DEPOSIT, SessionMode.FLIP_PAYMENT}:
                self._wallet.add_scanned_note(event.value)
                self._logger.info(
                    "WALLET_PENDING_ADD mode=%s value=%s pending_total=%s",
                    self._state.mode.value,
                    event.value,
                    self._wallet.preview_transaction_total(),
                )
            self._logger.info(
                "FLIP_COUNTED frame=%s state=%s class=%s value=%s confidence=%.2f vote_ratio=%.2f direction=%s bbox=%s metadata=%s mode=%s",
                event.frame_index,
                event.state,
                event.class_name,
                event.value,
                event.confidence,
                event.vote_ratio,
                event.direction,
                event.bbox,
                event.metadata_path,
                self._state.mode.value,
            )
            self._set_count_flash(event.class_name, event.value)

    def _draw_detections(self, frame, detections) -> None:
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = BOX_COLORS.get(detection.class_name, (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.class_name} {detection.confidence:.2f}"
            self._draw_label(frame, label, x1, y1, color)

    def _wire_voice_commands(self) -> None:
        assert self._voice is not None

        self._voice.register_handler(CommandId.START_SCAN, self._cmd_start_scan)
        self._voice.register_handler(CommandId.STOP_SCAN, self._cmd_stop_scan)
        self._voice.register_handler(CommandId.COUNT_TOTAL, self._cmd_count_total)
        self._voice.register_handler(CommandId.RESET_SESSION, self._cmd_reset_session)
        self._voice.register_handler(CommandId.STATUS_CHECK, self._cmd_status)
        self._voice.register_handler(CommandId.LAST_DETECTION, self._cmd_last_detection)
        self._voice.register_handler(CommandId.WALLET_BALANCE, self._cmd_wallet_balance)
        self._voice.register_handler(CommandId.SET_BALANCE, self._cmd_set_balance)
        self._voice.register_handler(CommandId.START_DEPOSIT, self._cmd_start_deposit)
        self._voice.register_handler(CommandId.FINISH_DEPOSIT, self._cmd_finish_deposit)
        self._voice.register_handler(CommandId.START_PAYMENT, self._cmd_start_payment)
        self._voice.register_handler(CommandId.FINISH_PAYMENT, self._cmd_finish_payment)
        self._voice.register_handler(CommandId.START_FLIP_SCAN, self._cmd_start_flip_scan)
        self._voice.register_handler(CommandId.START_FLIP_DEPOSIT, self._cmd_start_flip_deposit)
        self._voice.register_handler(CommandId.FINISH_FLIP_DEPOSIT, self._cmd_finish_flip_deposit)
        self._voice.register_handler(CommandId.START_FLIP_PAYMENT, self._cmd_start_flip_payment)
        self._voice.register_handler(CommandId.FINISH_FLIP_PAYMENT, self._cmd_finish_flip_payment)
        self._voice.register_handler(CommandId.CONFIRM, self._cmd_confirm)
        self._voice.register_handler(CommandId.CANCEL, self._cmd_cancel)
        self._voice.register_handler(CommandId.EXIT_APP, self._cmd_exit)

    def _cmd_start_scan(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_START_SCAN")
        self._start_mode(SessionMode.SCAN)
        return self._r("Scan started.", "تم بدء المسح.")

    def _cmd_stop_scan(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_STOP_SCAN mode=%s", self._state.mode.value)
        if self._state.mode in {
            SessionMode.DEPOSIT,
            SessionMode.PAYMENT,
            SessionMode.FLIP_DEPOSIT,
            SessionMode.FLIP_PAYMENT,
        }:
            self._wallet.cancel_transaction()
            self._logger.info("WALLET_PENDING_CANCEL reason=stop_scan")
        self._stop_detection(reset_session=False)
        return self._r("Scan stopped.", "تم ايقاف المسح.")

    def _cmd_count_total(self, _result: ListenerResult) -> str:
        total = self._counter.get_total()
        self._logger.info("CMD_COUNT_TOTAL total=%s", total)
        return self._r(f"Session total is {total} pounds.", f"اجمالي الجلسة {total} جنيه.")

    def _cmd_reset_session(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_RESET_SESSION")
        if self._wallet.has_pending_transaction():
            self._wallet.cancel_transaction()
            self._logger.info("WALLET_PENDING_CANCEL reason=reset_session")
        self._counter.reset()
        self._tracker.reset()
        self._flip_counter.reset()
        return self._r("Session reset.", "تم تصفير الجلسة.")

    def _cmd_status(self, _result: ListenerResult) -> str:
        balance = self._wallet.get_balance()
        stats = self._counter.get_statistics()
        self._logger.info(
            "CMD_STATUS mode=%s balance=%s total=%s counted_notes=%s",
            self._state.mode.value,
            balance,
            stats["total_amount"],
            stats["counted_notes"],
        )
        return self._r(
            f"Mode {self._state.mode.value.lower()}. Wallet balance is {balance} pounds.",
            f"الوضع {self._state.mode.value}. رصيد المحفظة {balance} جنيه.",
        )

    def _cmd_last_detection(self, _result: ListenerResult) -> str:
        event = self._counter.get_last_count_event()
        self._logger.info("CMD_LAST_DETECTION event=%s", event)
        if not event:
            return self._r("No note detected yet.", "لا توجد ورقة مكتشفة بعد.")
        return self._r(
            f"Last note was {event.value} pounds.",
            f"اخر ورقة كانت {event.value} جنيه.",
        )

    def _cmd_wallet_balance(self, _result: ListenerResult) -> str:
        balance = self._wallet.get_balance()
        self._logger.info("CMD_WALLET_BALANCE balance=%s", balance)
        return self._r(f"Wallet balance is {balance} pounds.", f"رصيد المحفظة {balance} جنيه.")

    def _cmd_set_balance(self, result: ListenerResult) -> str:
        amount = result.amount
        self._logger.info(
            "CMD_SET_BALANCE amount=%s waiting_for_amount=%s raw=%r",
            amount,
            self._state.waiting_for_balance_amount,
            result.raw_text,
        )
        if result.command_id is None and not self._state.waiting_for_balance_amount:
            return self._r("I did not understand.", "لم افهم الامر.")

        if amount is None:
            self._state.waiting_for_balance_amount = True
            return self._r(
                "Say set balance followed by the amount.",
                "قل اضبط الرصيد ثم المبلغ.",
            )

        self._wallet.set_balance(amount)
        self._logger.info("WALLET_PENDING_SET_BALANCE amount=%s", amount)
        self._state.waiting_for_balance_amount = False
        return self._r(
            f"Set balance to {amount} pounds? Say confirm or cancel.",
            f"تعيين الرصيد إلى {amount} جنيه؟ قل تأكيد أو الغاء.",
        )

    def _cmd_start_deposit(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_START_DEPOSIT")
        self._wallet.begin_transaction(TransactionType.DEPOSIT)
        self._start_mode(SessionMode.DEPOSIT)
        return self._r("Deposit started.", "تم بدء الايداع.")

    def _cmd_finish_deposit(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_FINISH_DEPOSIT mode=%s", self._state.mode.value)
        if self._state.mode != SessionMode.DEPOSIT:
            return self._r("No deposit is active.", "لا توجد عملية ايداع نشطة.")
        total = self._wallet.preview_transaction_total()
        self._logger.info("WALLET_DEPOSIT_PREVIEW total=%s", total)
        self._stop_detection(reset_session=False)
        return self._r(
            f"Deposit total is {total} pounds. Say confirm or cancel.",
            f"اجمالي الايداع {total} جنيه. قل تأكيد أو الغاء.",
        )

    def _cmd_start_payment(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_START_PAYMENT")
        self._wallet.begin_transaction(TransactionType.PAYMENT)
        self._start_mode(SessionMode.PAYMENT)
        return self._r("Payment started.", "تم بدء الدفع.")

    def _cmd_finish_payment(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_FINISH_PAYMENT mode=%s", self._state.mode.value)
        if self._state.mode != SessionMode.PAYMENT:
            return self._r("No payment is active.", "لا توجد عملية دفع نشطة.")
        total = self._wallet.preview_transaction_total()
        balance_after = self._wallet.get_balance() - total
        self._logger.info("WALLET_PAYMENT_PREVIEW total=%s balance_after=%s", total, balance_after)
        self._stop_detection(reset_session=False)
        if balance_after < 0:
            self._wallet.cancel_transaction()
            self._logger.info("WALLET_PENDING_CANCEL reason=payment_gt_balance")
            return self._r(
                "Payment is greater than wallet balance. Transaction cancelled.",
                "المبلغ اكبر من الرصيد. تم الغاء العملية.",
            )
        return self._r(
            f"Payment total is {total} pounds. Balance after payment will be {balance_after}. Say confirm or cancel.",
            f"اجمالي الدفع {total} جنيه. الرصيد بعد الدفع {balance_after}. قل تأكيد أو الغاء.",
        )

    def _cmd_start_flip_scan(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_START_FLIP_SCAN")
        if not self._config.flip.enabled:
            return self._r("Flip count is disabled.", "عد الرزمة غير مفعل.")
        self._start_mode(SessionMode.FLIP_SCAN)
        return self._r("Flip count started.", "تم بدء عد الرزمة.")

    def _cmd_start_flip_deposit(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_START_FLIP_DEPOSIT")
        if not self._config.flip.enabled:
            return self._r("Flip count is disabled.", "عد الرزمة غير مفعل.")
        self._wallet.begin_transaction(TransactionType.DEPOSIT)
        self._start_mode(SessionMode.FLIP_DEPOSIT)
        return self._r("Flip deposit started.", "تم بدء ايداع الرزمة.")

    def _cmd_finish_flip_deposit(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_FINISH_FLIP_DEPOSIT mode=%s", self._state.mode.value)
        if self._state.mode != SessionMode.FLIP_DEPOSIT:
            return self._r("No flip deposit is active.", "لا توجد عملية ايداع رزمة نشطة.")
        total = self._wallet.preview_transaction_total()
        self._logger.info("WALLET_FLIP_DEPOSIT_PREVIEW total=%s", total)
        self._stop_detection(reset_session=False)
        return self._r(
            f"Flip deposit total is {total} pounds. Say confirm or cancel.",
            f"اجمالي ايداع الرزمة {total} جنيه. قل تأكيد أو الغاء.",
        )

    def _cmd_start_flip_payment(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_START_FLIP_PAYMENT")
        if not self._config.flip.enabled:
            return self._r("Flip count is disabled.", "عد الرزمة غير مفعل.")
        self._wallet.begin_transaction(TransactionType.PAYMENT)
        self._start_mode(SessionMode.FLIP_PAYMENT)
        return self._r("Flip payment started.", "تم بدء دفع الرزمة.")

    def _cmd_finish_flip_payment(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_FINISH_FLIP_PAYMENT mode=%s", self._state.mode.value)
        if self._state.mode != SessionMode.FLIP_PAYMENT:
            return self._r("No flip payment is active.", "لا توجد عملية دفع رزمة نشطة.")
        total = self._wallet.preview_transaction_total()
        balance_after = self._wallet.get_balance() - total
        self._logger.info("WALLET_FLIP_PAYMENT_PREVIEW total=%s balance_after=%s", total, balance_after)
        self._stop_detection(reset_session=False)
        if balance_after < 0:
            self._wallet.cancel_transaction()
            self._logger.info("WALLET_PENDING_CANCEL reason=flip_payment_gt_balance")
            return self._r(
                "Payment is greater than wallet balance. Transaction cancelled.",
                "المبلغ اكبر من الرصيد. تم الغاء العملية.",
            )
        return self._r(
            f"Flip payment total is {total} pounds. Balance after payment will be {balance_after}. Say confirm or cancel.",
            f"اجمالي دفع الرزمة {total} جنيه. الرصيد بعد الدفع {balance_after}. قل تأكيد أو الغاء.",
        )

    def _cmd_confirm(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_CONFIRM has_pending=%s", self._wallet.has_pending_transaction())
        if not self._wallet.has_pending_transaction():
            return self._r("Nothing to confirm.", "لا يوجد شيء للتأكيد.")
        snapshot = self._wallet.commit_transaction()
        self._logger.info("WALLET_COMMIT balance=%s", snapshot.balance)
        self._counter.reset()
        self._tracker.reset()
        self._flip_counter.reset()
        return self._r(
            f"Confirmed. Wallet balance is {snapshot.balance} pounds.",
            f"تم التأكيد. رصيد المحفظة {snapshot.balance} جنيه.",
        )

    def _cmd_cancel(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_CANCEL has_pending=%s", self._wallet.has_pending_transaction())
        if self._wallet.has_pending_transaction():
            self._wallet.cancel_transaction()
            self._logger.info("WALLET_PENDING_CANCEL reason=voice_cancel")
        self._state.waiting_for_balance_amount = False
        self._stop_detection(reset_session=False)
        return self._r("Cancelled.", "تم الالغاء.")

    def _cmd_exit(self, _result: ListenerResult) -> str:
        self._logger.info("CMD_EXIT")
        self._state.shutdown_requested = True
        return self._r("Closing.", "جاري الاغلاق.")

    def _start_mode(self, mode: SessionMode) -> None:
        previous = self._state.mode
        self._counter.reset()
        self._tracker.reset()
        self._flip_counter.reset()
        self._state.mode = mode
        self._logger.info(
            "MODE_CHANGE %s -> %s reset_counter=True reset_tracker=True reset_flip=True",
            previous.value,
            mode.value,
        )

    def _stop_detection(self, reset_session: bool) -> None:
        previous = self._state.mode
        self._state.mode = SessionMode.IDLE
        self._tracker.reset()
        self._flip_counter.reset()
        self._counter.reset_tracking_state()
        if reset_session:
            self._counter.reset()
        self._logger.info(
            "MODE_CHANGE %s -> %s reset_session=%s reset_tracker=True reset_flip=True",
            previous.value,
            self._state.mode.value,
            reset_session,
        )

    def _draw_tracks(self, frame, tracks) -> None:
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = BOX_COLORS.get(track.class_name, (255, 255, 255))
            thickness = 3 if track.is_stable else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            text = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
            if track.is_stable:
                text += " stable"
            self._draw_label(frame, text, x1, y1, color)

    def _draw_flip_gate(self, frame) -> None:
        zones = self._flip_counter.zone_bounds(frame.shape[:2])
        orientation, gate_start, gate_end = zones.gate
        height, width = frame.shape[:2]
        gate_color = (0, 255, 255)
        enter_color = (80, 180, 255)
        exit_color = (120, 255, 120)
        overlay = frame.copy()
        if orientation == "horizontal":
            _, enter_start, enter_end = zones.enter
            _, exit_start, exit_end = zones.exit
            cv2.rectangle(overlay, (0, enter_start), (width, enter_end), enter_color, -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (0, enter_start), (width, enter_end), enter_color, 1)
            cv2.rectangle(frame, (0, exit_start), (width, exit_end), exit_color, 1)
            cv2.rectangle(frame, (0, gate_start), (width, gate_end), gate_color, 2)
            cv2.line(frame, (0, zones.count_line), (width, zones.count_line), gate_color, 2)
            self._draw_label(frame, "enter / count / exit", 12, gate_start + 24, gate_color)
            return
        _, enter_start, enter_end = zones.enter
        _, exit_start, exit_end = zones.exit
        cv2.rectangle(overlay, (enter_start, 0), (enter_end, height), enter_color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.rectangle(frame, (enter_start, 0), (enter_end, height), enter_color, 1)
        cv2.rectangle(frame, (exit_start, 0), (exit_end, height), exit_color, 1)
        cv2.rectangle(frame, (gate_start, 0), (gate_end, height), gate_color, 2)
        cv2.line(frame, (zones.count_line, 0), (zones.count_line, height), gate_color, 2)
        self._draw_label(frame, "enter / count / exit", gate_start + 6, 30, gate_color)

    def _draw_overlay(self, frame, fps: float) -> None:
        stats = self._counter.get_statistics()
        voice_state = self._voice.state.value if self._voice else "DISABLED"
        wallet = self._wallet.snapshot()
        height, width = frame.shape[:2]
        pending_label = wallet.pending_type.value if wallet.pending_type else "NONE"
        mode_color = self._mode_color()

        panel_w = min(310, max(260, width // 2 - 20))
        self._draw_panel(frame, 12, 12, panel_w, 222, border=mode_color)
        self._draw_text(frame, "Money Detection", 26, 40, 0.72, (245, 245, 245), 2)
        self._draw_badge(frame, self._state.mode.value, 26, 56, mode_color)
        self._draw_text(frame, "SESSION TOTAL", 26, 104, 0.45, (165, 175, 182), 1)
        self._draw_text(frame, f"{stats['total_amount']} EGP", 26, 142, 1.05, (255, 255, 255), 2)
        self._draw_metric(frame, "Notes", str(stats["counted_notes"]), 26, 174)
        self._draw_metric(frame, "Wallet", f"{wallet.balance} EGP", 26, 200)
        self._draw_metric(frame, "Pending", f"{pending_label} {wallet.pending_total} EGP", 26, 226)

        right_w = min(260, max(220, width // 3))
        right_x = max(12, width - right_w - 12)
        self._draw_panel(frame, right_x, 12, right_w, 204)
        self._draw_text(frame, "Live Detections", right_x + 14, 38, 0.58, (245, 245, 245), 2)
        if self._last_detections:
            for idx, detection in enumerate(self._last_detections[:5]):
                y = 72 + idx * 26
                color = BOX_COLORS.get(detection.class_name, (0, 255, 255))
                cv2.circle(frame, (right_x + 20, y - 5), 5, color, -1)
                text = f"{detection.class_name}  {detection.confidence:.2f}"
                self._draw_text(frame, text, right_x + 34, y, 0.48, (230, 235, 235), 1)
                self._draw_bar(
                    frame,
                    right_x + 34,
                    y + 7,
                    right_w - 54,
                    6,
                    min(1.0, max(0.0, detection.confidence)),
                    color,
                )
        else:
            self._draw_text(frame, "No notes detected", right_x + 14, 78, 0.5, (170, 180, 185), 1)

        if self._is_flip_mode() and self._last_flip_debug:
            debug_y = 230
            self._draw_panel(frame, right_x, debug_y, right_w, 184, border=mode_color)
            debug = self._last_flip_debug
            self._draw_text(frame, "Flip Engine", right_x + 14, debug_y + 26, 0.58, (245, 245, 245), 2)
            self._draw_metric(frame, "State", debug.state, right_x + 14, debug_y + 56)
            self._draw_metric(frame, "Reason", debug.reason, right_x + 14, debug_y + 82)
            self._draw_metric(frame, "Class", debug.best_class or "NONE", right_x + 14, debug_y + 108)
            self._draw_metric(frame, "Vote", f"{debug.vote_ratio:.2f}", right_x + 14, debug_y + 134)
            self._draw_bar(frame, right_x + 102, debug_y + 140, right_w - 122, 7, debug.vote_ratio, mode_color)
            direction = debug.direction or "-"
            self._draw_metric(frame, "Motion", f"{direction} {debug.motion_pixels:.0f}px", right_x + 14, debug_y + 164)

        bottom_h = 42
        self._draw_panel(frame, 12, height - bottom_h - 12, width - 24, bottom_h, alpha=0.62)
        self._draw_text(
            frame,
            f"Money detection running   FPS {fps:.1f}   Voice {voice_state}   Q exit   R reset",
            26,
            height - 28,
            0.52,
            (235, 240, 240),
            1,
        )

        self._draw_count_flash(frame)

    def _handle_key(self, key: int) -> None:
        if key in (27, ord("q")):
            self._state.shutdown_requested = True
        elif key == ord("r"):
            self._counter.reset()
            self._tracker.reset()
            self._flip_counter.reset()

    def _should_show_window(self) -> bool:
        return (not self._config.headless) or self._config.debug_window

    def _is_flip_mode(self) -> bool:
        return self._state.mode in {
            SessionMode.FLIP_SCAN,
            SessionMode.FLIP_DEPOSIT,
            SessionMode.FLIP_PAYMENT,
        }

    def _set_count_flash(self, class_name: str, value: int) -> None:
        color = BOX_COLORS.get(class_name, (0, 255, 255))
        self._last_count_flash = (f"Counted {value} EGP", color, time.monotonic())

    def _draw_count_flash(self, frame) -> None:
        if not self._last_count_flash:
            return
        text, color, created_at = self._last_count_flash
        age = time.monotonic() - created_at
        if age > 1.4:
            self._last_count_flash = None
            return
        height, width = frame.shape[:2]
        box_w, box_h = 250, 58
        x = max(12, (width - box_w) // 2)
        y = 76
        alpha = max(0.35, 0.85 - age * 0.25)
        self._draw_panel(frame, x, y, box_w, box_h, color=(20, 30, 32), border=color, alpha=alpha)
        self._draw_text(frame, text, x + 28, y + 38, 0.82, (255, 255, 255), 2)

    def _mode_color(self) -> tuple[int, int, int]:
        if self._state.mode in {SessionMode.DEPOSIT, SessionMode.FLIP_DEPOSIT}:
            return (60, 220, 120)
        if self._state.mode in {SessionMode.PAYMENT, SessionMode.FLIP_PAYMENT}:
            return (80, 150, 255)
        if self._state.mode in {SessionMode.SCAN, SessionMode.FLIP_SCAN}:
            return (0, 220, 255)
        return (160, 170, 175)

    def _draw_panel(
        self,
        frame,
        x: int,
        y: int,
        w: int,
        h: int,
        color: tuple[int, int, int] = (22, 25, 28),
        border: tuple[int, int, int] = (80, 90, 96),
        alpha: float = 0.74,
    ) -> None:
        height, width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)

    def _draw_badge(self, frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        w, h = size
        cv2.rectangle(frame, (x, y), (x + w + 18, y + h + 14), color, -1)
        cv2.rectangle(frame, (x, y), (x + w + 18, y + h + 14), (255, 255, 255), 1)
        self._draw_text(frame, text, x + 9, y + h + 8, 0.52, (10, 16, 18), 2)

    def _draw_metric(self, frame, label: str, value: str, x: int, y: int) -> None:
        value = self._shorten(value, 18)
        self._draw_text(frame, label, x, y, 0.46, (165, 175, 182), 1)
        self._draw_text(frame, value, x + 88, y, 0.54, (250, 250, 250), 2)

    def _draw_bar(
        self,
        frame,
        x: int,
        y: int,
        w: int,
        h: int,
        ratio: float,
        color: tuple[int, int, int],
    ) -> None:
        ratio = min(1.0, max(0.0, ratio))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 62, 68), -1)
        fill_w = int(w * ratio)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 130, 136), 1)

    def _draw_label(self, frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        text = self._shorten(text, 28)
        y = max(22, y)
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        w, h = size
        x2 = min(frame.shape[1] - 1, x + w + 10)
        y1 = max(0, y - h - 12)
        cv2.rectangle(frame, (x, y1), (x2, y + 4), (15, 18, 20), -1)
        cv2.rectangle(frame, (x, y1), (x2, y + 4), color, 1)
        self._draw_text(frame, text, x + 5, y - 5, 0.5, (245, 245, 245), 1)

    def _draw_text(
        self,
        frame,
        text: str,
        x: int,
        y: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    @staticmethod
    def _shorten(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)] + "..."

    def _r(self, english: str, arabic: str) -> str:
        if self._voice and self._voice.language == "ar":
            return arabic
        return english

    def _log_runtime_heartbeat(self, fps: float) -> None:
        now = time.monotonic()
        if now - self._last_runtime_diag < 5.0:
            return
        self._last_runtime_diag = now
        stats = self._counter.get_statistics()
        wallet = self._wallet.snapshot()
        voice_state = self._voice.state.value if self._voice else "DISABLED"
        self._logger.info(
            "RUNTIME_HEARTBEAT frame=%s fps=%.1f mode=%s voice=%s total=%s notes=%s wallet=%s pending=%s pending_total=%s",
            self._frame_index,
            fps,
            self._state.mode.value,
            voice_state,
            stats["total_amount"],
            stats["counted_notes"],
            wallet.balance,
            wallet.pending_type.value if wallet.pending_type else None,
            wallet.pending_total,
        )
