"""Dictation daemon: warm Whisper model behind a unix socket.

Logic
    startup  -> load the Whisper model once and keep it resident
    "start"  -> open the mic stream, buffer frames
    "stop"   -> close the stream, transcribe the buffer, copy to clipboard, notify """

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import config as config_mod

log = logging.getLogger("dictation")

IDLE = "idle"
RECORDING = "recording"
MAX_TRANSCRIPTIONS = 2
MAX_CLIENTS = 8
CLIENT_REQUEST_TIMEOUT_S = 5.0


class DictationDaemon:
    def __init__(self, cfg: config_mod.Config):
        self.cfg = cfg
        self.state = IDLE

        # The audio callback appends frames until the input stream is stopped
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._recording_started_at: float | None = None
        # Capture state changes share a lock; inference never holds it
        self._capture_lock = threading.RLock()
        self._recording_token = None
        self._recording_timer = None
        self._pending_transcriptions = 0
        self._closing = False

        self.model = self._load_model()
        log.info("model ready on %s", self.active_device)
        # One inference worker keeps transcripts and clipboard updates in order
        self._transcriptions = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcription")
        self._notifications = ThreadPoolExecutor(max_workers=1, thread_name_prefix="notification")
        self._notification_slots = threading.BoundedSemaphore(16)

    ##### MODEL #####

    def _load_model(self):
        # Load faster-whisper once, falling back to CPU if CUDA is unavailable
        from faster_whisper import WhisperModel

        name = self.cfg.get("model.name")
        device = self.cfg.get("model.device")
        compute_type = self.cfg.get("model.compute_type")

        attempts = [(device, compute_type)]
        if device != "cpu":
            attempts.append(("cpu", "int8"))
        for device, compute_type in attempts:
            log.info("loading %s on %s (%s)...", name, device, compute_type)
            started = time.monotonic()
            try:
                self.model = WhisperModel(name, device=device, compute_type=compute_type)
                log.info(
                    "model loaded on %s (%s) in %.1fs",
                    device, compute_type, time.monotonic() - started,
                )
                self._warm_up_model()
            except Exception:
                self.model = None
                if device == "cpu":
                    raise
                log.exception("GPU initialization failed; falling back to CPU")
            else:
                self.active_device = device
                return self.model

    def _warm_up_model(self) -> None:
        log.info("warming up transcription...")
        started = time.monotonic()
        audio = np.zeros(config_mod.WHISPER_SAMPLE_RATE, dtype=np.float32)

        # Silence must bypass VAD or it will skip the encoder and decoder
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.get("transcribe.language"),
            beam_size=self.cfg.get("transcribe.beam_size"),
            vad_filter=False,
            condition_on_previous_text=False,
            max_new_tokens=1,
            temperature=0.0,
        )
        # Inference runs lazily when the segments are consumed
        for _ in segments:
            pass

        if self.cfg.get("transcribe.vad_filter"):
            self._transcribe(audio)

        log.info("transcription warmed up in %.2fs", time.monotonic() - started)

    ##### CAPTURE #####

    def _start_capture(self) -> None:
        # Open the input stream and begin buffering frames
        self._frames = []
        captured_samples = 0
        max_samples = int(self.cfg.get("audio.max_duration_s") * config_mod.WHISPER_SAMPLE_RATE)
        deadline = self._recording_started_at + self.cfg.get("audio.max_duration_s")

        def _callback(indata, frames, time_info, status):
            nonlocal captured_samples
            # Enforce the limit even if the timer thread is delayed
            if time.monotonic() >= deadline:
                raise sd.CallbackStop
            remaining = max_samples - captured_samples
            chunk = indata[:remaining].copy()
            self._frames.append(chunk)
            captured_samples += len(chunk)
            if status:
                log.warning("audio status: %s", status)
            if captured_samples >= max_samples:
                raise sd.CallbackStop

        self._stream = sd.InputStream(
            samplerate=self.cfg.get("audio.sample_rate"),
            channels=self.cfg.get("audio.channels"),
            blocksize=self.cfg.get("audio.blocksize"),
            device=self.cfg.audio_device,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def _stop_capture(self) -> np.ndarray | None:
        # Close the stream, then return audio unless too short
        if self._stream is None:
            return None

        self._stream.stop()
        self._stream.close()
        self._stream = None

        # The callback has stopped -> drain buffer
        frames, self._frames = self._frames, []

        if not frames:
            return None

        audio = np.concatenate(frames)
        audio = audio.reshape(-1)

        # Ignore accidental key presses
        duration = len(audio) / config_mod.WHISPER_SAMPLE_RATE
        if duration < self.cfg.get("audio.min_duration_s"):
            return None

        log.info("captured %d frames (%.2fs)", len(frames), duration)
        return audio

    def _transcribe(self, audio: np.ndarray) -> str | None:
        # Transcribe an array
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg.get("transcribe.language"),
            beam_size=self.cfg.get("transcribe.beam_size"),
            vad_filter=self.cfg.get("transcribe.vad_filter"),
            condition_on_previous_text=self.cfg.get("transcribe.condition_on_previous_text"),
            initial_prompt=self.cfg.get("transcribe.initial_prompt") or None,
        )

        # Iterating segments completes inference. Segment text includes spacing
        text = "".join([segment.text for segment in segments]).strip()

        if text == "":
            return None

        # Finish the whole recording, not individual model segments
        if self.cfg.get("transcribe.ensure_final_period"):
            content = text.rstrip("\"'”’")
            if content and content[-1].isalnum():
                closing_quotes = text[len(content):]
                text = content + "." + closing_quotes

        return text

    ##### OUTPUT #####

    def _copy_to_clipboard(self, text: str) -> bool:
        if not self.cfg.get("output.clipboard"):
            return False
        """ wl-copy forks a clipboard owner. Pipes inherited by that child can
        keep communicate() waiting after the original process has exited """
        with tempfile.TemporaryFile(mode="w+t") as stderr:
            try:
                subprocess.run(
                    ["wl-copy", "--"], input=text, text=True, check=True,
                    stdout=subprocess.DEVNULL, stderr=stderr, timeout=5,
                )
            except (subprocess.SubprocessError, FileNotFoundError) as exc:
                stderr.seek(0)
                log.error("wl-copy failed: %s; %s", exc, stderr.read().strip() or "no stderr")
                return False
        return True

    def _notify(self, summary: str, body: str = "", *, transient: bool = False) -> None:
        if not self.cfg.get("output.notifications") or self._closing:
            return
        if not self._notification_slots.acquire(blocking=False):
            log.warning("notification queue is full")
            return
        payload = json.dumps({"summary": summary, "body": body, "transient": transient})
        try:
            future = self._notifications.submit(self._send_notification, payload)
        except RuntimeError:
            self._notification_slots.release()
            if not self._closing:
                raise
        else:
            future.add_done_callback(self._notification_finished)

    def _notification_finished(self, future: Future) -> None:
        self._notification_slots.release()

    @staticmethod
    def _send_notification(payload: str) -> None:
        helper = Path(__file__).with_name("notify.py")
        try:
            result = subprocess.run(
                ["/usr/bin/python3", str(helper)], input=payload, text=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5,
            )
            if result.returncode:
                log.warning("notification helper failed with exit code %d", result.returncode)
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning("notification helper failed: %s", type(exc).__name__)

    def _preview(self, text: str) -> str:
        limit = int(self.cfg.get("output.preview_chars"))
        if limit <= 0:
            return ""
        collapsed = " ".join(text.split())
        return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"

    ##### HANDLERS #####

    def handle(self, verb: str) -> str | Future:
        handlers = {
            "start": self.do_start,
            "stop": self.do_stop,
            "toggle": self.do_toggle,
            "status": self.do_status,
            "ping": lambda: f"pong {self.active_device}",
        }
        handler = handlers.get(verb)
        if handler is None:
            raise ValueError("unknown command")
        with self._capture_lock:
            if self._closing:
                raise RuntimeError("daemon is shutting down")
            try:
                return handler()
            except Exception:
                if verb in ("start", "stop", "toggle"):
                    self._reset_after_failure()
                raise

    def do_status(self) -> str:
        if self.state == RECORDING:
            return RECORDING
        return "transcribing" if self._pending_transcriptions else IDLE

    def do_start(self) -> str:
        if self.state == RECORDING:
            return "already recording"
        if self._pending_transcriptions >= MAX_TRANSCRIPTIONS:
            self._notify("Dictation busy", "Wait for a transcript before recording again", transient=True)
            raise RuntimeError("transcription queue is full; wait before recording again")
        self._recording_started_at = time.monotonic()
        self._start_capture()
        self.state = RECORDING
        token = object()
        self._recording_token = token
        remaining = self.cfg.get("audio.max_duration_s") - (time.monotonic() - self._recording_started_at)
        self._recording_timer = threading.Timer(max(0, remaining), self._expire_recording, args=(token,))
        self._recording_timer.daemon = True
        self._recording_timer.start()
        log.info("recording")
        self._notify("Recording…", transient=True)
        return "recording"

    def do_stop(self) -> str | Future:
        if self.state == IDLE:
            return "not recording"

        self.state = IDLE
        self._cancel_recording_timer()
        held = time.monotonic() - (self._recording_started_at or time.monotonic())
        audio = self._stop_capture()

        if audio is None or len(audio) == 0:
            log.info("no audio captured (held %.2fs)", held)
            self._notify("Nothing captured", transient=True)
            return "empty"

        self._notify("Transcribing…", transient=True)
        self._pending_transcriptions += 1
        try:
            return self._transcriptions.submit(self._transcribe_and_publish, audio)
        except Exception:
            self._pending_transcriptions -= 1
            raise

    def _expire_recording(self, token: object) -> None:
        with self._capture_lock:
            if self._closing or self.state != RECORDING or self._recording_token is not token:
                return
            self._notify("Recording limit reached", "Microphone stopped; processing the captured audio", transient=True)
            try:
                self.do_stop()
            except Exception:
                log.exception("failed to stop recording at the time limit")
                self._reset_after_failure()

    def _cancel_recording_timer(self) -> None:
        timer, self._recording_timer = self._recording_timer, None
        self._recording_token = None
        if timer is not None:
            timer.cancel()

    def _transcribe_and_publish(self, audio: np.ndarray) -> str:
        try:
            return self._publish_transcription(audio)
        except Exception:
            log.exception("transcription failed")
            self._notify("Transcription failed", "See the dictation service log", transient=True)
            raise
        finally:
            with self._capture_lock:
                self._pending_transcriptions -= 1

    def _publish_transcription(self, audio: np.ndarray) -> str:
        started = time.monotonic()
        text = self._transcribe(audio)
        elapsed = time.monotonic() - started
        duration = len(audio) / config_mod.WHISPER_SAMPLE_RATE
        log.info("transcribed %.2fs of audio in %.2fs", duration, elapsed)
        if self._closing:
            raise RuntimeError("daemon is shutting down")

        if not text:
            self._notify("No speech detected", transient=True)
            return "empty"

        copied = self._copy_to_clipboard(text)
        if not self.cfg.get("output.clipboard"):
            summary = "Transcribed"
        elif copied:
            summary = "Copied "
        else:
            summary = "Transcribed — clipboard copy failed"
        self._notify(summary, self._preview(text))
        return text

    def do_toggle(self) -> str | Future:
        return self.do_stop() if self.state == RECORDING else self.do_start()

    def _reset_after_failure(self) -> None:
        # Return to a known-good state after a handler blew up mid-recording
        self.state = IDLE
        self._cancel_recording_timer()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                log.exception("failed to close the input stream during cleanup")
        self._frames = []

    def close(self) -> None:
        with self._capture_lock:
            self._closing = True
            self._reset_after_failure()
        self._transcriptions.shutdown(wait=True)
        self._notifications.shutdown(wait=True, cancel_futures=True)

    ##### SERVER #####

    def serve(self, socket_path: Path) -> None:
        _clear_stale_socket(socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            bound_socket = socket_path.lstat()
            os.chmod(socket_path, 0o600)
            server.listen(8)
            server.settimeout(0.2)
            log.info("listening on %s", socket_path)
            # Only replies wait in workers; commands are applied in connection order
            clients = ThreadPoolExecutor(max_workers=MAX_CLIENTS, thread_name_prefix="dictation-client")
            slots = threading.BoundedSemaphore(MAX_CLIENTS)

            try:
                while not self._closing:
                    try:
                        conn, _ = server.accept()
                    except (InterruptedError, socket.timeout):
                        continue
                    if not slots.acquire(blocking=False):
                        conn.close()
                        continue
                    try:
                        result = self._receive_command(conn)
                        if result is None:
                            conn.close()
                            slots.release()
                            continue
                        clients.submit(self._serve_client, conn, result, slots)
                    except BaseException:
                        conn.close()
                        slots.release()
                        raise
            finally:
                self.close()
                clients.shutdown(wait=True)
                _unlink_socket(socket_path, bound_socket)
                log.info("socket removed, exiting")

    def _serve_client(
        self, conn: socket.socket, result: Future, slots: threading.BoundedSemaphore,
    ) -> None:
        try:
            with conn:
                self._send_reply(conn, result)
        finally:
            slots.release()

    def _receive_command(self, conn: socket.socket) -> Future | None:
        deadline = time.monotonic() + CLIENT_REQUEST_TIMEOUT_S
        try:
            data = bytearray()
            while b"\n" not in data and len(data) < 1024:
                remaining = deadline - time.monotonic()
                if self._closing or remaining <= 0:
                    return None
                # Poll for shutdown without extending the total request deadline
                conn.settimeout(min(0.2, remaining))
                try:
                    chunk = conn.recv(1024 - len(data))
                except socket.timeout:
                    continue
                if not chunk:
                    break
                data.extend(chunk)
        except OSError as exc:
            log.debug("client read failed: %s", type(exc).__name__)
            return None

        verb = data.decode(errors="replace").strip()
        if not verb:
            return None

        reply = Future()
        try:
            if len(data) >= 1024:
                raise ValueError("command is too long")
            result = self.handle(verb)
            if isinstance(result, Future):
                return result
            reply.set_result(result)
        except Exception as exc:
            reply.set_exception(exc)
        return reply

    def _send_reply(self, conn: socket.socket, result: Future) -> None:
        try:
            reply = {"ok": True, "text": result.result()}
        except Exception as exc:
            log.warning("command failed: %s", type(exc).__name__)
            reply = {"ok": False, "text": f"{type(exc).__name__}: {exc}"}
        try:
            conn.settimeout(5)
            conn.sendall(json.dumps(reply).encode() + b"\n")
        except OSError as exc:
            log.debug("client reply failed: %s", type(exc).__name__)


def _clear_stale_socket(socket_path: Path) -> None:
    # Remove a leftover socket file, but never one a live daemon is using
    try:
        original = socket_path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(original.st_mode):
        raise ValueError(f"refusing to remove a non-socket path: {socket_path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(1)
        probe.connect(str(socket_path))
    except (ConnectionRefusedError, FileNotFoundError):
        log.warning("removing stale socket at %s", socket_path)
        _unlink_socket(socket_path, original)
    else:
        raise SystemExit(f"a daemon is already listening on {socket_path}")
    finally:
        probe.close()


def _unlink_socket(socket_path: Path, original: os.stat_result) -> None:
    try:
        current = socket_path.lstat()
    except FileNotFoundError:
        return
    same_file = (current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)
    if not stat.S_ISSOCK(current.st_mode) or not same_file:
        raise ValueError(f"socket path changed; refusing to remove it: {socket_path}")
    socket_path.unlink()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="dictation-daemon", description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = config_mod.load(args.config)
    log.info("config: %s", cfg.source or "built-in defaults")

    daemon = DictationDaemon(cfg)

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    try:
        daemon.serve(cfg.socket_path)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
