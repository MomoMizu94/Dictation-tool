"""Dictation daemon: warm Whisper model behind a unix socket.

Logic
    startup  -> load the Whisper model once and keep it resident
    "start"  -> open the mic stream, buffer frames
    "stop"   -> close the stream, transcribe the buffer, copy to clipboard, notify """

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import config as config_mod

log = logging.getLogger("dictation")

IDLE = "idle"
RECORDING = "recording"


class DictationDaemon:
    def __init__(self, cfg: config_mod.Config):
        self.cfg = cfg
        self.state = IDLE

        # The audio callback appends frames until the input stream is stopped
        self._frames: list[np.ndarray] = []
        self._stream = None
        self._recording_started_at: float | None = None

        self.model = self._load_model()

    ##### MODEL #####

    def _load_model(self):
        # Load faster-whisper once, falling back to CPU if CUDA is unavailable
        from faster_whisper import WhisperModel

        name = self.cfg.get("model.name")
        device = self.cfg.get("model.device")
        compute_type = self.cfg.get("model.compute_type")

        log.info("loading %s on %s (%s)...", name, device, compute_type)
        started = time.monotonic()
        try:
            model = WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as exc:
            if device == "cpu":
                raise
            log.error("CUDA load failed (%s: %s)", type(exc).__name__, exc)
            log.error("FALLING BACK TO CPU -- expect much higher latency.")
            log.error("See README 'CUDA notes'; usually LD_LIBRARY_PATH is missing.")
            device, compute_type = "cpu", "int8"
            model = WhisperModel(name, device=device, compute_type=compute_type)

        log.info(
            "model ready on %s (%s) in %.1fs", device, compute_type,
            time.monotonic() - started,
        )
        self.active_device = device
        return model

    ##### CAPTURE #####

    def _start_capture(self) -> None:
        # Open the input stream and begin buffering frames
        self._frames = []

        def _callback(indata, frames, time_info, status):
            self._frames.append(indata.copy())
            if status:
                log.warning("audio status: %s", status)

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
        )

        # Iterating segments completes inference. Segment text includes spacing
        text = "".join([segment.text for segment in segments]).strip()

        if text == "":
            return None

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
        if not self.cfg.get("output.notifications"):
            return
        cmd = ["notify-send", "--app-name=Dictation", "--expire-time=2500"]
        if transient:
            cmd.append("--hint=int:transient:1")
        cmd += ["--", summary]
        if body:
            cmd.append(body)
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=5)
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            log.warning("notify-send failed: %s", exc)

    def _preview(self, text: str) -> str:
        limit = int(self.cfg.get("output.preview_chars"))
        if limit <= 0:
            return ""
        collapsed = " ".join(text.split())
        return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"

    ##### HANDLERS #####

    def handle(self, verb: str) -> str:
        handlers = {
            "start": self.do_start,
            "stop": self.do_stop,
            "toggle": self.do_toggle,
            "status": lambda: self.state,
            "ping": lambda: f"pong {self.active_device}",
        }
        handler = handlers.get(verb)
        if handler is None:
            return f"error: unknown verb {verb!r}"
        try:
            return handler()
        except Exception as exc:
            log.exception("handler for %r failed", verb)
            self._reset_after_failure()
            return f"error: {type(exc).__name__}: {exc}"

    def do_start(self) -> str:
        if self.state == RECORDING:
            return "already recording"
        self._start_capture()
        self.state = RECORDING
        self._recording_started_at = time.monotonic()
        log.info("recording")
        self._notify("Recording…", transient=True)
        return "recording"

    def do_stop(self) -> str:
        if self.state == IDLE:
            return "not recording"

        self.state = IDLE
        held = time.monotonic() - (self._recording_started_at or time.monotonic())
        audio = self._stop_capture()

        if audio is None or len(audio) == 0:
            log.info("no audio captured (held %.2fs)", held)
            self._notify("Nothing captured", transient=True)
            return "empty"

        started = time.monotonic()
        text = self._transcribe(audio)
        elapsed = time.monotonic() - started
        duration = len(audio) / config_mod.WHISPER_SAMPLE_RATE
        log.info("transcribed %.2fs of audio in %.2fs", duration, elapsed)

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

    def do_toggle(self) -> str:
        return self.do_stop() if self.state == RECORDING else self.do_start()

    def _reset_after_failure(self) -> None:
        # Return to a known-good state after a handler blew up mid-recording
        self.state = IDLE
        self._frames = []
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                log.exception("failed to close the input stream during cleanup")

    ##### SERVER #####

    def serve(self, socket_path: Path) -> None:
        _clear_stale_socket(socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)
            server.listen(8)
            log.info("listening on %s", socket_path)

            try:
                while True:
                    try:
                        conn, _ = server.accept()
                    except InterruptedError:
                        continue
                    with conn:
                        self._serve_one(conn)
            finally:
                socket_path.unlink(missing_ok=True)
                log.info("socket removed, exiting")

    def _serve_one(self, conn: socket.socket) -> None:
        conn.settimeout(5)
        try:
            data = conn.recv(1024)
        except socket.timeout:
            log.warning("client sent nothing within the timeout")
            return

        verb = data.decode(errors="replace").strip()
        if not verb:
            return

        reply = self.handle(verb)
        try:
            conn.sendall(reply.encode() + b"\n")
        except BrokenPipeError:
            log.debug("client disconnected before the reply for %r", verb)


def _clear_stale_socket(socket_path: Path) -> None:
    # Remove a leftover socket file, but never one a live daemon is using
    if not socket_path.exists():
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(1)
        probe.connect(str(socket_path))
    except (ConnectionRefusedError, FileNotFoundError):
        log.warning("removing stale socket at %s", socket_path)
        socket_path.unlink(missing_ok=True)
    else:
        raise SystemExit(f"a daemon is already listening on {socket_path}")
    finally:
        probe.close()


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
