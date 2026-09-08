# Config loading: TOML on disk, merged over the defaults below

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, dict[str, Any]] = {
    "model": {
        "name": "large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "blocksize": 1024,
        "device": "",
        "min_duration_s": 0.3,
    },
    "transcribe": {
        "language": "en",
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    },
    "daemon": {
        "socket_path": "$XDG_RUNTIME_DIR/dictation.sock",
    },
    "output": {
        "clipboard": True,
        "notifications": True,
        "preview_chars": 80,
    },
}

# Required input sample rate (without resampling)
WHISPER_SAMPLE_RATE = 16000


class Config:
    def __init__(self, data: dict[str, dict[str, Any]], source: Path | None):
        self._data = data
        self.source = source

    def get(self, path: str) -> Any:
        section, _, key = path.partition(".")
        return self._data[section][key]

    @property
    def socket_path(self) -> Path:
        return Path(os.path.expandvars(self.get("daemon.socket_path")))

    @property
    def audio_device(self) -> int | str | None:
        dev = self.get("audio.device")
        if dev == "" or dev is None:
            return None
        if isinstance(dev, str) and dev.isdigit():
            return int(dev)
        return dev


def _config_path() -> Path | None:
    # config.local.toml wins over config.toml
    for name in ("config.local.toml", "config.toml"):
        candidate = REPO_ROOT / name
        if candidate.exists():
            return candidate
    return None


def load(path: Path | None = None) -> Config:
    source = path if path is not None else _config_path()
    merged = {section: dict(values) for section, values in DEFAULTS.items()}

    if source is not None:
        with source.open("rb") as handle:
            on_disk = tomllib.load(handle)
        for section, values in on_disk.items():
            if section not in merged:
                raise ValueError(f"{source}: unknown config section [{section}]")
            unknown = set(values) - set(merged[section])
            if unknown:
                raise ValueError(
                    f"{source}: unknown key(s) in [{section}]: {', '.join(sorted(unknown))}"
                )
            merged[section].update(values)

    return Config(merged, source)
