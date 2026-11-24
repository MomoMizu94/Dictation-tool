""" Connects to the daemon socket, sends one verb, prints the reply.

This is what the Hyprland keybind runs, so it must stay cheap to start """

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from . import config as config_mod

VERBS = ("start", "stop", "toggle", "status", "ping")


class DaemonError(RuntimeError):
    pass


def send(socket_path, verb: str, timeout: float = 60.0) -> str:
    # Send `verb` to the daemon and return its single-line reply

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(socket_path))
        sock.sendall(verb.encode() + b"\n")
        sock.shutdown(socket.SHUT_WR)

        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)

    try:
        reply = json.loads(b"".join(chunks))
    except (ValueError, UnicodeDecodeError) as exc:
        raise DaemonError("invalid daemon reply; restart the daemon after upgrading") from exc
    if not isinstance(reply, dict):
        raise DaemonError("invalid daemon reply")
    if type(reply.get("ok")) is not bool or not isinstance(reply.get("text"), str):
        raise DaemonError("invalid daemon reply")
    if not reply["ok"]:
        raise DaemonError(reply["text"])
    return reply["text"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dictation-client", description=__doc__)
    parser.add_argument("verb", choices=VERBS)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="seconds to wait for a reply"
    )
    args = parser.parse_args(argv)

    try:
        cfg = config_mod.load(args.config)
        reply = send(cfg.socket_path, args.verb, timeout=args.timeout)
    except socket.timeout:
        print("dictation-client: daemon did not reply in time", file=sys.stderr)
        return 1
    except (OSError, ValueError, DaemonError) as exc:
        print(f"dictation-client: {exc}", file=sys.stderr)
        return 1

    if reply:
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
