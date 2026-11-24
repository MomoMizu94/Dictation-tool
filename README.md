# Dictation-tool

Hold **Super+D**, speak, then release to copy the transcript to your clipboard.
English dictation runs locally with faster-whisper. A background service keeps the
model loaded for quick responses.

## Setup

Requires Hyprland with Lua configuration, Python 3.11+, PortAudio, wl-clipboard,
and the system Python D-Bus module for notifications. The default setup uses an
NVIDIA GPU; CPU mode is also available.

Keep the checkout at `~/Documents/GitHub/Dictation-tool`, or update the paths in
[the service](systemd/dictation-daemon.service) and [the bindings](hyprland-binds.lua).
Run these commands from the repository:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
/usr/bin/python3 -c 'import dbus'
```

## Installation

If upgrading an older enabled service, run
`systemctl --user disable dictation-daemon.service` before creating these links:

```bash
mkdir -p ~/.local/bin ~/.config/systemd/user
ln -sf "$PWD/bin/dictation-client" ~/.local/bin/dictation-client
ln -sf "$PWD/systemd/dictation-daemon.service" ~/.config/systemd/user/
systemctl --user daemon-reload
```

Copy the contents of [hyprland-binds.lua](hyprland-binds.lua) into
`~/.config/hypr/hyprland.lua`, replacing any existing dictation bindings.
Then run from a Hyprland terminal:

```bash
hyprctl reload
./bin/dictation-session-start
```

The first start downloads and warms up the model, which can take a few minutes.
Check readiness with `./bin/dictation-client ping`; `pong cuda` means it is ready
on the GPU. Hyprland starts the service at login, so do not use `systemctl enable`.

## Configuration

Copy `config.toml` to `config.local.toml` for local settings. See
[config.toml](config.toml) for all options.

| Setting | Use |
| --- | --- |
| `audio.device` | Leave empty for the default microphone, or enter a device name |
| `audio.max_duration_s` | Recording limit, default 120 seconds |
| `transcribe.ensure_final_period` | Set to `false` when dictating fragments or code |
| `output.preview_chars` | Set to `0` to hide transcript text in notifications |

Keep audio at 16000 Hz and one channel. For CPU mode, set `model.device = "cpu"`
and `model.compute_type = "int8"`. Restart after changing settings:

```bash
systemctl --user restart dictation-daemon.service
```

Audio and transcripts stay local. Clipboard managers and notification history can
retain text.

## Tips

Free the model's VRAM by stopping the service:

```bash
systemctl --user stop dictation-daemon.service
```

Start it again with `./bin/dictation-session-start`. Super+D does not restart a
stopped service.

- Check state: `./bin/dictation-client status`
- Watch logs: `journalctl --user -u dictation-daemon -f`
