# Dictation-tool

Push-to-talk dictation for Hyprland/Wayland. Hold a key, speak, release. Transcript
lands on your clipboard. Local inference with faster-whisper, English only.

A persistent daemon keeps the Whisper model resident in VRAM &
a thin client bound to press/release drives it over a unix socket.

Example:

```
Hyprland  SUPER+D press    ->  dictation-client start  ->\
                                                          }-> daemon (warm model)
Hyprland  SUPER+D release  ->  dictation-client stop   ->/         |
                                                                   v
                                                       wl-copy + notify-send
```

## Setup

Requires Python 3.11+, PortAudio, `wl-copy`/`wl-paste` (wl-clipboard), and
`notify-send`. The tested system uses Python 3.14, PipeWire, and Hyprland with Lua
configuration. The CUDA dependencies below are for the tested NVIDIA setup. They can
vary depending on your setup.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then run the CUDA smoke test before anything else:

```bash
./bin/dictation-daemon --verbose
```

It will download `large-v3-turbo` (~1.6 GB) and, on first run only, JIT-compile PTX for
your GPU. You want to see:

```
model ready on cuda (float16) in X.Xs
listening on /run/user/1000/dictation.sock
```

In another terminal, confirm it with `./bin/dictation-client ping` → `pong cuda`. The daemon sits on about
2.4 GB of VRAM. If it says `FALLING BACK TO CPU`, read the next section.

## CUDA notes

If CUDA can't be made to work, set `device = "cpu"` and `compute_type = "int8"` in
`config.toml`.

## Audio capture

The microphone opens when recording starts and closes before transcription. Audio stays
in memory and is passed directly to faster-whisper as a mono float32 array at 16 kHz.
Keep `audio.sample_rate = 16000` and `audio.channels = 1`.

The tested PipeWire setup accepts this format directly. An empty `audio.device` follows
the system default microphone; use a device name such as `"Scarlett"` to select one.

## Manual verification

If the service is installed, leave it running. Otherwise, launch
`./bin/dictation-daemon --verbose` in a separate terminal first.

```bash
./bin/dictation-client ping           # expect "pong cuda"
./bin/dictation-client start
# ...speak...
./bin/dictation-client stop           # prints the transcript as a notification
wl-paste                              # same text
```

`status` reports `idle`/`recording`; `toggle` is there if you ever want a tap-to-toggle
bind instead of push-to-talk.

## Install

Run these commands from the repository. The service assumes the checkout is at
`~/Documents/GitHub/Dictation-tool`. The Lua snippet reads `HOME` and assumes the same
checkout location. Adjust the service's `ExecStart` and the snippet's startup-helper
path if you keep the repo elsewhere. Stop any foreground daemon
with Ctrl-C before starting the service.

If upgrading from the old enabled unit, first run
`systemctl --user disable dictation-daemon.service`. Disabling removes the unit symlink;
the commands below recreate it.

```bash
mkdir -p ~/.local/bin ~/.config/systemd/user
ln -sf "$PWD/bin/dictation-client" ~/.local/bin/dictation-client
ln -sf "$PWD/systemd/dictation-daemon.service" ~/.config/systemd/user/
systemctl --user daemon-reload
```

Then append the contents of [hyprland-binds.lua](hyprland-binds.lua) to the KEYBINDINGS
section of `~/.config/hypr/hyprland.lua` and `hyprctl reload`. The snippet includes a
session startup hook as well as the keybinds. Start it once for the current session:

```bash
./bin/dictation-session-start
systemctl --user status dictation-daemon.service
```

The service is static: Hyprland starts it at login, so do not use `systemctl enable`.
Starting from `default.target` runs before Hyprland and can leave the daemon without
`WAYLAND_DISPLAY`. Importing the environment later does not update an already running
process. The startup helper imports the display first, then restarts the service.
This bare Hyprland setup does not activate `graphical-session.target`.

Logs: `journalctl --user -u dictation-daemon -f`

## Freeing VRAM

To stop the installed daemon and release the model's VRAM:

```bash
systemctl --user stop dictation-daemon.service
```

Dictation stays unavailable until you start the daemon again or begin a new Hyprland
session, when the startup hook starts it automatically. `SUPER+D` does not restart it.

To start it again, run this from the repository in a terminal in your Wayland session:

```bash
./bin/dictation-session-start
```

Allow a few seconds for the model to load before recording. Check readiness with
`./bin/dictation-client ping`; `pong cuda` means it is ready on the GPU.

## Configuration

[config.toml](config.toml) is documented inline. Copy it to `config.local.toml` to
override without dirtying the repo — the daemon prefers the local file when present.
Unknown sections or keys are a hard error rather than a silent no-op.

## Launcher syntax check

```bash
bash -n bin/dictation-client bin/dictation-daemon bin/dictation-session-start
```

This checks shell syntax only. Use the manual verification above to check dictation
and clipboard output.
