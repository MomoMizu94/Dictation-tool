"""Send a desktop notification using JSON received on stdin."""

import html
import json
import sys


def main() -> int:
    try:
        import dbus

        payload = json.load(sys.stdin)
        hints = {}
        if payload["transient"]:
            hints["transient"] = dbus.Boolean(True, variant_level=1)
        bus = dbus.SessionBus(private=True)
        try:
            notifications = bus.get_object(
                "org.freedesktop.Notifications", "/org/freedesktop/Notifications",
                introspect=False,
            )
            notifications.Notify(
                "Dictation", dbus.UInt32(0), "", payload["summary"],
                html.escape(payload["body"]), dbus.Array([], signature="s"),
                dbus.Dictionary(hints, signature="sv"), dbus.Int32(2500),
                dbus_interface="org.freedesktop.Notifications", timeout=2,
            )
        finally:
            bus.close()
    except Exception as exc:
        # D-Bus errors can echo the payload, so only report their type
        print(f"notification failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
