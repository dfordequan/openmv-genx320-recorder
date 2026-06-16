"""CLI dispatcher: `genx320 {list,record,replay,analyze}`."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from . import analyze as analyze_mod
from . import cameras as cam_mod
from . import record as record_mod
from . import replay as replay_mod


def _cmd_list(_argv: List[str]) -> int:
    cams = cam_mod.find_cameras()
    if not cams:
        print("no OpenMV / MicroPython USB-CDC devices found")
        return 1
    print(f"found {len(cams)} candidate camera(s):")
    for c in cams:
        print(f"  {c}")
    return 0


def _select_port(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    cams = cam_mod.find_cameras()
    if not cams:
        raise SystemExit(
            "error: no OpenMV cameras found. plug one in, or pass --port "
            "explicitly. run `genx320 list` to see what's connected."
        )
    if len(cams) > 1:
        msg = "error: multiple OpenMV cameras found, pick one with --port:\n"
        for c in cams:
            msg += f"  --port {c.port}  ({c.description}, sn={c.serial_number})\n"
        raise SystemExit(msg.rstrip())
    return cams[0].port


def _cmd_record(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="genx320 record",
        description="Record from an OpenMV + GenX320. "
                    "Stops on Ctrl+C (or after --duration).",
    )
    ap.add_argument("--mode", choices=("events", "histo"), default="events",
                    help="recording mode (default: events). "
                         "'events' = raw asynchronous events (N×6 uint16). "
                         "'histo' = 320×320 grayscale event-histogram frames "
                         "accumulated on-chip, like a normal camera.")
    ap.add_argument("--port", default=None,
                    help="serial device (default: auto-detect)")
    ap.add_argument("--duration", type=float, default=None,
                    help="capture duration in seconds (default: until Ctrl+C)")
    ap.add_argument("--output", "-o", default=None,
                    help="output .npz path (default: recording_TIMESTAMP.npz)")
    ap.add_argument("--evt-res", type=int, default=2048,
                    help="[events mode] per-ioctl event buffer size "
                         "(pow2 in [1024, 65536], default 2048)")
    ap.add_argument("--framerate", type=int, default=30,
                    help="[histo mode] target frame rate in Hz (default 30). "
                         "Higher rates may exceed USB throughput.")
    ap.add_argument("--transport", choices=("auto", "repl", "omv"),
                    default="auto",
                    help="[histo mode] how to fetch frames. 'omv' uses the "
                         "framed OMV_PROTOCOL (firmware v5+) for ~3x the "
                         "REPL throughput. 'repl' falls back to the legacy "
                         "raw-REPL+base64 path that works on any firmware. "
                         "'auto' (default) tries OMV first then REPL.")
    ap.add_argument("--no-status", action="store_true",
                    help="suppress the live status line")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the GenX320 capability probe before recording")
    args = ap.parse_args(argv)

    port = _select_port(args.port)

    if not args.no_verify and args.mode == "events":
        print(f"[record] verifying GenX320 event-mode APIs on {port} …")
        err = cam_mod.confirm_genx320(port)
        if err:
            print(f"error: {err}", file=sys.stderr)
            print(
                "hint: this firmware may not expose the GenX320 event-mode "
                "APIs. update via OpenMV IDE → Tools → Install latest "
                "development firmware.\n"
                "histogram mode (`--mode histo`) works on older firmware too.",
                file=sys.stderr,
            )
            return 2

    if args.mode == "events":
        events, meta, out_path = record_mod.record_events(
            port=port,
            output_path=args.output,
            duration_s=args.duration,
            evt_res=args.evt_res,
            show_status=not args.no_status,
        )
        record_mod.print_events_summary(events, meta, out_path)
    else:
        transport = _resolve_transport(args.transport, port)
        if transport == "omv":
            frames, ts, meta, out_path = record_mod.record_histo_omv(
                port=port,
                output_path=args.output,
                duration_s=args.duration,
                framerate=args.framerate,
                show_status=not args.no_status,
            )
        else:
            frames, ts, meta, out_path = record_mod.record_histo(
                port=port,
                output_path=args.output,
                duration_s=args.duration,
                framerate=args.framerate,
                show_status=not args.no_status,
            )
        record_mod.print_histo_summary(frames, ts, meta, out_path)
    return 0


def _resolve_transport(requested: str, port: str) -> str:
    """Pick a transport for histo mode.

    'omv' or 'repl': used as-is.
    'auto': probe OMV_PROTOCOL by sending PROTO_SYNC; fall back to REPL if it
            doesn't respond.
    """
    if requested != "auto":
        return requested

    # Auto-detect: quick OMV sync attempt.
    from . import omv_protocol as omv
    try:
        with omv.OmvProtocol(port, timeout=0.5) as p:
            p.sync(retries=1, timeout=0.5)
        print("[record] auto-detected transport: omv "
              "(firmware speaks OMV_PROTOCOL)")
        return "omv"
    except Exception:
        print("[record] auto-detected transport: repl "
              "(firmware does not speak OMV_PROTOCOL)")
        return "repl"


def _cmd_replay(argv: List[str]) -> int:
    return replay_mod.run(argv)


def _cmd_analyze(argv: List[str]) -> int:
    return analyze_mod.run(argv)


_COMMANDS = {
    "list": _cmd_list,
    "record": _cmd_record,
    "replay": _cmd_replay,
    "analyze": _cmd_analyze,
}


def _print_help() -> None:
    print(
        f"genx320 — OpenMV + Prophesee GenX320 event recorder (v{__version__})\n"
        "\n"
        "Usage: genx320 <command> [options]\n"
        "\n"
        "Commands:\n"
        "  list     show connected OpenMV / MicroPython cameras\n"
        "  record   record events to a .npz file (Ctrl+C to stop)\n"
        "  replay   play back a recorded .npz file as a video\n"
        "  analyze  diagnose whether a recording dropped events\n"
        "\n"
        "Run `genx320 <command> --help` for command-specific options."
    )


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_help()
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in _COMMANDS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        _print_help()
        return 1
    return _COMMANDS[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
