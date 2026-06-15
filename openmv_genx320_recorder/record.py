"""Record raw events from an OpenMV + GenX320.

Records until Ctrl+C (or until --duration expires) and saves the events to a
.npz file. Streams events from the camera while running so peak host RAM is
proportional to the recording length × event rate, not bounded by a fixed
buffer on the device.
"""

from __future__ import annotations

import base64
import datetime as dt
import signal
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import serial

from . import cameras as cam_mod
from . import format as fmt


# --------------------------------------------------------------------------
# On-camera MicroPython script
# --------------------------------------------------------------------------

# Each call to IOCTL_GENX320_READ_EVENTS fills an (EVT_RES, 6) uint16 buffer;
# we ship the populated rows over USB-CDC as a single base64 line. The "C N
# <b64>" framing is dead simple to parse incrementally on the host side.
#
# DURATION_MS is set by the host. For "record until Ctrl+C" the host passes a
# very large value (e.g. 24h) and interrupts the script with Ctrl+C.
_CAPTURE_SCRIPT = r"""
import sys, time, binascii
import csi
from ulab import numpy as np

EVT_RES = {evt_res}
DURATION_MS = {duration_ms}

events = np.zeros((EVT_RES, 6), dtype=np.uint16)

cam = csi.CSI(cid=csi.GENX320)
cam.reset()
try:
    cam.ioctl(csi.IOCTL_GENX320_CALIBRATE, 200, 0.5)
except BaseException:
    pass
cam.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, EVT_RES)

sys.stdout.write("<<<HEADER>>>\n")
sys.stdout.write("evt_res=%d\n" % EVT_RES)
sys.stdout.write("<<<STREAM>>>\n")

t0_ms = time.ticks_ms()
t0_us = time.ticks_us()
total = 0
iters = 0
saturated = 0
neg_returns = 0
max_n = 0
try:
    while time.ticks_diff(time.ticks_ms(), t0_ms) < DURATION_MS:
        n = cam.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)
        if n < 0:
            neg_returns += 1
        elif n > 0:
            if n > max_n:
                max_n = n
            if n == EVT_RES:
                saturated += 1
            chunk = events[:n].tobytes()
            b64 = binascii.b2a_base64(chunk, newline=False).decode()
            sys.stdout.write("C %d %s\n" % (n, b64))
            total += n
        iters += 1
except KeyboardInterrupt:
    pass

t1_us = time.ticks_us()
sys.stdout.write("<<<FOOTER>>>\n")
sys.stdout.write("events=%d\n" % total)
sys.stdout.write("iters=%d\n" % iters)
sys.stdout.write("saturated_iters=%d\n" % saturated)
sys.stdout.write("neg_returns=%d\n" % neg_returns)
sys.stdout.write("max_n_per_call=%d\n" % max_n)
sys.stdout.write("elapsed_us=%d\n" % time.ticks_diff(t1_us, t0_us))
sys.stdout.write("<<<END>>>\n")
"""


# --------------------------------------------------------------------------
# Streaming reader (background thread)
# --------------------------------------------------------------------------

class _LineReader:
    """Read serial in a background thread, deliver complete \\n-terminated lines."""

    def __init__(self, ser: serial.Serial) -> None:
        self._ser = ser
        self._buf = bytearray()
        self._lines: List[bytes] = []
        self._lock = threading.Lock()
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._alive = False
        self._thread.join(timeout=2.0)

    def get_lines(self) -> List[bytes]:
        with self._lock:
            out, self._lines = self._lines, []
            return out

    def _run(self) -> None:
        local = bytearray()
        while self._alive:
            try:
                chunk = self._ser.read(65536)
            except serial.SerialException:
                break
            if chunk:
                local.extend(chunk)
                while b"\n" in local:
                    line, _, rest = local.partition(b"\n")
                    with self._lock:
                        self._lines.append(bytes(line))
                    local = bytearray(rest)
            else:
                time.sleep(0.005)
        if local:
            with self._lock:
                self._lines.append(bytes(local))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

class RecordingResult(Tuple[np.ndarray, dict]):
    """Tuple subtype just for type hint clarity."""


def record(
    port: str,
    output_path: Optional[str] = None,
    duration_s: Optional[float] = None,
    evt_res: int = 2048,
    show_status: bool = True,
) -> Tuple[np.ndarray, dict, str]:
    """Record events from one camera into a .npz file.

    Args:
        port: serial device, e.g. "/dev/ttyACM0" or "COM3".
        output_path: where to save the .npz. Defaults to recording_YYYYMMDD_HHMMSS.npz.
        duration_s: fixed capture duration. None = record until Ctrl+C.
        evt_res: per-ioctl event buffer size on the device (pow2 in [1024, 65536]).
        show_status: print a live "X events captured (Y ev/s)" line.

    Returns:
        (events, metadata, output_path) tuple. The events array is also saved to disk.
    """
    if output_path is None:
        output_path = f"recording_{dt.datetime.now():%Y%m%d_%H%M%S}.npz"

    # Use a 24-hour internal timeout when the user wants Ctrl+C-driven capture;
    # the host's SIGINT handler interrupts well before that.
    duration_ms = (
        int(duration_s * 1000) if duration_s is not None
        else 24 * 60 * 60 * 1000
    )

    script = _CAPTURE_SCRIPT.format(evt_res=evt_res, duration_ms=duration_ms)

    print(f"[record] opening {port} …")
    ser = cam_mod.open_raw_repl(port)
    reader = _LineReader(ser)

    # SIGINT → interrupt the on-camera script so it emits the footer.
    stop_requested = threading.Event()

    def _sigint_handler(_sig, _frame):
        if not stop_requested.is_set():
            stop_requested.set()
            sys.stdout.write("\n[record] Ctrl+C — stopping camera …\n")
            sys.stdout.flush()
            cam_mod.send_ctrl_c(ser)

    prev_sigint = signal.signal(signal.SIGINT, _sigint_handler)

    chunks: List[np.ndarray] = []
    meta: dict = {}
    saw_header = False
    in_stream = False
    saw_footer = False
    saw_end = False
    chunks_received = 0
    chunks_malformed = 0
    total_decoded = 0

    if duration_s is None:
        print("[record] no --duration set, recording until Ctrl+C")
    else:
        print(f"[record] capturing for {duration_s:.1f} s")

    last_print = time.time()
    last_count = 0
    started_streaming_at: Optional[float] = None

    try:
        reader.start()
        cam_mod.start_exec(ser, script)

        # Cap how long we wait after the script ends or after Ctrl+C.
        post_ctrlc_deadline: Optional[float] = None

        while True:
            for line in reader.get_lines():
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                if text == "<<<HEADER>>>":
                    saw_header = True
                    continue
                if text == "<<<STREAM>>>":
                    in_stream = True
                    started_streaming_at = time.time()
                    continue
                if text == "<<<FOOTER>>>":
                    in_stream = False
                    saw_footer = True
                    continue
                if text == "<<<END>>>":
                    saw_end = True
                    continue

                if in_stream and text.startswith("C "):
                    chunks_received += 1
                    parts = text.split(" ", 2)
                    if len(parts) != 3:
                        chunks_malformed += 1
                        continue
                    try:
                        n = int(parts[1])
                        raw = base64.b64decode(parts[2])
                    except (ValueError, base64.binascii.Error):
                        chunks_malformed += 1
                        continue
                    if len(raw) != n * 6 * 2:
                        chunks_malformed += 1
                        continue
                    chunks.append(
                        np.frombuffer(raw, dtype=np.uint16).reshape(n, 6).copy()
                    )
                    total_decoded += n
                    continue

                if "=" in text and not in_stream:
                    k, _, v = text.partition("=")
                    try:
                        meta[k.strip()] = int(v.strip())
                    except ValueError:
                        meta[k.strip()] = v.strip()
                    continue
                # Anything else (e.g. "CSI: Calibrating - 0%") just gets dropped.

            if saw_end:
                break

            if stop_requested.is_set():
                if post_ctrlc_deadline is None:
                    post_ctrlc_deadline = time.time() + 3.0  # 3 s grace
                if time.time() > post_ctrlc_deadline:
                    print("[record] camera did not emit footer; saving what we have")
                    break

            if show_status and started_streaming_at:
                now = time.time()
                if now - last_print > 0.5:
                    rate = (total_decoded - last_count) / max(now - last_print, 1e-9)
                    sys.stdout.write(
                        f"\r[record] {total_decoded:>9} events  ({rate:>8.0f} ev/s)  "
                    )
                    sys.stdout.flush()
                    last_print = now
                    last_count = total_decoded

            time.sleep(0.02)
    finally:
        if show_status:
            sys.stdout.write("\n")
            sys.stdout.flush()
        signal.signal(signal.SIGINT, prev_sigint)
        reader.stop()
        cam_mod.close_raw_repl(ser)

    if chunks:
        events = np.concatenate(chunks, axis=0)
    else:
        events = np.zeros((0, 6), dtype=np.uint16)

    meta["chunks_received"] = chunks_received
    meta["chunks_malformed"] = chunks_malformed
    meta["decoded_events"] = total_decoded
    meta["host_capture_started"] = dt.datetime.now().isoformat()
    meta["host_port"] = port
    meta["stopped_by_user"] = bool(stop_requested.is_set())
    meta["duration_s"] = meta.get("elapsed_us", 0) / 1e6 if meta.get("elapsed_us") else duration_s

    fmt.save_recording(output_path, events, meta)

    return events, meta, output_path


def print_summary(events: np.ndarray, meta: dict, output_path: str) -> None:
    n = events.shape[0]
    print(f"[record] saved {n} events → {output_path}")
    if n == 0:
        return
    types = events[:, 0]
    on_count = int((types == 1).sum())
    off_count = int((types == 0).sum())
    t_us = fmt.events_to_microseconds(events)
    span_s = (int(t_us[-1]) - int(t_us[0])) / 1e6
    rate = n / max(span_s, 1e-9)
    print(f"[record]   ON={on_count}  OFF={off_count}")
    print(
        f"[record]   x∈[{int(events[:, 4].min())},{int(events[:, 4].max())}]  "
        f"y∈[{int(events[:, 5].min())},{int(events[:, 5].max())}]"
    )
    print(f"[record]   timespan ≈ {span_s:.3f} s, average ≈ {rate:.0f} ev/s")
    sat = meta.get("saturated_iters", 0)
    iters = meta.get("iters", 0)
    if iters:
        print(
            f"[record]   sensor saturation: {sat}/{iters} reads at EVT_RES "
            f"({100*sat/max(1,iters):.1f}%)"
        )
