"""Record from an OpenMV + GenX320 in either event mode or histogram mode.

Records until Ctrl+C (or until --duration expires) and saves to a .npz file.
Both modes use the same streaming framing (one chunk per output line) — only
the per-chunk payload format differs:
  - mode='events': "C N <b64>"  (N events × 6 uint16, decoded events)
  - mode='histo':  "F T <b64>"  (one 320×320 uint8 grayscale frame at t=T µs)

Peak host RAM is proportional to recording length × rate, not bounded by a
fixed on-device buffer.
"""

from __future__ import annotations

import base64
import datetime as dt
import signal
import struct
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import serial

from . import cameras as cam_mod
from . import format as fmt
from . import omv_protocol as omv


# --------------------------------------------------------------------------
# On-camera MicroPython script
# --------------------------------------------------------------------------

# Each call to IOCTL_GENX320_READ_EVENTS fills an (EVT_RES, 6) uint16 buffer;
# we ship the populated rows over USB-CDC as a single base64 line. The "C N
# <b64>" framing is dead simple to parse incrementally on the host side.
#
# DURATION_MS is set by the host. For "record until Ctrl+C" the host passes a
# very large value (e.g. 24h) and interrupts the script with Ctrl+C.
_EVENTS_SCRIPT = r"""
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


# Histogram-mode capture script. The GenX320 on this firmware accumulates events
# into a 320×320 grayscale "event histogram" frame at whatever framerate is set
# (default uses the sensor's natural cadence). Each frame is base64-emitted as
# one "F <t_us> <b64>" line; host parses incrementally.
_HISTO_SCRIPT = r"""
import sys, time, binascii
import csi

WIDTH = 320
HEIGHT = 320
FRAMERATE = {framerate}
DURATION_MS = {duration_ms}

cam = csi.CSI(cid=csi.GENX320)
cam.reset()
cam.pixformat(csi.GRAYSCALE)
cam.framesize((WIDTH, HEIGHT))
try:
    cam.framerate(FRAMERATE)
    fr_ok = "ok"
except BaseException as e:
    fr_ok = repr(e)

sys.stdout.write("<<<HEADER>>>\n")
sys.stdout.write("mode=histo\n")
sys.stdout.write("width=%d\n" % WIDTH)
sys.stdout.write("height=%d\n" % HEIGHT)
sys.stdout.write("framerate_set=%d\n" % FRAMERATE)
sys.stdout.write("framerate_status=%s\n" % fr_ok)
sys.stdout.write("<<<STREAM>>>\n")

t0_ms = time.ticks_ms()
t0_us = time.ticks_us()
frames = 0
try:
    while time.ticks_diff(time.ticks_ms(), t0_ms) < DURATION_MS:
        img = cam.snapshot()
        t_us = time.ticks_diff(time.ticks_us(), t0_us)
        b64 = binascii.b2a_base64(bytes(img), newline=False).decode()
        sys.stdout.write("F %d %s\n" % (t_us, b64))
        frames += 1
except KeyboardInterrupt:
    pass

t1_us = time.ticks_us()
sys.stdout.write("<<<FOOTER>>>\n")
sys.stdout.write("frames=%d\n" % frames)
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

def _stream_until_done(
    ser: serial.Serial,
    script: str,
    duration_s: Optional[float],
    show_status: bool,
    on_chunk,
    chunk_prefix: str,
    status_unit: str,
):
    """Run `script` on the camera, parse the stream, return (meta, stop_requested).

    `on_chunk(parts)` is called for every line that starts with `chunk_prefix`,
    with parts already split into [prefix, ...]. It returns either an int
    (count of decoded units, e.g. events/frames) or None on parse failure.
    """
    reader = _LineReader(ser)
    stop_requested = threading.Event()

    def _sigint_handler(_sig, _frame):
        if not stop_requested.is_set():
            stop_requested.set()
            sys.stdout.write("\n[record] Ctrl+C — stopping camera …\n")
            sys.stdout.flush()
            cam_mod.send_ctrl_c(ser)

    prev_sigint = signal.signal(signal.SIGINT, _sigint_handler)

    meta: dict = {}
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
    post_ctrlc_deadline: Optional[float] = None

    try:
        reader.start()
        cam_mod.start_exec(ser, script)

        while True:
            for line in reader.get_lines():
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                if text == "<<<HEADER>>>":
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

                if in_stream and text.startswith(chunk_prefix + " "):
                    chunks_received += 1
                    parts = text.split(" ", 2)
                    if len(parts) != 3:
                        chunks_malformed += 1
                        continue
                    decoded_n = on_chunk(parts)
                    if decoded_n is None:
                        chunks_malformed += 1
                    else:
                        total_decoded += decoded_n
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
                    post_ctrlc_deadline = time.time() + 3.0
                if time.time() > post_ctrlc_deadline:
                    print("[record] camera did not emit footer; saving what we have")
                    break

            if show_status and started_streaming_at:
                now = time.time()
                if now - last_print > 0.5:
                    rate = (total_decoded - last_count) / max(now - last_print, 1e-9)
                    sys.stdout.write(
                        f"\r[record] {total_decoded:>9} {status_unit}  "
                        f"({rate:>8.0f} {status_unit}/s)  "
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

    meta["chunks_received"] = chunks_received
    meta["chunks_malformed"] = chunks_malformed
    meta["decoded_events"] = total_decoded  # generic "decoded count" name kept
    meta["stopped_by_user"] = bool(stop_requested.is_set())
    return meta


def record_events(
    port: str,
    output_path: Optional[str] = None,
    duration_s: Optional[float] = None,
    evt_res: int = 2048,
    show_status: bool = True,
) -> Tuple[np.ndarray, dict, str]:
    """Record raw events into a .npz file."""
    if output_path is None:
        output_path = f"recording_{dt.datetime.now():%Y%m%d_%H%M%S}.npz"

    duration_ms = (
        int(duration_s * 1000) if duration_s is not None
        else 24 * 60 * 60 * 1000
    )

    script = _EVENTS_SCRIPT.format(evt_res=evt_res, duration_ms=duration_ms)

    print(f"[record] opening {port} … (mode: events)")
    ser = cam_mod.open_raw_repl(port)
    chunks: List[np.ndarray] = []

    def _on_event_chunk(parts):
        try:
            n = int(parts[1])
            raw = base64.b64decode(parts[2])
        except (ValueError, base64.binascii.Error):
            return None
        if len(raw) != n * 6 * 2:
            return None
        chunks.append(np.frombuffer(raw, dtype=np.uint16).reshape(n, 6).copy())
        return n

    meta = _stream_until_done(
        ser, script, duration_s, show_status,
        on_chunk=_on_event_chunk, chunk_prefix="C", status_unit="events",
    )

    events = (np.concatenate(chunks, axis=0)
              if chunks else np.zeros((0, 6), dtype=np.uint16))

    meta["host_capture_started"] = dt.datetime.now().isoformat()
    meta["host_port"] = port
    meta["mode"] = fmt.MODE_EVENTS
    meta["duration_s"] = (meta.get("elapsed_us", 0) / 1e6
                          if meta.get("elapsed_us") else duration_s)

    fmt.save_events(output_path, events, meta)
    return events, meta, output_path


def record_histo(
    port: str,
    output_path: Optional[str] = None,
    duration_s: Optional[float] = None,
    framerate: int = 30,
    show_status: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Record histogram-mode frames into a .npz file."""
    if output_path is None:
        output_path = f"recording_{dt.datetime.now():%Y%m%d_%H%M%S}.npz"

    duration_ms = (
        int(duration_s * 1000) if duration_s is not None
        else 24 * 60 * 60 * 1000
    )

    script = _HISTO_SCRIPT.format(framerate=framerate, duration_ms=duration_ms)

    print(f"[record] opening {port} … (mode: histo, framerate={framerate})")
    ser = cam_mod.open_raw_repl(port)
    frames: List[np.ndarray] = []
    timestamps: List[int] = []

    def _on_frame_chunk(parts):
        try:
            t_us = int(parts[1])
            raw = base64.b64decode(parts[2])
        except (ValueError, base64.binascii.Error):
            return None
        if len(raw) != 320 * 320:
            return None
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(320, 320).copy())
        timestamps.append(t_us)
        return 1

    meta = _stream_until_done(
        ser, script, duration_s, show_status,
        on_chunk=_on_frame_chunk, chunk_prefix="F", status_unit="frames",
    )

    if frames:
        frames_arr = np.stack(frames, axis=0)
        ts_arr = np.array(timestamps, dtype=np.int64)
    else:
        frames_arr = np.zeros((0, 320, 320), dtype=np.uint8)
        ts_arr = np.zeros((0,), dtype=np.int64)

    meta["host_capture_started"] = dt.datetime.now().isoformat()
    meta["host_port"] = port
    meta["mode"] = fmt.MODE_HISTO
    meta["framerate_requested"] = framerate
    meta["duration_s"] = (meta.get("elapsed_us", 0) / 1e6
                          if meta.get("elapsed_us") else duration_s)

    fmt.save_frames(output_path, frames_arr, ts_arr, meta)
    return frames_arr, ts_arr, meta, output_path


# --------------------------------------------------------------------------
# OMV_PROTOCOL transport for histogram mode
# --------------------------------------------------------------------------
# On firmware v5.x boards (where legacy USBDBG was replaced with the new
# framed OMV_PROTOCOL), the stream channel can deliver raw frames at much
# higher rates than the REPL+base64 path (~70+ FPS measured for 320x320
# uint8 frames vs ~24 FPS via REPL). This implementation pulls frames over
# OMV_PROTOCOL's stream channel and saves to the same npz schema.

GENX320_CHIP_ID = 0xB0602003  # GENX320_ID_MP — used as stream-source key

# On-camera script for histo mode under OMV_PROTOCOL. We DON'T base64-emit
# anything from Python — the firmware writes frames into the stream FB via
# framebuffer_update_preview() automatically as long as snapshot() keeps
# running.
_OMV_HISTO_SCRIPT = r"""
import csi
cam = csi.CSI(cid=csi.GENX320)
cam.reset()
cam.pixformat(csi.GRAYSCALE)
cam.framesize((320, 320))
cam.framerate({framerate})
print('init done')
while True:
    cam.snapshot()
"""


def _wait_for_stdout(p: omv.OmvProtocol, marker: str,
                     timeout_s: float) -> str:
    """Poll the stdout channel until `marker` appears or timeout."""
    acc = ""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sz = p.channel_size(omv.CHAN_STDOUT)
        if isinstance(sz, int) and sz > 0:
            data = p.channel_read(omv.CHAN_STDOUT, 0, sz)
            acc += data.decode(errors="replace")
            if marker in acc:
                return acc
        else:
            time.sleep(0.02)
    return acc


def record_histo_omv(
    port: str,
    output_path: Optional[str] = None,
    duration_s: Optional[float] = None,
    framerate: int = 100,
    show_status: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict, str]:
    """Record histogram-mode frames via OMV_PROTOCOL (firmware v5.x).

    Same return shape as `record_histo()`. Auto-falls-back to the REPL path
    is handled at the CLI layer (see __main__.py).
    """
    if output_path is None:
        output_path = f"recording_{dt.datetime.now():%Y%m%d_%H%M%S}.npz"

    duration_ms = (
        int(duration_s * 1000) if duration_s is not None
        else 24 * 60 * 60 * 1000
    )

    print(f"[record] opening {port} via OMV_PROTOCOL "
          f"(mode: histo, framerate={framerate})")

    frames: List[np.ndarray] = []
    timestamps_us: List[int] = []
    n_size_polls = 0
    n_lock_fail = 0
    n_read_fail = 0
    chunks_received = 0
    chunks_malformed = 0  # framebuffer header sanity errors

    stop_requested = threading.Event()
    p = omv.OmvProtocol(port, timeout=2.0)
    p.__enter__()  # we manage entry/exit manually to allow SIGINT cleanup

    def _sigint_handler(_sig, _frame):
        if not stop_requested.is_set():
            stop_requested.set()
            sys.stdout.write("\n[record] Ctrl+C — stopping camera …\n")
            sys.stdout.flush()

    prev_sigint = signal.signal(signal.SIGINT, _sigint_handler)

    last_print = time.time()
    last_count = 0
    t_capture_start: Optional[float] = None
    success = False

    try:
        # 1. Handshake + capability negotiation (disable ACK+CRC for speed).
        p.sync()
        p.set_caps(crc_enabled=False, seq_enabled=True, ack_enabled=False)

        # 2. Kill main.py / any user script so we own the MicroPython runtime.
        p.stdin_stop()
        time.sleep(0.4)
        # Drain any banner/output left from previous runs.
        for _ in range(8):
            sz = p.channel_size(omv.CHAN_STDOUT)
            if isinstance(sz, int) and sz > 0:
                p.channel_read(omv.CHAN_STDOUT, 0, sz)
            else:
                break

        # 3. Configure the stream channel.
        p.channel_ioctl(omv.CHAN_STREAM, omv.IOCTL_STREAM_SOURCE,
                        arg=struct.pack("<I", GENX320_CHIP_ID))
        p.channel_ioctl(omv.CHAN_STREAM, omv.IOCTL_STREAM_RAW_CFG,
                        arg=struct.pack("<II", 320, 320))
        p.channel_ioctl(omv.CHAN_STREAM, omv.IOCTL_STREAM_RAW_CTRL,
                        arg=struct.pack("<I", 1))
        p.stream_enable(True)

        # 4. Start the snapshot loop on-device.
        script = _OMV_HISTO_SCRIPT.format(framerate=framerate)
        p.stdin_exec(script)
        banner = _wait_for_stdout(p, "init done", timeout_s=3.0)
        if "init done" not in banner:
            print(f"[record] WARNING: didn't see 'init done' marker; "
                  f"stdout so far: {banner!r}")

        # 5. Stream frames until duration / Ctrl+C.
        if duration_s is None:
            print("[record] no --duration set, recording until Ctrl+C")
        else:
            print(f"[record] capturing for {duration_s:.1f} s")

        capture_deadline = (
            time.time() + duration_s if duration_s is not None
            else float("inf")
        )
        t_capture_start = time.time()

        while not stop_requested.is_set() and time.time() < capture_deadline:
            try:
                sz = p.channel_size(omv.CHAN_STREAM)
            except Exception as e:
                print(f"\n[record] stream size err: {e}")
                break
            n_size_polls += 1

            if not isinstance(sz, int) or sz <= 0:
                time.sleep(0.001)
                continue

            try:
                if not p.channel_lock(omv.CHAN_STREAM):
                    n_lock_fail += 1
                    continue
            except Exception as e:
                print(f"\n[record] lock err: {e}")
                break

            try:
                payload = p.channel_read(omv.CHAN_STREAM, 0, sz)
                t_frame = time.time()
            except Exception as e:
                p.channel_unlock(omv.CHAN_STREAM)
                n_read_fail += 1
                continue

            p.channel_unlock(omv.CHAN_STREAM)
            chunks_received += 1

            # Frame layout: 32-byte aligned header + 320*320 image bytes.
            if len(payload) < 32 + 320 * 320:
                chunks_malformed += 1
                continue

            # Header fields we care about for round-trip diagnostics.
            w, h, pixfmt, comp_size, offset = struct.unpack(
                "<IIIII", payload[:20]
            )
            (fps_dev,) = struct.unpack("<f", payload[20:24])

            if w != 320 or h != 320:
                chunks_malformed += 1
                continue

            # data starts at the 32-byte cache-aligned boundary.
            frame = np.frombuffer(
                payload[32:32 + 320 * 320], dtype=np.uint8
            ).reshape(320, 320).copy()
            frames.append(frame)
            timestamps_us.append(
                int((t_frame - t_capture_start) * 1_000_000)
            )

            if show_status and (time.time() - last_print) > 0.5:
                now = time.time()
                rate = (len(frames) - last_count) / max(now - last_print, 1e-9)
                sys.stdout.write(
                    f"\r[record] {len(frames):>6} frames  "
                    f"({rate:>5.1f} fps, dev_fps={fps_dev:.1f})  "
                )
                sys.stdout.flush()
                last_print = now
                last_count = len(frames)

        success = True

    finally:
        if show_status:
            sys.stdout.write("\n")
            sys.stdout.flush()
        signal.signal(signal.SIGINT, prev_sigint)
        # Best-effort cleanup.
        try:
            p.stdin_stop()
        except Exception:
            pass
        try:
            p.stream_enable(False)
        except Exception:
            pass
        try:
            p.__exit__(None, None, None)
        except Exception:
            pass

    if frames:
        frames_arr = np.stack(frames, axis=0)
        ts_arr = np.array(timestamps_us, dtype=np.int64)
    else:
        frames_arr = np.zeros((0, 320, 320), dtype=np.uint8)
        ts_arr = np.zeros((0,), dtype=np.int64)

    elapsed_s = (
        timestamps_us[-1] / 1e6 if timestamps_us
        else (duration_s if duration_s else 0.0)
    )
    meta = {
        "mode": fmt.MODE_HISTO,
        "transport": "omv_protocol",
        "framerate_requested": framerate,
        "frames": len(frames),
        "decoded_events": len(frames),  # generic name shared with events path
        "chunks_received": chunks_received,
        "chunks_malformed": chunks_malformed,
        "size_polls": n_size_polls,
        "lock_failures": n_lock_fail,
        "read_failures": n_read_fail,
        "host_capture_started": dt.datetime.now().isoformat(),
        "host_port": port,
        "stopped_by_user": bool(stop_requested.is_set()),
        "duration_s": elapsed_s,
    }

    fmt.save_frames(output_path, frames_arr, ts_arr, meta)
    return frames_arr, ts_arr, meta, output_path


# Back-compat alias for the original API.
record = record_events


def print_events_summary(events: np.ndarray, meta: dict, output_path: str) -> None:
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


def print_histo_summary(
    frames: np.ndarray, timestamps_us: np.ndarray,
    meta: dict, output_path: str,
) -> None:
    n = frames.shape[0]
    print(f"[record] saved {n} frames → {output_path}")
    if n == 0:
        return
    span_s = (int(timestamps_us[-1]) - int(timestamps_us[0])) / 1e6
    fps = n / max(span_s, 1e-9)
    print(
        f"[record]   frame_shape={frames.shape[1:]}  dtype={frames.dtype}"
    )
    print(f"[record]   timespan ≈ {span_s:.3f} s, average ≈ {fps:.1f} FPS "
          f"(requested {meta.get('framerate_requested', '?')})")
    mb = frames.nbytes / (1024 * 1024)
    print(f"[record]   raw frame data: {mb:.1f} MB ({frames.nbytes} bytes)")


# Back-compat alias.
print_summary = print_events_summary
