# openmv-genx320-recorder

Plug-and-play recorder for the **OpenMV RT1062 + Prophesee GenX320** event
camera, over USB-CDC. No ROS, no Metavision SDK, no firmware build — just
`pip install` and `genx320 record`.

## What it does

- Auto-detects connected OpenMV boards (Linux / macOS / Windows)
- **Two recording modes**:
  - `events`: raw asynchronous events (`(N, 6)` uint16: `type, sec, ms, us, x, y`)
  - `histo`: on-chip event-histogram frames (`(F, 320, 320)` uint8 — looks like
    a normal grayscale video)
- Records until `Ctrl+C` (or a fixed `--duration`)
- Saves as a compressed `.npz` with a metadata dict
- Plays back recordings as a 320×320 video (auto-detects mode)
- Diagnoses dropped events / frames (USB integrity, sensor FIFO saturation,
  timestamp / inter-frame continuity)

## Requirements

- **Hardware**: an OpenMV RT1062 board with a Prophesee GenX320 camera module
- **Firmware**: OpenMV firmware that exposes the GenX320 event-mode APIs
  (`csi.GENX320_MODE_EVENT`, `csi.IOCTL_GENX320_SET_MODE`,
  `csi.IOCTL_GENX320_READ_EVENTS`). Install via OpenMV IDE →
  *Tools → Install latest development firmware* if your board doesn't have it.
- **Python ≥ 3.9** with `numpy` and `pyserial` (auto-installed); `matplotlib`
  for replay / analyze plots.

## Install

```bash
git clone https://github.com/yourname/openmv-genx320-recorder.git
cd openmv-genx320-recorder
pip install -e ".[viz]"     # [viz] adds matplotlib for replay/analyze
```

Or with `pipx` for an isolated CLI install:

```bash
pipx install ".[viz]"
```

Linux: add yourself to the `dialout` group so you don't need `sudo` for the
serial port:

```bash
sudo usermod -aG dialout $USER
# log out and back in
```

## Quick start

```bash
# 1. plug in the camera, then check it's detected
genx320 list

# 2. record raw events (Ctrl+C to stop)
genx320 record
# → recording_YYYYMMDD_HHMMSS.npz

# 2b. or record histogram frames (works on older firmware too)
genx320 record --mode histo --framerate 30

# 3. play it back (auto-detects events vs histo)
genx320 replay recording_*.npz

# 4. check whether anything was dropped
genx320 analyze recording_*.npz
```

## Commands

### `genx320 list`

Lists connected boards that look like OpenMV / MicroPython USB-CDC devices.

### `genx320 record [options]`

Records to a `.npz` file. Streams chunks to the host during capture, so RAM
use grows with recording length rather than being bounded by a fixed on-device
buffer.

| flag | default | meaning |
|---|---|---|
| `--mode {events,histo}` | `events` | recording mode (see below) |
| `--port` | auto-detect | serial device, e.g. `/dev/ttyACM0`, `COM3` |
| `--duration N` | until Ctrl+C | fixed capture length in seconds |
| `--output PATH`, `-o` | `recording_TIMESTAMP.npz` | output file |
| `--evt-res N` | 2048 | [events] per-ioctl event buffer (pow2 in [1024, 65536]) |
| `--framerate N` | 30 | [histo] target frame rate (actual cap ≈ 20–24 FPS) |
| `--no-status` | off | suppress the live status line |
| `--no-verify` | off | skip the GenX320 capability probe |

**Mode `events`** — raw asynchronous events read via
`IOCTL_GENX320_READ_EVENTS`. Requires newer OpenMV firmware (see
[Troubleshooting](#troubleshooting)). Output array: `events[N, 6]` uint16
with columns `[type, sec, ms, us, x, y]`.

**Mode `histo`** — on-chip event-histogram frames (`csi.snapshot()` in
GRAYSCALE pixformat). This is the same mode the OpenMV IDE shows for the
GenX320 by default. Works on any firmware that exposes `csi.CSI(cid=csi.GENX320)`.
Output arrays: `frames[F, 320, 320]` uint8, `frame_timestamps_us[F]` int64.

### `genx320 replay FILE [options]`

| flag | default | meaning |
|---|---|---|
| `--fps N` | — | playback frame rate; sets bin width to 1000/fps |
| `--bin-ms N` | 20 | time bin width in ms (ignored if `--fps` is given) |
| `--speed X` | 1.0 | playback speed multiplier (1.0 = real time) |
| `--save PATH` | — | render to mp4 / gif instead of a window |

### `genx320 analyze FILE [options]`

Reports pipeline integrity, on-device FIFO saturation, timestamp monotonicity,
event rate, suspected stall gaps, and hot-pixel concentration. Renders a
two-panel plot: events/ms over time, and cumulative event count. The final
**verdict** line summarises whether to trust the recording.

## File format

Both modes save to a `.npz` with a shared `metadata` dict (object array of one
Python dict). The data arrays differ by mode.

### Event-mode files

- `events`: `(N, 6)` `uint16`, columns are `[type, sec, ms, us, x, y]`
  - `type`: 1 = `PIX_ON_EVENT`, 0 = `PIX_OFF_EVENT`
  - `sec`, `ms`, `us`: split timestamp; total µs = `sec·1e6 + ms·1000 + us`
  - `x`, `y`: pixel coordinates (0..319 on GenX320)

```python
import numpy as np
d = np.load("recording.npz", allow_pickle=True)
events = d["events"]                  # (N, 6) uint16
meta = d["metadata"].item()           # dict, meta["mode"] == "events"
t_us = (events[:, 1].astype(np.int64) * 1_000_000
        + events[:, 2].astype(np.int64) * 1000
        + events[:, 3].astype(np.int64))
```

### Histogram-mode files

- `frames`: `(F, 320, 320)` `uint8` grayscale frames
- `frame_timestamps_us`: `(F,)` `int64` — µs since stream start

```python
import numpy as np
d = np.load("recording.npz", allow_pickle=True)
frames = d["frames"]                       # (F, 320, 320) uint8
ts_us = d["frame_timestamps_us"]           # (F,) int64
meta = d["metadata"].item()                # dict, meta["mode"] == "histo"
```

Or use the bundled loader, which dispatches on mode:

```python
from openmv_genx320_recorder.format import load_recording, detect_mode
mode = detect_mode("recording.npz")        # "events" or "histo"
data, meta = load_recording("recording.npz")
# data is `events` array (event mode) or `(frames, timestamps_us)` tuple (histo mode)
```

## Throughput

**Event mode**: USB-CDC on the RT1062 caps at roughly **12.8 MB/s** (TinyUSB
overhead). At 12 bytes per decoded event, that's about **1 MEv/s** sustained
per camera. For high event rates, increase `--evt-res` (e.g. 8192 or 16384) so
each ioctl returns a fuller batch and the sensor FIFO is drained faster.

**Histogram mode**: on the OpenMV firmware tested (build `cd4fb3ad60`),
streaming 320×320 grayscale frames via the raw REPL caps at **~20–24 FPS**
regardless of the requested `--framerate`. The bottleneck is not the sensor —
`csi.snapshot()` itself can do 300+ FPS — but the per-frame
`binascii.b2a_base64()` + `sys.stdout.write` over USB-CDC (≈ 40 ms/frame on
the device CPU). If you need higher histo-mode FPS (e.g. 100 FPS like the
OpenMV IDE achieves), use the binary USBDBG protocol directly (see
[openmv/openmv](https://github.com/openmv/openmv) source) — this recorder
prioritises a clean REPL protocol over peak histo throughput.

## Troubleshooting

**`error: GenX320 APIs not exposed`** — your firmware predates the event-mode
API. Update via OpenMV IDE → *Tools → Install latest development firmware*.

**`error: no OpenMV cameras found`** — check the cable and `dmesg | tail`
(Linux) for USB enumeration. Run `genx320 list` to confirm the board is seen.

**`sensor FIFO saturated X% of reads`** in `analyze` — re-record with
`--evt-res 4096` or `8192`. The default of 2048 is sized for moderate event
rates; busy scenes need more headroom.

**Hot pixels dominate the recording** — re-run with a brighter scene or
include some motion at the start so the on-startup `IOCTL_GENX320_CALIBRATE`
pass has more activity to threshold against.

## License

MIT — see [LICENSE](LICENSE).
