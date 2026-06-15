"""Visualize a recorded .npz event stream as a video.

Each frame is a 320×320 event histogram: ON events add to a per-pixel counter,
OFF events subtract, then we map the result to grayscale centered on 128.
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterator, Tuple

import numpy as np

from . import format as fmt


def bin_events_to_frames(
    events: np.ndarray,
    bin_us: int,
    width: int = 320,
    height: int = 320,
    brightness: int = 128,
    contrast: int = 32,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (t_center_us, frame_uint8) per time bin."""
    if events.shape[0] == 0:
        return
    t = fmt.events_to_microseconds(events)
    t0, t1 = int(t[0]), int(t[-1])
    n_bins = max(1, (t1 - t0 + bin_us - 1) // bin_us)
    bin_idx = ((t - t0) // bin_us).astype(np.int64)

    for b in range(n_bins):
        lo = int(np.searchsorted(bin_idx, b, side="left"))
        hi = int(np.searchsorted(bin_idx, b + 1, side="left"))
        center = t0 + b * bin_us + bin_us // 2
        if hi <= lo:
            yield center, np.full((height, width), brightness, dtype=np.uint8)
            continue
        types = events[lo:hi, 0]
        xs = np.clip(events[lo:hi, 4], 0, width - 1).astype(np.int64)
        ys = np.clip(events[lo:hi, 5], 0, height - 1).astype(np.int64)
        delta = np.where(types == 1, 1, -1).astype(np.int32)
        acc = np.zeros((height, width), dtype=np.int32)
        np.add.at(acc, (ys, xs), delta)
        frame = np.clip(brightness + acc * contrast, 0, 255).astype(np.uint8)
        yield center, frame


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="genx320 replay",
        description="Play back a recorded .npz event stream.",
    )
    ap.add_argument("file", help="path to recording_*.npz")
    ap.add_argument(
        "--fps", type=float, default=None,
        help="frame rate in Hz — sets the time-bin width to 1000/fps "
             "(default: 50). Overrides --bin-ms.",
    )
    ap.add_argument(
        "--bin-ms", type=float, default=None,
        help="time bin width in milliseconds (default: 20)",
    )
    ap.add_argument(
        "--speed", type=float, default=1.0,
        help="playback speed multiplier (default: 1.0 = real time)",
    )
    ap.add_argument(
        "--save", default=None,
        help="render to mp4 / gif instead of opening a window",
    )
    ap.add_argument(
        "--max-frames", type=int, default=None,
        help="cap frame count (debug)",
    )
    args = ap.parse_args(argv)

    try:
        events, meta = fmt.load_recording(args.file)
    except FileNotFoundError:
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 1

    print(f"loaded {events.shape[0]} events from {args.file}")
    if not events.size:
        print("(empty recording)")
        return 1

    if args.fps:
        bin_ms = 1000.0 / args.fps
    elif args.bin_ms:
        bin_ms = args.bin_ms
    else:
        bin_ms = 20.0
    bin_us = int(bin_ms * 1000)
    print(f"bin width: {bin_ms:.2f} ms  (≈ {1000.0 / bin_ms:.1f} FPS at speed=1.0)")

    frames = list(bin_events_to_frames(events, bin_us))
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        print("(no frames to play)", file=sys.stderr)
        return 1
    print(f"binned into {len(frames)} frames")

    interval_ms = max(1, int(bin_ms / args.speed))

    try:
        import matplotlib
    except ImportError:
        print(
            "error: matplotlib is required for replay. install with: "
            "pip install matplotlib  (or 'pip install \"openmv-genx320-recorder[viz]\"')",
            file=sys.stderr,
        )
        return 2
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        frames[0][1], cmap="gray", vmin=0, vmax=255, interpolation="nearest"
    )
    txt = ax.set_title("")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(i):
        t_us, frame = frames[i]
        im.set_array(frame)
        txt.set_text(f"t = {t_us / 1e6:.3f} s  ({i + 1}/{len(frames)})")
        return im, txt

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=interval_ms, blit=False, repeat=True,
    )

    if args.save:
        print(f"rendering to {args.save} …")
        fps = max(1, int(1000 / interval_ms))
        if args.save.endswith(".gif"):
            anim.save(args.save, writer="pillow", fps=fps)
        else:
            anim.save(args.save, writer="ffmpeg", fps=fps)
        print("done")
    else:
        plt.show()
    return 0
