"""Post-hoc loss diagnosis for a .npz event recording.

Reports on USB transfer integrity, on-device FIFO saturation, timestamp
monotonicity, event rate over time, and hot-pixel concentration. Renders an
event-rate plot so visual inspection can confirm or refute the heuristics.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

from . import format as fmt


def _detect_quiet_gaps(
    t_us: np.ndarray, gap_us: int, neighbor_rate_min: float
) -> List[Tuple[int, int, int, float]]:
    """Find time gaps that look like USB/sensor stalls.

    A real stall has high event rate right up to the gap boundary on BOTH
    sides; scene-driven quiet has gradual ramps. We require >= neighbor_rate_min
    ev/s in a ±5 ms shoulder window on each side.

    Returns [(gap_start_us, gap_end_us, gap_width_us, edge_rate), ...].
    """
    if t_us.size < 2:
        return []
    diffs = np.diff(t_us)
    big = np.where(diffs >= gap_us)[0]
    out = []
    SHOULDER_US = 5_000
    for i in big:
        gap_start = int(t_us[i])
        gap_end = int(t_us[i + 1])
        width = gap_end - gap_start
        before_lo = int(np.searchsorted(t_us, gap_start - SHOULDER_US, side="left"))
        after_hi = int(np.searchsorted(t_us, gap_end + SHOULDER_US, side="right"))
        before_n = (i + 1) - before_lo
        after_n = after_hi - (i + 1)
        before_rate = before_n / (SHOULDER_US / 1e6)
        after_rate = after_n / (SHOULDER_US / 1e6)
        edge_rate = min(before_rate, after_rate)
        if edge_rate >= neighbor_rate_min:
            out.append((gap_start, gap_end, width, edge_rate))
    return out


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="genx320 analyze",
        description="Diagnose whether a recording dropped events.",
    )
    ap.add_argument("file")
    ap.add_argument("--gap-ms", type=float, default=20.0,
                    help="report gaps wider than this (default 20 ms)")
    ap.add_argument("--gap-neighbor-rate", type=float, default=200_000.0,
                    help="only flag gaps with ±5 ms shoulder rate >= this "
                         "ev/s (default 200000)")
    ap.add_argument("--save", default=None,
                    help="save the event-rate plot to a PNG/PDF instead of "
                         "opening a window")
    ap.add_argument("--no-plot", action="store_true",
                    help="text-only output, no plot")
    args = ap.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 1

    try:
        mode = fmt.detect_mode(args.file)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if mode == fmt.MODE_HISTO:
        return _analyze_histo(args)

    events, meta = fmt.load_events(args.file)
    n = events.shape[0]

    print(f"=== file: {args.file} (event mode) ===")
    print(f"events: {n}")
    print(f"metadata: {meta}")
    print()

    if n == 0:
        print("(empty recording)")
        return 0

    # --- 1. Pipeline integrity ---------------------------------------------
    print("--- 1. pipeline integrity ---")
    dev_total = meta.get("events")
    dec_total = meta.get("decoded_events")
    chunks_rx = meta.get("chunks_received")
    chunks_bad = meta.get("chunks_malformed")
    if dev_total is not None and dec_total is not None:
        delta = dev_total - dec_total
        status = "OK" if delta == 0 else f"MISMATCH ({delta} events unaccounted)"
        print(f"  device reported : {dev_total}")
        print(f"  host decoded    : {dec_total}  → {status}")
    if chunks_rx is not None:
        print(f"  chunks received : {chunks_rx}")
        print(f"  chunks malformed: {chunks_bad}  "
              f"({'OK' if chunks_bad == 0 else 'WARNING'})")
    neg = meta.get("neg_returns", 0)
    if neg:
        print(f"  WARNING: {neg} ioctl calls returned a negative error code")
    print()

    # --- 2. Saturation hint ------------------------------------------------
    print("--- 2. on-device saturation ---")
    sat = meta.get("saturated_iters")
    iters = meta.get("iters")
    evt_res = meta.get("evt_res")
    max_n = meta.get("max_n_per_call")
    if sat is not None and iters:
        pct = 100 * sat / iters
        verdict = ("OK — sensor FIFO appears drained" if pct < 5 else
                   "SUSPICIOUS — frequent buffer-cap returns" if pct < 30 else
                   "LIKELY DROPS — buffer cap was hit often")
        print(f"  EVT_RES                  : {evt_res}")
        print(f"  ioctl iterations         : {iters}")
        print(f"  iters returning EVT_RES  : {sat} ({pct:.1f}%) → {verdict}")
        print(f"  max events per call seen : {max_n}")
        if pct >= 5:
            print("  hint: re-record with a larger --evt-res (e.g. 4096 or 8192)")
    print()

    # --- 3. Timestamp monotonicity ----------------------------------------
    print("--- 3. timestamp monotonicity ---")
    t_us = fmt.events_to_microseconds(events)
    diffs = np.diff(t_us)
    n_decreasing = int((diffs < 0).sum())
    max_backstep = int(-diffs.min()) if n_decreasing else 0
    sec_max = int(events[:, 1].max())
    ms_max = int(events[:, 2].max())
    us_max = int(events[:, 3].max())
    print(f"  range: sec∈[0,{sec_max}]  ms∈[0,{ms_max}]  us∈[0,{us_max}]")
    if n_decreasing == 0:
        print(f"  monotonic non-decreasing: OK ✓")
    else:
        pct = 100 * n_decreasing / max(1, len(diffs))
        print(f"  {n_decreasing} backward steps ({pct:.2f}% of pairs), "
              f"max back-jump {max_backstep} µs")
        if sec_max >= 65535:
            print("  → likely uint16 wrap on sec column (recording > 18 h)")
        elif max_backstep < 100:
            print("  → small jitter only; events within the same ioctl batch can "
                  "be reordered. Not a loss indicator.")
        else:
            print("  → large back-jumps. Sort events by timestamp before "
                  "downstream use.")
    print()

    # --- 4. Event rate ----------------------------------------------------
    print("--- 4. event rate over time ---")
    span_us = int(t_us[-1] - t_us[0])
    avg_rate = n / max(span_us / 1e6, 1e-9)
    bin_us = 1000
    nbins = max(1, span_us // bin_us + 1)
    bin_idx = ((t_us - t_us[0]) // bin_us).astype(np.int64)
    rate_per_ms = np.bincount(bin_idx, minlength=nbins)
    peak = int(rate_per_ms.max())
    p99 = int(np.percentile(rate_per_ms, 99))
    p50 = int(np.percentile(rate_per_ms, 50))
    print(f"  timespan          : {span_us / 1e6:.3f} s")
    print(f"  average rate      : {avg_rate:.0f} ev/s")
    print(f"  per-ms median/p99 : {p50} / {p99}")
    print(f"  per-ms peak       : {peak}")

    gaps = _detect_quiet_gaps(t_us, int(args.gap_ms * 1000),
                              args.gap_neighbor_rate)
    if gaps:
        print(f"  candidate gaps (> {args.gap_ms} ms with edge rate "
              f">= {args.gap_neighbor_rate:.0f} ev/s on both sides):")
        for gs, ge, w, r in gaps[:10]:
            print(f"    t={gs/1e6:.3f}s..{ge/1e6:.3f}s "
                  f"(gap {w/1000:.1f} ms, edge rate {r:.0f} ev/s)")
        if len(gaps) > 10:
            print(f"    … {len(gaps) - 10} more")
        print(f"  (informational — interpret using verdict below)")
    else:
        print(f"  no candidate gaps found ✓")
    print()

    # --- 5. Hot pixels ----------------------------------------------------
    print("--- 5. hot pixels ---")
    flat = events[:, 5].astype(np.int64) * 320 + events[:, 4].astype(np.int64)
    counts = np.bincount(flat, minlength=320 * 320)
    top = np.argsort(counts)[::-1][:5]
    pct_in_top = 100 * counts[top].sum() / n
    print(f"  top 5 pixels hold {pct_in_top:.1f}% of all events")
    for px in top:
        y, x = divmod(int(px), 320)
        print(f"    ({x:3d}, {y:3d}): {int(counts[px])} events "
              f"({100*counts[px]/n:.1f}%)")
    if pct_in_top > 30:
        print("  hint: try a stronger CALIBRATE pass (iterations=1000, sigma=0.3)")
    print()

    # --- verdict ----------------------------------------------------------
    print("--- verdict ---")
    issues = []
    if dev_total is not None and dec_total is not None and dev_total != dec_total:
        issues.append(f"USB transfer mismatch ({dev_total - dec_total} events)")
    if chunks_bad:
        issues.append(f"{chunks_bad} malformed chunks")
    if neg:
        issues.append(f"{neg} ioctl errors")
    if sat and iters and (sat / iters) >= 0.3:
        issues.append(f"sensor FIFO saturated {100*sat/iters:.0f}% of reads "
                      f"— increase --evt-res")
    if not issues:
        print("  ✓ pipeline integrity OK — no evidence of dropped events")
        if gaps:
            print(f"  (note: section 4 flagged {len(gaps)} quiet gap(s) — "
                  "most likely scene-driven, not stalls. Check the cumulative "
                  "plot: real drops are flat steps between steep climbs.)")
    else:
        print("  ✗ issues detected:")
        for s in issues:
            print(f"    - {s}")
    print()

    if args.no_plot:
        return 0

    try:
        import matplotlib
    except ImportError:
        print("(matplotlib not installed; install with 'pip install matplotlib' "
              "for plots)", file=sys.stderr)
        return 0
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t_ms_axis = (np.arange(nbins) * bin_us + t_us[0]) / 1e6
    axes[0].plot(t_ms_axis, rate_per_ms, lw=0.5)
    axes[0].set_ylabel("events / ms")
    axes[0].set_title(f"{args.file}: event rate ({avg_rate:.0f} ev/s avg)")
    for gs, ge, _, _ in gaps:
        axes[0].axvspan(gs / 1e6, ge / 1e6, color="red", alpha=0.3)
    axes[1].plot(t_us / 1e6, np.arange(1, n + 1), lw=0.5)
    axes[1].set_ylabel("cumulative events")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=100)
        print(f"plot saved to {args.save}")
    else:
        plt.show()
    return 0


def _analyze_histo(args) -> int:
    frames, ts_us, meta = fmt.load_frames(args.file)
    n = frames.shape[0]

    print(f"=== file: {args.file} (histo mode) ===")
    print(f"frames: {n}")
    print(f"metadata: {meta}")
    print()

    if n == 0:
        print("(empty recording)")
        return 0

    # --- pipeline integrity (frames as units) -----------------------------
    print("--- 1. pipeline integrity ---")
    dev_total = meta.get("frames")
    dec_total = meta.get("decoded_events")  # generic 'decoded count' name
    chunks_rx = meta.get("chunks_received")
    chunks_bad = meta.get("chunks_malformed")
    if dev_total is not None and dec_total is not None:
        delta = dev_total - dec_total
        status = "OK" if delta == 0 else f"MISMATCH ({delta} frames unaccounted)"
        print(f"  device reported : {dev_total}")
        print(f"  host decoded    : {dec_total}  → {status}")
    if chunks_rx is not None:
        print(f"  chunks received : {chunks_rx}")
        print(f"  chunks malformed: {chunks_bad}  "
              f"({'OK' if chunks_bad == 0 else 'WARNING'})")
    print()

    # --- frame rate analysis ---------------------------------------------
    print("--- 2. frame rate ---")
    span_us = int(ts_us[-1] - ts_us[0])
    span_s = span_us / 1e6
    fps = (n - 1) / max(span_s, 1e-9)
    print(f"  timespan        : {span_s:.3f} s")
    print(f"  frames captured : {n}")
    print(f"  achieved FPS    : {fps:.1f}")
    print(f"  requested FPS   : {meta.get('framerate_requested', '?')}")
    if n >= 2:
        dts = np.diff(ts_us)
        median_dt_us = int(np.median(dts))
        max_dt_us = int(dts.max())
        min_dt_us = int(dts.min())
        print(f"  inter-frame Δt  : median={median_dt_us} µs, "
              f"min={min_dt_us} µs, max={max_dt_us} µs")
        # Flag dropped frames: any Δt > 2× the median is suspicious.
        gaps = dts > (2 * median_dt_us)
        n_gaps = int(gaps.sum())
        if n_gaps:
            print(f"  WARNING: {n_gaps} inter-frame gaps > 2× median "
                  "(possible dropped frame slots, USB stall, or sensor stall)")
    print()

    # --- per-frame brightness sanity -------------------------------------
    print("--- 3. frame content ---")
    means = frames.reshape(n, -1).mean(axis=1)
    print(f"  per-frame mean: median={float(np.median(means)):.1f}  "
          f"min={float(means.min()):.1f}  max={float(means.max()):.1f}")
    blank = int((means < 1.0).sum())
    sat = int((means > 254.0).sum())
    if blank:
        print(f"  WARNING: {blank} frames are nearly all-black")
    if sat:
        print(f"  WARNING: {sat} frames are nearly all-white")
    print()

    # --- verdict ---------------------------------------------------------
    print("--- verdict ---")
    issues = []
    if dev_total is not None and dec_total is not None and dev_total != dec_total:
        issues.append(f"frame transfer mismatch ({dev_total - dec_total} frames)")
    if chunks_bad:
        issues.append(f"{chunks_bad} malformed chunks")
    if not issues:
        print("  ✓ pipeline integrity OK — all frames accounted for")
    else:
        print("  ✗ issues detected:")
        for s in issues:
            print(f"    - {s}")
    print()

    if args.no_plot or n < 2:
        return 0

    try:
        import matplotlib
    except ImportError:
        print("(matplotlib not installed; install with 'pip install matplotlib' "
              "for plots)", file=sys.stderr)
        return 0
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t_s = ts_us[1:] / 1e6
    axes[0].plot(t_s, 1e6 / np.maximum(np.diff(ts_us), 1), lw=0.6)
    axes[0].set_ylabel("instantaneous FPS")
    axes[0].set_title(f"{args.file}: frame timing ({fps:.1f} FPS avg)")
    axes[1].plot(ts_us / 1e6, means, lw=0.6)
    axes[1].set_ylabel("mean pixel value")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=100)
        print(f"plot saved to {args.save}")
    else:
        plt.show()
    return 0
