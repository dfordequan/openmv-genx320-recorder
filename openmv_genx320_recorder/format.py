"""On-disk schema and I/O for recordings (event-mode + histogram-mode)."""

from __future__ import annotations

import datetime as dt
from typing import Tuple, Union

import numpy as np


SCHEMA_VERSION = 2

MODE_EVENTS = "events"
MODE_HISTO = "histo"

# Columns of the events array. Matches the layout returned by
# IOCTL_GENX320_READ_EVENTS in the OpenMV firmware:
#   [0] type — 1 = PIX_ON_EVENT, 0 = PIX_OFF_EVENT
#   [1] sec  — uint16 seconds since stream start
#   [2] ms   — uint16 milliseconds within the second (0..999)
#   [3] us   — uint16 microseconds within the millisecond (0..999)
#   [4] x    — uint16 pixel column (0..319 on GenX320)
#   [5] y    — uint16 pixel row    (0..319 on GenX320)
EVENT_COLUMNS = ["type", "sec", "ms", "us", "x", "y"]


def _common_meta(metadata: dict, mode: str) -> dict:
    meta = dict(metadata)
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta.setdefault("mode", mode)
    meta.setdefault("sensor", "GenX320")
    meta.setdefault("board", "OpenMV RT1062")
    meta.setdefault("host_saved_at", dt.datetime.now().isoformat())
    return meta


def save_events(path: str, events: np.ndarray, metadata: dict) -> None:
    """Save an (N, 6) uint16 event array to a .npz file."""
    if events.dtype != np.uint16 or events.ndim != 2 or events.shape[1] != 6:
        raise ValueError(
            f"events must be (N, 6) uint16, got shape={events.shape} "
            f"dtype={events.dtype}"
        )
    meta = _common_meta(metadata, MODE_EVENTS)
    meta.setdefault("columns", EVENT_COLUMNS)
    np.savez_compressed(
        path, events=events, metadata=np.array(meta, dtype=object)
    )


def save_frames(
    path: str,
    frames: np.ndarray,
    timestamps_us: np.ndarray,
    metadata: dict,
) -> None:
    """Save an (F, H, W) uint8 frame array + (F,) int64 timestamp array."""
    if frames.dtype != np.uint8 or frames.ndim != 3:
        raise ValueError(
            f"frames must be (F, H, W) uint8, got shape={frames.shape} "
            f"dtype={frames.dtype}"
        )
    if timestamps_us.ndim != 1 or timestamps_us.shape[0] != frames.shape[0]:
        raise ValueError(
            f"timestamps_us must be (F,) matching frames[0]; got "
            f"shape={timestamps_us.shape}, frames[0]={frames.shape[0]}"
        )
    meta = _common_meta(metadata, MODE_HISTO)
    meta.setdefault("frame_shape", list(frames.shape[1:]))
    np.savez_compressed(
        path,
        frames=frames,
        frame_timestamps_us=timestamps_us.astype(np.int64),
        metadata=np.array(meta, dtype=object),
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def detect_mode(path: str) -> str:
    """Return MODE_EVENTS or MODE_HISTO based on the file contents/metadata."""
    d = np.load(path, allow_pickle=True)
    if "metadata" in d.files:
        meta = d["metadata"].item()
        m = meta.get("mode")
        if m in (MODE_EVENTS, MODE_HISTO):
            return m
    if "frames" in d.files:
        return MODE_HISTO
    if "events" in d.files:
        return MODE_EVENTS
    raise ValueError(f"{path}: cannot determine recording mode")


def load_events(path: str) -> Tuple[np.ndarray, dict]:
    """Load an event-mode recording."""
    d = np.load(path, allow_pickle=True)
    if "events" not in d.files:
        raise ValueError(f"{path}: not an event-mode recording")
    events = d["events"]
    meta = d["metadata"].item() if "metadata" in d.files else {}
    return events, meta


def load_frames(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load a histogram-mode recording → (frames, timestamps_us, metadata)."""
    d = np.load(path, allow_pickle=True)
    if "frames" not in d.files:
        raise ValueError(f"{path}: not a histogram-mode recording")
    frames = d["frames"]
    ts = d["frame_timestamps_us"] if "frame_timestamps_us" in d.files else None
    if ts is None:
        ts = np.arange(frames.shape[0], dtype=np.int64)
    meta = d["metadata"].item() if "metadata" in d.files else {}
    return frames, ts, meta


def load_recording(
    path: str,
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], dict]:
    """Mode-agnostic loader.

    Returns (events, meta) for event-mode files,
    or ((frames, timestamps_us), meta) for histogram-mode files.
    """
    mode = detect_mode(path)
    if mode == MODE_EVENTS:
        return load_events(path)
    frames, ts, meta = load_frames(path)
    return (frames, ts), meta


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def events_to_microseconds(events: np.ndarray) -> np.ndarray:
    """Combine the (sec, ms, us) columns into a single int64 µs timeline."""
    sec = events[:, 1].astype(np.int64)
    ms = events[:, 2].astype(np.int64)
    us = events[:, 3].astype(np.int64)
    return sec * 1_000_000 + ms * 1000 + us


# Back-compat alias for callers still using the v1 name. New code should use
# save_events() directly.
save_recording = save_events
COLUMNS = EVENT_COLUMNS
