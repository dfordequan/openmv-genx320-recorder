"""On-disk schema and I/O for recorded event streams."""

from __future__ import annotations

import datetime as dt
from typing import Any, Tuple

import numpy as np


SCHEMA_VERSION = 1

# Columns of the events array. Matches the layout returned by
# IOCTL_GENX320_READ_EVENTS in the OpenMV firmware:
#   [0] type — 1 = PIX_ON_EVENT, 0 = PIX_OFF_EVENT
#   [1] sec  — uint16 seconds since stream start
#   [2] ms   — uint16 milliseconds within the second (0..999)
#   [3] us   — uint16 microseconds within the millisecond (0..999)
#   [4] x    — uint16 pixel column (0..319 on GenX320)
#   [5] y    — uint16 pixel row    (0..319 on GenX320)
COLUMNS = ["type", "sec", "ms", "us", "x", "y"]


def save_recording(path: str, events: np.ndarray, metadata: dict) -> None:
    """Save an (N, 6) uint16 event array + metadata dict to a .npz file."""
    if events.dtype != np.uint16 or events.ndim != 2 or events.shape[1] != 6:
        raise ValueError(
            f"events must be (N, 6) uint16, got shape={events.shape} "
            f"dtype={events.dtype}"
        )
    meta = dict(metadata)
    meta.setdefault("schema_version", SCHEMA_VERSION)
    meta.setdefault("columns", COLUMNS)
    meta.setdefault("sensor", "GenX320")
    meta.setdefault("board", "OpenMV RT1062")
    meta.setdefault("host_saved_at", dt.datetime.now().isoformat())
    np.savez_compressed(
        path, events=events, metadata=np.array(meta, dtype=object)
    )


def load_recording(path: str) -> Tuple[np.ndarray, dict]:
    """Load a recording saved with save_recording()."""
    d = np.load(path, allow_pickle=True)
    events = d["events"]
    meta = d["metadata"].item() if "metadata" in d.files else {}
    return events, meta


def events_to_microseconds(events: np.ndarray) -> np.ndarray:
    """Combine the (sec, ms, us) columns into a single int64 µs timeline."""
    sec = events[:, 1].astype(np.int64)
    ms = events[:, 2].astype(np.int64)
    us = events[:, 3].astype(np.int64)
    return sec * 1_000_000 + ms * 1000 + us
