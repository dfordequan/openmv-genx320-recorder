"""Build the optional C fragment decoder.

The recorder works without this — there's a pure-Python fallback for the
fragment reassembly loop — but the C path lifts histo-mode recording from
~52 FPS to ~80 FPS (or higher with smaller frames) at 320×320 grayscale.

Run after install:
    python -m openmv_genx320_recorder._build_decoder
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "_omv_decoder.c")
    out = os.path.join(here, "_omv_decoder.so")

    if not os.path.exists(src):
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    cc = os.environ.get("CC", "cc")
    cmd = [
        cc, "-O2", "-Wall", "-Wextra",
        "-shared", "-fPIC",
        "-o", out, src,
    ]
    print(" ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print(f"error: '{cc}' not found. set CC=<your compiler> or install gcc/clang.",
              file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"error: compile failed ({e})", file=sys.stderr)
        return 3

    print(f"built {out}")

    # Verify it loads.
    try:
        from . import omv_protocol as op
        # Reload to pick up the freshly-built so.
        import importlib
        importlib.reload(op)
        if op.has_native_decoder():
            print("native decoder loaded ✓")
        else:
            print("warning: built .so but couldn't load it via ctypes",
                  file=sys.stderr)
            return 4
    except Exception as e:
        print(f"warning: post-build load check failed: {e}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
