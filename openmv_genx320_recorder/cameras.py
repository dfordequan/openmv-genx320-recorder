"""Camera discovery and low-level OpenMV REPL transport.

Detection works cross-platform via pyserial's list_ports (Linux /dev/ttyACM*,
macOS /dev/tty.usbmodem*, Windows COMx).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import serial
from serial.tools import list_ports


OPENMV_INDICATORS = ("OpenMV", "MicroPython")


@dataclass
class CameraInfo:
    port: str             # e.g. "/dev/ttyACM0" or "COM3"
    description: str      # human-readable
    serial_number: str    # USB serial, useful when multiple are connected

    def __str__(self) -> str:
        sn = f"  (sn={self.serial_number})" if self.serial_number else ""
        return f"{self.port}  {self.description}{sn}"


def find_cameras() -> List[CameraInfo]:
    """Return every connected port that looks like an OpenMV/MicroPython board.

    Doesn't verify GenX320 attachment — that requires a REPL probe which is
    slower. Use confirm_genx320() per-camera if precision is needed.
    """
    cams: List[CameraInfo] = []
    for p in list_ports.comports():
        fields = " ".join(
            str(x) for x in
            (p.description, p.manufacturer or "", p.product or "")
        )
        if any(ind in fields for ind in OPENMV_INDICATORS):
            cams.append(CameraInfo(
                port=p.device,
                description=p.description or "",
                serial_number=p.serial_number or "",
            ))
    return cams


# --------------------------------------------------------------------------
# Raw REPL transport
# --------------------------------------------------------------------------

def _read_until(ser: serial.Serial, marker: bytes, timeout_s: float) -> bytes:
    buf = bytearray()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        chunk = ser.read(65536)
        if chunk:
            buf += chunk
            if marker in buf:
                return bytes(buf)
        else:
            time.sleep(0.005)
    return bytes(buf)


def open_raw_repl(port: str, baudrate: int = 115200,
                  timeout: float = 0.1) -> serial.Serial:
    """Open the serial port, interrupt any running main.py, enter raw REPL."""
    ser = serial.Serial(port, baudrate, timeout=timeout)
    # Send Ctrl+C twice to interrupt anything currently executing.
    ser.write(b"\r\x03\x03")
    time.sleep(0.3)
    ser.reset_input_buffer()
    # Enter raw REPL (Ctrl+A).
    ser.write(b"\x01")
    _read_until(ser, b"raw REPL", 2.0)
    time.sleep(0.05)
    ser.reset_input_buffer()
    return ser


def close_raw_repl(ser: serial.Serial) -> None:
    """Send Ctrl+B to leave raw REPL, then close the port."""
    try:
        ser.write(b"\x02")
        time.sleep(0.1)
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass


def exec_raw(ser: serial.Serial, code: str, timeout_s: float = 15.0
             ) -> tuple[str, str]:
    """Execute `code` in raw REPL, wait for it to finish, return (stdout, stderr).

    Use only for short, bounded scripts. For long-running scripts where you want
    to read output incrementally, drive the REPL directly with start_exec() and
    a streaming reader.
    """
    ser.write(code.encode())
    ser.write(b"\x04")  # Ctrl+D = "go"
    raw = _read_until(ser, b"\x04>", timeout_s)
    body = raw[2:] if raw.startswith(b"OK") else raw
    parts = body.split(b"\x04")
    stdout = parts[0].decode(errors="replace")
    stderr = parts[1].decode(errors="replace") if len(parts) > 1 else ""
    return stdout, stderr


def start_exec(ser: serial.Serial, code: str) -> None:
    """Begin executing `code` in raw REPL without waiting for completion.

    The caller is responsible for draining stdout incrementally (e.g. via a
    background thread) and for sending Ctrl+C (0x03) to interrupt.
    """
    ser.write(code.encode())
    ser.write(b"\x04")


def send_ctrl_c(ser: serial.Serial) -> None:
    """Interrupt a script running in raw REPL."""
    try:
        ser.write(b"\x03")
    except Exception:
        pass


# --------------------------------------------------------------------------
# Capability check
# --------------------------------------------------------------------------

GENX320_PROBE = r"""
import csi
try:
    print('OK' if csi.GENX320 and csi.IOCTL_GENX320_SET_MODE and
          csi.IOCTL_GENX320_READ_EVENTS else 'NO')
except BaseException as e:
    print('NO:' + repr(e))
"""


def confirm_genx320(port: str, timeout_s: float = 5.0) -> Optional[str]:
    """Return None if the board exposes GenX320 event-mode APIs, else an error string."""
    try:
        ser = open_raw_repl(port)
    except Exception as e:
        return f"could not open {port}: {e}"
    try:
        out, err = exec_raw(ser, GENX320_PROBE, timeout_s=timeout_s)
        if "OK" in out:
            return None
        return f"GenX320 APIs not exposed: stdout={out!r} stderr={err!r}"
    finally:
        close_raw_repl(ser)
