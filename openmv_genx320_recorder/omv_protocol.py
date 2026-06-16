"""Client for the OpenMV firmware-v5 "OMV_PROTOCOL".

A framed binary protocol that replaced the legacy single-byte USBDBG in OpenMV
firmware v5.x (PR-equivalent commit 515fd76 on 2025-09-28). Spec lives in the
firmware tree at protocol/omv_protocol.h.

Wire layout:
    HEADER (10 bytes):
        SYNC[2]    little-endian uint16 = 0xD5AA
        SEQ[1]     monotonic per packet (mod 256)
        CHAN[1]    channel ID (0=transport, 1=stdin, 2=stdout, 3=stream, ...)
        FLAGS[1]   ACK / NAK / RTX / ACK_REQ / FRAGMENT / EVENT
        OPCODE[1]  protocol/system/channel opcode
        LEN[2]     little-endian uint16 = payload length (NOT including CRC32)
        CRC16[2]   little-endian uint16 = CRC16(first 8 header bytes)
    PAYLOAD (only if LEN > 0):
        DATA[LEN]
        CRC32[4]   little-endian uint32 = CRC32(payload bytes)

CRC16: poly=0xF94F, init=0xFFFF, MSB-first, no reflect, no xorout
CRC32: poly=0xFA567D89, init=0xFFFFFFFF, MSB-first, no reflect, no xorout

The protocol is intentionally simple: send a packet → camera ACKs or
responds. ACK_REQ requests an explicit ACK packet; events are unsolicited
packets with the EVENT flag set.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import serial


# --------------------------------------------------------------------------
# CRC tables
# --------------------------------------------------------------------------

_CRC16_POLY = 0xF94F
_CRC16_INIT = 0xFFFF
_CRC32_POLY = 0xFA567D89
_CRC32_INIT = 0xFFFFFFFF


def _make_crc16_table() -> list[int]:
    t = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            if c & 0x8000:
                c = ((c << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                c = (c << 1) & 0xFFFF
        t.append(c)
    return t


def _make_crc32_table() -> list[int]:
    t = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            if c & 0x80000000:
                c = ((c << 1) ^ _CRC32_POLY) & 0xFFFFFFFF
            else:
                c = (c << 1) & 0xFFFFFFFF
        t.append(c)
    return t


_CRC16_TABLE = _make_crc16_table()
_CRC32_TABLE = _make_crc32_table()


def crc16(data: bytes, init: int = _CRC16_INIT) -> int:
    c = init
    for b in data:
        c = ((c << 8) ^ _CRC16_TABLE[((c >> 8) ^ b) & 0xFF]) & 0xFFFF
    return c


def crc32(data: bytes, init: int = _CRC32_INIT) -> int:
    c = init
    for b in data:
        c = ((c << 8) ^ _CRC32_TABLE[((c >> 24) ^ b) & 0xFF]) & 0xFFFFFFFF
    return c


# --------------------------------------------------------------------------
# Protocol constants (mirrors omv_protocol.h)
# --------------------------------------------------------------------------

SYNC_WORD = 0xD5AA
HEADER_SIZE = 10
MAGIC_BAUDRATE = 921600
MAX_PAYLOAD = 4096 - HEADER_SIZE - 4

# Flags
FLAG_ACK = 1 << 0
FLAG_NAK = 1 << 1
FLAG_RTX = 1 << 2
FLAG_ACK_REQ = 1 << 3
FLAG_FRAGMENT = 1 << 4
FLAG_EVENT = 1 << 5

# Opcodes
OP_PROTO_SYNC = 0x00
OP_PROTO_GET_CAPS = 0x01
OP_PROTO_SET_CAPS = 0x02
OP_PROTO_STATS = 0x03
OP_PROTO_VERSION = 0x04

OP_SYS_RESET = 0x10
OP_SYS_BOOT = 0x11
OP_SYS_INFO = 0x12
OP_SYS_EVENT = 0x13
OP_SYS_MEMORY = 0x14

OP_CHANNEL_LIST = 0x20
OP_CHANNEL_POLL = 0x21
OP_CHANNEL_LOCK = 0x22
OP_CHANNEL_UNLOCK = 0x23
OP_CHANNEL_SHAPE = 0x24
OP_CHANNEL_SIZE = 0x25
OP_CHANNEL_READ = 0x26
OP_CHANNEL_WRITE = 0x27
OP_CHANNEL_IOCTL = 0x28
OP_CHANNEL_EVENT = 0x29

# Channel IDs (reserved)
CHAN_TRANSPORT = 0
CHAN_STDIN = 1
CHAN_STDOUT = 2
CHAN_STREAM = 3
CHAN_PROFILE = 4

# Channel-IOCTL commands
IOCTL_STDIN_STOP = 0x01
IOCTL_STDIN_EXEC = 0x02
IOCTL_STDIN_RESET = 0x03

IOCTL_STREAM_CTRL = 0x00
IOCTL_STREAM_RAW_CTRL = 0x01
IOCTL_STREAM_RAW_CFG = 0x02
IOCTL_STREAM_SOURCE = 0x03

# Status codes
STATUS_SUCCESS = 0
STATUS_FAILED = 1
STATUS_INVALID = 2
STATUS_TIMEOUT = 3
STATUS_BUSY = 4
STATUS_CHECKSUM = 5
STATUS_SEQUENCE = 6
STATUS_OVERFLOW = 7
STATUS_FRAGMENT = 8
STATUS_UNKNOWN = 9


@dataclass
class Packet:
    """A decoded OMV_PROTOCOL packet."""
    sync: int
    sequence: int
    channel: int
    flags: int
    opcode: int
    length: int
    crc16_header: int
    payload: bytes  # data only — payload CRC32 is validated before discard

    @property
    def is_ack(self) -> bool:
        return bool(self.flags & FLAG_ACK)

    @property
    def is_nak(self) -> bool:
        return bool(self.flags & FLAG_NAK)

    @property
    def is_event(self) -> bool:
        return bool(self.flags & FLAG_EVENT)


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------

def build_packet(
    sequence: int, channel: int, opcode: int, payload: bytes = b"",
    flags: int = 0, crc_enabled: bool = True,
) -> bytes:
    """Serialise a packet for transmission."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    header_no_crc = struct.pack(
        "<HBBBBH",
        SYNC_WORD,
        sequence & 0xFF,
        channel & 0xFF,
        flags & 0xFF,
        opcode & 0xFF,
        len(payload),
    )
    header_crc = crc16(header_no_crc) if crc_enabled else 0
    header = header_no_crc + struct.pack("<H", header_crc)
    if not payload:
        return header
    payload_crc = crc32(payload) if crc_enabled else 0
    return header + payload + struct.pack("<I", payload_crc)


def parse_header(buf10: bytes, crc_enabled: bool = True) -> Packet:
    """Decode a 10-byte header (no payload yet)."""
    if len(buf10) != HEADER_SIZE:
        raise ValueError(f"header must be {HEADER_SIZE} bytes, got {len(buf10)}")
    sync, seq, chan, flags, opcode, length, crc = struct.unpack(
        "<HBBBBHH", buf10
    )
    if sync != SYNC_WORD:
        raise ValueError(f"bad sync: 0x{sync:04X} != 0x{SYNC_WORD:04X}")
    if crc_enabled:
        computed = crc16(buf10[:8])
        if crc != computed:
            raise ValueError(
                f"header CRC mismatch: sent=0x{crc:04X} computed=0x{computed:04X}"
            )
    return Packet(sync, seq, chan, flags, opcode, length, crc, b"")


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class OmvProtocol:
    """Synchronous host-side client for OMV_PROTOCOL.

    Usage:
        with OmvProtocol("/dev/ttyACM0") as p:
            p.sync()                          # PROTO_SYNC handshake
            chans = p.channel_list()
            p.stdin_exec("import csi; ...")
            ...
    """

    def __init__(self, port: str, baudrate: int = MAGIC_BAUDRATE,
                 timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._seq = 0
        # Negotiated caps — start with device defaults (all on); flip after SET_CAPS.
        self._crc_enabled = True
        self._ack_enabled = True
        self._seq_enabled = True

    def __enter__(self) -> "OmvProtocol":
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        # Drain anything left in the kernel buffer from a previous owner.
        self._ser.reset_input_buffer()
        return self

    def __exit__(self, *exc) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    # ----- low-level send / recv ------------------------------------------

    def _next_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) & 0xFF
        return s

    def _send(self, channel: int, opcode: int, payload: bytes = b"",
              flags: int = 0) -> int:
        assert self._ser is not None
        seq = self._next_seq()
        pkt = build_packet(seq, channel, opcode, payload, flags,
                           crc_enabled=self._crc_enabled)
        self._ser.write(pkt)
        return seq

    def _read_exact(self, n: int, deadline: float) -> bytes:
        assert self._ser is not None
        buf = bytearray()
        while len(buf) < n:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out reading {n} bytes (got {len(buf)})"
                )
            self._ser.timeout = min(remaining, 0.5)
            chunk = self._ser.read(n - len(buf))
            if chunk:
                buf += chunk
        return bytes(buf)

    def _hunt_sync(self, deadline: float) -> None:
        """Discard bytes until we hit the SYNC word."""
        assert self._ser is not None
        sync_lo = SYNC_WORD & 0xFF
        sync_hi = (SYNC_WORD >> 8) & 0xFF
        last = -1
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("timed out hunting for SYNC")
            self._ser.timeout = min(remaining, 0.2)
            b = self._ser.read(1)
            if not b:
                continue
            if last == sync_lo and b[0] == sync_hi:
                return
            last = b[0]

    def _recv(self, timeout: float = 2.0) -> Packet:
        """Receive one packet (header + payload + CRC32 if any)."""
        assert self._ser is not None
        deadline = time.time() + timeout
        self._hunt_sync(deadline)
        # We've consumed the 2 SYNC bytes; read the remaining 8 of the header.
        rest = self._read_exact(HEADER_SIZE - 2, deadline)
        header = struct.pack("<H", SYNC_WORD) + rest
        pkt = parse_header(header, crc_enabled=self._crc_enabled)
        if pkt.length == 0:
            return pkt
        payload = self._read_exact(pkt.length, deadline)
        # Read trailing CRC32 (4 bytes are always present when length > 0, but
        # they're zero when crc_enabled was off on the sender).
        crc_bytes = self._read_exact(4, deadline)
        if self._crc_enabled:
            (payload_crc,) = struct.unpack("<I", crc_bytes)
            computed = crc32(payload)
            if payload_crc != computed:
                raise ValueError(
                    f"payload CRC mismatch: sent=0x{payload_crc:08X} "
                    f"computed=0x{computed:08X}"
                )
        pkt.payload = payload
        return pkt

    def _ack(self, opcode: int, sequence: int, channel: int = 0) -> None:
        """Send an ACK packet that the device is waiting for.

        The device's send_packet() path sets ACK_REQ on every data packet (any
        flags without ACK/NAK/EVENT). If we don't ACK, it retransmits and
        eventually resets its sequence counter. ACKs themselves are in the
        NO_ACK class so this won't recurse.
        """
        assert self._ser is not None
        pkt = build_packet(sequence, channel, opcode, payload=b"",
                           flags=FLAG_ACK, crc_enabled=self._crc_enabled)
        self._ser.write(pkt)

    def _request(self, channel: int, opcode: int, payload: bytes = b"",
                 flags: int = 0, timeout: float = 2.0) -> Packet:
        """Send a request, drop unsolicited events, auto-ACK data responses,
        and reassemble fragmented responses transparently.

        Large responses (size > max_payload) are split by the firmware into
        multiple packets, all but the last carrying FLAG_FRAGMENT. We ACK each
        fragment (the firmware's send_packet path waits for an ACK per
        fragment) and concatenate their payloads, returning a single Packet
        whose .payload contains the full reassembled data.
        """
        seq_sent = self._send(channel, opcode, payload, flags)
        if getattr(self, "_debug", False):
            print(f"  → tx: opcode=0x{opcode:02X} chan={channel} seq={seq_sent} "
                  f"len={len(payload)}")

        accumulated = bytearray()
        first_pkt: Optional[Packet] = None
        while True:
            pkt = self._recv(timeout=timeout)
            if getattr(self, "_debug", False):
                print(f"  ← rx: opcode=0x{pkt.opcode:02X} flags=0x{pkt.flags:02X} "
                      f"seq={pkt.sequence} chan={pkt.channel} len={pkt.length}")
            if pkt.is_event:
                # SYS_EVENT carrying SOFT_REBOOT (0x02) means the device
                # called omv_protocol_reset() — ctx.sequence is now 0, so we
                # must reset ours too or we'll drift by however many requests
                # we've sent.
                if (pkt.opcode == OP_SYS_EVENT
                        and pkt.payload[:2] == b"\x02\x00"):
                    self._seq = 0
                continue

            is_data = not pkt.is_ack and not pkt.is_nak
            if self._ack_enabled and is_data:
                self._ack(pkt.opcode, pkt.sequence, pkt.channel)

            if first_pkt is None:
                first_pkt = pkt

            accumulated.extend(pkt.payload)

            if pkt.flags & FLAG_FRAGMENT:
                # More fragments coming; continue reading.
                continue

            # Final fragment (or single-packet response). Each fragment that
            # arrived above incremented the device's ctx.sequence, but our
            # _seq only incremented once when we sent the request. Re-sync:
            # device's ctx.sequence is now `pkt.sequence + 1`, so our next
            # request should use that.
            self._seq = (pkt.sequence + 1) & 0xFF

            first_pkt.payload = bytes(accumulated)
            first_pkt.length = len(first_pkt.payload)
            return first_pkt

    # ----- high-level helpers --------------------------------------------

    def sync(self, retries: int = 3, timeout: float = 1.0) -> Packet:
        """Send PROTO_SYNC and return the response packet.

        Resets our sequence counter to 0 after a successful SYNC, because the
        device's SYNC handler does `ctx.sequence = 0` AFTER sending its ACK,
        so it expects our next request to have seq=0 too.
        """
        last_err: Optional[Exception] = None
        for _ in range(retries):
            try:
                # Use seq=0 explicitly for the SYNC packet so we have a known
                # starting point regardless of how many requests came before.
                self._seq = 0
                self._send(0, OP_PROTO_SYNC, b"")
                while True:
                    pkt = self._recv(timeout=timeout)
                    if pkt.is_event:
                        continue
                    # SYNC's response is the ACK at seq=0; device then resets
                    # its counter to 0 again — match that.
                    self._seq = 0
                    return pkt
            except (TimeoutError, ValueError) as e:
                last_err = e
                if self._ser is not None:
                    self._ser.reset_input_buffer()
        raise RuntimeError(f"PROTO_SYNC failed after {retries} retries: {last_err}")

    def get_caps(self) -> Tuple[dict, Packet]:
        pkt = self._request(0, OP_PROTO_GET_CAPS)
        # omv_protocol_caps_t: bitfields(32) + uint16 max_payload + 10 bytes reserved
        if len(pkt.payload) < 16:
            return {}, pkt
        bits, max_payload = struct.unpack("<IH", pkt.payload[:6])
        caps = {
            "crc_enabled": bool(bits & 1),
            "seq_enabled": bool(bits & 2),
            "ack_enabled": bool(bits & 4),
            "event_enabled": bool(bits & 8),
            "max_payload": max_payload,
        }
        return caps, pkt

    def set_caps(self, crc_enabled: bool = True, seq_enabled: bool = True,
                 ack_enabled: bool = True, event_enabled: bool = True,
                 max_payload: int = 4082) -> Packet:
        """Negotiate protocol capabilities. Disabling seq/ack simplifies the
        host-side state machine but loses the protocol's reliability features —
        only safe over USB where the kernel CDC layer already provides
        ordering and CRC.
        """
        bits = ((1 if crc_enabled else 0)
                | (2 if seq_enabled else 0)
                | (4 if ack_enabled else 0)
                | (8 if event_enabled else 0))
        # omv_protocol_caps_t: u32 bits, u16 max_payload, u8[10] reserved
        caps_payload = struct.pack("<IH10s", bits, max_payload, b"\x00" * 10)
        pkt = self._request(0, OP_PROTO_SET_CAPS, payload=caps_payload)
        # The device ACKs THEN updates caps. So this response was processed
        # under the old caps. Flip ours now to match the device's new state.
        if not pkt.is_nak:
            self._crc_enabled = crc_enabled
            self._seq_enabled = seq_enabled
            self._ack_enabled = ack_enabled
        return pkt

    def get_version(self) -> Tuple[dict, Packet]:
        pkt = self._request(0, OP_PROTO_VERSION)
        if len(pkt.payload) < 16:
            return {}, pkt
        proto = tuple(pkt.payload[0:3])
        boot = tuple(pkt.payload[3:6])
        fw = tuple(pkt.payload[6:9])
        return {"protocol": proto, "bootloader": boot, "firmware": fw}, pkt

    def channel_list(self) -> list[dict]:
        """Return a list of {id, flags, name} dicts."""
        pkt = self._request(0, OP_CHANNEL_LIST)
        # entries: struct {uint8 id; uint8 flags; char name[14]} → 16 bytes each
        entries = []
        for i in range(0, len(pkt.payload), 16):
            chunk = pkt.payload[i:i + 16]
            if len(chunk) < 16:
                break
            cid, cflags = chunk[0], chunk[1]
            name = chunk[2:].split(b"\0", 1)[0].decode(errors="replace")
            entries.append({"id": cid, "flags": cflags, "name": name})
        return entries

    def sys_info(self) -> Tuple[dict, Packet]:
        pkt = self._request(0, OP_SYS_INFO)
        if len(pkt.payload) < 76:
            return {}, pkt
        f = struct.unpack("<IIIIIIIIIIIIIIIIIII", pkt.payload[:76])
        return {
            "cpu_id": f[0],
            "dev_id": f[1:4],
            "usb_id": f[4],
            "chip_id": f[5:8],
            "hw_caps": f[10:12],
            "flash_size_kb": f[12],
            "ram_size_kb": f[13],
            "frame_buffer_size_kb": f[14],
            "stream_buffer_size_kb": f[15],
        }, pkt

    # ----- channel ops ----------------------------------------------------
    #
    # Channel ID lives in the packet HEADER (.channel field), not the payload.
    # Payload formats:
    #   SIZE / SHAPE / LOCK / UNLOCK : no payload
    #   READ / WRITE                 : omv_protocol_channel_io_t = {u32 offset, u32 length, bytes}
    #   IOCTL                        : omv_protocol_channel_ioctl_t = {u32 request, bytes}

    def channel_size(self, channel: int) -> int:
        pkt = self._request(channel, OP_CHANNEL_SIZE)
        if pkt.is_nak or len(pkt.payload) < 4:
            return -1
        return struct.unpack("<I", pkt.payload[:4])[0]

    def channel_shape(self, channel: int) -> list[int]:
        pkt = self._request(channel, OP_CHANNEL_SHAPE)
        if pkt.is_nak:
            return []
        n = len(pkt.payload) // 4
        if n == 0:
            return []
        return list(struct.unpack(f"<{n}I", pkt.payload[:n * 4]))

    def channel_lock(self, channel: int) -> bool:
        pkt = self._request(channel, OP_CHANNEL_LOCK)
        return not pkt.is_nak

    def channel_unlock(self, channel: int) -> bool:
        pkt = self._request(channel, OP_CHANNEL_UNLOCK)
        return not pkt.is_nak

    def channel_read(self, channel: int, offset: int, length: int) -> bytes:
        body = struct.pack("<II", offset, length)
        pkt = self._request(channel, OP_CHANNEL_READ, payload=body, timeout=4.0)
        if pkt.is_nak:
            raise RuntimeError(
                f"CHANNEL_READ NAK: status={pkt.payload[:2].hex()}"
            )
        return pkt.payload

    def channel_write(self, channel: int, offset: int, data: bytes) -> Packet:
        body = struct.pack("<II", offset, len(data)) + data
        return self._request(channel, OP_CHANNEL_WRITE, payload=body, timeout=4.0)

    def channel_ioctl(self, channel: int, cmd: int, arg: bytes = b"") -> Packet:
        body = struct.pack("<I", cmd) + arg
        return self._request(channel, OP_CHANNEL_IOCTL, payload=body)

    def stdin_exec(self, code: str) -> Packet:
        """Upload `code` to stdin channel and trigger exec."""
        self.channel_write(CHAN_STDIN, 0, code.encode("utf-8"))
        return self.channel_ioctl(CHAN_STDIN, IOCTL_STDIN_EXEC)

    def stdin_stop(self) -> Packet:
        return self.channel_ioctl(CHAN_STDIN, IOCTL_STDIN_STOP)

    def stream_enable(self, enable: bool = True) -> Packet:
        return self.channel_ioctl(
            CHAN_STREAM, IOCTL_STREAM_CTRL,
            arg=struct.pack("<I", 1 if enable else 0),
        )
