/*
 * Fast OMV_PROTOCOL fragment reassembly.
 *
 * Reads a chain of FRAGMENT-flagged response packets from a serial fd and
 * concatenates their payloads into out_buf, stopping at the first
 * non-FRAGMENT non-event packet. The host's per-frame round-trip cost drops
 * from ~12-15 ms (pure Python: byte-by-byte SYNC hunt + per-fragment
 * struct.unpack + accumulated.extend) to <1 ms — enough to lift histo
 * recording from ~52 FPS to the ~100 FPS the sensor produces.
 *
 * Build:
 *   cc -O2 -shared -fPIC -o _omv_decoder.so _omv_decoder.c
 *
 * Assumes crc_enabled=False has been negotiated via SET_CAPS — header CRC16
 * and payload CRC32 are skipped, not validated. The kernel CDC layer already
 * provides byte-level integrity on USB; the host-side CRC was duplicate work.
 */

#define _DEFAULT_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/select.h>
#include <time.h>

#define INBUF_SIZE   (1 << 16)   /* 64 KB read-side ring */
#define SYNC_LO      0xAA
#define SYNC_HI      0xD5
#define HEADER_SZ    10

#define FLAG_ACK      (1 << 0)
#define FLAG_NAK      (1 << 1)
#define FLAG_FRAGMENT (1 << 4)
#define FLAG_EVENT    (1 << 5)

/* Error codes returned to Python (negative). */
#define ERR_READ     -1
#define ERR_TIMEOUT  -2
#define ERR_OVERFLOW -3

typedef struct {
    int fd;
    uint8_t buf[INBUF_SIZE];
    size_t pos;
    size_t len;
    struct timespec deadline;
} dec_t;

static int64_t ns_remaining(const struct timespec *d) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return ((int64_t)(d->tv_sec - now.tv_sec)) * 1000000000LL
           + ((int64_t)(d->tv_nsec - now.tv_nsec));
}

/* Make at least `need` bytes available at &buf[pos]. Returns 0 on success,
 * negative on timeout/read error. */
static int ensure(dec_t *s, size_t need)
{
    if (s->len - s->pos >= need) return 0;

    /* Compact. */
    if (s->pos > 0) {
        size_t left = s->len - s->pos;
        memmove(s->buf, s->buf + s->pos, left);
        s->pos = 0;
        s->len = left;
    }

    while (s->len < need) {
        int64_t ns_left = ns_remaining(&s->deadline);
        if (ns_left <= 0) return ERR_TIMEOUT;

        struct timeval tv;
        if (ns_left > 100000000LL) {            /* cap each select at 100 ms */
            tv.tv_sec = 0;
            tv.tv_usec = 100000;
        } else {
            tv.tv_sec = ns_left / 1000000000LL;
            tv.tv_usec = (ns_left % 1000000000LL) / 1000LL;
            if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1;
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(s->fd, &rfds);
        int sel = select(s->fd + 1, &rfds, NULL, NULL, &tv);
        if (sel < 0) {
            if (errno == EINTR) continue;
            return ERR_READ;
        }
        if (sel == 0) continue;                /* will re-check deadline */

        ssize_t n = read(s->fd, s->buf + s->len, INBUF_SIZE - s->len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return ERR_READ;
        }
        if (n == 0) continue;
        s->len += (size_t)n;
    }
    return 0;
}

/*
 * Read a fragmented response into out_buf.
 *
 *   fd          POSIX file descriptor (open serial port)
 *   out_buf     destination buffer for concatenated payloads
 *   out_cap     out_buf capacity in bytes
 *   timeout_ms  total deadline for the whole read
 *
 * Output args (may be NULL):
 *   last_opcode, last_seq, last_chan, last_flags  from the final (non-FRAGMENT) packet
 *
 * Returns total payload bytes written on success; negative error code
 * otherwise (see ERR_* above).
 */
ssize_t omv_read_fragments(int fd,
                           uint8_t *out_buf, size_t out_cap,
                           uint32_t timeout_ms,
                           uint8_t *last_opcode, uint8_t *last_seq,
                           uint8_t *last_chan, uint8_t *last_flags)
{
    dec_t s;
    memset(&s, 0, sizeof(s));
    s.fd = fd;
    clock_gettime(CLOCK_MONOTONIC, &s.deadline);
    s.deadline.tv_sec += timeout_ms / 1000;
    s.deadline.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (s.deadline.tv_nsec >= 1000000000L) {
        s.deadline.tv_sec += 1;
        s.deadline.tv_nsec -= 1000000000L;
    }

    size_t out_pos = 0;

    for (;;) {
        /* SYNC hunt — bulk-scan the buffer. */
        int rc = ensure(&s, 2);
        if (rc < 0) return rc;

        int sync_found = 0;
        for (;;) {
            while (s.pos + 1 < s.len) {
                if (s.buf[s.pos] == SYNC_LO && s.buf[s.pos + 1] == SYNC_HI) {
                    sync_found = 1;
                    break;
                }
                s.pos++;
            }
            if (sync_found) break;
            rc = ensure(&s, 2);
            if (rc < 0) return rc;
        }

        /* Full header. */
        rc = ensure(&s, HEADER_SZ);
        if (rc < 0) return rc;

        const uint8_t *h = s.buf + s.pos;
        uint8_t seq    = h[2];
        uint8_t chan   = h[3];
        uint8_t flags  = h[4];
        uint8_t opcode = h[5];
        uint16_t length = (uint16_t)h[6] | ((uint16_t)h[7] << 8);
        /* h[8..9] = header CRC16 — we negotiated crc_enabled=false; skipped. */

        s.pos += HEADER_SZ;

        /* Payload + trailing CRC32 (4 bytes when length > 0). */
        size_t need = (size_t)length + (length > 0 ? 4U : 0U);
        rc = ensure(&s, need);
        if (rc < 0) return rc;

        /* Events: discard payload, keep hunting. */
        if (flags & FLAG_EVENT) {
            s.pos += need;
            continue;
        }

        /* Append to output (drop trailing CRC32). */
        if (out_pos + length > out_cap) return ERR_OVERFLOW;
        memcpy(out_buf + out_pos, s.buf + s.pos, length);
        out_pos += length;
        s.pos += need;

        if (flags & FLAG_FRAGMENT) {
            continue;
        }

        if (last_opcode) *last_opcode = opcode;
        if (last_seq)    *last_seq    = seq;
        if (last_chan)   *last_chan   = chan;
        if (last_flags)  *last_flags  = flags;
        return (ssize_t)out_pos;
    }
}
