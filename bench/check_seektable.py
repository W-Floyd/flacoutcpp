#!/usr/bin/env python3
"""Validate a FLAC file's SEEKTABLE against the frames actually in the file.

This encoder rewrites the block partition, so a SEEKTABLE copied through from
the input names byte offsets that fall inside our frames rather than at their
headers. `flac -t` cannot see that -- it never seeks -- and neither can a
bit-exactness diff, so this is the check that covers it.

For every non-placeholder seek point it asserts, by parsing the frame header
the point aims at:

  * the offset lands on a frame sync code, not mid-frame;
  * that frame's coded number resolves to the point's target sample; and
  * the point's frame_samples field equals the frame's real block size.

It also asserts the format's ordering rules: real points strictly ascending by
sample number, placeholders only at the tail.

  usage: check_seektable.py <file.flac>
  exit:  0 all points valid | 1 a point is wrong | 2 unreadable
         77 no SEEKTABLE present (nothing to check, not a failure)
"""
import struct
import sys

PLACEHOLDER = 0xFFFFFFFFFFFFFFFF
POINT_BYTES = 18

# Frame-header block size codes -> samples. 6 and 7 defer to a field stored
# after the coded frame/sample number; 0 is reserved.
BLOCKSIZE_CODE = {1: 192, 6: -8, 7: -16}
for _c in range(2, 6):
    BLOCKSIZE_CODE[_c] = 576 << (_c - 2)
for _c in range(8, 16):
    BLOCKSIZE_CODE[_c] = 256 << (_c - 8)


def utf8_coded(buf, i):
    """Decode FLAC's UTF-8-style coded number. Returns (value, byte length)."""
    b = buf[i]
    if b < 0x80:
        return b, 1
    n = 0
    while b & (0x80 >> n):
        n += 1
    if n < 2 or n > 7:
        raise ValueError("bad coded number")
    v = b & (0x7F >> n)
    for k in range(1, n):
        if (buf[i + k] & 0xC0) != 0x80:
            raise ValueError("bad continuation byte")
        v = (v << 6) | (buf[i + k] & 0x3F)
    return v, n


def parse_metadata(d):
    """Return (seektable_payload, audio_offset). Payload is None if absent."""
    if d[:4] != b"fLaC":
        raise ValueError("not a FLAC file")
    p, table = 4, None
    while True:
        if p + 4 > len(d):
            raise ValueError("truncated metadata")
        last = d[p] & 0x80
        btype = d[p] & 0x7F
        ln = int.from_bytes(d[p + 1:p + 4], "big")
        if btype == 3 and table is None:
            table = d[p + 4:p + 4 + ln]
        p += 4 + ln
        if last:
            break
    return table, p


def frame_at(d, off):
    """Parse the frame header at absolute offset `off`.

    Returns (first_sample, block_size). Raises if it is not a frame header.
    """
    if off + 16 > len(d):
        raise ValueError("offset past end of file")
    if d[off] != 0xFF or (d[off + 1] & 0xFC) != 0xF8:
        raise ValueError("no frame sync at offset")
    variable = d[off + 1] & 0x01          # blocking strategy bit
    bs_code = d[off + 2] >> 4
    if bs_code == 0:
        raise ValueError("reserved block size code")

    number, n = utf8_coded(d, off + 4)
    tail = off + 4 + n                     # where the deferred fields start

    bs = BLOCKSIZE_CODE[bs_code]
    if bs == -8:
        bs = d[tail] + 1
    elif bs == -16:
        bs = int.from_bytes(d[tail:tail + 2], "big") + 1

    # A variable-blocksize stream codes the sample number directly; a fixed one
    # codes the frame number, which only becomes a sample number via the block
    # size -- constant across the stream, so this is exact for those files.
    return (number if variable else number * bs), bs


def main(path):
    try:
        d = open(path, "rb").read()
        table, audio = parse_metadata(d)
    except (OSError, ValueError) as e:
        print("  ERROR  %s: %s" % (path, e))
        return 2

    if table is None:
        return 77
    if len(table) % POINT_BYTES:
        print("  INVALID  %s: SEEKTABLE length %d is not a multiple of %d"
              % (path, len(table), POINT_BYTES))
        return 1

    bad = 0
    checked = 0
    seen_placeholder = False
    prev_sample = -1

    for i in range(len(table) // POINT_BYTES):
        sample, offset, samples = struct.unpack(
            ">QQH", table[i * POINT_BYTES:(i + 1) * POINT_BYTES])

        if sample == PLACEHOLDER:
            seen_placeholder = True
            continue
        if seen_placeholder:
            print("  INVALID  %s: point %d is real but follows a placeholder"
                  % (path, i))
            bad += 1
            continue
        if sample <= prev_sample:
            print("  INVALID  %s: point %d sample %d does not exceed %d"
                  % (path, i, sample, prev_sample))
            bad += 1
        prev_sample = sample

        try:
            got_sample, got_bs = frame_at(d, audio + offset)
        except (ValueError, IndexError, KeyError) as e:
            print("  INVALID  %s: point %d (sample %d, offset %d): %s"
                  % (path, i, sample, offset, e))
            bad += 1
            continue

        if got_sample != sample:
            print("  INVALID  %s: point %d claims sample %d but the frame at "
                  "offset %d starts at %d" % (path, i, sample, offset, got_sample))
            bad += 1
        elif samples != got_bs:
            print("  INVALID  %s: point %d claims %d frame_samples but the "
                  "frame holds %d" % (path, i, samples, got_bs))
            bad += 1
        else:
            checked += 1

    if bad:
        print("  INVALID  %s: %d of %d seek points are wrong"
              % (path, bad, bad + checked))
        return 1
    print("  seek ok   %s (%d points)" % (path.rsplit("/", 1)[-1], checked))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
