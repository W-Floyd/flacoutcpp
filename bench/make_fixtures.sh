#!/usr/bin/env bash
# Generate the audio fixtures used by check.sh and compare.sh.
#
# These are synthetic on purpose: they are reproducible from this script alone,
# so a reference recorded on one machine is meaningful on another. They are NOT
# representative of real music — see bench/README.md before drawing conclusions
# about compression ratio from them.
#
# Usage: bench/make_fixtures.sh [outdir]     (default: bench/fixtures)
set -euo pipefail

OUT=${1:-"$(cd "$(dirname "$0")" && pwd)/fixtures"}
mkdir -p "$OUT"

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }

# A tone-plus-noise mix. The noise floor keeps it from being trivially
# compressible, so the optimizer actually explores the search space.
EXPR="0.4*sin(2*PI*440*t)+0.2*sin(2*PI*1319*t)*sin(2*PI*3*t)+0.05*random(0)|0.4*sin(2*PI*443*t)+0.15*sin(2*PI*880*t)+0.05*random(1)"
SRC="aevalsrc=$EXPR:s=44100:d=20"
SRC96="aevalsrc=$EXPR:s=96000:d=20"

gen() { # outfile, extra ffmpeg args...
  local out=$1; shift
  ffmpeg -y -loglevel error -f lavfi -i "$SRC" "$@" -c:a flac "$OUT/$out"
}

gen96() { # outfile, extra ffmpeg args...
  local out=$1; shift
  ffmpeg -y -loglevel error -f lavfi -i "$SRC96" "$@" -c:a flac "$OUT/$out"
}

# Correctness fixtures (check.sh). Long enough to span several DP nodes and
# exercise more than one block size.
gen stereo_1s.flac  -t 1  -ac 2 -sample_fmt s16
gen stereo_4s.flac  -t 4  -ac 2 -sample_fmt s16
gen mono_2s.flac    -t 2  -ac 1 -sample_fmt s16
gen s24_2s.flac     -t 2  -ac 2 -sample_fmt s32 -bits_per_raw_sample 24
gen short.flac      -t 0.01 -ac 2 -sample_fmt s16   # < 1024 samples: short-stream path

# Micro fixtures (compare.sh default). ~0.25 s, so an exhaustive A/B round trip
# is seconds rather than minutes — short enough to keep in the edit/measure
# loop. Still 10 DP nodes, so the parallel phase is real work rather than one
# block on one thread. Use the longer fixtures above to confirm anything that
# looks like a win here, especially for scheduling changes: tail effects barely
# show up at this size.
gen micro.flac      -t 0.25 -ac 2 -sample_fmt s16
gen micro_s24.flac  -t 0.25 -ac 2 -sample_fmt s32 -bits_per_raw_sample 24

# 24-bit/96 kHz hi-res. Two things these exercise that nothing above does: a
# sample rate outside STREAMINFO's common set (so the frame header takes the
# extra-bits path, which frame_bits() has to price exactly), and 2.2x the
# samples per second of wall-clock, so a fixed-duration block spans fewer
# musical events and the DP sees a different block-size distribution.
gen96 hr24_2s.flac      -t 2    -ac 2 -sample_fmt s32 -bits_per_raw_sample 24
gen96 micro_hr24.flac   -t 0.25 -ac 2 -sample_fmt s32 -bits_per_raw_sample 24

# Full-length (3 min) synthetic tracks, one per content regime, for testing
# content-dependent behaviour (e.g. adaptive window selection) at realistic
# stream lengths. Still synthetic — the usual caveat about mispredicting real
# music applies; use them for robustness, not for compression conclusions.
gen3() { # outfile, source-expr
  local out=$1; shift
  ffmpeg -y -loglevel error -f lavfi -i "aevalsrc=$1:s=44100:d=180" \
         -ac 2 -sample_fmt s16 -c:a flac "$OUT/$out"
}
gen3 syn3m_mix.flac       "0.4*sin(2*PI*440*t)+0.2*sin(2*PI*1319*t)*sin(2*PI*3*t)+0.05*random(0)|0.4*sin(2*PI*443*t)+0.15*sin(2*PI*880*t)+0.05*random(1)"
gen3 syn3m_tonal.flac     "0.35*sin(2*PI*440*t)+0.25*sin(2*PI*659*t)+0.15*sin(2*PI*1319*t)+0.005*random(0)|0.35*sin(2*PI*442*t)+0.25*sin(2*PI*661*t)+0.15*sin(2*PI*880*t)+0.005*random(1)"
gen3 syn3m_noise.flac     "0.35*random(0)+0.08*sin(2*PI*440*t)|0.35*random(1)+0.08*sin(2*PI*443*t)"
gen3 syn3m_transient.flac "0.7*sin(2*PI*880*t)*exp(-25*mod(t\,0.4))+0.02*random(0)|0.7*sin(2*PI*662*t)*exp(-25*mod(t\,0.4))+0.02*random(1)"

# A fixture carrying a SEEKTABLE, which ffmpeg does not write. The encoder
# rewrites the block partition, so seek points copied through from the input
# name offsets that land inside our frames rather than at their headers --
# invisible to both `flac -t` and a bit-exactness diff, so check.sh validates
# these outputs with check_seektable.py instead.
#
# The targets are chosen to be awkward on purpose. metaflac snaps each to a
# frame start of *this* file (4096-aligned), which our variable 1024-grid
# partition does not share, so the rebuild has to re-snap every one. The pair
# at 90000/94000 lands inside a single larger output frame and must collapse to
# one point, and 900000 is past the end of a 176400-sample stream, so it has to
# become a trailing placeholder.
if command -v metaflac >/dev/null; then
  cp "$OUT/stereo_4s.flac" "$OUT/seek_4s.flac"
  metaflac --add-seekpoint=1000  --add-seekpoint=5000   --add-seekpoint=90000 \
           --add-seekpoint=94000 --add-seekpoint=176000 --add-seekpoint=900000 \
           "$OUT/seek_4s.flac"
else
  echo "warning: metaflac not found — skipping seek_4s.flac (check.sh will" \
       "skip its seek-table cases)" >&2
fi

ls -l "$OUT"
