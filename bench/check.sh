#!/usr/bin/env bash
# Bit-exactness harness for optimizations that must not change the bitstream.
#
#   bench/check.sh record <build-dir>    # snapshot outputs as the reference
#   bench/check.sh verify <build-dir>    # diff a new build against that snapshot
#
# `record` against a known-good build, change the encoder, then `verify`. Any
# byte difference is a failure: this gates changes meant to be purely a speedup,
# so `cmp` is the whole test. Outputs are also run through `flac -t`, so a change
# that corrupts both reference and candidate identically still gets caught.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODE=${1:-}
BUILD=${2:-}
FIX=${FIXTURES:-"$HERE/fixtures"}

[ -n "$MODE" ] && [ -n "$BUILD" ] || { echo "usage: $0 {record|verify} <build-dir>" >&2; exit 2; }
BIN="$BUILD/flacoutcpp"
[ -x "$BIN" ] || { echo "error: no executable at $BIN" >&2; exit 2; }
[ -d "$FIX" ] || { echo "error: no fixtures at $FIX — run bench/make_fixtures.sh" >&2; exit 2; }

case "$MODE" in
  record) OUT="$HERE/reference" ;;
  verify) OUT="$HERE/.current" ;;
  *) echo "usage: $0 {record|verify} <build-dir>" >&2; exit 2 ;;
esac
rm -rf "$OUT"; mkdir -p "$OUT"

# Each case pins a distinct path through the encoder: exhaustive vs heuristic,
# stereo vs mono, 16- vs 24-bit, the <1024-sample short-stream path, and an
# explicit -w list (which selects a different window set than either default).
run() { local name=$1; shift
  "$BIN" -q "$@" "$OUT/$name.flac" || { echo "FAIL: encoder returned nonzero for $name" >&2; exit 1; }
}
# All search-path cases pass -R: they pin the *search*, and letting the
# input's own frames compete would splice ffmpeg-encoded frames into the
# reference. The ru_* cases below cover the reuse path itself.
run ex_stereo -R -e "$FIX/stereo_1s.flac"
run ex_mono   -R -e "$FIX/mono_2s.flac"
run ex_24     -R -e "$FIX/s24_2s.flac"
run ex_short  -R -e "$FIX/short.flac"
run ex_win    -R -e -w hann,tukey020,punchouttukey2_067 "$FIX/stereo_1s.flac"
run he_stereo -R    "$FIX/stereo_4s.flac"
run he_mono   -R    "$FIX/mono_2s.flac"
run he_24     -R    "$FIX/s24_2s.flac"
run he_short  -R    "$FIX/short.flac"
# Ranked exact search (-e -c N; plain -c N before the flags composed). Its
# output is a deliberate compression/speed trade, so it is not comparable to
# bare -e — but it must still be stable and decode losslessly. A reference for
# these can only be recorded from the commit that introduced ranked search or
# later; older builds reject the flag and `record` will fail.
run rk_stereo -R -e -c 8 "$FIX/stereo_1s.flac"
run rk_mono   -R -e -c 4 "$FIX/mono_2s.flac"
run rk_24     -R -e -c 8 "$FIX/s24_2s.flac"
run rk_short  -R -e -c 8 "$FIX/short.flac"
run rk_win    -R -e -c 2 -w hann,tukey020 "$FIX/stereo_1s.flac"
# Analytic precision ladder (-L N). Off by default, so the cases above already
# pin the full-ladder path; these pin the model that picks the rungs. Retuning
# it is expected to move them — re-record after confirming the size delta went
# the intended way, as with rk_*.
#
# -L 1, not the recommended -L 2, on purpose: at -L 2 the model agrees with the
# full ladder on every synthetic fixture here (byte-identical output), so a -L 2
# case would pass even against a broken model. -L 1 forces it to commit to one
# rung and does diverge — 1 B, 6 B and 155 B respectively.
run ld_stereo -R -L 1 "$FIX/stereo_1s.flac"
run ld_24     -R -L 1 "$FIX/s24_2s.flac"
run ld_short  -R -L 1 "$FIX/short.flac"
# Effort dial: pins the level -> (candidates, rungs) table itself. -E 3 is
# (-c 24 -L 1) and stereo_1s is one of the fixtures where -L 1 diverges from
# the full ladder, so this case moves if either half of the mapping changes.
run ef_stereo -R -E 3 "$FIX/stereo_1s.flac"
# The documented exact-DP recipe: -E under -e, where the level's -a is dropped
# and only its -c/-L survive. Pins that interaction, not just the mapping.
run ef_ex     -R -e -E 0 "$FIX/stereo_1s.flac"
# Exact-DP effort levels (10-12). These pin two things at once: that the level
# selects exact DP itself, and that at a finite -c it uses the shortlist rather
# than all 26 windows (the gate in Optimizer::Optimizer).
run ef_x10    -R -E 10 "$FIX/stereo_1s.flac"
run ef_x12    -R -E 12 "$FIX/mono_2s.flac"
# Runtime-loaded window (-w custom:<file>): pins the knot parser and the
# interpolation onto both table and non-table block sizes. The knot file is
# committed next to this script, so these are reproducible anywhere.
run cw_stereo -R    -w "custom:$HERE/windows/example_taper.txt",hann "$FIX/stereo_4s.flac"
run cw_short  -R -e -w "custom:$HERE/windows/example_taper.txt"      "$FIX/short.flac"
# Block-size ladder (-b). bs_ceil raises the ceiling past the built-in 16384,
# so it pins the on-the-fly window path for a size with no precomputed table.
# bs_floor lowers the floor and therefore changes the DP's node spacing, which
# is what the reachability rule constrains. bs_max reaches 65520, the largest
# attainable size — 65535 is odd, so no valid grid lands on it, and a ladder
# whose smallest entry does not divide the rest strands the final node (that
# bug shipped a 99-byte file once; the parser rejects it now).
run bs_ceil  -R -e -b 1024,2048,4096,8192,16384,32768 "$FIX/stereo_1s.flac"
run bs_floor -R -e -b 256,512,1024,2048,4096          "$FIX/mono_2s.flac"
run bs_max   -R    -b 5040,10080,20160,65520          "$FIX/stereo_4s.flac"
# Frame reuse (the default) — heuristic splice path and exact-DP reuse-edge
# path, both over the ffmpeg-encoded fixtures so input frames actually
# compete.
run ru_he           "$FIX/stereo_4s.flac"
run ru_ex -e -c 8   "$FIX/stereo_1s.flac"
run ru_24 -e -c 8   "$FIX/s24_2s.flac"
# SEEKTABLE rebuild. The input's seek points name the input's partition, and
# this encoder rewrites it, so copying them through yields offsets that land
# inside our frames. Neither `flac -t` (which never seeks) nor `cmp` can see
# that, hence the separate validation pass below. Three partitions of the same
# audio, so the rebuild has to re-snap the points three different ways.
if [ -f "$FIX/seek_4s.flac" ]; then
  run sk_he                       "$FIX/seek_4s.flac"
  run sk_ex    -e -c 8            "$FIX/seek_4s.flac"
  run sk_bs -R -b 1024,2048,4096  "$FIX/seek_4s.flac"
fi

fail=0
for f in "$OUT"/*.flac; do
  flac -t -s "$f" 2>/dev/null || { echo "  INVALID  $(basename "$f") — does not decode"; fail=1; }
done

# --- Seek tables: do the points name real frames? ----
# Structural check, over every output that carries a table (most do not).
if command -v python3 >/dev/null 2>&1; then
  tables=0
  for f in "$OUT"/*.flac; do
    python3 "$HERE/check_seektable.py" "$f"
    case $? in
      0)  tables=$((tables + 1)) ;;
      77) ;;                       # no SEEKTABLE here, nothing to validate
      1)  tables=$((tables + 1)); fail=1 ;;   # present but wrong: still covered
      *)  fail=1 ;;
    esac
  done
  # A fixture that lost its table would make this pass by testing nothing.
  if [ -f "$FIX/seek_4s.flac" ] && [ "$tables" -eq 0 ]; then
    echo "  INVALID  no output carried a SEEKTABLE — seek coverage is vacuous"
    fail=1
  fi
else
  echo "  SKIP      python3 not found — seek tables unchecked"
fi

# Behavioural check: drive libFLAC's own seek through the rebuilt table and
# confirm it lands where the same seek into the fixture does. A table can parse
# cleanly and still misdirect, and this is what a player would actually hit.
if [ -f "$FIX/seek_4s.flac" ]; then
  for c in sk_he sk_ex sk_bs; do
    [ -f "$OUT/$c.flac" ] || continue
    for range in 0:4096 92160:96256 172032:176400; do
      from=${range%%:*}; to=${range##*:}
      cmp -s <(flac -d -s --skip="$from" --until="$to" -c "$FIX/seek_4s.flac" 2>/dev/null) \
             <(flac -d -s --skip="$from" --until="$to" -c "$OUT/$c.flac"      2>/dev/null) \
        || { echo "  INVALID  $c.flac — seek to $from..$to decodes differently than the input"
             fail=1; }
    done
  done
fi

if [ "$MODE" = record ]; then
  [ $fail -eq 0 ] || { echo "refusing to record a reference that does not decode"; exit 1; }
  echo "recorded $(ls "$OUT" | wc -l | tr -d ' ') reference outputs in $OUT"
  exit 0
fi

REF="$HERE/reference"
[ -d "$REF" ] || { echo "error: no reference — run '$0 record <build-dir>' first" >&2; exit 2; }
for f in "$REF"/*.flac; do
  n=$(basename "$f")
  if cmp -s "$f" "$OUT/$n"; then
    echo "  ok        $n"
  else
    echo "  MISMATCH  $n  ($(wc -c <"$f" | tr -d ' ') -> $(wc -c <"$OUT/$n" | tr -d ' ') bytes)"
    fail=1
  fi
done

if [ $fail -eq 0 ]; then echo "ALL BIT-EXACT"; else echo "FAILURES — output changed"; fi
exit $fail
