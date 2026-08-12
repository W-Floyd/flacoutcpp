# Pure-GPU FLAC encoding: what the transformation actually is

Scope: raw PCM goes into device memory, an encoded FLAC byte stream comes out,
and the host does no per-candidate, per-subframe or per-frame work in between.
**Goal is GPU-residency, not optimality** — a pure-GPU encoder that compresses
slightly worse than `-e` is the target; one that needs a host round trip per
subframe is not, however well it compresses.

Read `GPU_PLAN.md` first for what `-G` already is and where it broke. This
document is about the other 40% of the pipeline.

## The one invariant that matters, and the constraint that dissolves

Everything in the encoder upstream of *"here are the integer LPC coefficients"*
is a **heuristic**. The window, the autocorrelation, Levinson-Durbin, the
coefficient quantization — none of it has to be reproducible, or even
numerically good, for the output to be a valid lossless FLAC. It only has to
produce `int32_t q_coeffs[32]` and a shift.

Everything downstream is **exact integer arithmetic that the decoder mirrors**:

```
residual[i] = s[i] - (int32_t)((sum_j qc[j] * s[i-1-j]) >> shift)
```

Get that right and the file decodes bit-exactly, whatever nonsense chose `qc`.

This is the load-bearing observation for a pure-GPU port, because it kills the
constraint that otherwise ends the project on the author's hardware:

- `apply_window`, `autocorrelation`, `compute_lpc_all_orders` and
  `quantize_lpc_coeffs` are all `double` on the CPU, and the whole project is
  built `-ffp-contract=off` because a last-bit FMA difference flips which
  candidate wins.
- **Metal has no `double` type at all.** On Apple silicon via MoltenVK,
  `shaderFloat64` is not coming. On discrete parts fp64 exists at 1/16–1/64
  of fp32 rate.

Under a bit-exactness requirement that is fatal. Under "GPU-residency, not
optimality" it is a non-issue: run those four stages in **fp32**, accept that a
different candidate sometimes wins, and the output is still lossless. What it
costs is compression, and that cost is separately measurable — the existing
`FLACOUT_DUMP_FP32RANK` result already says it is small in the place it was
measured (the exact winner ranked 0 in the fp32 Rice ordering on 97.5–100% of
58.3M candidates; keeping the top 4 and refining exactly cost **zero bits** on
every fixture including real music).

Corollary: **`bench/check.sh` cannot gate this.** `-G` is gateable today
precisely because it is bit-exact with the CPU. A pure-GPU path is a different
encoder, and its regression net has to be "decodes to the input's md5" plus a
size table, not `cmp`.

## Stage table

Fixed-block configuration (one block size, no variable-block DP) unless noted.
"New" = does not exist in any form today.

| # | stage | GPU shape | status |
|---|---|---|---|
| 0 | de-interleave + sign-extend raw PCM to planar `int32` | 1 lane/sample, pure gather | new, trivial |
| 1 | wasted-bit detect (`ctz` of OR-reduce per subframe) | workgroup reduce | new, trivial |
| 2 | stereo decorrelation → 4 signals (L, R, M, S) | 1 lane/sample × 4 | new, trivial |
| 3 | window × autocorrelation, lags 0..32 | workgroup per (block, signal, window); 33 accumulators, subgroup-reduced | new, easy |
| 4 | Levinson-Durbin, all orders | **one subgroup per solve, lane j holds a[j]** | new, delicate |
| 5 | `lpc_log2cmax` + quantize with error feedback | 1 lane per (order, precision); 32 sequential steps | new, easy but sequential |
| 6 | residual + Rice cost per candidate | `shaders/sweep.comp` | **exists, bit-exact** |
| 7 | argmin over candidates; Constant/Verbatim/Fixed(0..4) alternatives | reduction + 6 cheap kernels | partly exists (CPU) |
| 8 | stereo-mode choice per block (min over 4 pairings) | trivial reduce | new, trivial |
| 9 | materialize the winner's residuals | 1 lane/sample, exact int64 | new, trivial |
| 10 | bit-length prefix scan → absolute bit offset per code | hierarchical scan | **new, the real work** |
| 11 | scatter Rice codes by `atomicOr` | 1 lane/residual | **new, the real work** |
| 12 | frame headers, subframe headers, coefficients | 1 lane/frame | new, fiddly |
| 13 | CRC-8 (header), CRC-16 (frame) | 1 lane/frame, or combine-reduce | new, easy |
| 14 | frame byte-length scan → final file offsets | scan over frames | new, trivial |
| 15 | MD5 of interleaved PCM | **cannot be parallelized** | see below |

Stages 0–2, 7–9, 14 are bandwidth-bound trivia. Stages 3–5 are the fp64
question above, now answered. Stage 6 exists. **Stages 10–13 are the genuinely
new engineering**, and 15 is the one honest exception to purity.

## Why this is worth doing: the host round trip is the binding constraint

Not "the GPU is faster". The current `-G` is at best parity on a discrete part
(`TODO.md`), and the reason is structural rather than a tuning miss:

- `GpuEvaluator::Candidate` is 34 `int32_t` = **136 bytes per candidate**, and
  the host uploads one per candidate priced. A full sweep is 26 windows × 32
  orders × 8 precisions = **6656 candidates per subframe**; the `FP32RANK` run
  logged 58.3M candidates over 8845 subframes of a few seconds of audio, i.e.
  **7.9 GB of candidate upload** for that. The shader's own header estimates
  ~10⁹ candidates for a 3-minute track under `-e`: ~136 GB.
- The host must **quantize every candidate before it can offer it**, and
  `eval_candidate` (quantize + delta residual update) is **33% of `-e`
  runtime**. So the CPU cannot get out of the way even in principle: it is
  doing a third of the work just to feed the device.
- `would_accept()` exists to stop that batch-build cost being wasted, and duty
  cycling makes dispatch nondeterministic (trap 11).

Moving stages 3–5 onto the device replaces 136 bytes/candidate with **~132
bytes per (block, signal, window)** — one autocorrelation — and the candidate
explosion happens entirely in device memory. That is the whole point. The
speed follows from deleting the traffic, not from the arithmetic.

Second-order but real: it also makes the GPU path *simpler*. No slots, no duty
cycle, no `min_batch`, no CPU fallback path to keep in sync with a shader. One
command buffer per chunk, barriers between stages, one readback.

## Stage 4 in detail: Levinson without an 8 KB private array

The CPU version keeps `double ld_a[33][33]` (~8 KB) on the stack. A private
array of that size per GPU lane is not a spill, it is a rout — `sweep.comp`'s
own notes record that a bare `int qc[32]` cost a quarter of the 128-register
budget at SIMD32 and had to be removed.

Only two rows of `ld_a` are ever live (order-1 and order). The decomposition
that fits the hardware is **one subgroup per solve, with lane `j` holding
`a[j]`** — order ≤ 32 and subgroup = 32 is not a coincidence worth wasting:

- `lambda = autoc[ord] - sum_j a[j]*autoc[ord-j]` → one `subgroupAdd`.
- the reversal update `a'[j] = a[j] - k*a[ord-j]` → one
  `subgroupShuffle` (index `ord-j`) plus an FMA.
- ~32 sequential steps of ~3 subgroup ops each: ~100 ops per solve, **zero
  private arrays, zero shared memory**.

Parallelism comes from the batch, not from within a solve: a 3-minute track at
4096-sample blocks gives ~1938 blocks × 4 signals × N windows solves, which is
5k–50k independent subgroups. Occupancy is fine.

The instability early-out (`|k| >= 1` → zero the remaining orders) is
subgroup-uniform, so it stays a clean branch. `out_err[]` per order — the
ranking quantity — falls out for free and can drive a device-side candidate
shortlist, which is how `-c`/`-p` survive the port.

## Stages 10–13 in detail: the bitstream is the interesting problem

This is standard GPU variable-length entropy coding (the shape GPU JPEG and ANS
encoders use), and FLAC's framing is unusually kind to it. Two facts from
`frame_writer.cpp` set the whole design:

1. **`bw.align()` before CRC-16 means every frame is a whole number of bytes.**
   Frames are therefore *independently* placeable: a scan over frame byte
   lengths gives each frame's final file offset, and packing can write straight
   into the final output buffer. One device→host copy produces the file.
2. **Within a frame, subframe boundaries are bit-level** — no padding between
   them. So inside a frame you need genuine bit-granular packing.

The two-pass shape:

**Pass A (lengths).** Each Rice code is `(u >> k) + 1 + k` bits; escape
partitions are fixed `escape_bps` each. `sweep.comp` already computes exactly
these sums per partition (`total0`/`total1`) — the winner's totals are a
by-product, not new work. Hierarchically scan: residual within partition →
partition within subframe → subframe within frame → frame within file. Add
header, coefficient and warm-up widths (closed-form from `SubframeParams`) and
the frame-end pad. Each residual now knows its absolute bit offset; each frame
knows its byte offset.

**Pass B (scatter).** One lane per residual: build the code in a `uint64_t`,
shift into position, `atomicOr` the one or two `uint32_t` words it touches.
Correctness does not depend on ordering — distinct codes occupy **disjoint bit
ranges**, and OR over disjoint ranges is commutative. The output buffer must be
zeroed first, which it is anyway.

The one wart: a long unary run (`u >> k` can be hundreds of bits on a badly
chosen `k`) spans many words. Handle it as a zero-fill plus a terminating `1`
bit — the zeros need no write at all into a zeroed buffer, so a long code is
*one* `atomicOr` for the stop bit and one for the remainder. Long codes get
cheaper, not more expensive.

Headers (stage 12) are ~10 bytes per frame plus ~2–60 bytes of coefficients;
one lane per frame writing bytes serially is correct and invisible in the
profile. `write_utf8` and `blocksize_code` port verbatim.

**CRC (stage 13).** CRC-8 covers the header (≤ 16 bytes, one lane, done).
CRC-16 covers header+payload, so it runs after pass B. One lane per frame with
a slice-by-4 table is a few thousand iterations over a few thousand frames —
adequate. If it ever matters, CRC is linear: `CRC(A||B)` combines from
`CRC(A)`, `CRC(B)` and `|B|`, so it is a subgroup reduce over chunks with a
precomputed shift basis. Do the simple thing first.

## Stage 15: MD5 is the one place purity must give

MD5 chains 64-byte blocks. It is not parallelizable, at all, and a single GPU
lane grinding ~500k serial rounds for a 3-minute track would be the slowest
thing in the pipeline by orders of magnitude.

Three options, and the middle one is the answer:

1. **One GPU lane.** ~500k sequential block compressions for a 3-minute track,
   each a few hundred instructions on a serial dependency chain: **~0.25 s
   estimated**, i.e. ~6x slower than the host pass, while occupying a lane and
   blocking the pipeline instead of overlapping it. Don't.
2. **Keep it on the host, overlapped.** It is a single pass over the input PCM
   at memory bandwidth (~40 ms for a 3-minute 16/44.1 stereo track) and it has
   no dependency on anything the GPU does. Run it on a host thread while the
   device encodes; wall-clock cost is zero. The host still does *no per-frame
   work* — the purity claim survives intact.
3. **Write zeros.** All-zero MD5 in STREAMINFO is legal and means "unknown";
   `flac -t` then reports the signature as unavailable and skips the check.
   Free, and strictly worse for users. Offer it as a flag at most.

Take option 2. State in the docs that MD5 is host-side by design and why.

## Memory: this forces chunking

Whole-stream residency does not scale, and the existing measurements say so
directly (`CLAUDE.md`, Memory footprint: a 74-minute single-track rip projects
to ~5.5 GB under `-e` on the CPU already). For a device buffer set:

| buffer | bytes/sample/channel | 3-min stereo | 74-min stereo |
|---|---|---|---|
| planar PCM | 4 | 63 MB | 1.5 GB |
| 4 decorrelated signals | 8 | 127 MB | 3.1 GB |
| winner residuals | 4 | 63 MB | 1.5 GB |
| packed output | ~2 | ~32 MB | ~0.8 GB |

So: **chunk the stream into ~10–30 s segments** and pipeline them. With a fixed
block size this is exactly free (frames are independent). With variable blocks
it costs one partition decision per chunk boundary, which is negligible if
chunk boundaries are placed on the block grid. Chunking also gets you
compute/transfer overlap for nothing.

## What about the variable-block DP?

It is a shortest path over ~7700 nodes × 5 candidates and runs in
*microseconds* — `O(N×K)` sequential. Two honest choices:

- Read the cost table back (a few tens of KB) and run the DP on the host. Costs
  one small readback per chunk and no per-frame host work. Not a purity
  violation in any sense that matters.
- Or run it as a single-workgroup sequential kernel, 5 lanes wide, ~7700
  launch-latency-bound steps. Doable, pointless, but it makes the "no host in
  the loop" claim literal.

Do the first. Revisit only if a profile says the readback synchronization
stalls the pipeline.

## Cost model: predicted, then measured

**Built and measured (M4 Max, 2026-08-11).** The stage-share prediction held
exactly; the throughput estimate did not.

`FLACOUT_PG_PROFILE=1`, master mix (8.3M samples/channel, stereo 16-bit),
serialised submits so the shares are readable:

| stage | share |
|---|---|
| **sweep (6)** | **92.4%** |
| crc (12) | 2.4% |
| autoc (2) | 1.5% |
| prepare (1) | 1.4% |
| rice (8) | 0.8% |
| pack (11) | 0.3% |
| levinson (3), quant (4), fixed (5), select (7), layout (9), frame (10) | 0.2% each |

So the architectural claim above is confirmed: **every new stage is a rounding
error and the pre-existing sweep kernel is the whole cost.** The bit packer —
the part flagged as the real engineering risk — is 0.3%.

**The ~71 ms estimate was ~14x optimistic, and the reason is worth recording.**
It came from this repo's own "1835 GMAC/s exact-integer" figure
(`FLACOUT_DUMP_FP32RANK`'s commit message). That number is not what this kernel
achieves. Measured on the same device:

| path | rate |
|---|---|
| `-G` (existing kernel, `-e -c 0 -L 0`, music_10s) | 3303.9 GMAC in 25.11 s = **131.6 GMAC/s** |
| `-P` sweep (master mix) | ~70 GMAC in 0.665 s = **105 GMAC/s** |

The two agree to within a factor of 1.25, so the pure-GPU sweep is running at
the kernel's real speed — the 1835 figure must have been a microbenchmark of the
residual loop alone, without the partition search. **Do not use it to size
anything.** ~110-130 GMAC/s is the number.

That reframes the whole exercise: at 32 orders x 4 windows the sweep is a
*deeper* search than `flac -8` performs, and a 16-core NEON CPU with
hand-vectorised residual and Rice loops is genuinely competitive with this
device on it. Speed has to come from pricing fewer candidates, not from the
device being faster per candidate.

### The order shortlist is the lever (`--pg-orders`)

Levinson already produces the per-order prediction error, so ranking orders and
sweeping only the best few costs one extra store and a 32-iteration loop in
`pg_quant` — no extra dispatch, no compaction. Skipped candidates are marked and
the sweep bails before its first subgroup op.

Master mix, against all 32 orders:

| orders | bytes | device s | speedup | size |
|---|---|---|---|---|
| 32 | 16954903 | 0.677 | 1.00x | — |
| 16 | 16955999 | 0.389 | 1.74x | +0.006% |
| **8** | **16959643** | **0.245** | **2.77x** | **+0.028%** |
| 4 | 16969221 | 0.163 | 4.15x | +0.084% |
| 2 | 16984374 | 0.121 | 5.60x | +0.174% |
| 1 | 17005700 | 0.103 | 6.58x | +0.300% |

8 is the default. This is the same shape as the CPU's `-c` frontier and for the
same reason.

### Attacking the sweep: it is not ALU-bound, and two "obvious" fixes prove it

The sweep started at 92.4% of device time, so it is the only stage worth
optimising. Three changes were tried. **The order they are listed in is the
order of increasing payoff, and it is the reverse of the order they look
promising in.**

1. **Skip empty bit-planes** (bit-exact). The kernel ballotted all 32 planes per
   32-sample chunk; the highest plane any lane populates is
   `findMSB(subgroupMax(u))`, and skipping above it is exactly equivalent
   (lane j owns plane j, so a lane above the top keeps `myBallot == 0`, and a
   zero `B_j` contributes nothing to the scan, the `bitCount`, or `Hcur`).
   Real residuals occupy ~10-12 planes, not 32. **Worth 2.7%.**

2. **Narrow the dot-product accumulator to int32** where
   `sum|qc| <= INT32_MAX >> (effBps-1)` — the same bound the CPU uses to pick
   `lpc_residual_blocked<int32_t>`. Metal has no 64-bit integer ALU, so int64 on
   Apple silicon is synthesised from 32-bit ops, and this looked like the big
   one. Verified exact (four fixtures byte-identical). **Worth zero.**

3. **Cap the sweep's partition-order search** (`--pg-pcap`). **Worth 3.4x.**

Two null results in a row are the finding, not a disappointment: **this kernel
is bound by cross-lane operations, not by arithmetic.** At order 8 a 32-sample
chunk costs ~16 arithmetic ops for the residual against ~80 subgroup ops --
~12 plane ballots plus two `closePartition` calls, each a 5-step shuffle scan
with two `subgroupMin`s and its own ballots. The dot product is not the cost.

That is also why the `FP32RANK` route (sweep in fp32, exactly refine the top K)
is much less attractive than it looks: fp32 would speed up the arithmetic the
kernel is not spending its time on, and leave every ballot and shuffle in place.
It was worth reaching for when the sweep was believed to be an int64 MAC loop.

The partition cap works because closes are `2^(P+1)-1` and the kernel is made of
them -- 511 at P=8, 31 at P=4, 7 at P=2. Master mix:

| pcap | bytes | sweep s | speedup | size |
|---|---|---|---|---|
| 8 | 16959643 | 0.1948 | 1.00x | — |
| 6 | 16961874 | 0.0832 | 2.34x | +0.013% |
| **4** | **16964581** | **0.0575** | **3.39x** | **+0.029%** |
| 3 | 16965880 | 0.0522 | 3.73x | +0.037% |
| 2 | 16969536 | 0.0508 | 3.83x | +0.058% |
| 1 | 16978990 | 0.0492 | 3.96x | +0.114% |

Sizes barely move because **the cap is a ranking approximation only** — stage 8
re-prices the winner with the full P=8 search, so the cap can change which
candidate wins but never what the bitstream says a partition costs. Identical
argument to `--gpu-partition-cap`, and here it is worth far more, because
nothing else in this kernel is the bottleneck. Default 4.

After both, the profile has flattened and the sweep is no longer the whole cost
(master mix, serialised submits, so small stages carry ~1.2 ms of submit
overhead each):

| stage | share |
|---|---|
| sweep | 55.0% |
| **crc** | **15.3%** |
| autoc | 8.2% |
| prepare | 7.6% |
| rice | 5.1% |
| everything else | ~1% each |

### CRC-16 was a parallelism problem, not an algorithm problem (10.7x)

At 15.3% it was the second bottleneck, and the diagnosis is worth keeping
because the number that mattered was not a share:

```
96 MB/s   on an M4 Max
```

A chunk holds at most 256 frames, so a lane-per-frame kernel has **256 threads
for ~1.8 MB**, each walking ~7 KB serially from a different region of the buffer.
It was short of parallelism by two orders of magnitude, and uncoalesced besides.
No table trick (slice-by-4, slice-by-8) addresses that — they divide the 7 KB
walk, they do not add threads.

Splitting *within* a frame needs CRC's one exploitable property, and it is the
property MD5 lacks: **it is linear over GF(2).** With init 0 and no final xor,
`CRC(M) = M*x^16 mod P`, so for `M = A||B` with `|B| = n` bytes:

```
CRC(A||B) = CRC(A) * x^(8n)  +  CRC(B)      (mod P, + is xor)
```

So every lane CRCs its own slice from init 0 and a tree reduction folds them --
one workgroup per frame, 8 levels for 256 lanes. The only new primitive is
multiplication in `GF(2)[x]/P` (a 16-iteration carryless multiply); the `x^(8n)`
factor is square-and-multiply on `x^8`. The reduction carries `(crc, length)`
rather than `crc` alone, so it stays associative when the last slice is short --
~8 `powx8` calls per lane against the ~7000 byte-steps they replace.

**0.0192 s → 0.0018 s, 10.7x**, ~96 MB/s → ~1 GB/s. 1.8 ms is at this
profiler's submit-overhead floor, so the true cost is now below what it can
resolve. Verified end to end rather than by argument: `flac -t` validates every
frame's CRC-16, so a wrong fold fails on the first frame of any fixture.

Profile after the sweep cap and this (master mix; small stages carry ~1.2 ms of
submit overhead each, so their shares are inflated):

| stage | share |
|---|---|
| sweep | 61.5% |
| prepare | 9.7% |
| autoc | 9.4% |
| rice | 6.3% |
| pack | 2.2% |
| crc | 2.0% |
| everything else | ~1.5% each |

### End-to-end, against the CPU

Interleaved best-of-5, quiet machine, `-n` on every arm. Times include decode,
MD5 and file I/O, which are common fixed costs (decode alone is 0.109 s on the
master mix, so the device is the bulk of `-P`).

| fixture | `-P` wall | CPU default | speedup | size |
|---|---|---|---|---|
| short (<1024 samples) | 0.038 s | 0.021 s | **0.57x** | — |
| micro | 0.048 s | 0.074 s | 1.54x | — |
| music_3s | 0.051 s | 0.092 s | 1.81x | +0.83% |
| music_10s | 0.062 s | 0.136 s | 2.18x | +0.40% |
| music_20s | 0.063 s | 0.202 s | 3.23x | +0.14% |
| MLKDream | 0.211 s | 1.473 s | **6.97x** | +1.12% |
| master mix | 0.154 s | 0.995 s | 6.46x | +0.99% |

**`-P` loses on very short inputs and that is structural.** A ~16 ms fixed cost
cannot be amortised over a 20 ms file, so the CPU path wins below roughly a
second of audio. Left as-is rather than papered over with a silent fallback: `-P`
means the encode ran on the device, and quietly not doing that would make the
flag untrustworthy. It is a throughput tool.

### Startup latency: cache what you can, overlap the rest

Fixed cost was ~26 ms, which is ~37% of a 10-second file's wall clock — the
reason short inputs stalled at ~2x while long ones reached 6x. Measured with
`FLACOUT_PG_INITTIME=1` rather than guessed at:

| phase | before | after |
|---|---|---|
| `vkCreateInstance` (loader + MoltenVK) | 13.8 ms | 12.3 ms |
| enumerate + `vkCreateDevice` | 1.7 ms | 1.5 ms |
| **12 compute pipelines** | **10.5 ms** | **1.8 ms** |
| descriptors, command pool, buffers, windows | 0.2 ms | 0.2 ms |

Two fixes, because the two big terms need different ones:

- **The pipeline compiles are cacheable.** A `VkPipelineCache` seeded from
  `$XDG_CACHE_HOME/flacoutcpp/pg_pipelines.bin` (200 KB) takes 12.4 ms → 2.0 ms,
  so only the first run on a given machine and driver pays. Safe by construction:
  Vulkan validates the blob's header and cache entries are keyed by shader hash,
  so a driver upgrade or an edited shader is a silent miss, not a hazard. Written
  via a temp file and renamed so a concurrent run cannot see a partial cache.
  `FLACOUT_PG_CACHE=-` disables it. The twelve pipelines are also created in one
  `vkCreateComputePipelines` call rather than twelve, which lets the driver batch
  whatever it still has to compile.
- **Instance creation is not cacheable, but nothing about it depends on the
  audio.** So Vulkan now comes up *after* the decode threads are already running,
  and its ~12 ms sits beside the decode instead of in front of it.

Net: master mix 0.170 → **0.154 s**, MLKDream 0.237 → **0.211 s**, music_20s
0.076 → 0.063 s. `vkCreateInstance` is now the floor and there is nothing
portable to do about it.

### The pipeline: one producer, two consumers

Decode was the binding constraint once the kernels were fast (0.109 s against
0.106 s of device time), and it does not want to move to the GPU. Decoding is
*not* the mirror of encoding: encoding variable-length codes is parallel because
the lengths are computable independently and a prefix scan turns them into
offsets, whereas **decoding cannot know where code i+1 starts until code i is
decoded** — a set bit is a stop bit only if it is not inside a previous
remainder field. Frame-boundary discovery parallelises (speculative sync scan
validated by CRC-8, CRC-16 and sample-number chaining), and LPC reconstruction
parallelises across subframes, but the Rice decode is a genuine serial
dependency. Meanwhile libFLAC's decode scales ~7x across cores for free
(8 concurrent decodes of the master mix: 0.124 s wall, 0.0155 s each) and the
CPU is *idle* while the device works. Moving it onto the GPU would take work off
an idle unit and put it on the saturated one.

So `-P` drives its own streaming decode instead. `write_callback` writes into a
**preallocated** buffer at the published offset and republishes the count with
release ordering; the encoder acquires it and may read anything below. That is
the entire synchronisation on the sample data — no lock. Chunk N is submitted as
soon as its samples exist.

**MD5 belongs on a third thread, and getting this wrong cost most of the win.**
It was first folded into the decode thread on the theory that the samples are
already hot. But MD5 is compute-bound at a few hundred MB/s, not memory-bound,
so that serialised ~66 ms of hashing behind a 109 ms decode: the master mix went
0.239 → 0.221 s instead of the predicted ~0.13 s. Split back out as its own
consumer of the same counter (and batched, since `update()` at 6 bytes a call is
mostly call overhead), it disappears: 0.221 → **0.170 s**.

What is left is not encode work:

```
0.170 s  =  0.025 (Vulkan init + 12 pipeline compiles)
          + 0.109 (decode, overlapped with 0.106 device)
          + ~0.036 (metadata, 17 MB file write, process startup)
```

Two consequences. The ~25 ms fixed cost is why short inputs only reach ~2x
(it is 37% of music_10s's wall clock) — a pipeline cache would address it.
And **threading the decode is now pointless**: `max(0.109, 0.106)` becomes
`max(0.016, 0.106)`, worth 3 ms, because the device is once again the bound.

Losslessness verified by decode on all 18 fixtures (`flac -t` ok and decoded
audio MD5 equal to the input's), including 24-bit, mono, and the `<1024`-sample
short-stream path.

### The tonal gap: fp32, but not where it looked

This was the last real defect and it took four wrong hypotheses to corner, so the
eliminations are worth keeping.

**Where the gap is.** Held against the CPU at a fixed 4096-block partition:

| fixture | gap |
|---|---|
| syn3m_noise | +0.02% |
| syn3m_transient | +1.71% |
| syn3m_mix | +5.53% |
| syn3m_tonal | **+12.23%** |

Entirely tonal content, and only ~1.9% of `syn3m_mix`'s original 7.62% was the
fixed block size. Eliminated one at a time, each measured: **windows** (1 vs 4 vs
8), **precisions** (including the low rungs the CPU actually picks), the **order
shortlist** (8 vs 32 orders differ by 0.05%), the **partition cap**, and
**instability** — a reach histogram showed all 4096 solves reaching order 32, so
no candidate was missing. A `-P` run was also worse than `flac -5`, which uses one
window and max order 8, ruling out "the CPU just searches harder".

**The mono test localised it.** Extracting one channel to a mono file put `-P`
within **+0.37%** of the CPU and *ahead* of `flac -8`. So the LPC analysis was
fine and the whole gap lived in the **side channel** — where `-P` chose FIXED for
525 of 1938 subframes and the CPU chose LPC for all of them.

**The mechanism.** Dumping such a block:

```
err/err0:  1:0.0068  2:3.3e-05  3:2.6e-06  4:2.2e-06  5:9.8e-07  6:9.3e-06 ...
LPC costs: o4:28152  o8:24714  o16:25984  o32:36743     FIXED f4:19356
```

`err[5] = 9.8e-7` is a **60 dB prediction gain**, sitting on fp32's 1e-7 floor.
Past that the reflection coefficients are noise, `|k| >= 1` fires, and the retry
fills the higher orders with something worse — note err *rising* at order 6 and
order 32 costing more than order 8. So fp32 was the problem after all, but in the
**autocorrelation**, whose ~1e-7 accuracy bounds the resolvable prediction gain.
Real music sits at err/err0 ~ 4e-5, a 400x margin, which is why it barely cared.

**Two things that did not work, in order.** Upgrading the *recursion* alone to
double-float made it worse (12610817 → 12722154): fed a noisy autocorrelation it
faithfully solves the noisy problem, stops aborting, and loses the retry's
accidental regularisation. A global ridge is also a bad trade (see the table in
pg_levinson.comp).

**And the trap that hid the fix.** With the autocorrelation *and* the recursion
both in double-float, the measured accuracy did not move — still ~1e-8. MoltenVK
compiles Metal with **fast math** by default, which reassociates floating point
and therefore deletes the very error terms double-float is built from:
`fma(a, b, -a*b)` folds to zero. With fast math on, the double-float code is not
merely useless but *harmful*, because it adds rounding without adding precision.

```
                autocorrelation rel. err     syn3m_tonal (1 window)
plain fp32      1e-8 .. 7e-8                 12729698
df, fast math   1e-8 .. 7e-8                 12902040   <- worse than fp32
df, no fast math  3e-15 .. 5e-15             11857452
```

`MVK_CONFIG_FAST_MATH_ENABLED=0` is now set in-process before instance creation
(Apple only, and not if the user already chose). Diagnosing this needed a host
reference that replicates the device's *fp32* windowing exactly — comparing
against a double-windowed signal shows a ~1e-7 difference from the window
coefficients alone and says nothing about the sum.

**Result.** Gaps against the CPU at the same partition: syn3m_tonal +12.23% →
**+3.88%**, s24_2s +19% → **+4.08%**, stereo_4s → +4.58%, syn3m_mix → +3.94%.
Real music improved too, against the CPU *default*: music_20s +0.14% → **+0.03%**,
music_10s +0.40% → +0.22%, master mix +0.99% → +0.77%.

**Cost.** The autocorrelation went 8.6 → 19.7 ms, 2.3x rather than the ~13x the
op count suggests, because that stage is shared-memory bound and the extra
arithmetic is nearly free. Device total 91.6 → 104.4 ms; master mix wall clock
0.154 → 0.166 s (6.46x → 6.01x). Trading ~8% of time for 0.22% of size is far
better than anything on the CPU's own frontier, where the whole estimated-DP dial
spans ~0.1% for 6.4x.

## Absolute throughput: parallel decode, then the real candidate frontier

Two changes, and the second one caught a measurement mistake worth recording.

### The decode had to be parallelised to make anything else pay

With `FLACOUT_PG_WALL=1`, the master mix's 148 ms broke down as ~3 ms buffer
allocation, ~30 ms Vulkan init (overlapped with decode), and a 115 ms
encode phase that was *waiting on the decode* — libFLAC's single-threaded decode
was ~109 ms against ~99 ms of device time. Being co-bound is why cutting either
alone had bought nothing.

Frames are independent once you know where they start, and a seek is how a worker
finds out. The stream is split into block-aligned contiguous ranges; worker *i*
seeks to its start and decodes until it passes its end. Each frame is placed by
**its own sample number** (from the frame header, not a running counter) and
clipped to the worker's range, so there is exactly one writer per sample — a seek
lands on the frame *containing* the target, so the first frame usually starts
before it, and clipping is what keeps two workers from both writing those samples.

The published counter advances only as a **contiguous prefix** of ranges
completes, which preserves the "anything below this is readable" rule that the
encoder and the MD5 consumer both depend on.

```
decode threads    1      2      4      6      8     12
decode done at  99.8   59.3   33.6   23.1   18.2   16.4  ms
```

5.5x at 8 threads (the default is `min(cores/2, 8)`; `FLACOUT_PG_DECODE_THREADS`
overrides). Decode stops being the bound: 18 ms against ~99 ms of device work.
Vulkan init also fell from 30 ms to 18 ms, because `vkCreateInstance` had been
contending with the old single decode thread (24.7 ms under load, 12.3 ms idle).

### Windows are worth more than orders — but measure *device* time, not sweep time

The default was 4 windows x 8 orders. Sweeping the two axes on MLKDream showed it
**dominated in both directions**, and the frontier is much flatter in orders than
in windows.

The trap: a first pass measured only *sweep* time, concluded 10 windows x 2 orders
(sweep 44 ms against 61 ms) and made the encoder **slower overall** — because the
**autocorrelation also scales with window count** (14 ms at 4 windows, 30 ms at
10), and it is the second-largest stage. On total device time:

| config | total | MLKDream | master mix |
|---|---|---|---|
| 4w x 4o | 107.3 ms | 28032077 | 16930720 |
| 4w x 8o | 127.2 ms | 28027341 | 16927410 |
| **6w x 3o** | **109.4 ms** | **28008023** | **16925777** |
| 8w x 3o | 123.9 ms | 28005095 | 16896703 |
| 10w x 3o | 140.3 ms | 28003888 | 16883922 |

6w x 3o was faster *and* smaller than the old default on both fixtures, so it
became the default for a while. **It is now 10 windows x 2 orders** — the whole
CPU shortlist, at two orders — which the corpus says is strictly better.

### 10 x 2, and all 188 tracks agree

Re-swept as a 2-D frontier on real music instead of one axis at a time. A window
costs **~9-10% of device time** (album track, 6 → 10 windows: autoc 43.8 → 59.6 ms,
levinson 15.4 → 24.4, sweep 63.9 → 95.7, total 137.5 → 193.5); only `pg_rice` is
flat, since it prices the winner alone. Below ~6 windows the cost is invisible in
wall clock, buried under process startup and decode — 1, 2, 4 and 6 windows all
finish an album track in 0.151–0.153 s.

What windows buy, at 3 orders on that track: 6→8 **-0.081%**, 8→10 **-0.052%**,
10→12 -0.002%, 12→16 -0.004%. **Ten is the knee**, and standard shapes past the
shortlist (blackman, hamming, bartlett, nuttall, flattop, connes) add nothing —
independently reproducing what the CPU shortlist concluded.

Orders barely register beside that. `8w x 2o` (0.179 s, -0.077%) **dominates
`6w x 6o`** (0.216 s, -0.006%) on both axes, and `6x2` → `6x6` is -0.011% for 1.4x.

Album corpus, 188 tracks, `10x2` against `6x3`:

| | |
|---|---|
| 6x3 | 3666771700 |
| **10x2** | **3661740884** |
| delta | **-0.1372%, 4.8 MiB** |
| smaller / unchanged / larger | **188 / 0 / 0** |
| corpus wall clock | 20.20 s → 21.08 s (1.04x) |

Per album it spans -0.115% (Shovel Knight, 48 mono tracks) to -0.179% (MDK), with
the 24-bit album (Parklands) at -0.136%. **No track regressed**, which is unusual
here — and it disposes of the one worry the fixtures raised: `s24_2s` grows 0.5%
at two orders, and that is a small-fixture artifact rather than a bit-depth
effect.

**The two knobs are not independent.** Two orders is only safe because the
shortlist widened: `6w x 2o` alone is +0.005% on the album track. Do not tune one
without the other, exactly as `-c` and `-L` behave on the CPU side.

### Result

| fixture | `-P` | at first commit | CPU default | speedup | size vs 4w8o |
|---|---|---|---|---|---|
| music_10s | 0.059 s | 0.075 s | 0.138 s | 2.32x | -0.01% |
| music_20s | 0.065 s | 0.084 s | 0.200 s | 3.06x | -0.02% |
| MLKDream | 0.164 s | 0.421 s | 1.485 s | **9.04x** | -0.07% |
| master mix | 0.145 s | 0.412 s | 0.990 s | 6.83x | -0.01% |
| syn3m_mix | 0.144 s | 0.395 s | 0.941 s | 6.54x | +0.49% |
| s24_2s | 0.052 s | — | 0.089 s | 1.71x | +1.24% |

Faster everywhere and smaller on all real music; the two synthetic fixtures pay
for the shallower order search and want `--pg-orders 8` back. Against the first
working commit this is **2.6-2.8x** on the long fixtures.

## Cross-platform: Intel Arc A380 (Mesa ANV), and a portability bug it exposed

First run of `-P` on anything but Apple. Built for linux/amd64 with
`docker/Dockerfile.amd64` (bookworm on purpose -- its glibc 2.36 runs on the test
host's 2.42, which is the direction that works) and shipped to the Debian/Arc
A380 host.

**It works and it is lossless.** `subgroup 32 pinned, shaderInt64` via
`VK_EXT_subgroup_size_control`, and every output decoded back on the Mac matches
the input's audio MD5 and passes `flac -t`.

**But the double-float arithmetic was silently dead there.** The autocorrelation
came out at **3e-8 relative, not 3e-15** -- no better than plain fp32. Mesa ANV
reassociates floating point just as MoltenVK's fast math does, so
`MVK_CONFIG_FAST_MATH_ENABLED=0` had been solving the problem on exactly one
platform. Because the double-float code adds rounding without adding precision
when that happens, Intel was getting the *worse* variant.

**The portable fix is the `precise` qualifier**, which emits SPIR-V
`NoContraction` on the marked expressions. It has to cover every intermediate the
error terms flow through -- `twoSum`, `twoProd`, `dfNorm` alone was not enough,
because `dfMul` and `dfDiv` compute their low parts inline and those reassociated
instead. With all of them marked:

| fixture | Apple | Intel | before, Intel |
|---|---|---|---|
| music_10s | 1664755 | 1664758 | 1669227 |
| MLKDream | 28008006 | 28008003 | — |
| syn3m_tonal | 11620328 | 11623318 | 12458677 |
| s24_2s | 298076 | 299477 | 336838 |

The two platforms now agree to within 0.03% on real music and 0.5% on the
synthetic fixtures, against 7-13% apart before. It is also a **win on Apple**
(syn3m_tonal -0.50%, s24_2s -0.61%), and it works there with fast math forced back
*on* -- so `precise` supersedes the MoltenVK setting, which is kept only as belt
and braces.

### Performance: the A380 is ~12x slower than an M4 Max here

Device timestamps, MLKDream. The CPU side of that host was contended (jellyfin at
289%, load 7.7) so wall clock is worthless, but the GPU was idle -- jellyfin holds
no `/dev/dri` descriptors -- so the device numbers stand.

| stage | M4 Max | Arc A380 |
|---|---|---|
| sweep | 43.9 ms (49%) | 491 ms (45%) |
| autoc | 22.8 ms (26%) | 285 ms (26%) |
| rice | 12.4 ms (14%) | 132 ms (12%) |
| **total** | **89.4 ms** | **1087 ms** |

The *shape* is portable -- the same three stages in the same proportions -- but the
absolute speed is 12x apart, more than the ~3.5x raw fp32 ratio between the parts.
`GPU_PLAN.md` proposed a width-agnostic bit-plane mapping as the fix, on the
theory that the kernels spill at SIMD32 and the driver would rather use SIMD16.
**The spills are real and the theory is wrong.** See below.

### Register pressure is not the Arc's problem, and the width rewrite cannot fix it

Measured with `INTEL_DEBUG=cs` (needs `MESA_SHADER_CACHE_DISABLE=true`, or the
shaders never recompile and nothing prints), Mesa 26.1.5, music_3s:

| kernel | instr | spills:fills | share of device time |
|---|---|---|---|
| **sweep** | 9391 | **449:747** | 42.6% |
| **autoc** | 12538 | **198:737** | 29.6% |
| **rice** | 8122 | **350:588** | 11.4% |
| frame | 1565 | 29:22 | 0.1% |
| the other 9 | — | 0:0 | — |

So three kernels spill and they are 84% of device time. Everything after that
point contradicts the plan:

**1. The dominant state is width-invariant, so narrowing the subgroup cannot
shrink it.** The fold's live state is `Bacc/Nacc/Hacc/total0/total1`, and the
divergent part of it is *32 planes x NLEV levels* of counters. That is 1152 bytes
per subgroup-candidate whatever the width: at SIMD32 it is `Bacc[9]` per lane
(9 dwords x 4 GRFs), at SIMD16 each lane owns two planes (`Bacc[2][9]`, 18 dwords
x 2 GRFs), at SIMD8 four planes (36 dwords x 1 GRF). **36 GRFs in every case.**
The rewrite redistributes the state across lanes; it does not reduce it.

**2. Cutting the pressure barely helps anyway.** Compiling the sweep with a
partition ceiling of 4 instead of 8 -- which is a genuine reduction, since the
default `--pg-pcap` is already 4 and levels 5-8 are dead weight -- cuts the kernel
by a third and the spills by 40%, and buys almost nothing:

| build | instr | cycles (compiler estimate) | spills:fills | MLKDream wall |
|---|---|---|---|---|
| `PG_SWEEP_MAXP=8` | 9389 | 44.8M | 449:747 | 1.320 s |
| `PG_SWEEP_MAXP=4` | 6417 | 24.1M | 267:479 | **1.291 s (1.02x)** |

Output byte-identical, and on Apple the same build is 0.109 s either way, i.e.
exactly nothing. **-32% instructions and -40% spills bought 2%**, so scratch
traffic is not what this device is waiting on, and the compiler's cycle estimate
is not predictive of it.

**3. The two devices disagree about the tap loop's shape.** Breaking the
accumulator chain was worth 1.26x on Apple (925fcc1). On the Arc, `NACC` 8 / 4 / 1
measures 1.385 / 1.322 / 1.323 s -- fewer accumulators is *slightly better*. Both
knobs now exist (`-DFLACOUT_PG_DEFS=-DNACC=4`, `-DPG_SWEEP_MAXP=4`) precisely
because the right value is per-device.

**And cross-lane ops are not it either.** That was the last surviving
hypothesis -- ballots and the fold are ~50% of the sweep on Apple, and cross-lane
throughput is the obvious structural difference between an M4 Max and an
8-Xe-core A380 at a 2.8x raw-FLOPS ratio. Tested with the same ablation on both
devices (`-DFLACOUT_PG_DEFS=-DPG_ABLATE=n`, timing-only builds; 1 drops the LPC
dot product, 2 drops every ballot, shuffle and the partition fold). MLKDream,
sweep stage:

| build | M4 Max | Arc A380 | Arc/M4 |
|---|---|---|---|
| full kernel | 44.2 ms | 483.9 ms | 10.9x |
| no dot product | 18.4 ms | 227.8 ms | 12.4x |
| **no ballots, no fold** | 16.5 ms | 225.9 ms | **13.7x** |

**The ratio is flat.** Strip the cross-lane work entirely and the Arc is still
13.7x slower on what is left -- an integer MAC loop with no subgroup op in it --
so no primitive is disproportionately slow on that device. The two halves also
measure at 53%/53% there against 58%/63% on Apple, i.e. the same shape. Note the
shares exceed 100% on both, which is the latency-bound signature already recorded:
removing either half lets the other's latency hide.

A uniformly ~12x slower device is a throughput story, not a mapping story. The
int-MAC-only variant being the *worst* ratio points at int32 multiply rate on
Xe-HPG rather than anything the kernel chooses to do.

### 12x is not the hardware gap: 4x is, and int32 multiply is the other 3x

Tested by swapping only the arithmetic. `PG_ABLATE=3` is ablation 2's kernel --
no ballots, no fold -- with the integer MAC loop rewritten in fp32. MLKDream,
sweep stage:

| MAC loop only, no cross-lane work | M4 Max | Arc A380 | Arc/M4 |
|---|---|---|---|
| int32 | 16.5 ms | 226.6 ms | **13.7x** |
| fp32 | 19.9 ms | 80.0 ms | **4.0x** |

On the Arc the same loop in fp32 is **2.83x faster**. On the M4 it is 0.83x, i.e.
slightly *slower* -- integer multiply is full rate there and the conversions are
pure overhead.

4.0x is about what the parts predict: an M4 Max 40-core GPU is ~16.4 TFLOPS fp32
(5120 ALUs, ~1.6 GHz) against the A380's ~5.0 TFLOPS (1024 ALUs at a measured
2450 MHz), a 3.3x ratio, with the rest in clocks and occupancy. **So the answer
to "is 12x expected for this hardware gap" is no -- 3-4x is expected, and the
excess is one instruction.** Xe-HPG executes 32-bit integer multiply at reduced
rate; Apple does not.

This also corrects the framing used above and in `GPU_PLAN.md`: comparing the
observed gap against the "raw fp32 ratio" was the wrong basis, because the kernel
that matters is an *integer* MAC loop.

**It suggests one Intel-specific trade, and it is a real one.** The sweep's dot
product only *ranks* candidates -- `pg_rice` re-derives the winner's residual
exactly -- so pricing it in fp32 would cost compression, never losslessness,
exactly like the fp32 analysis stages. On the full kernel the arithmetic is
partly hidden behind the fold, so the ceiling is roughly 484 -> ~340 ms of sweep
(~1.15x overall on that device), against an unmeasured size cost from mis-ranking.
That is a much better lead than the width rewrite, and it is the opposite
conclusion from the FP32RANK note above -- which was measured on Apple, where the
arithmetic is not the bottleneck and int32 is free. Per-device, not universal.

### What is left on Intel, and it is small

Per-stage Intel/Apple ratios on the same default (10x2, MLKDream), which is the
only way to see which kernels are slow *for that device* rather than slow because
the device is slow. The baseline is ~13-16x for the integer-heavy kernels and
~4x for the fp32-heavy ones (`levinson` 4.1x, `crc` 3.1x), exactly as the
int32-multiply finding above predicts:

| stage | Apple | Arc | ratio | |
|---|---:|---:|---:|---|
| sweep | 36.7 ms | 563.4 ms | 15.4x | at baseline |
| autoc | 32.5 ms | 534.5 ms | 16.4x | at baseline |
| rice | 10.0 ms | 134.7 ms | 13.5x | at baseline |
| **quant** | **0.7 ms** | **136.2 ms** | **194x** | **anomalous** |
| prepare | 0.8 ms | 32.9 ms | 41x | anomalous |
| select | 0.2 ms | 5.8 ms | 29x | anomalous, but 0.4% |
| pack | 1.3 ms | 25.7 ms | 20x | mildly anomalous |
| levinson | 13.1 ms | 53.3 ms | 4.1x | fp32 parity |

Bringing all four anomalies to the device's own baseline is worth **~11% of Arc
device time**, and `pg_quant` alone is ~8% of it. That is the entire remaining
prize, and **two obvious explanations for it are already refuted** (timing-only
ablations, `-DFLACOUT_PG_DEFS=-DPG_QUANT_NOLOG=1` / `-DPG_QUANT_NOSTORE=1`):

- *Transcendentals.* The order-shortlist loop calls `log2` 32 times per lane and
  every one of a solve's 32 order-lanes recomputes the same 32 scores. Removing
  the `log2`: **136.2 -> 138.2 ms.** Not it.
- *Scattered stores.* Each lane writes 40 dwords and lane-adjacent candidates are
  160 bytes apart, so the writes coalesce badly. Collapsing the store to one
  dword: **136.2 -> 140.7 ms.** Not it either.

What is left to suspect is the uncoalesced `lpcf` reads (lane-adjacent lanes are
128 bytes apart) or plain latency on the 32-step error-feedback chain at low
occupancy -- and separating those needs a shader profiler, not timestamps.
**Bounded at ~8% for that kernel, on a device where `-P` already beats its host's
CPU path by 5.45x, this is where I would stop.**

**Conclusion: do not build the width-agnostic bit-plane mapping.** Three
independent measurements say it addresses nothing -- the state it would
redistribute is width-invariant, cutting that state 40% for real buys 2%, and the
cross-lane work it would make cheaper is not where the device's disadvantage
lies. It remains a large, delicate change to a kernel that is a cost-model
contract. If Intel throughput is ever worth chasing, profile a *per-primitive*
microbenchmark (int32 MAC, load, ballot) against the M4 first and aim at whatever
that says, because the kernel-level ablation has now been exhausted.

### What the Arc offers that Apple does not, and whether any of it helps

Queried rather than assumed:

- **`VK_KHR_shader_float_controls2`** -- present. Would let the fast-math mode be
  set per instruction rather than relying on `precise`; the `precise` fix already
  works, so this is a fallback if a driver ever ignores `NoContraction`.
- **`integerDotProduct4x8BitPackedSignedAccelerated = true`** (DP4a). The *only*
  accelerated integer dot product: 8-bit, 16-bit and 32-bit are all false. The LPC
  dot product is 15-bit coefficients times up to 25-bit samples, so using it needs
  limb decomposition -- 6-8 DP4a ops per 4 taps against 4 plain int32 MACs -- and
  the ablation above already showed the multiplies are not the bottleneck. **No.**
- **`shaderFloat64 = false`.** No hardware double even on a discrete part (the
  `true` in `vulkaninfo` is llvmpipe), so double-float emulation remains the only
  route to precision. No shortcut.
- **`VK_KHR_shader_subgroup_extended_types`** -- `subgroupAdd` on int64, which is
  the primitive that was missing when the exact-int64 autocorrelation was
  rejected. It does not fix that idea's real blocker (windowed values must stay
  under ~24 bits, impossible at 24-bit input), but it removes one of the two.
- **48 KB shared memory** against Apple's 32 KB -- the autocorrelation could stage
  larger tiles, though it is tiled precisely so that it does not have to.
- **`VK_KHR_cooperative_matrix`** -- present, but it enumerated no configurations
  here, and the precision available to matrix engines (fp16/int8) suits neither
  the autocorrelation nor the exact integer residual.

## How to profile this: use timestamp queries, not serialised submits

`FLACOUT_PG_PROFILE=1` originally fenced after every stage. That is easy and it
lies in two directions: it charges each stage a submit (~1.2 ms here, which is
most of a small kernel's apparent cost) and it cannot distinguish a stage that is
*computing* from one that is *waiting*.

The in-API answer is **timestamp queries**, and this device supports them fully
(`vulkaninfo`): `timestampComputeAndGraphics = true`, `timestampValidBits = 64`,
`timestampPeriod = 1` (ticks are nanoseconds), plus `VK_EXT_host_query_reset` and
`VK_EXT_calibrated_timestamps`. `pipelineStatisticsQuery` is false and
`VK_KHR_performance_query` is absent, so counters are not available — timestamps
are what there is.

The shape: a `VK_QUERY_TYPE_TIMESTAMP` pool of `S_COUNT+1` marks,
`vkCmdResetQueryPool` at the top of the command buffer, one
`vkCmdWriteTimestamp(BOTTOM_OF_PIPE)` after each stage's barrier (after, so the
mark measures the stage rather than the launch), then `vkGetQueryPoolResults` with
`64_BIT | WAIT_BIT` once the fence signals. Stage *i* costs `ts[i+1]-ts[i]`, times
`timestampPeriod`. **One submit, no serialisation, device clock.**

**It corrected the map this document had been using.** MLKDream, same build:

| stage | serialised | timestamps |
|---|---|---|
| sweep | 56.9 ms (57.7%) | **43.9 ms (49.1%)** |
| autoc | 17.1 ms (17.4%) | **22.8 ms (25.5%)** |
| **rice** | 5.5 ms (5.6%) | **12.4 ms (13.9%)** |
| prepare | 6.8 ms (6.9%) | 2.8 ms (3.1%) |
| select / fixed / layout / frame | ~1.2 ms each | 0.1-0.5 ms each |
| total | 98.6 ms | 89.4 ms |

Every small stage was almost entirely submit overhead, and two stages were
*understated*: the double-float autocorrelation costs more than it appeared
(25.5%, not 17%), and **`pg_rice` is 13.9%, nearly three times the 5.6% recorded
above** — which makes it the third-largest stage rather than a rounding error, and
a real target. The sum is now meaningful too: 89.4 ms against a 103 ms encode
phase, the difference being host-side work (PCM staging, readback, file writes).

**What timestamps still cannot see** is inside a kernel — occupancy, register
spills, shared-memory throughput. Several conclusions here were inferred from
ablation instead (the sweep is latency-bound; `pg_rice` has no registers to
spare). For that:

- **Apple**: Xcode's Metal debugger — capture the workload
  (`MTL_CAPTURE_ENABLED=1`, or attach Xcode), which gives per-encoder timing,
  occupancy and per-line shader cost. `xctrace record --template 'Metal System
  Trace'` works headlessly. MoltenVK routes through Metal, so these apply to a
  Vulkan build unchanged.
- **Intel**: `INTEL_DEBUG=cs` prints the chosen SIMD width and spill/fill counts
  (GPU_PLAN.md already used it to find 511:1444 spills in `sweep.comp`).
- **AMD / Nvidia**: Radeon GPU Profiler, Nsight Graphics.

## The autocorrelation's 25% is not cheaply reducible — three failures

With timestamps showing `pg_autoc` at 25.5% (a cost the double-float fix
introduced), two obvious routes were tried and **both were reverted**. Neither
failed for the reason it looked like it would.

**Wrong framing first.** "Use double-float only for the lags Levinson actually
cancels on" does not exist as an optimisation: `lambda = autoc[ord] - sum_j
A[j]*autoc[ord-1-j]` mixes every lag at O(autoc[0]), so the precision requirement
is uniform across lags. There is no subset to spare.

**Failure 1: relaxed accumulation.** `dfAdd` renormalises every step (3 of its 11
ops). Deferring that — twoSum the high parts, let the low parts accumulate in
fp32, normalise once at the end — keeps the accuracy (measured 3e-15, unchanged)
and makes the kernel **25% faster: 19.3 → 14.4 ms**. It makes the whole encoder
**1.4x slower**:

| stage | before | relaxed |
|---|---|---|
| autoc | 19.3 ms | **14.4 ms** |
| sweep | 36.2 ms | **94.1 ms** |
| total | 75.1 ms | 122.5 ms |

The slightly different autocorrelation shifts which orders the shortlist ranks
highest, the ranking moves toward higher orders, and the sweep's residual work is
linear in the order it prices. **A kernel-local speedup in the analysis is not a
program-level speedup, because the analysis decides how much work the sweep does.**
This is the trap to remember from the whole exercise; the first measurement of it
looked like a clean win because only `autoc` was timed.

**Failure 2: fewer lags.** The autocorrelation's cost is linear in the lag count,
which is `maxOrder + 1`, so capping the order should buy it back. It cannot be
done with a push constant: the lag loop bounds a private `acc[33]` array, and
making the bound dynamic stopped the array living in registers — **45.8 ms against
19.3 ms**, spilled to scratch. A specialization constant is the documented fix
(`sweep.comp` says so for its dead fp32 ladder) and it is **fragile here**: sizing
the array by it produced a kernel that ran in 2.3 ms with visibly wrong output
(+2.5% bytes) through MoltenVK, and even keeping the array fixed-size while
bounding only the loops changed the output at `maxOrder = 32`, where every code
path should be identical.

That last one is the useful finding: **this kernel's loop bounds must stay
compile-time.** A dynamic bound stops the unrolling, which changes instruction
selection in the double-float arithmetic, which changes the coefficients. So a
max-order knob is not free — it perturbs the numerics even where it should be a
no-op — and an order cap would have to be accepted as an output change rather than
a pure speed knob.

**Failure 3 (earlier, same shape): shared-memory coefficients in the sweep.**
0.0436 s against a 0.0431-0.0438 s baseline. The hardware broadcast of a
subgroup-uniform load is already free.

Sizes for the record, from the (spilled, so timing-invalid but size-valid) capped
runs on MLKDream: `maxOrder` 24 costs +0.26%, 16 costs +0.81%, 8 costs +1.9%. Even
if the speed were free, the exchange rate is poor next to the window/order
frontier above.

## Inside the sweep: it was the dependency chain, not the fold

The fold was the suspected bottleneck. It is not, and ablation said so before any
of it was restructured. Deliberately-wrong builds, timing only, MLKDream sweep:

| build | sweep |
|---|---|
| baseline | 43.8 ms |
| no LPC dot product | 24.0 ms |
| no bit-plane ballots, no partition fold | 22.1 ms |

So ~45% dot product, ~50% ballots+fold. Splitting the second further with the
`--pg-pcap` sweep (pcap 4 → 2 costs 12% of the sweep) puts the **fold at only
~15%** — about 7% of device time, and already exposed as a knob. Re-tuning it on
the current default confirms it is well placed: pcap 4 → 3 buys 3% of device time
for +0.005% size, 8 costs 44% more time for -0.03%.

**Three things measured as exactly nothing**, and each looked like the answer:

- **Coefficients in shared memory.** `sweep.comp` rejected a *private* array here
  (32 registers at SIMD32, a quarter of the budget), but shared memory is 128
  bytes per subgroup and had never been tried. 0.0436 s against a 0.0431-0.0438 s
  baseline. The hardware broadcast of a subgroup-uniform load really is free.
- **Removing all but one sample load** from the tap loop: 0.0411 s against
  0.0412 s. The samples are L1-resident and coalesced; the loads are free.
- **Removing the multiplies** but keeping the loads: no gain either.

Which is the whole finding. The tap loop costs 45% of the kernel while neither its
loads nor its multiplies cost anything, so what it costs is **latency**:
`sum += qc[j] * s[i-1-j]` serialises `ord` additions on one accumulator with no
instruction-level parallelism to hide the chain.

Integer addition is associative, and the narrow path's bound already rules out
overflow, so regrouping into independent accumulators is **bit-exact rather than a
trade** — verified byte-identical on six fixtures:

| accumulators | sweep |
|---|---|
| 1 (chain) | 41.2 ms |
| 4 | 34.5 ms |
| **8** | **32.8 ms** |

The wide (int64) path keeps four: an int64 accumulator costs two registers, and
eight of them spilled. **`pg_rice` was left alone entirely** — the same change
made it 2x slower (5.5 → 10.7 ms), because that kernel already carries five
NLEV-deep arrays plus a 4 KB shared table and has no registers to spare. Same
transformation, opposite sign, one kernel apart.

| fixture | `-P` | before | CPU default | speedup |
|---|---|---|---|---|
| music_20s | 0.064 s | 0.065 s | 0.198 s | 3.08x |
| MLKDream | 0.151 s | 0.164 s | 1.477 s | **9.79x** |
| master mix | 0.141 s | 0.145 s | 0.997 s | 7.05x |
| syn3m_mix | 0.140 s | 0.144 s | 0.940 s | 6.73x |

## Block size: the cap was worth raising, a device-side DP is not

The residual gap after the double-float fix was ~3.9%, of which the fixed
partition looked like ~1.9%. Measured before building anything, and the numbers
said something different from what was expected.

**What a variable partition is actually worth.** CPU, fixed 4096 against its own
DP ladder — and note CLAUDE.md trap 10 forbids the master mix here, since its 188
artificial splices flatter anything that prices block boundaries:

| fixture | fixed 4096 | variable DP | prize |
|---|---|---|---|
| music_10s | 1658877 | 1661262 | **-0.14%** |
| music_20s | 3418654 | 3422229 | **-0.10%** |
| album track | 26582623 | 26540185 | +0.16% |
| MLKDream | 27879091 | 27717435 | +0.58% |
| syn3m_mix | 14041475 | 13768845 | +1.94% |

On two real fixtures the CPU's *estimated* variable DP is **worse than just using
4096**, which is consistent with the 0.65% partition regret CLAUDE.md documents
for the estimator. Held at exact DP instead (`-e -c 8 -L 1`, fixed vs ladder) the
honest ceiling is **0.30% (music_10s) and 0.38% (music_20s)**.

**And a fixed size cannot capture it.** Sweeping the CPU across fixed sizes:

| fixture | b4096 | b8192 | b16384 | variable |
|---|---|---|---|---|
| music_10s | **1658877** | 1661637 | 1665603 | 1661262 |
| MLKDream | 27879091 | 27906511 | 27989814 | **27717435** |
| syn3m_mix | 14041475 | 13861552 | **13770499** | 13768845 |

So syn3m_mix's entire 1.9% is just "use 16384", while MLKDream's 0.58% needs a
genuine *mixture* — its DP picks 1024 (1160 frames), 2048 (1925), 4096 (1447),
8192 (516) and 16384 (396).

**Conclusion, and it is a decision not to build.** A device-side DP over
{4096, 8192, 16384} on a 4096 grid costs **~7x the analysis work** — the spans do
not share windows or autocorrelations, so work is `sum over sizes of
total_samples * S/step`. Seven times the analysis for a measured ceiling of
0.30-0.58% on real music would make `-P` slower than the CPU path it is currently
6x faster than. The full `{1024..16384}` ladder the CPU uses is 31x. Not worth
it, and this is the number to re-check before anyone tries.

**What was worth doing** is removing the arbitrary 4096 cap, which existed only
because `pg_autoc` staged the whole windowed block in 16 KB of shared memory. It
now walks the block in 2048-sample tiles with a 32-sample halo (lag l needs
x[i+l]), so each sample is read ~1.016 times and any size up to 16384 works.
Verified output-neutral at 4096 on four fixtures, and lossless at 256, 1024,
4096, 8192 and 16384 across all 18.

`-P` sizes with the cap raised:

| fixture | 4096 | 8192 | 16384 |
|---|---|---|---|
| music_10s | **1664955** | 1679087 | 1706676 |
| MLKDream | 28027341 | **27998911** | 28061945 |
| syn3m_mix | 14594422 | 14438215 | **14342723** |

Default stays 4096 — it is best on real music, and 16384 costs +0.4% there. It
costs ~10-15% more time as well, since chunk buffers are bounded by samples
rather than frames.

Note what the syn3m_mix column says about the *remaining* gap: at 4096 it is
+3.9% against the CPU and at 16384 it is +4.2%. The gap is flat in block size, so
what is left is search depth and analysis, not the partition.

## `precise` is a portability fix that must not be paid for twice (2.2x)

The `precise` qualifiers that made the double-float math survive Mesa ANV cost
**5x in `pg_autoc` on Apple, for byte-identical output**. The regression shipped
because that commit re-quoted the M4 Max stage table above instead of
re-measuring it — and it is the largest single number in this document's history,
so treat every stage table here as pinned to the commit that produced it.

Measured, device timestamps, before and after compiling the qualifiers out:

| fixture | autoc | device total | wall | bytes |
|---|---|---|---|---|
| master mix | 173.8 → **33.8 ms** | 213 → 86 ms | 0.260 → 0.120 s | identical |
| MLKDream | 110 → **15.0 ms** | 161 → 65 ms | 0.220 → 0.130 s | identical |

The output is byte-identical on Apple, which is what says the two builds are the
same computation there rather than a size/speed trade — and it also rules out the
failure-1 trap below, where a cheaper analysis silently hands the sweep more work.

**The split between the two kernels is not symmetric, and only one of them is
worth the money.** `precise` in `pg_levinson` costs ~8 ms on the master mix and is
worth keeping (without it: s24_2s +0.6%, syn3m_tonal +0.05%, master mix +149 B).
`precise` in `pg_autoc` costs 140 ms and buys nothing on a driver that does not
reassociate. So `pg_autoc.comp` is compiled twice — as-is, and with
`PG_NO_PRECISE` — and the host picks.

**It picks by asking the device, not by recognising it.** `shaders/pg_probe.comp`
runs the fast variant's own expressions on values whose fp32 results are inexact
and reports whether the error terms survived; zero means the compiler reassociated
and the safe build is bound instead. A vendor allowlist would be wrong the first
time anyone runs this on an untested driver, and it could not see
`MVK_CONFIG_FAST_MATH_ENABLED` being set by hand — which is exactly the case that
makes the fast variant unsafe on Apple. The probe is one dispatch at init and the
banner reports which build won (`df native` / `df via NoContraction`).

Intel is unaffected: it fails the probe and gets what 332ac61 gave it.

## The host was standing in the device's way (1.14-1.19x)

`runChunk` submitted and immediately waited, so the ~8 MB PCM `memcpy` for the
next chunk and the ~2 MB file write for the last one both happened with the device
idle — 109 ms of encode phase against 86 ms of device time on the master mix.

Two slots fix it, and the ordering is the whole design:

```
stage chunk i's PCM        <- overlaps chunk i-1 running
wait for chunk i-1
submit chunk i             <- device restarts before any host work
read back and write i-1    <- overlaps chunk i running
```

Waiting before submitting is deliberate. It means **no two chunks ever execute at
once**, so only the buffers the host touches (`B_PCM`, `B_OUT`) need duplicating
and every intermediate stays single-slot — +16 MB, against +85 MB for a second
full set. Everything the host reads out of a shared buffer (`B_TOTAL`, `B_FINFO`)
is read after the fence and before the next submit.

`FLACOUT_PG_NOPIPE=1` turns it off, which is how it was measured: master mix
1.14x, MLKDream 1.19x, syn3m_mix 1.14x, music_20s 1.01x (too few chunks to
amortise anything). Profiling forces it off — the serialised mode fences inside
the recording and there is one timestamp pool.

### Both together

Interleaved best-of-5, `-n` on every arm, separate pipeline caches per binary:

| fixture | 332ac61 | now | speedup |
|---|---|---|---|
| master mix | 0.269 s | **0.112 s** | **2.39x** |
| MLKDream | 0.231 s | 0.110 s | 2.10x |
| syn3m_mix | 0.262 s | 0.108 s | 2.43x |
| music_20s | 0.056 s | 0.041 s | 1.37x |
| music_10s | 0.045 s | 0.037 s | 1.21x |
| music_3s | 0.035 s | 0.032 s | 1.10x |

All 18 fixtures byte-identical to 332ac61 and verified lossless (`flac -t` plus
decoded-audio MD5). The short fixtures gain least because the ~20 ms fixed
startup, not the device, is what bounds them.

**The stage map has moved again** (master mix, timestamps): sweep 41.9%, autoc
34.6%, levinson ~5%, rice ~6%, everything else under 2%. The window/order
frontier in "Windows are worth more than orders" was tuned when a window cost 5x
what it now costs in `pg_autoc`, so it is stale in the direction of buying more
windows, and is the next thing to re-sweep.

## Batch mode: the fixed cost is a quarter of a corpus run

`flacoutcpp <in-dir> <out-dir>` walks the tree and encodes every `.flac` into the
mirrored path. It exists for `-P` specifically, because the per-invocation cost is
large next to a track:

```
27.6 ms   per invocation, measured on a 0.02-second file (best of 15)
 5.2 s    x188 tracks -- of a 21.6 s corpus run
```

Most of it is `vkCreateInstance` and the pipeline cache load, and **none of it
depends on the audio**: the device buffers are sized from (channels, bps, block
size, windows, chunk length) alone. So `shared_pure_gpu_encoder()` keeps one
context alive across files and rebuilds it only when the shape or the config
changes. Corpus: **21.6 s per-file, 17.1 s batched**, 188/188 outputs verified
with `flac -t`.

Storage is not the constraint and was checked rather than assumed: the corpus run
moves 164 MB/s read and 158 MB/s written, against 2.9 GB/s measured write
throughput on this machine, and the decode of the largest track finishes 24 ms
into a 142 ms encode.

**`-P` still does no frame reuse**, so it can emit a file larger than its input.
Measured over the corpus, paired by path: **9 of 188 tracks are larger, by 292 KiB
in total**, worst case +0.575%. A whole-file copy-through backstop would move the
corpus from -1.318% to -1.326%, i.e. it buys the *guarantee* and almost none of
the bytes. Frame-level splice is not reachable without giving up the fixed grid:
`-P`'s boundaries rarely coincide with the input's, and its parallel decoder does
not record input frame byte ranges.

## Build order

Each step ends somewhere testable. Do not skip to step 4.

1. **Bit-packer first, against CPU-chosen parameters.** Take the existing
   encoder's `BlockParams` output, upload it, and have the GPU do stages 9–14.
   Verify byte-identical against `FrameWriter`. This is the highest-risk
   component and it is testable in complete isolation, with `cmp` as the oracle.
2. **Stages 0–2 + 5** (de-interleave, wasted bits, decorrelation, quantize).
   Verify the quantized coefficients match the CPU's to the last bit — they
   will, if quantize is done in fp32 *and* the CPU comparison arm is also fp32;
   otherwise verify by size delta only.
3. **Stages 3–4** (window/autocorr/Levinson in fp32). Output now diverges from
   the CPU. Switch the oracle from `cmp` to decode-md5 plus a size table over
   the album corpus. Expect a small loss; measure it and write it down.
4. **Fuse into one command buffer per chunk, delete the host round trip.**
   This is where the win is; steps 1–3 will each look like a wash or worse in
   isolation.
5. **Then** consider re-adding search depth (device-side ranking from
   `out_err`, the top-K exact refine that `FP32RANK` measured as free, the
   variable-block DP).

## Traps specific to this work

- **`bench/check.sh` is not the net here** (see above). Build the losslessness +
  size-table harness *before* step 3, not after it diverges.
- **Every timing claim needs trap 9's hygiene** (`pgrep -x flacoutcpp`, steady
  load), and per trap 11 a `-G`-style path needs more reps than a CPU one.
- **A GPU run that fell back is indistinguishable from a GPU run.** `-G`
  already had to learn this — read the counter, not the exit code.
- **fp32 divergence is content-dependent.** `FP32RANK` found 24-bit real music
  the worst case (more sample bits eat more mantissa) and 16-bit real music
  perfect. Judge the fp32 loss on 24-bit content, and per trap 2 never on
  synthetics.
- **`FRAME_OVERHEAD` does not exist any more.** Anything the packer computes
  about frame size must call the same logic `FrameWriter::frame_bits` does, or
  the two drift and the cost model contract breaks silently.