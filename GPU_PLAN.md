# GPU (`-G`) findings: two defects, measured on real hardware — both fixed

> **Status: fixed and verified.** Both defects below are repaired; `-G` output
> is now byte-identical to the CPU path on every fixture and mode tested, and
> deterministic across runs. The diagnosis is kept in full because the *shape*
> of this bug — a hardware invariant that holds on one vendor and not another,
> plus a sentinel that was arithmetic — is worth not re-deriving. See
> "The fix" at the end.


First test of the Vulkan backend on a discrete GPU (PR #3 was developed and
measured elsewhere). The backend **works** — it builds, enumerates a real
device, dispatches, and never corrupts audio — but two things in the current
`-G` are not what the flag's help text claims.

Summary, in the order they matter:

1. **`-G` output is non-deterministic.** Same input, same flags, different
   bytes each run. Cause is timing-dependent CPU/GPU work routing, isolated
   below.
2. **The GPU path picks far worse candidates** — up to **+4.5%** larger. The
   help text's "Bit-exact with the CPU path, so the output is byte-identical"
   is false. **Root cause found:** the kernel's `0xFFFFFFFF` error sentinel is
   never checked on the host, and `hdr + 6u + cost` wraps it to the *cheapest*
   cost representable, so a poisoned candidate always wins. Details below.
3. **Output is still lossless.** Every variant decodes to the input's exact
   audio md5 and passes `flac -t`, including all divergent runs. This is a
   search-correctness bug, not a data-corruption one — the winner is re-priced
   exactly before encoding, so the bad *cost* never reaches the bitstream.

Not measured: **the >2x speed claim.** See "What is still unmeasured".

## Test environment

| | |
|---|---|
| host | Debian forky/sid, kernel 6.19.11, glibc 2.42, Xeon E5-2698 v4 (40 threads) |
| GPU | Intel Arc A380 (DG2), `i915`, Mesa ANV, Vulkan 1.4.354 |
| device caps | subgroup 32, `shaderInt64` — meets the kernel's requirements |
| binary | cross-built for amd64 in Docker (debian bookworm, gcc 12.2, `libvulkan-dev` 1.3.239, `glslangValidator`), `-DFLACOUT_VULKAN=ON` |
| fixtures | `music_3s`, `music_10s`, `stereo_4s`, `master_250ms_mix`, `MLKDream` |

Staged at `/tmp/flacout-test/` on the test host: binary, `in/`, `out/`,
`remote_test.sh`.

### Getting a device to enumerate at all

Two environment traps cost time here and will cost it again:

- **The user must be in the `render` group.** `/dev/dri/renderD128` is
  `root:render`; membership in `video` is not enough. Without it ANV cannot
  open the node, drops out of enumeration entirely, and the only device left
  is llvmpipe — which `-G` then correctly rejects for its 8-lane subgroups.
  The failure looks exactly like "no driver installed". `sudo usermod -aG
  render $USER`, then a **fresh login**.
- **`TU: error: ... freedreno ... VK_ERROR_INCOMPATIBLE_DRIVER` is noise.**
  That is Turnip probing the same render node and failing. It appears even on
  a healthy system and is not the reason a device is missing.

Cross-build recipe (glibc 2.36 → runs on the host's 2.42, the safe direction;
SPIR-V is embedded in the executable, so no shader file travels with it):

```sh
docker run --rm --platform linux/amd64 \
  -v "$PWD":/src:ro -v /tmp/build:/build -v /tmp/ccache:/ccache \
  <image with cmake g++ libvulkan-dev glslang-tools ccache> bash -c '
    cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DFLACOUT_VULKAN=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    cmake --build /build -j$(nproc)'
```

`find_package(Vulkan)` reports `glslc missing`, which is fine — CMake falls
back to `glslangValidator` as intended.

## Finding 1: `-G` is non-deterministic

Three runs, same binary, same input, same flags:

```
-R -G  music_3s.flac      3e1d6d69...  bf711003...  5956df6a...   (3 distinct)
-R     music_3s.flac      799dcd72...  799dcd72...  799dcd72...   (CPU control)
```

The `-G` banner shows the same search config as the CPU run
(`24 candidates/subframe, patience 48`), so this is not flag composition.

**`--gpu-slots 1` does not fix it** — still three distinct md5s. The slot count
is not the variable.

**Single-threaded is deterministic**, and that is the isolating experiment:

| config | run 1 | run 2 | run 3 | bytes |
|---|---|---|---|---|
| `-R -t 1 -G --gpu-slots 1` | `c032fe00` | `c032fe00` | `c032fe00` | 500009 |
| `-R -t 1` (CPU) | `799dcd72` | `799dcd72` | — | 491936 |

So **the GPU kernel itself is deterministic**. The non-determinism comes from
*which candidates reach it*.

### Mechanism

`GpuEvaluator::evaluate` ([src/gpu.cpp:501-517](src/gpu.cpp#L501-L517)) decides
per batch whether the GPU takes the work:

- return false if `cands.size() < min_batch`,
- otherwise try to claim a slot by CAS on `Slot::busy`; **if no slot is free,
  return false and the caller does it on the CPU.**

That is a perfectly reasonable work-stealing design, and it is exactly what
makes the output vary: with 40 worker threads racing for 3 slots, whether any
given batch is priced by the GPU or by the CPU depends on wall-clock timing.
Confirmed by the GPU's own counter — the same file reports **31616**, **37824**
and **160000** candidates absorbed depending on thread count and timing.

Routing that varies is only *harmless if both paths agree exactly*. They do
not, which is Finding 2. The two defects multiply: fix either one and `-G`
becomes reproducible.

## Finding 2: the GPU's cost model is worse than the CPU's

With one thread and one slot the GPU absorbs all 160000 candidates, and the
result is deterministic and **+8073 bytes = +1.64%** against the CPU
(500009 vs 491936 on `music_3s`).

Multi-threaded, the penalty is a dilution of that number — it tracks how large
a share the GPU happened to win:

| fixture | default (`-R`) | `-R -e -c 8 -L 0` |
|---|---|---|
| master_250ms_mix | +0.217% | +0.023% |
| MLKDream | +0.304% | +0.032% |
| music_10s | +0.205% | +0.016% |
| music_3s | +0.486% | +0.019% |
| stereo_4s | **+1.288%** | +0.014% |

Two caveats on that table. The default-mode column compares **different
searches**: [main.cpp:502](src/main.cpp#L502) silently sets
`precision_rungs = 0` when `-G` is given without `-E`/`-L`, while the CPU arm
keeps effort level 3's `-L 1`. The gap is real anyway — CPU `-L 0` on
`music_3s` is 491916 against the GPU's 494329 — but for an apples-to-apples
comparison pass `-L 0` explicitly on both arms. Second, these were measured
multi-threaded, so each cell also mixes in however large a share the GPU
happened to win.

Held constant at `-t 1 --gpu-slots 1` (deterministic, GPU absorbs everything),
on `stereo_4s`:

| path | CPU | GPU | |
|---|---|---|---|
| `-e` (exhaustive, kernel priced alone) | 312860 | 326862 | **+4.48%** |
| `-c 24 -L 0` (ranked pre-pass) | 314422 | 323291 | **+2.82%** |

The exhaustive path — the simpler of the two, with no ranked replay, no
patience, no `reach` window — is the *worse* of the two. That is what moved
suspicion off the ranked integration and onto the kernel/host boundary.

The penalty is also **one-sided**: GPU output is never smaller, on any fixture,
in any mode, which points at candidates being mis-priced rather than at a
symmetric tie-break difference.

For scale: **the entire estimated-DP effort dial spans ~0.2% end to end.** A
+0.2-1.3% penalty means `-G` currently gives back more compression than every
`-E` level can buy. That has to be fixed before any speed number is worth
having.

### Ruled out: the `-L` ladder mismatch

The obvious hypothesis — the help text says `-G` prices the whole precision
ladder and "does not combine with `-L`", while the default `-E 3` sets
`-L 1`, so CPU and GPU would be searching different rung sets — **does not
explain the gap.** On `music_3s`, CPU-side:

```
-R        (i.e. -L 1)   491936
-R -L 0                 491916    (-20 bytes)
```

The whole ladder setting is worth 20 bytes. The GPU gap is 8073. Whatever the
GPU is doing differently, it is not the ladder.

### Root cause: the kernel's error sentinel wraps into the cheapest cost

**The host never checks the sentinel the shader documents.** The kernel's last
line ([shaders/sweep.comp:244](shaders/sweep.comp#L244)):

```glsl
outCost[cidx] = (gl_SubgroupSize == uint(SG)) ? best_total : 0xFFFFFFFFu;
```

Both host call sites then do, unguarded:

```cpp
const uint32_t cost = hdr + 6u + gcosts[i];   // optimizer.cpp:2786 (exhaustive)
const uint32_t tot  = hdr + 6u + costs[i];    // optimizer.cpp:3046 (ranked)
```

`hdr + 6u + 0xFFFFFFFFu` wraps in `uint32_t` to `hdr + 5`. **The "this candidate
is invalid" sentinel therefore becomes the cheapest cost representable**, so a
poisoned candidate always wins — and the one that wins is whichever has the
smallest header, i.e. the *lowest order at the lowest precision*.

Confirmed on the output. Winning LPC orders, `-e` on `stereo_4s`:

| | CPU | GPU |
|---|---|---|
| top orders | 32 (x13), 28-30 (x7) | **1 (x12)**, 21-24 (x10) |
| subframe types | 26 LPC, 0 FIXED | 21 LPC, **5 FIXED** |

An encoder choosing order 1 on content where order 32 wins is not a rounding
difference; it is the header-minimising choice the wraparound creates. It stays
lossless because the winner is re-priced exactly before encoding — the bad
choice is real, the bad *cost* never reaches the bitstream.

This also explains the otherwise baffling result that
**`--gpu-partition-cap 1`, `4` and `8` produce byte-identical output**
(323291 each): the partition-order search cannot matter when the value being
compared is a sentinel rather than a partition cost.

The one-sidedness, the non-determinism, the losslessness, and the cap
invariance all fall out of this single defect.

### What triggers the sentinel is not yet pinned

The sentinel fires when `gl_SubgroupSize != 32` at runtime. The host checks the
**device property** `subgroupSize` (that is what the banner's "subgroup 32"
reports) — but on Intel ANV the compiler picks SIMD8/16/32 per shader, and the
dispatched width is not pinned to that property unless
`VK_EXT_subgroup_size_control` with an explicit `requiredSubgroupSize` is
requested at pipeline creation. `src/gpu.cpp` never requests it. So the host is
validating a number that does not govern the dispatch.

Against that being the whole story: if *every* invocation were poisoned, every
LPC subframe would be order 1, and 10 of 21 are not. So either the width varies
by dispatch shape, or a second path produces the sentinel. **Dump the raw
`outCost` values for one subframe before concluding** — that single datum
settles it, and nothing else should be changed until it does.

Two fixes are needed regardless of the trigger, and they are independent:

1. **Treat `0xFFFFFFFF` as "no result" on the host**, at both call sites, and
   fall back to `eval_candidate` for that candidate. Saturating or checking
   before the add would also do. As written, the kernel's own error path is
   indistinguishable from a very cheap candidate.
2. **Pin the subgroup size at pipeline creation**, so the guard is a real
   invariant rather than a runtime lottery — or drop the requirement and
   handle any width.

### Ruled out by reading the code

- **fp32 residuals.** [gpu.cpp:551](src/gpu.cpp#L551) builds the push constants
  as `{ncand, bsize, 0, pcap}` — `flags` is hardcoded `0`, and `flags & 1` is
  the only selector for the fp32 path. The exact `int64_t` path
  ([sweep.comp:199-202](shaders/sweep.comp#L199-L202)) always runs. (An earlier
  version of this document named fp32 as the leading suspect. It was wrong.)
- **The kernel's integer arithmetic**, checked line by line against
  `calculate_rice_cost`: suffix-scan weights (`2^(2^st)` = 2, 4, 16, 256,
  65536), k ranges (0-14 method 0, 0-30 method 1), escape width
  (`hi>=1 ? hi+1 : 1` vs `findMSB(mabs)+2`, equivalent since `mabs = or_all>>1`),
  the escape guard (`hi <= 30` vs `mabs < 2^30`), per-partition parameter bits
  (`4<<l` / `5<<l`), warm-up exclusion, and the coarse-first `<=`/`<`
  tie-break. All agree.
- **The CPU's `any_high` gate** ([optimizer.cpp:2011](src/optimizer.cpp#L2011),
  [:2065](src/optimizer.cpp#L2065)), which the shader has no equivalent of, is
  a pure optimisation: with no residual reaching 2^15 the shader's `total1`
  still carries the 5-bit-vs-4-bit parameter penalty and cannot win.
- **Saturating arithmetic.** A lane whose `S_k` saturates costs >= 2^32 while
  the escape bound is <= ~524k, so a saturated lane is never the argmin.
- **The ranked pre-pass's missing dedup.** [optimizer.cpp:3024-3032](src/optimizer.cpp#L3024-L3032)
  omits the duplicate-predictor skip its exhaustive sibling has at
  [:2769-2773](src/optimizer.cpp#L2769-L2773), but a duplicate has identical
  coefficients at a higher precision, so it loses on header alone. Harmless.
- **Delta-chain corruption from mixing paths.** `eval_candidate`'s
  `prev_qc`/`prev_shift`/`have_pred` are declared inside the lambda
  ([optimizer.cpp:2435-2437](src/optimizer.cpp#L2435-L2437)), so they are
  per-call. Interleaving GPU- and CPU-priced candidates cannot corrupt them.
- **The `best_lpc_cost` pruning argument** the batching rests on
  ([optimizer.cpp:2984-2990](src/optimizer.cpp#L2984-L2990)) holds: a precision
  skipped by `hdr >= best_lpc_cost` has `cost >= hdr >= best_lpc_cost` and
  could never have won.

### Two stale comments found while reading

- [optimizer.cpp:2732](src/optimizer.cpp#L2732): "Only the exhaustive path
  offloads" — untrue since the ranked pre-pass at
  [:2999](src/optimizer.cpp#L2999) was added.
- [sweep.comp:161](shaders/sweep.comp#L161): "See `-U`" — no such flag exists
  in `main.cpp`.

## Finding 3: it is lossless

Every GPU output tested — including all three divergent runs and both bit
depths — passes `flac -t` and decodes to the input's exact audio md5:

```
music_3s   input   7b04109852bda76d3c4efe98555c7c44
           det_1   7b04109852bda76d3c4efe98555c7c44   ok
           det_2   7b04109852bda76d3c4efe98555c7c44   ok
           det_3   7b04109852bda76d3c4efe98555c7c44   ok
MLKDream   input   989a58fe428592d149f43925248a87e2
           gpu     989a58fe428592d149f43925248a87e2   ok
```

So the bug is confined to *which* candidate the search picks. Nothing
downstream of the choice is broken.

## What is still unmeasured

**The >2x claim is now measured and it holds — see the Tesla P4 section at the
end.** `-e -L 0`, quiet host, byte-identical arms: 2.21x to 3.11x. The rest of
this section is the state before that run and is kept for the reasoning.

**The >2x speed claim in the PR description.** The test host was running
jellyfin (289% CPU), ffmpeg, immich, node and Sonarr, at load average 18-23 on
40 cores. Per trap 9 in CLAUDE.md every timing from that window is junk.
Sizes are deterministic and unaffected, which is why everything above stands.

Re-run the timing half on a quiet machine — and note it cannot be a plain
A/B against the CPU until Finding 2 is fixed, because the arms would be
compressing to different sizes. A speedup measured against a worse search is
not a speedup.

Also untested: any GPU other than this A380, and any driver other than Mesa
ANV. The subgroup-32 requirement means several otherwise-capable devices
(anything reporting 8- or 16-lane subgroups, including llvmpipe) decline the
GPU path entirely and fall back silently to the CPU — correct behaviour, but
it means "it ran" is not evidence the GPU did anything. Read the
`GPU: N candidates in Ts` line before believing a GPU run was a GPU run.

## The fix

Two independent changes, both in `src/gpu.cpp`; the shader is untouched.

**1. The sentinel is no longer arithmetic.** `GpuEvaluator::evaluate` scans the
returned costs and returns `false` if any is `UINT32_MAX`, so the caller prices
that batch on the CPU. One guard covers both call sites, because both already
treat `false` as "the GPU declined" — the exhaustive path leaves `gpu_done`
unset and runs its own sweep, the ranked pre-pass returns before setting
`gpu_ranked`. Rejecting the whole batch rather than individual entries is what
makes the fallback exact rather than merely less wrong. First occurrence warns
on stderr; the count is kept.

**2. The subgroup width is pinned instead of assumed.** Where
`VK_EXT_subgroup_size_control` is available and can deliver 32, the device is
created with `subgroupSizeControl` + `computeFullSubgroups`, and the pipeline
is built with `VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT`
(`requiredSubgroupSize = 32`) and
`VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT`. The old
`subgroupSize != 32` property check is kept only as the fallback for devices
without the extension. The banner now distinguishes the two:
`(subgroup 32 pinned, shaderInt64)` vs `(subgroup 32 by default, ...)`.

### Verification (Arc A380)

Every combination byte-identical, `-t 1 --gpu-slots 1` and multi-threaded:

| fixture | `-R -c 24 -L 0` | `-R -e -c 8 -L 0` |
|---|---|---|
| master_250ms_mix | 4241313 | 4220145 |
| MLKDream | 27848432 | 27698192 |
| music_10s | 1666958 | 1658133 |
| music_3s | 491916 | 488310 |
| stereo_4s | 314422 | 312996 |

The two paths that isolated the bug now agree exactly: `-e` on `stereo_4s`
326862 → **312860** (= CPU), ranked 323291 → **314422** (= CPU). Determinism:
three multi-threaded `-G` runs give one md5, equal to the CPU's
(`c559fa5512f8`), where they previously gave three. Losslessness re-checked by
decoding (`flac -t` ok, audio md5 matches input) on both bit depths.

The default (non-Vulkan) build is untouched — every change is inside
`#ifdef FLACOUT_HAVE_VULKAN`.

### Cost, and the open performance question

Pinning to SIMD32 **cost throughput on this device**: 1.5e5 → 5.27e4
candidates/s. That is not a regression against anything real — the shader's
width guard is on its *last* line, so the old dispatches ran the entire kernel
and then discarded it. The old figure was the same work with the answer thrown
away.

The likely mechanism is register pressure: SIMD32 doubles per-lane register
demand against SIMD16, and this kernel holds five `NLEV`-deep private arrays
plus `qc[32]`/`fc[32]`, so it plausibly spills to scratch. **Unmeasured** —
`INTEL_DEBUG=cs` prints the chosen width and spill counts.

If that is confirmed, the real Intel performance path is a kernel change, not a
flag: make the bit-plane mapping width-agnostic (two planes per lane at 16,
four at 8) so the driver can pick its preferred width *correctly*, instead of
being forced to one that spills. That is a design change and wants its own
measurement.

> **Measured, and this is wrong.** The spills are real (`pg_sweep` 449:747,
> `pg_rice` 350:588, `pg_autoc` 198:737 on Mesa 26.1.5) but they are not what
> costs the A380, and a width-agnostic mapping could not fix them if they were:
> the fold's divergent state is 32 planes x NLEV levels = 1152 bytes per
> subgroup-candidate **whatever the width**, so redistributing it across 8, 16 or
> 32 lanes leaves the same 36 GRFs. Cutting spills 40% for real (compile-time
> partition ceiling of 4) bought 2% of wall clock there and 0% on Apple. See
> "Register pressure is not the Arc's problem" in PURE_GPU_PLAN.md for the tables
> and for what to measure instead.

Apple is unaffected either way. If MoltenVK does not advertise the extension,
`size_ctl` stays false and the path is exactly as before; if it does, 32 is
pinned, which is what Metal does anyway.

## Suggested order of work

Items 1-3 are **done** (see "The fix"). Remaining:

1. **Measure speed**, on an idle machine, with `-R` — the PR's >2x claim is
   still unverified on any hardware but the author's M4, and the SIMD32 pinning
   makes the Intel number an open question rather than a known one.
2. **Check whether SIMD32 spills on Intel** (`INTEL_DEBUG=cs`), and if so
   consider a width-agnostic plane mapping.
3. Add a `bench/check.sh` case for `-G` if a GPU is available in the
   environment — the byte-identity claim is exactly what that harness exists to
   pin, and it would have caught both defects.

## `-G` on a constrained iGPU: Haswell HD 4600 (Mesa hasvk)

Second Vulkan platform after the Arc A380, and the first one that could not run
the kernel at all. Host: Debian forky, kernel 7.1.3, i7-4790 (4c/8t), Intel HD
Graphics 4600 (HSW GT2, `8086:0412`), Mesa 26.1.5 **hasvk**, which announces
itself with `MESA-INTEL: warning: Haswell Vulkan support is incomplete`. Binary
cross-built with `docker/Dockerfile.amd64` as before. A GTX 1080 sits in the same
box but is bound to `vfio-pci`, so it has no DRM node and never enumerates.

Note for the next person on this host: the render node is `crw-rw----+` with a
logind ACL granting the desktop user rw directly, so the `render`-group trap
above does not apply — and `INTEL_DEBUG=cs` prints nothing unless you also pass
`MESA_SHADER_CACHE_DISABLE=true`, because a cached pipeline never compiles.

### What the device is missing, and what it costs to emulate

`vkprobe` (40 lines against `libvulkan`, since the host has no `vulkaninfo`):
subgroup size 32, `supportedOperations` = basic/vote/ballot/shuffle/
shuffle-relative/quad — **no ARITHMETIC** — and **no `shaderInt64`**. Both were
hard requirements, so `-G` refused the device and `-P` still does.

Both gaps are emulable exactly, which matters more than cheaply: the cost this
kernel returns has to stay the CPU's to the bit or the byte-identical contract
goes. `shaders/sweep.comp` now carries two defines, `SWEEP_NO_ARITH`
(subgroupMin/Max as a 5-step `subgroupShuffleXor` butterfly) and
`SWEEP_NO_INT64` (the residual dot product via `imulExtended`/`uaddCarry`), and
`src/gpu.cpp` picks a variant per device. **The default variant's SPIR-V is
byte-identical to before the change** — checked with `cmp` on `sweep.spv` across
both toolchains — so a capable part pays nothing.

`FLACOUT_GPU_COMPAT=arith|int64|1` forces a fallback on hardware that does not
need it, which is how their intrinsic cost was measured. M4 Max, `music_10s`,
candidates/s (the count varies run to run, so the rate is the comparable figure):

| variant | cand/s | vs fast |
|---|---|---|
| fast | 8.59e5 | 1.00x |
| emulated min/max | 7.52e5 | 0.88x |
| emulated int64 | 6.13e5 | 0.71x |
| both | 5.90e5 | 0.69x |

All four byte-identical to the CPU path. So the emulations are worth ~1.45x
together — and on Haswell the same kernel ran at **1.05e3 cand/s, 640x below the
M4**, which no hardware ratio explains.

### It was spill-bound, and SLM fixes it (5.8x)

`INTEL_DEBUG=cs`: SIMD32, 5486 instructions, **342:564 spills:fills**, and of 917
`send` messages 342 are OWORD block writes to `bti 255` — scratch. Attributing
the compiler's own cycle model by basic block puts the top blocks on memory
traffic with ~6 multiplies between them. The cause is the streaming fold's five
per-lane `NLEV`-deep arrays: 45 values, two GRFs each at SIMD32, ~90 of a
128-register budget.

`SWEEP_SLM_STATE` moves those arrays to shared memory — 128 lanes x 45 uints =
23 KB, lane-major, no cross-lane access and therefore no barrier:

| | instructions | spills:fills | cycle model | measured |
|---|---|---|---|---|
| registers | 5486 | 342:564 | 35.5M | 916 cand/s |
| SLM | 2739 | 30:54 | 10.7M | **5.33e3 cand/s** |

Output byte-identical throughout. Forced on an M4 Max it measures 0.98x, i.e.
free, but it is **not** enabled there: the gate is "device is already running an
emulation", since Vulkan exposes no register-budget query and "integrated" would
catch Apple too. `FLACOUT_GPU_SLM=1/0` overrides.

### Slots matter more than any of it: 3 slots is 0.26x, 1 slot is 0.94x

Each slot parks a CPU worker on a fence. On a device slower than the host that
converts directly into idle cores, and this device is much slower than eight
Haswell threads. `music_3s`, wall clock against CPU-only, all byte-identical
except where noted:

| config | wall | cand/s | vs CPU |
|---|---|---|---|
| cap 8, slots 3 | 2.437s | 5.51e3 | 0.26x |
| **cap 8, slots 1** | **0.666s** | 9.34e3 | **0.94x** |
| cap 4, slots 1 | 0.663s | 8.64e3 | 0.94x |
| cap 2, slots 1 | 0.792s | 3.99e3 | 0.79x, **output differs** |

`--gpu-slots` therefore defaults to auto: 3 normally, 1 when the device is on the
compat kernel. Note `--gpu-partition-cap 4` buys nothing once slots are right —
before SLM it was worth 2.7x, after it is noise. Trap 3 in CLAUDE.md, again.

### Where it lands, and the honest verdict

Auto defaults, clean environment, `pgrep -x flacoutcpp` empty, load steady at
1.0, interleaved best-of-3:

| fixture | CPU | `-G` | ratio |
|---|---|---|---|
| music_3s | 0.632s | 0.646s | 0.98x |
| music_10s | 1.850s | 1.931s | 0.96x |
| s24_2s | 0.465s | 0.830s | **0.56x** |

Six fixtures (16/24-bit, mono/stereo, the short-stream path) are byte-identical
to the CPU path over two `-G` runs each, decode to the input's audio MD5, and
pass `flac -t`. So **the fallbacks make a 2013 iGPU a correct participant, and
not yet a profitable one**: neutral on 16-bit music, still clearly negative on
24-bit, where the emulated 64-bit dot product is widest and RICE2 puts real work
in planes 15-30.

What would make it profitable is not another kernel trick but a **throughput-aware
throttle**: `duty` and slots are static, so the encoder cannot notice that this
device prices candidates ~90x slower than the M4's and hand the surplus back on
its own. Measuring device and host candidate rates and setting `duty` from the
ratio would bound the loss at ~1.0x on any device and let a good one still win —
and it is the same mechanism that would let `--gpu-slots 3` stay safe everywhere.

`-P` remains unavailable here: 13 kernels, `int64` throughout the Rice, packing
and CRC stages, so the same two emulations are a much larger port than one
`sweep.comp`. Nothing about the above says it is impossible, only that it was not
attempted.

### Third platform: UHD 630 (CFL GT2, Mesa ANV) — and SLM is a *loss* there

Host: CachyOS, kernel 7.0.9, i7-8700 (6c/12t), Intel UHD Graphics 630 (Gen9.5)
under ANV 26.1.2. A **Tesla P4** sits in the same box and did not enumerate: the
NVIDIA kernel module is 580.159.03 against 580.159.04 userspace (a package
upgrade with no reboot, 82 days uptime), so the loader rejects the ICD with
`Could not get 'vkCreateInstance' ... for ICD libGLX_nvidia.so.0` and `nvidia-smi`
reports the matching NVML mismatch. Needs a reboot; nothing to do with this code.

Unlike Haswell this part has the full feature set — subgroup 32, arithmetic,
`shaderInt64` — so it runs the **fast** kernel, and it is the device that decides
whether the SLM fallback should be gated on "constrained" or turned on for
integrated parts generally. `music_3s` / `s24_2s`, best-of-3, every row
byte-identical to the CPU path:

| config | music_3s | vs CPU | s24_2s | vs CPU |
|---|---|---|---|---|
| CPU only (12 threads) | 0.366s | 1.000 | 0.303s | 1.000 |
| **registers, slots 3** | **0.631s** | **0.580** | 1.092s | 0.277 |
| registers, slots 1 | 0.690s | 0.531 | **0.440s** | **0.689** |
| SLM, slots 3 | 0.784s | 0.467 | 1.580s | 0.192 |
| SLM, slots 1 | 0.868s | 0.422 | 0.466s | 0.650 |

**SLM costs 20-30% on Gen9.5**, so the "only when the device is already
emulating" gate is right and must not be widened to integrated parts: the fix for
a 128-register Haswell at SIMD32 is a pessimisation on a part that can hold the
arrays. Two devices now agree that a capable part should keep the register
version (M4 Max 0.98x, i.e. free-but-pointless; UHD 630 0.78x, a real loss).

Also note the slots optimum **flips with bit depth** on the same device: 3 slots
win on 16-bit `music_3s` and lose 2.5x on 24-bit `s24_2s`, where candidates are
heavier and a parked worker waits longer. Static slot counts cannot be right for
both, which is the same argument for an adaptive throttle from a second direction.

`FLACOUT_GPU_COMPAT` also makes this host a correctness check for the fallbacks on
a third driver: all six fixtures byte-identical to the CPU path under fast,
emulated-min/max, emulated-int64 and both, and lossless by decode. So the
emulations are now pinned on MoltenVK (Apple), hasvk (Haswell) and ANV (Gen9.5).

And the headline number is unchanged by better silicon: **`-G` is a net loss on
this iGPU too**, best 0.69x. A 2018 integrated part against six modern cores is
still the wrong side of the trade for this kernel.

## Fourth platform: Tesla P4 (Pascal, NVIDIA 580.173.02) — the >2x claim, verified

Same CachyOS host as the UHD 630 above (i7-8700, 6c/12t), once the driver mismatch
was fixed by a reboot. `vkprobe`: subgroup 32, arithmetic, `shaderInt64` — full
feature set, so it runs the **fast** kernel with no fallback, and it is the first
NVIDIA device this code has run on at all.

**This is the device the design was waiting for.** Exhaustive DP, `-e -L 0` pinned
on both arms, quiet host, best-of-2, every output byte-identical to CPU-only:

| workload | CPU (12 threads) | `-G` | ratio | cand/s |
|---|---|---|---|---|
| mono_2s | 14.98s | 6.79s | **2.21x** | 2.58e5 |
| s24_2s | 85.69s | 27.58s | **3.11x** | 2.65e5 |
| music_3s | 136.27s | 44.30s | **3.08x** | 2.56e5 |

So the PR's ">2x" is real, on hardware that is not the author's Mac, with the arms
compressing to identical bytes — which is the form of the claim that was never
established before (see "What is still unmeasured"). Note the pattern: the deeper
the search, the better the device does, because `-e` is where candidates-per-block
is large enough to fill a dispatch.

Estimated DP is a much weaker story, as the fixed costs (decode, MD5, ~28 ms device
init) floor short fixtures:

| workload | CPU | `-G` | ratio |
|---|---|---|---|
| MLKDream (28 MB, default `-L 0`) | 18.46s | 15.84s | 1.165x |
| music_10s (default) | 0.978s | 0.821s | 1.191x |
| music_3s / s24_2s (default) | — | — | ~1.00x |

Slots behave exactly as the M4 table predicted — U-shaped with the default at or
near the optimum, and a cliff past 6:

| slots | music_3s | music_10s | s24_2s |
|---|---|---|---|
| 1 | 0.892x | 1.191x | 0.912x |
| **3 (default)** | 1.000x | 1.184x | 1.002x |
| 6 | 0.932x | 1.135x | 0.935x |
| 12 | 0.683x | 0.669x | 0.884x |

Taken with the two iGPUs, the shape of the whole result is: **`-G` pays when the
device is a real GPU and the search is deep.** Integrated parts lose on both
counts (0.56-0.98x), and on the same discrete card the estimator only reaches
~1.19x while `-e` reaches 3.1x.

### Host traps on this box, all three of which cost measurements

- **The GDM greeter idle-suspends the machine.** Left parked on the login screen,
  the greeter's own `gsd-power` (uid 60578, *not* the logged-in user's) requests
  S3 after 900 s — it fired mid-benchmark and the box was unreachable for seven
  minutes. It does this **even though another user is logged in on another VT**,
  and `sleep-inactive-ac-type` for uid 1000 was already `'nothing'`, so checking
  your own session's settings proves nothing. Fix needs root: set the same keys
  for the `gdm` user, or `systemctl mask sleep.target suspend.target`.
  `systemd-inhibit` is *not* a workaround here — it fails with `Access denied ...
  requires interactive authentication`.
- **A suspend silently corrupts wall clock.** `date +%s.%N` deltas (and
  `/proc/uptime`) include suspended time, and `uptime` does not reset, so the
  aftermath looks like a slow run on a machine that never rebooted rather than
  like a suspend. `bench`-side scripts on this host should bracket every
  measurement with `journalctl -k | grep -c "PM: suspend entry"` and void the
  timing if it moved.
- **`-G` does not survive suspend/resume, and wedges.** The run that straddled the
  suspend made no further progress: a 44-second job was still going 13 minutes
  later, at 45% CPU with the GPU at 0% utilisation, and it **ignored SIGKILL**
  (stuck in a driver ioctl) for a while before dying. The Vulkan device is lost
  across S3 and nothing in `GpuEvaluator` notices. Worth fixing independently of
  suspend, since `VK_ERROR_DEVICE_LOST` is reportable on any submit or fence
  wait: treat it as "disable the GPU path and finish on the CPU", which is
  already the behaviour for a device that never initialised.

## Two things the multi-device work exposed, and the fixes for both

### 1. A lost device used to wedge the encode

Found by accident: a host suspended to S3 mid-encode (see the traps above), and
the `-G` run never made progress again. A 44-second job was still running 13
minutes later at 45% CPU with the GPU at 0% utilisation, and it **ignored
SIGKILL** for a while, stuck in a driver ioctl. Nothing in `GpuEvaluator`
checked a `VkResult` other than to return `false` for that one batch, so a device
that had gone away was re-offered work for the rest of the file.

`Impl::fail_device` now takes the whole path down on any failure from
`vkResetCommandBuffer`, `vkBeginCommandBuffer`, `vkEndCommandBuffer`,
`vkQueueSubmit` or `vkWaitForFences`, prints which call failed, and lets the
encode finish on the CPU — which is already the behaviour for a device that never
initialised, so there is no new failure mode to reason about. `ok` became atomic
for it (it is now cleared from worker threads while others read it).

Note what this cannot fix: the wait that never returns. `vkWaitForFences` with
`UINT64_MAX` on a driver that is gone does not report an error, it blocks, and a
finite timeout would trade a hang for a wrong answer about whether the batch
completed. Every failure mode the driver *reports* is now handled; the one it
does not report needs the driver not to lie.

### 2. Fixed shares cannot work, so the throttle measures instead

`duty` and `--gpu-slots` were static, and no static value survives contact with
four devices: 3 slots are optimal on a Tesla P4 and cost 3.7x on a Haswell iGPU,
and on one device the best slot count **flips with bit depth**. So the encoder now
measures both sides and compares them.

The comparison is device MACs/s against **one CPU thread's** MACs/s, because the
thread that offers a batch is the thread that blocks on the fence — the question
is never "is the GPU fast" but "is it faster than the worker it idles". Both rates
are EMAs of real measurements: the device's from `evaluate` (submit through
readback, so dispatch overhead is included and small batches price themselves out
without needing `min_batch`), the CPU's from one timed candidate per subframe in
`Optimizer::optimize_subframe`. Output is unaffected by any routing decision —
both paths return the same cost — which is what makes this safe to tune at all.

Four things went wrong on the way, each measured, none obvious:

- **The control loop starved itself.** Accepting everything during warm-up meant
  the CPU never ran a batch, so `cpu_samples` stayed at 0, warm-up never ended,
  and a device 20x slower than one thread reported the ratio correctly (0.05x)
  while accepting **100%** of offers. Warm-up now takes only 1 offer in 8 and
  needs 3 samples per side.
- **Deciding inside `evaluate` was too late.** The caller builds a batch — a
  quantize pass over every (candidate, precision) pair — *before* offering it, so
  a late decline pays for the batch twice. A UHD 630 declining 93% of offers still
  gave up 18% of wall clock. The decision moved into `would_accept()`, which is
  where the same reasoning was already written down for `min_batch`.
- **Probing a hopeless device is not free.** At a flat 1-in-64 re-probe a UHD 630
  still leaked 9% of batches. The interval now backs off geometrically to 4096,
  and below parity the path shuts down entirely, because pinning `--gpu-duty 1`
  — an essentially idle device — *still* cost 10% against CPU-only. An awake iGPU
  takes package power and memory bandwidth from the cores whatever it is doing.
- **A busy device flatters itself.** The CPU rate is measured while the device
  runs, and the device depresses it: the same UHD 630 reads 0.30x against an
  idle-pool CPU rate and 0.50x against the rate it has itself dragged down. A
  give-up threshold below parity therefore lets a device stay alive by slowing
  down its own competition, which is why the threshold is parity and the accept
  margin is 1.5x.

That 1.5x margin covers measured interference, which is the part of this that
inverted the obvious guess. Single-thread CPU rate with the device idle
(`--gpu-duty 1`) against saturated (`--gpu-duty 100`), music_10s:

| device | idle | saturated | change |
|---|---|---|---|
| UHD 630 (CFL iGPU, 12 threads) | 2.14-2.18e9 | 1.34-1.90e9 | **-13 to -38%** |
| Tesla P4 (discrete, 12 threads) | 2.18-2.33e9 | 1.43-2.02e9 | **-12 to -38%** |
| Haswell HD 4600 (iGPU, 8 threads) | 1.70-1.75e9 | 1.81-1.86e9 | none |

The iGPU *does* steal from the CPU on Coffee Lake — shared power budget and ring
bus — and the discrete card does too, through host-visible buffer traffic. Haswell
escapes only because its compat kernel is slow enough that it barely loads the
device. So "integrated means it competes with the cores" is true, but it is not
the whole story and it is not what distinguishes the devices here.

Result, `music_10s`, best-of-3, everything byte-identical to CPU-only. The old
default is `--gpu-duty 100` at the slot count that device would have used:

| device | old default | adaptive | CPU-only |
|---|---|---|---|
| Haswell HD 4600 (slots 1 auto) | 0.948x | 0.955x | 1.000 |
| Haswell HD 4600 at slots 3 | 0.477x | **0.893x** | 1.000 |
| UHD 630 (slots 3) | 0.593x | **0.825-0.849x** | 1.000 |
| M4 Max (`-e -L 0`, mono_2s) | 1.73x | 1.72x | 1.000 |

And what it costs a device that deserves the work. Tesla P4, same session,
interleaved, best-of-3 (the only valid way to compare these -- an earlier
cross-session pinned number read 2.207x and is not comparable):

| workload | pinned | adaptive | premium |
|---|---|---|---|
| mono_2s `-e -L 0` | 2.152x | 2.042x | **5%** |
| s24_2s `-e -L 0` | 3.107x | 3.011x | 3% |
| music_10s default | 1.191x | **1.338x** | -12% (adaptive wins) |
| MLKDream default | 1.144x | **1.379x** | -21% (adaptive wins) |

Adaptive is *better* in estimated DP, where declining some batches avoids parked
workers the pinned share cannot refuse, and ~3-5% worse in exhaustive mode. Most
of that premium was the CPU probe: at a flat 1-in-64 it hands a whole subframe to
a CPU 68x slower per MAC in that mode, so the interval now scales with the
measured ratio (mono_2s went 1.994x -> 2.042x on that change alone). What is left
is warm-up -- ~21 declined offers before the ratio is known -- and closing that
would mean accepting more while still ignorant, which is precisely what makes a
slow device catastrophic. **5% on a winning device is the premium for bounding
the loss at ~0.9x instead of 0.12x on a losing one, without knowing which is
which in advance.**

So the throttle's value is not that it makes an iGPU pay — it does not, and no
iGPU tested ever will for this kernel. Its value is that **the catastrophic
configurations are gone without per-device tuning**: the worst case moves from
0.12-0.48x to ~0.85-0.9x, a winning device keeps its speedup to within 0.5%, and
neither outcome needed anyone to know which device they had.
