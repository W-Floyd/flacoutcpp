/**
 * @file flacoutcpp.hpp
 * @brief Public C++ library API for flacoutcpp.
 *
 * flacoutcpp is a FLAC re-encoder that achieves better compression than
 * `flac --best` by exhaustively searching the LPC parameter space
 * (26 standard apodization windows × 32 orders × 8 quantization precisions ×
 * 4 stereo modes per block) and selecting the globally optimal variable block-size
 * partition via dynamic programming.
 *
 * ### Typical usage
 * @code
 * #include "flacoutcpp.hpp"
 *
 * flacoutcpp::Config cfg;
 * cfg.max_threads = 0;          // use all logical CPUs
 * cfg.copy_metadata = true;     // preserve VORBIS_COMMENT, PICTURE, …
 *
 * bool ok = flacoutcpp::optimise("input.flac", "output.flac", cfg);
 * @endcode
 *
 * The library is header-only at this level; link against `libflacout_lib`.
 */

#ifndef FLACOUTCPP_HPP
#define FLACOUTCPP_HPP

#include <string>
#include <vector>
#include "optimizer.hpp"

/// Top-level namespace for the flacoutcpp library.
namespace flacoutcpp {

/// @cond
struct Config;
/// @endcond

/**
 * @brief Configuration for a single optimise-and-encode run.
 *
 * All fields have sensible defaults; zero-initialising the struct is valid.
 */
struct Config {
    /**
     * @brief Copy non-audio metadata blocks to the output file.
     *
     * When @c true (default), VORBIS_COMMENT, PICTURE, PADDING, and other
     * metadata blocks present in the source file are replicated verbatim in
     * the output.  STREAMINFO is always rewritten from scratch.
     */
    bool copy_metadata = true;

    /**
     * @brief Apodization windows to test during LPC optimisation.
     *
     * An empty vector (the default) enables all 26 standard windows under
     * exhaustive mode, which yields maximum compression at the cost of higher
     * CPU usage.  Supply a smaller set to trade compression for speed.
     * Experimental windows (WindowType values from EXPERIMENTAL_BEGIN on)
     * are used only when named here explicitly.
     *
     * @see WindowType for the list of available windows.
     */
    std::vector<WindowType> windows;

    /**
     * @brief DP block-size ladder (`-b`).
     *
     * Empty (the default) uses the built-in `{1024, 2048, 4096, 8192, 16384}`.
     * Every entry must be a multiple of 16 and within [16, 65520]: the DP
     * places nodes every GCD-of-the-ladder samples and each frame spans
     * exactly one candidate, so a size that is not a multiple of that step
     * could never land on a node. 65535 is therefore unreachable — it is odd;
     * 65520 is the largest usable size.
     */
    std::vector<uint32_t> dp_candidates;

    /**
     * @brief Maximum number of worker threads.
     *
     * Set to @c 0 (default) to use all logical CPUs reported by the OS.
     * The DP block-evaluation phase is embarrassingly parallel and scales
     * linearly with thread count.
     */
    unsigned max_threads = 0;

    /**
     * @brief Exact-search mode: price every block-partitioning choice exactly.
     *
     * When true, every (position, block size) pair in the partitioning DP is
     * fully encoded rather than estimated from granule autocorrelations, all
     * four stereo modes are fully evaluated per block, and the default window
     * set widens to all 26 standard windows. Can be extremely slow.
     *
     * Orthogonal to @ref max_candidates: this option decides how *blocks* are
     * priced; @c max_candidates decides how deep the per-subframe LPC search
     * goes within whatever blocks get encoded.
     */
    bool exhaustive = false;

    /**
     * @brief Ranked-search budget: (window, order) pairs evaluated per subframe.
     *
     * Levinson-Durbin already computes the prediction error at every order as
     * a by-product; the ranking uses it to estimate each (window, order)
     * pair's cost up front, and only the best @c max_candidates of them are
     * fully evaluated (each across the whole precision ladder). @c 0 means no
     * limit — every pair is fully evaluated, the classic exhaustive sweep.
     *
     * The default of 24 is effort level 3 (see @ref apply_effort): the old
     * default of 8 with a full precision ladder measured *off* the frontier in
     * both directions — level 3 is faster and smaller. Costs ~0.03% size on
     * real music for ~1.75x speed over the unlimited sweep. Known weak spot: near-white high-bps content, where
     * the Levinson errors are almost flat across orders and the ranking is
     * noise — 24-bit whitenoise fixtures grew 2-2.8% at 8 and needed 32 for
     * parity. Real music does not behave that way.
     */
    unsigned max_candidates = 24;

    /**
     * @brief Consecutive non-improving candidates before the ranked scan stops.
     *
     * @ref max_candidates alone cuts the ranked list at a fixed depth, which
     * assumes the ranking is right about what lies below the cut. It often is
     * not: measured over every candidate, the winner's rank is heavy-tailed —
     * rank 0 takes 56% of subframes on a 24-bit pure sine but the tail reaches
     * rank 59, and on real music only ~51% of winners fall inside rank 7. The
     * misses are not near-ties with rank 0, so no widening around the top can
     * reach them.
     *
     * Patience uses the exact costs the scan is already computing as its own
     * stopping signal: descend the ranked list and keep going while it is
     * still producing improvements, stopping only after this many consecutive
     * candidates fail to beat the best so far. @ref max_candidates becomes a
     * floor rather than a ceiling. Subframes the ranking ordered well stop
     * near the floor; the ones it ordered badly pay for the tail they need.
     *
     * @c -1 (default) means 2 x @ref max_candidates, which costs ~1.34x time
     * for ~59% of the unlimited sweep's remaining compression on real music.
     * @c 0 disables patience and restores the plain top-N cut. Irrelevant when
     * @ref max_candidates is 0, since every candidate is evaluated anyway.
     */
    int patience = -1;

    /**
     * @brief Precision rungs actually encoded per candidate (0 = all 8).
     *
     * Each surviving (window, order) candidate is priced at LPC coefficient
     * precisions 8..15, and every rung costs a full residual + Rice pass —
     * the largest single multiplier on the cost of a candidate, and never
     * pruned in practice.
     *
     * The ladder can be predicted instead of run. For a predictor @c c the
     * windowed residual energy is `E_a + (c-a)'R(c-a)`, with @c a the
     * Levinson solution and @c R the windowed autocorrelation — both already
     * computed. That makes a rung an O(order²) quadratic form rather than an
     * O(block × order) pass, so all 8 can be scored for a few percent of one
     * rung and only the best few encoded for real.
     *
     * Measured in the encoder, as a fraction of file size given up against
     * the full ladder at the same @ref max_candidates: 1 rung +0.019%,
     * **2 rungs +0.009%**, 3 rungs +0.005%, alike on 16- and 24-bit material.
     * The speedup is ~1.3x, not the ~4x the rung count suggests: the
     * precision-delta path already makes 7 of the 8 rungs cheap, and skipping
     * rungs breaks that chain, so what is actually saved is the Rice pass.
     *
     * The point is the *frontier*, not the speed. Spending 1.3x on a deeper
     * @ref max_candidates beats plain @ref max_candidates at equal time
     * everywhere the two overlap: on the 188-track master mix, `-c 24 -L 2`
     * is 0.058% smaller than the `-c 8` default at 1.03x its speed, where
     * plain `-c 12` buys only 0.031% and costs 1.16x. Full-ladder search
     * still wins on size alone at unlimited time, which is why this is off by
     * default. See PRECISION_LADDER_PLAN.md.
     *
     * Defaults to 1 as part of effort level 3; @c 0 encodes every rung, which
     * is what @ref exhaustive resets it to and what the CLI restores whenever
     * @c -e is given without an explicit @c -L.
     */
    unsigned precision_rungs = 1;

    /**
     * @brief Coefficient-lattice refinement sweeps (`-Q`, experimental).
     *
     * The search quantizes LPC coefficients by one fixed rule — round the
     * Levinson solution with error feedback — and never revisits the integer
     * vector it lands on. That rule minimizes the quantization error's
     * quadratic form, not the Rice cost actually being paid, so a neighbouring
     * lattice point can be cheaper.
     *
     * When non-zero, the winning candidate of each subframe is refined by
     * coordinate descent: each tap is tried at +-1 and any perturbation that
     * strictly lowers the exact cost is adopted, repeating until a full sweep
     * finds nothing or this many sweeps have run. It can only shrink a
     * subframe, never grow one.
     *
     * @c 0 (default) disables it, which is bit-exact with the pre-existing
     * behaviour. Costs 2 x order residual+Rice passes per sweep per subframe.
     */
    unsigned lattice_sweeps = 0;

    /**
     * @brief Evaluate LPC candidates on the GPU (`-G`, experimental).
     *
     * Offloads the per-candidate Rice cost to a Vulkan compute shader that
     * reproduces Optimizer::calculate_rice_cost exactly, so the encoded file
     * is byte-identical to a CPU-only run and bench/check.sh still applies.
     * Requires a build configured with -DFLACOUT_VULKAN=ON and a device with
     * a 32-lane subgroup and shaderInt64; otherwise the run reports why and
     * falls back to the CPU.
     */
    bool use_gpu = false;

    /**
     * @brief Smallest candidate batch worth sending to the GPU (`--gpu-min-batch`).
     *
     * Below this the subframe is encoded on the CPU, because a dispatch costs
     * more than it saves. 0 forces every batch onto the GPU — the setting to
     * use when measuring where a new device's crossover actually is.
     */
    unsigned gpu_min_batch = 0;

    /**
     * @brief Cap on the GPU's partition-order search (`--gpu-partition-cap`).
     *
     * 8 is the format maximum and the default: the GPU then reproduces
     * calculate_rice_cost exactly and the output is byte-identical to a
     * CPU-only run. Lower values make the kernel markedly faster (its cost is
     * dominated by partition closes, of which there are 2^(P+1)-1) at the
     * price of possibly ranking a candidate wrong. The winner is still
     * re-priced exactly on the CPU, so a cap can never mis-state a cost or
     * break losslessness — it can only pick a slightly worse candidate.
     */
    unsigned gpu_partition_cap = 8;

    /**
     * @brief Dispatches in flight (`--gpu-slots`, 1-16).
     *
     * 0 = auto, which is 3 on a device running the fast kernel and **1** on a
     * constrained one. Each slot parks a CPU worker on a fence, so a device
     * slower than the host turns extra slots into idle cores: on a Haswell iGPU
     * three slots cost 3.7x against one, at 8 CPU threads.
     *
     * See GpuEvaluator::set_slots.
     */
    unsigned gpu_slots = 0;

    /**
     * @brief Percent of subframes the GPU accepts (`--gpu-duty`, 1-100), or 0
     *        for the adaptive throttle.
     *
     * 0 (the default) lets GpuEvaluator compare its own measured throughput
     * against one CPU thread's and take work only while it is winning. No fixed
     * share can be right: measured, the best value moves with the device (a
     * Haswell iGPU wants almost none, a Tesla P4 wants all of it) and even with
     * the bit depth of the input on one device.
     */
    unsigned gpu_duty = 0;

    /**
     * @brief Fully GPU-resident encoder (`-P`). A different encoder, not a
     *        backend for the one above.
     *
     * Every stage from PCM to packed frame bytes runs as a compute dispatch: the
     * host uploads samples, waits, and writes out the result. Nothing else here
     * applies when this is set — no variable-block DP, no ranked search, no
     * patience, no precision ladder, no frame reuse, and the analysis runs in
     * fp32 rather than double.
     *
     * Output is lossless (the residual is exact integer arithmetic once the
     * coefficients are integers) but is **not** bit-identical to the CPU path,
     * so bench/check.sh cannot gate it. Verify by decoding. See PURE_GPU_PLAN.md.
     */
    bool pure_gpu = false;

    /// Fixed frame size for `-P`. Multiple of 256, in [256, 16384] — the 4096 cap
    /// this comment used to state was lifted; 4096 is still the default because it
    /// measured best on real music.
    uint32_t pg_block_size = 4096;

    /// LPC precisions swept per (window, order) under `-P`. Empty → {15}.
    std::vector<int> pg_precisions;

    /// Orders swept per (block, signal, window) under `-P`, out of 32, ranked by
    /// Levinson error. Trades against the window count rather than standing
    /// alone -- 2 is the default only because the shortlist is 10 windows wide;
    /// at 6 windows it costs size. See PureGpuEncoder::Config::orders.
    uint32_t pg_orders = 2;

    /// Sweep-side Rice partition-order cap under `-P` (1-8). Ranking only; see
    /// PureGpuEncoder::Config::partition_cap. 4 measured 3.4x for +0.029%.
    uint32_t pg_partition_cap = 4;

    /// Frames per device chunk under `-P`; bounds peak device memory.
    uint32_t pg_blocks_per_chunk = 256;

    /**
     * @brief Effort level 0-9: one dial across the measured size/time frontier.
     *
     * @ref max_candidates and @ref precision_rungs are not independent — they
     * buy the same thing (a better-chosen predictor) at different exchange
     * rates, and which mix is efficient shifts with the budget. Measured on
     * the 188-track master mix, the frontier is "spend everything on depth,
     * one rung" until depth saturates around @c max_candidates 128 (past that,
     * patience has exhausted the 256-candidate pool and extra depth is
     * byte-identical), and only then does buying rungs pay.
     *
     * This dial walks that frontier so callers do not have to. It is not a new
     * search mode: each level is exactly a (@ref max_candidates,
     * @ref precision_rungs) pair, with @ref patience left at its 2x default.
     *
     * @c -1 (default) means no effort level: the individual knobs stand as
     * they are, and behaviour is unchanged.
     *
     * @see apply_effort
     */
    int effort = -1;

    /**
     * @brief Adaptive per-subframe window selection (experimental).
     *
     * Estimated-DP modes only: instead of the fixed 4-window shortlist, each
     * encoded block picks a 4-window set from its cached granule statistics
     * (stationarity, transient position, spectral tilt) at identical analysis
     * cost. Incompatible with @ref exhaustive and an explicit @ref windows
     * list, both of which define their own window sets — the CLI clears this
     * default rather than erroring when either is present, and only a
     * *explicit* @c -a is treated as a contradiction there.
     *
     * On by default as part of effort level 3: swept as a third frontier
     * dimension it appears in 16 of 21 efficient points, and the album that
     * historically regressed under it is -0.219% with no track worse.
     */
    bool adaptive_windows = true;

    /**
     * @brief Reuse input frames that beat the new encoding (default: on).
     *
     * The input file arrives already partitioned into frames whose exact
     * compressed sizes are known. Wherever the input's frames tile a span
     * of the chosen partition in fewer bytes than the re-encoded frames,
     * the input frames are spliced into the output (payload verbatim,
     * header rewritten to this stream's conventions, CRCs recomputed);
     * under exact-DP mode they also compete inside the partitioning DP as
     * exact-cost edges.
     *
     * Splicing alone is not the guarantee. It picks the cheaper side per
     * segment, but the input side is priced as *rewritten* frames, and
     * rewriting emits a variable-blocksize sample number where a
     * fixed-blocksize input carried a frame number — 1-2 UTF-8 bytes more per
     * frame. A segment also falls back to the re-encode regardless of size if
     * any of its input frames fails to rewrite. So splicing bounds the output
     * against what the search could emit, not against the input.
     *
     * The whole-file fallback is what closes that: if the finished file is
     * still larger, the input ships instead — copied verbatim under
     * @ref copy_metadata, or, under @c -n, as the input's audio frames
     * verbatim beneath a fresh STREAMINFO-only header, since @c -n asks for
     * the metadata to be dropped and not for the guarantee to go with it.
     * Together these guarantee re-encoding never grows a file.
     *
     * Disable only to measure the raw search without the input as a
     * competitor (bench/check.sh does this to pin search behavior).
     */
    bool reuse_frames = true;

    /**
     * @brief Warn (stderr) when the input's own frames beat the re-encode.
     *
     * If any input frames were reused — or the whole input was copied
     * through — the search lost to whatever encoder produced the input
     * somewhere. The warning reports how many frames and the input's
     * encoder vendor string (from its VORBIS_COMMENT block) when present.
     * Prints even with @ref verbose off; requires @ref reuse_frames, whose
     * comparison machinery is what detects superiority.
     */
    bool warn_superior = false;

    /**
     * @brief Print progress and statistics to stdout during the run.
     *
     * Set to @c false to suppress progress/statistics output — useful when
     * flacoutcpp is embedded as a library and the caller manages its own UI.
     * Errors are always printed to stderr regardless of this setting, since
     * the boolean return value alone does not say *why* encoding failed.
     */
    bool verbose = true;
};

/**
 * @brief Re-encode a FLAC file with exhaustive compression optimisation.
 *
 * Decodes @p input_path using libFLAC, runs the variable-block-size DP
 * optimizer, and serializes the result to @p output_path using a custom
 * bit-accurate FLAC frame writer.  The output is a fully valid FLAC stream
 * (verified by `flac --test`) with an MD5 audio signature in STREAMINFO.
 *
 * @param input_path   Path to the source FLAC file.
 * @param output_path  Path for the output FLAC file (created or overwritten).
 * @param config       Optimisation parameters (see Config).
 * @return             @c true on success, @c false on any decode/encode error.
 *
 * @note CPU time is proportional to @c (audio_duration × num_candidates ×
 *       num_windows × max_lpc_order) / @c max_threads.  For reference, a
 *       0.5-second test file takes ~18 minutes on a 4-core Linux runner.
 */
bool optimise(const std::string& input_path,
              const std::string& output_path,
              const Config&      config = {});

/**
 * @brief Set @ref Config::max_candidates, @ref Config::precision_rungs and
 *        @ref Config::adaptive_windows from an effort level 0-9.
 *
 * The levels are points measured on the size/time frontier of the 188-track
 * master mix (`bench/fixtures/master_1s_mix.flac`), lowest effort first:
 *
 * Sizes are relative to **level 3, which is the default**; times are for the
 * master mix.
 *
 * | level | candidates | rungs | time | mix | excerpts |
 * |---|---|---|---|---|---|
 * | 0 | 2 | 1 | 0.65 s | +0.146% | +0.076% |
 * | 1 | 8 | 1 | 0.71 s | +0.058% | +0.034% |
 * | 2 | 16 | 1 | 0.78 s | +0.011% | +0.008% |
 * | **3** | **24** | **1** | **0.85 s** | **—** | **—** |
 * | 4 | 32 | 1 | 0.95 s | −0.010% | −0.012% |
 * | 5 | 48 | 1 | 1.12 s | −0.024% | −0.028% |
 * | 6 | 64 | 1 | 1.26 s | −0.030% | −0.032% |
 * | 7 | 64 | 2 | 1.60 s | −0.041% | −0.040% |
 * | 8 | 64 | 3 | 1.94 s | −0.046% | −0.043% |
 * | 9 | 0 (no limit) | 0 (full ladder) | 4.82 s | −0.055% | −0.051% |
 *
 * Every level also enables @ref Config::adaptive_windows: on the 3-D frontier
 * (56 configs over candidates x rungs x adaptive), 16 of the 21 efficient
 * points use it, and every point past the fastest corner does. Its value does
 * shrink as the search deepens — −0.019% at the default, −0.008 to −0.011% at
 * each level — but it never stops paying, and its ~1.03-1.06x cost is the
 * cheapest compression on the table. The album that historically regressed
 * under `-a` (17 tracks, the case that motivated its patience scaling) is
 * −0.219% at level 4 with **no track worse** than the default.
 *
 * Levels rise monotonically in both time and compression on *both* corpora,
 * which is the property that makes the dial worth having. Note how little the
 * top half buys: level 9 costs 5.7x level 3's time for 0.055%, because the
 * candidate pool saturates — the dial's range is bounded by the estimated DP
 * it lives in, not by the levels.
 *
 * Levels 0-2 are bunched in time because fixed work (decode, MD5, the
 * granule DP) floors the runtime — the dial cannot go below that.
 *
 * @param cfg    Configuration to modify in place.
 * @param level  Effort level; out-of-range values are rejected.
 * @return       @c true if @p level was valid and applied.
 *
 * @note @ref Config::patience is left alone, so it resolves to its
 *       2 x @ref Config::max_candidates default — what every level was
 *       measured with. Patience 0 measured off the frontier at equal time, so
 *       the dial does not go there.
 * @note These levels tune the search *within* a mode. They are measured under
 *       the estimated DP and are not a substitute for @ref Config::exhaustive,
 *       which is worth an order of magnitude more (0.59% on 24-bit music,
 *       2.32% on 16-bit, against ~0.1% for this whole dial). Under
 *       @ref Config::exhaustive the levels still apply, and level 0 (a
 *       2-candidate search under exact pricing) reaches ~76% of a full
 *       exhaustive run's gain at ~80x less time. To keep nearly all of it
 *       instead, leave @ref Config::max_candidates at 0 and set
 *       @ref Config::precision_rungs to 1: ~99.5% of the gain at ~5x less
 *       time.
 * @note Adaptive windows contradict @ref Config::exhaustive and an explicit
 *       @ref Config::windows list, each of which defines its own window set.
 *       This function reads both and declines to set @ref
 *       Config::adaptive_windows when either is in play, so **set those
 *       first** — a level is a preference, not a request, and `-E 5 -e` should
 *       mean "level 5's depth under exact DP" rather than an error.
 */
bool apply_effort(Config& cfg, int level);

} // namespace flacoutcpp

#endif // FLACOUTCPP_HPP
