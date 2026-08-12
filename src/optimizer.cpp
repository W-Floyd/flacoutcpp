#include "optimizer.hpp"
#include "frame_writer.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <sstream>
#include <mutex>
#include <numeric>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace flacoutcpp {

#ifdef FLACOUT_INSTRUMENT
#include <cstdio>
struct InstrCounters {
    std::atomic<uint64_t> subframes{0};
    std::atomic<uint64_t> windows_run{0};
    std::atomic<uint64_t> order_iters{0};       // (window,order) pairs entered
    std::atomic<uint64_t> order_pruned_break{0};// orders skipped by hdr_min break
    std::atomic<uint64_t> prec_iters{0};        // (window,order,prec) entered
    std::atomic<uint64_t> prec_pruned_break{0};
    std::atomic<uint64_t> overflow_skips{0};    // candidates with >=1 clamped tap (kept, not skipped)
    std::atomic<uint64_t> residual_calls{0};    // residual loops actually run
    std::atomic<uint64_t> residual_delta{0};    // of which: precision-ladder delta updates
    std::atomic<uint64_t> residual_macs{0};     // int64 multiply-accumulates (full computes only)
    std::atomic<uint64_t> rice_calls{0};
    std::atomic<uint64_t> rice_scan_samples{0}; // residuals scanned in sums pass
    std::atomic<uint64_t> rice_k_ops{0};        // (u>>k) accumulations
    std::atomic<uint64_t> rice_fold_ops{0};
    std::atomic<uint64_t> rice_sums2_parts{0};   // partitions taking the RICE2 pass
    std::atomic<uint64_t> rice_sums2_samples{0}; // residuals it rescans, scalar
    std::atomic<uint64_t> rice_chunk_fast{0};  // 32-bit lane accumulation held
    std::atomic<uint64_t> rice_chunk_slow{0};  // OR proved it could wrap; redone in 64-bit
    std::atomic<uint64_t> autoc_macs{0};
    std::atomic<uint64_t> window_samples{0};
    std::atomic<uint64_t> lattice_evals{0};     // -Q: +-1 perturbations costed
    std::atomic<uint64_t> lattice_accepts{0};   // of which: adopted (cost dropped)
    std::atomic<uint64_t> win_order_hist[33]{};
    std::atomic<uint64_t> best_order_hist[33]{};
    std::atomic<uint64_t> best_prec_hist[16]{};
    std::atomic<uint64_t> best_window_hist[(size_t)WindowType::COUNT]{};
    ~InstrCounters() {
        std::fprintf(stderr, "\n===== INSTRUMENTATION =====\n");
        std::fprintf(stderr, "subframes optimized     : %llu\n", (unsigned long long)subframes);
        std::fprintf(stderr, "windows evaluated       : %llu\n", (unsigned long long)windows_run);
        std::fprintf(stderr, "window samples          : %llu\n", (unsigned long long)window_samples);
        std::fprintf(stderr, "autocorrelation MACs    : %llu\n", (unsigned long long)autoc_macs);
        std::fprintf(stderr, "(win,ord) entered       : %llu   pruned-by-break: %llu\n",
                     (unsigned long long)order_iters, (unsigned long long)order_pruned_break);
        std::fprintf(stderr, "(win,ord,prec) entered  : %llu   pruned-by-break: %llu\n",
                     (unsigned long long)prec_iters, (unsigned long long)prec_pruned_break);
        std::fprintf(stderr, "overflow clamps         : %llu\n", (unsigned long long)overflow_skips);
        std::fprintf(stderr, "residual loops run      : %llu   (delta updates: %llu)\n",
                     (unsigned long long)residual_calls, (unsigned long long)residual_delta);
        std::fprintf(stderr, "residual MACs           : %llu\n", (unsigned long long)residual_macs);
        {
            unsigned long long ev = lattice_evals, ac = lattice_accepts;
            std::fprintf(stderr, "lattice evals/accepts   : %llu / %llu (%.4f%% adopted)\n",
                         ev, ac, ev ? 100.0 * (double)ac / (double)ev : 0.0);
        }
        std::fprintf(stderr, "rice calls              : %llu\n", (unsigned long long)rice_calls);
        std::fprintf(stderr, "rice residuals scanned  : %llu\n", (unsigned long long)rice_scan_samples);
        std::fprintf(stderr, "rice (u>>k) ops         : %llu\n", (unsigned long long)rice_k_ops);
        std::fprintf(stderr, "rice fold ops           : %llu\n", (unsigned long long)rice_fold_ops);
        std::fprintf(stderr, "rice sums2 parts/samples: %llu / %llu\n",
                     (unsigned long long)rice_sums2_parts,
                     (unsigned long long)rice_sums2_samples);
        {
            unsigned long long f = rice_chunk_fast, sl = rice_chunk_slow;
            std::fprintf(stderr, "rice chunks fast/slow   : %llu / %llu (%.4f%% fell back)\n",
                         f, sl, (f + sl) ? 100.0 * (double)sl / (double)(f + sl) : 0.0);
        }
        std::fprintf(stderr, "WINNING order histogram:\n");
        for (int i = 1; i <= 32; ++i)
            if (best_order_hist[i]) std::fprintf(stderr, "   ord %2d: %llu\n", i, (unsigned long long)best_order_hist[i]);
        std::fprintf(stderr, "WINNING precision histogram:\n");
        for (int i = 0; i < 16; ++i)
            if (best_prec_hist[i]) std::fprintf(stderr, "   prec %2d: %llu\n", i, (unsigned long long)best_prec_hist[i]);
        std::fprintf(stderr, "WINNING window histogram:\n");
        for (int i = 0; i < (int)WindowType::COUNT; ++i)
            if (best_window_hist[i]) std::fprintf(stderr, "   %-22s: %llu\n",
                window_to_name((WindowType)i).c_str(), (unsigned long long)best_window_hist[i]);
        std::fprintf(stderr, "===========================\n");
    }
};
static InstrCounters g_instr;
#define INSTR(x) do { x; } while (0)
#else
#define INSTR(x) do {} while (0)
#endif

#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif
#ifndef M_E
#define M_E   2.71828182845904523536
#endif

// ============================================================
// WindowType utilities
// ============================================================

// ------------------------------------------------------------
// Runtime-loaded ("custom:") window registry
// ------------------------------------------------------------
// Filled by register_custom_window() before encoding starts, then read-only:
// worker threads index it lock-free. Each entry carries its own coefficient
// tables for the DP block sizes, so custom windows never touch the global
// window table (which is sized for the compiled-in shapes only) and cost
// nothing at all when unused.

struct CustomWindow {
    bool                loaded = false;
    std::string         path;
    std::vector<double> knots;
    std::vector<double> tables;     ///< coefficients for each DP size, concatenated
    std::vector<size_t> offsets;    ///< start of each DP size within `tables`
    std::vector<double> energy;     ///< sum(w^2) per DP size
    std::vector<double> zero_frac;  ///< fraction of exact zeros per DP size
};

static CustomWindow g_custom[MAX_CUSTOM_WINDOWS];

static inline bool window_is_custom(WindowType wt) {
    return (int)wt >= (int)WindowType::CUSTOM_BEGIN && (int)wt < (int)WindowType::COUNT;
}

static inline int custom_index(WindowType wt) {
    return (int)wt - (int)WindowType::CUSTOM_BEGIN;
}

std::vector<WindowType> all_window_types(bool include_experimental) {
    std::vector<WindowType> out;
    const int end = include_experimental ? (int)WindowType::COUNT
                                         : (int)WindowType::EXPERIMENTAL_BEGIN;
    for (int i = 0; i < end; ++i) {
        // Custom slots hold a shape only once something registers one; an
        // empty slot is not a window and must never reach the search.
        if (window_is_custom((WindowType)i) && !g_custom[custom_index((WindowType)i)].loaded)
            continue;
        out.push_back((WindowType)i);
    }
    return out;
}

std::string window_to_name(WindowType wt) {
    switch (wt) {
        case WindowType::RECTANGULAR:              return "rect";
        case WindowType::BARTLETT:                 return "bartlett";
        case WindowType::BARTLETT_HANN:            return "bartletthann";
        case WindowType::BLACKMAN:                 return "blackman";
        case WindowType::BLACKMAN_HARRIS_4TERM_92DB: return "blackmanharris";
        case WindowType::CONNES:                   return "connes";
        case WindowType::FLATTOP:                  return "flattop";
        case WindowType::GAUSS_025:                return "gauss025";
        case WindowType::GAUSS_0125:               return "gauss0125";
        case WindowType::HAMMING:                  return "hamming";
        case WindowType::HANN:                     return "hann";
        case WindowType::KAISER_BESSEL:            return "kaiserbessel";
        case WindowType::NUTTALL:                  return "nuttall";
        case WindowType::TRIANGLE:                 return "triangle";
        case WindowType::WELCH:                    return "welch";
        case WindowType::TUKEY_005:                return "tukey005";
        case WindowType::TUKEY_010:                return "tukey010";
        case WindowType::TUKEY_020:                return "tukey020";
        case WindowType::TUKEY_050:                return "tukey050";
        case WindowType::TUKEY_075:                return "tukey075";
        case WindowType::TUKEY_090:                return "tukey090";
        case WindowType::PARTIAL_TUKEY_2_000:      return "partialtukey2";
        case WindowType::PARTIAL_TUKEY_2_033:      return "partialtukey2_033";
        case WindowType::PARTIAL_TUKEY_2_067:      return "partialtukey2_067";
        case WindowType::PUNCHOUT_TUKEY_2_033:     return "punchouttukey2_033";
        case WindowType::PUNCHOUT_TUKEY_2_067:     return "punchouttukey2_067";
        case WindowType::LANCZOS:                  return "lanczos";
        case WindowType::BOHMAN:                   return "bohman";
        case WindowType::PARZEN:                   return "parzen";
        case WindowType::PLANCKTAPER_010:          return "plancktaper010";
        case WindowType::PLANCKTAPER_025:          return "plancktaper025";
        case WindowType::PARTIAL_TUKEY_3_1:        return "partialtukey3_1";
        case WindowType::PARTIAL_TUKEY_3_2:        return "partialtukey3_2";
        case WindowType::PARTIAL_TUKEY_3_3:        return "partialtukey3_3";
        case WindowType::PUNCHOUT_TUKEY_3_1:       return "punchouttukey3_1";
        case WindowType::PUNCHOUT_TUKEY_3_2:       return "punchouttukey3_2";
        case WindowType::PUNCHOUT_TUKEY_3_3:       return "punchouttukey3_3";
        case WindowType::PARTIAL_TUKEY_3H_000:     return "partialtukey3h_000";
        case WindowType::PARTIAL_TUKEY_3H_033:     return "partialtukey3h_033";
        case WindowType::PARTIAL_TUKEY_3H_067:     return "partialtukey3h_067";
        case WindowType::PUNCHOUT_TUKEY_3H_025:    return "punchouttukey3h_025";
        case WindowType::PUNCHOUT_TUKEY_3H_050:    return "punchouttukey3h_050";
        case WindowType::PUNCHOUT_TUKEY_2_000:     return "punchouttukey2_000";
        case WindowType::EXPDECAY_2:               return "expdecay2";
        case WindowType::EXPDECAY_4:               return "expdecay4";
        case WindowType::EXPATTACK_2:              return "expattack2";
        case WindowType::EXPATTACK_4:              return "expattack4";
        case WindowType::ATTACKDECAY_005:          return "attackdecay005";
        case WindowType::ATTACKDECAY_010:          return "attackdecay010";
        case WindowType::ATTACKDECAY_020:          return "attackdecay020";
        case WindowType::DPSS_2:                   return "dpss2";
        case WindowType::DPSS_3:                   return "dpss3";
        case WindowType::DPSS_4:                   return "dpss4";
        default: break;
    }
    if (window_is_custom(wt)) {
        const CustomWindow& cw = g_custom[custom_index(wt)];
        return cw.loaded ? "custom:" + cw.path
                         : "custom" + std::to_string(custom_index(wt)) + "(empty)";
    }
    return "unknown";
}

WindowType window_from_name(const std::string& raw) {
    std::string n;
    for (char c : raw) n += (char)std::tolower((unsigned char)c);
    // Remove common separators so "blackman_harris" and "blackmanharris" both work
    std::string clean;
    for (char c : n) if (c != '_' && c != '-' && c != ' ') clean += c;

    if (clean == "rect"       || clean == "rectangular") return WindowType::RECTANGULAR;
    if (clean == "bartlett")                              return WindowType::BARTLETT;
    if (clean == "bartletthann" || clean == "bh")        return WindowType::BARTLETT_HANN;
    if (clean == "blackman")                              return WindowType::BLACKMAN;
    if (clean == "blackmanharris" || clean == "bh4")     return WindowType::BLACKMAN_HARRIS_4TERM_92DB;
    if (clean == "connes")                                return WindowType::CONNES;
    if (clean == "flattop")                               return WindowType::FLATTOP;
    if (clean == "gauss025")                              return WindowType::GAUSS_025;
    if (clean == "gauss0125")                             return WindowType::GAUSS_0125;
    if (clean == "hamming")                               return WindowType::HAMMING;
    if (clean == "hann")                                  return WindowType::HANN;
    if (clean == "kaiserbessel" || clean == "kb")         return WindowType::KAISER_BESSEL;
    if (clean == "nuttall")                               return WindowType::NUTTALL;
    if (clean == "triangle")                              return WindowType::TRIANGLE;
    if (clean == "welch")                                 return WindowType::WELCH;
    if (clean == "tukey005")                              return WindowType::TUKEY_005;
    if (clean == "tukey010")                              return WindowType::TUKEY_010;
    if (clean == "tukey020")                              return WindowType::TUKEY_020;
    if (clean == "tukey050" || clean == "tukey")          return WindowType::TUKEY_050;
    if (clean == "tukey075")                              return WindowType::TUKEY_075;
    if (clean == "tukey090")                              return WindowType::TUKEY_090;
    if (clean == "partialtukey2")                         return WindowType::PARTIAL_TUKEY_2_000;
    if (clean == "partialtukey2033")                      return WindowType::PARTIAL_TUKEY_2_033;
    if (clean == "partialtukey2067")                      return WindowType::PARTIAL_TUKEY_2_067;
    if (clean == "punchouttukey2000")                     return WindowType::PUNCHOUT_TUKEY_2_000;
    if (clean == "punchouttukey2033")                     return WindowType::PUNCHOUT_TUKEY_2_033;
    if (clean == "punchouttukey2067")                     return WindowType::PUNCHOUT_TUKEY_2_067;
    if (clean == "lanczos")                               return WindowType::LANCZOS;
    if (clean == "bohman")                                return WindowType::BOHMAN;
    if (clean == "parzen")                                return WindowType::PARZEN;
    if (clean == "plancktaper010")                        return WindowType::PLANCKTAPER_010;
    if (clean == "plancktaper025")                        return WindowType::PLANCKTAPER_025;
    if (clean == "partialtukey31")                        return WindowType::PARTIAL_TUKEY_3_1;
    if (clean == "partialtukey32")                        return WindowType::PARTIAL_TUKEY_3_2;
    if (clean == "partialtukey33")                        return WindowType::PARTIAL_TUKEY_3_3;
    if (clean == "punchouttukey31")                       return WindowType::PUNCHOUT_TUKEY_3_1;
    if (clean == "punchouttukey32")                       return WindowType::PUNCHOUT_TUKEY_3_2;
    if (clean == "punchouttukey33")                       return WindowType::PUNCHOUT_TUKEY_3_3;
    if (clean == "partialtukey3h000")                     return WindowType::PARTIAL_TUKEY_3H_000;
    if (clean == "partialtukey3h033")                     return WindowType::PARTIAL_TUKEY_3H_033;
    if (clean == "partialtukey3h067")                     return WindowType::PARTIAL_TUKEY_3H_067;
    if (clean == "punchouttukey3h025")                    return WindowType::PUNCHOUT_TUKEY_3H_025;
    if (clean == "punchouttukey3h050")                    return WindowType::PUNCHOUT_TUKEY_3H_050;
    if (clean == "expdecay2")                             return WindowType::EXPDECAY_2;
    if (clean == "expdecay4")                             return WindowType::EXPDECAY_4;
    if (clean == "expattack2")                            return WindowType::EXPATTACK_2;
    if (clean == "expattack4")                            return WindowType::EXPATTACK_4;
    if (clean == "attackdecay005")                        return WindowType::ATTACKDECAY_005;
    if (clean == "attackdecay010")                        return WindowType::ATTACKDECAY_010;
    if (clean == "attackdecay020")                        return WindowType::ATTACKDECAY_020;
    if (clean == "dpss2")                                 return WindowType::DPSS_2;
    if (clean == "dpss3")                                 return WindowType::DPSS_3;
    if (clean == "dpss4")                                 return WindowType::DPSS_4;
    return WindowType::COUNT; // unrecognised
}

// ============================================================
// Optimizer constructor
// ============================================================

// The DP's candidate block sizes (shared with find_optimal_block_partitioning).
// Window coefficient tables for these sizes are precomputed once; any other
// size (the remainder block, the short-stream path) computes on the fly.
// The estimated-DP shortlist: six dense tapers plus the partial/punchout pair
// at each offset. Also what exact DP falls back to at a small -c.
std::vector<WindowType> default_shortlist() {
    return {WindowType::TUKEY_050, WindowType::HANN,
            WindowType::WELCH,    WindowType::RECTANGULAR,
            WindowType::TUKEY_005, WindowType::TUKEY_020,
            WindowType::PARTIAL_TUKEY_2_033,  WindowType::PARTIAL_TUKEY_2_067,
            WindowType::PUNCHOUT_TUKEY_2_033, WindowType::PUNCHOUT_TUKEY_2_067};
}

static const uint32_t DP_CANDIDATES[] = { 1024, 2048, 4096, 8192, 16384 };
static constexpr size_t NUM_DP_CANDIDATES = 5;

Optimizer::Optimizer(uint32_t channels, uint32_t bps, uint32_t sample_rate,
                     std::vector<WindowType> windows,
                     unsigned max_threads,
                     bool exhaustive,
                     bool verbose,
                     unsigned max_candidates,
                     bool adaptive_windows,
                     unsigned patience,
                     unsigned precision_rungs,
                     std::vector<uint32_t> dp_candidates,
                     unsigned lattice_sweeps,
                     bool     use_gpu,
                     unsigned gpu_min_batch,
                     unsigned gpu_partition_cap,
                     unsigned gpu_slots,
                     unsigned gpu_duty)
    : m_channels(channels), m_bps(bps), m_sample_rate(sample_rate),
      m_max_threads(max_threads),
      m_exhaustive(exhaustive), m_verbose(verbose), m_max_candidates(max_candidates),
      m_adaptive(adaptive_windows), m_patience(patience),
      m_precision_rungs(precision_rungs), m_lattice_sweeps(lattice_sweeps),
      m_use_gpu(use_gpu)
{
    if (m_use_gpu) {
        m_gpu.reset(new GpuEvaluator());
        m_gpu->set_min_batch(gpu_min_batch);
        m_gpu->set_partition_cap((int)gpu_partition_cap);
        m_gpu->set_slots((int)gpu_slots);
        m_gpu->set_duty((int)gpu_duty);
        if (m_verbose) {
            if (m_gpu->available())
                std::fprintf(stderr, "GPU: %s\n", m_gpu->why().c_str());
            else
                std::fprintf(stderr, "GPU unavailable (%s); using the CPU path\n",
                             m_gpu->why().c_str());
        }
    }
    // Block-size ladder. Empty means the built-in one, whose coefficient
    // tables are precomputed; any other size falls back to computing window
    // coefficients per block (correct, just slower), so a custom ladder costs
    // no memory.
    if (dp_candidates.empty())
        m_dp_candidates.assign(std::begin(DP_CANDIDATES), std::end(DP_CANDIDATES));
    else
        m_dp_candidates = std::move(dp_candidates);
    std::sort(m_dp_candidates.begin(), m_dp_candidates.end());
    m_dp_candidates.erase(std::unique(m_dp_candidates.begin(), m_dp_candidates.end()),
                          m_dp_candidates.end());

    // The node spacing is the GCD: every candidate must be walkable from one
    // node to another. Validation in main.cpp keeps every entry a multiple of
    // 16, so the GCD is too — which the estimated path needs, since it indexes
    // 16-sample granules by node.
    uint32_t g = m_dp_candidates.front();
    for (uint32_t c : m_dp_candidates) g = std::gcd(g, c);
    m_dp_step = g;
    // The smallest candidate must equal the step, or nodes between reachable
    // positions are dead and the DP can fail to reach the end of the stream
    // at all. main.cpp rejects such ladders; this catches library callers.
    assert(m_dp_candidates.front() == m_dp_step &&
           "every -b block size must be a multiple of the smallest");

    if (windows.empty()) {
        // Exact-DP mode (-e) affords the widest window set; the ranking pays
        // per candidate evaluated, not per window offered, so offering more
        // windows there only adds options.
        //
        // The heuristic shortlist carries six dense tapers plus the
        // partial/punchout pair at each of the two offsets. The sparse half
        // used to be left out because analysis (windowing + autocorrelation)
        // is paid per window on every block; measured, that cost is invisible
        // next to the budget-capped exact evaluation — 8 windows encode the
        // 188-track master mix in 0.90 s against 0.92 s for 4 — while the
        // compression is not: -0.113% on that corpus, -0.225% on held-out
        // excerpts, -0.53% on the 3-minute synthetic percussion fixture, and
        // not one fixture worse.
        //
        // It is specifically the *punchout* half that pays (-17989 B on the
        // master mix alone, against -10129 B for the partials alone), and the
        // *sparse* side does not want to grow further: adding the 3-partition
        // families on top gives back 2618 B, because at a fixed candidate
        // budget more sparse windows crowd each other out of the ranked top-N.
        //
        // The narrow Tukeys are the exception, and they are dense, which is
        // why they do not crowd. The list used to jump straight from
        // RECTANGULAR to TUKEY_050 with nothing in between, and that gap in
        // *taper width* — not any missing shape family — was the last thing
        // left on the table: 005 and 020 sit at different points in it rather
        // than substituting for each other, so the pair beats either alone
        // (master mix -0.0177%, against -0.0144% for 005 and -0.0117% for
        // 020; 12-fixture total -0.0212% against -0.0126% / -0.0147%). Album
        // corpus, 3 tracks from each of 9 albums: -0.0159% with **no track
        // worse**, and syn3m_noise — where every ranking-based knob here gives
        // something back — is -0.0053%. Cost is 1.02-1.04x, the same exchange
        // rate as -a. Adaptive windows are near-orthogonal to it: the deltas
        // move under 0.0005% with -A.
        //
        // 38 novel shapes were screened against this list first (asymmetric
        // Tukeys, sign-changing windows, multi-lobe combs, tapered sub-rects,
        // Hann-Poisson/Cauchy/Riesz), fed in through the -w custom: loader.
        // None reached even +tukey005: measured as a 9th window the best of
        // them was worth -0.0041%. Every shape that scored well against the
        // *old* four-window list was multi-lobe, i.e. redundant with the
        // partial/punchout pair already here — which is exactly why the
        // 4-window baseline must not be used to judge a new window.
        // Exact DP widens to all 26 only at an *unlimited* candidate budget.
        // The ranking pays per candidate evaluated, not per window offered, so
        // at -c 0 extra windows are free options — but at any finite -c they
        // crowd better candidates out of the top N and cost both size and
        // time. On the 188-track mix the shortlist wins on both axes at every
        // budget measured: -e -E 0 is -0.520% at 7.7x the default against
        // -0.413% at 15.4x for the wide set, and -e -c 24 -L 1 is -0.610% at
        // 24x against -0.598% at 41x. Real music (music_10s) prefers the
        // shortlist at -c 2 through 48 and only loses by 0.0065% at -c 0.
        //
        // The synthetic 24-bit fixture disagrees — s24_2s wants the wide set
        // from -c 16 on, by up to 0.49% — which is exactly the trap in
        // CLAUDE.md: synthetic noise floors mispredict window choice. Tuned to
        // the real-music result deliberately.
        //
        // Bare -e implies -c 0, so it is unaffected; an explicit -w overrides
        // all of this.
        if (full_search() && m_max_candidates == 0) {
            m_windows = all_window_types();
        } else if (full_search()) {
            m_windows = default_shortlist();
        } else {
            m_windows = default_shortlist();
        }
    } else {
        m_windows = std::move(windows);
    }
}

// ============================================================
// Window application
// ============================================================
// All window formulas up to EXPERIMENTAL_BEGIN match libFLAC's window.c
// exactly; experimental windows past it are our own additions.

// Note for future profilers: a 16-thread `sample` of -c once attributed ~24%
// of runtime to the trig here; a single-threaded profile put it at ~2%. The
// 24% was heterogeneous-core sampling distortion — verify shares
// single-threaded before optimizing this function. The table cache below is
// NOT here to skip the trig (that saved nothing measurable); it exists so
// apply_window stays a tiny multiply loop at its call site, which keeps the
// register allocator honest in the surrounding analyse_window code — the
// wide-band autocorrelation loop spills without it.

// DPSS (Slepian) window: the eigenvector for the largest eigenvalue of the
// symmetric tridiagonal Slepian matrix (diag ((N-1-2i)/2)^2 cos(2piW),
// off-diag i(N-i)/2, W = NW/N). Everything is fixed-count for determinism:
// 100 Sturm-bisection steps pin the eigenvalue (interval shrinks by 2^100,
// far below double resolution), then 4 inverse-iteration solves recover the
// eigenvector — no tolerance loops, straight double arithmetic, so it is on
// the same footing as the closed-form windows under -ffp-contract=off. Cost
// is ~100 O(N) passes at table-build time; the on-the-fly path (non-DP
// sizes, short streams) pays it per block, which is rare and still cheap
// next to encoding the block.
static void compute_dpss(uint32_t N, double NW, double* out)
{
    const double W = NW / (double)N;
    const double c = std::cos(2.0 * M_PI * W);
    std::vector<double> d(N), e(N); // e[i] couples rows i-1 and i; e[0] unused
    for (uint32_t i = 0; i < N; ++i) {
        double k = ((double)(N - 1) - 2.0 * (double)i) / 2.0;
        d[i] = k * k * c;
        e[i] = (double)i * (double)(N - i) / 2.0;
    }

    // Gershgorin interval containing every eigenvalue.
    double lo = d[0], hi = d[0], scale = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        double r = std::abs(e[i]) + (i + 1 < N ? std::abs(e[i + 1]) : 0.0);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
        scale = std::max(scale, std::abs(d[i]) + r);
    }

    // Sturm-sequence bisection down to the largest eigenvalue: count_below(s)
    // is the number of eigenvalues < s via the LDL^T pivot signs.
    auto count_below = [&](double s) {
        uint32_t cnt = 0;
        double q = 1.0;
        for (uint32_t i = 0; i < N; ++i) {
            q = (i == 0) ? d[0] - s : d[i] - s - e[i] * e[i] / q;
            if (q == 0.0) q = -1e-300; // pivot guard; count a zero pivot as below
            if (q < 0.0) ++cnt;
        }
        return cnt;
    };
    for (int it = 0; it < 100; ++it) {
        double mid = 0.5 * (lo + hi);
        if (count_below(mid) == N) hi = mid; else lo = mid;
    }
    const double lam = hi; // λmax to machine precision

    // Inverse iteration: repeatedly solve (T - λI) x = v (Thomas algorithm
    // with a relative pivot floor — the matrix is singular by construction,
    // which is what makes the iteration converge) and renormalize.
    const double piv_floor = 1e-14 * scale;
    std::vector<double> v(N), m(N), z(N);
    for (uint32_t i = 0; i < N; ++i) // Hann start: right symmetry, no zeros inside
        v[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * (double)i / (double)(N - 1));
    for (int it = 0; it < 4; ++it) {
        m[0] = d[0] - lam;
        if (std::abs(m[0]) < piv_floor) m[0] = piv_floor;
        z[0] = v[0];
        for (uint32_t i = 1; i < N; ++i) {
            double f = e[i] / m[i - 1];
            m[i] = d[i] - lam - f * e[i];
            if (std::abs(m[i]) < piv_floor) m[i] = piv_floor;
            z[i] = v[i] - f * z[i - 1];
        }
        v[N - 1] = z[N - 1] / m[N - 1];
        for (uint32_t i = N - 1; i-- > 0;)
            v[i] = (z[i] - e[i + 1] * v[i + 1]) / m[i];
        double mx = 0.0;
        for (uint32_t i = 0; i < N; ++i) mx = std::max(mx, std::abs(v[i]));
        for (uint32_t i = 0; i < N; ++i) v[i] /= mx;
    }

    // Peak-normalize to +1 (eigenvector sign is arbitrary; DPSS is unimodal).
    double peak = 0.0;
    for (uint32_t i = 0; i < N; ++i)
        if (std::abs(v[i]) > std::abs(peak)) peak = v[i];
    for (uint32_t i = 0; i < N; ++i) out[i] = v[i] / peak;
}

// Sample a registered knot vector at N points. The knots span the whole block,
// so knot j sits at position j/(K-1) and sample i at i/(N-1).
//
// Two properties the arithmetic is arranged to guarantee exactly, because
// experiments compare custom windows against compiled-in ones: a knot vector
// of all-equal values reproduces that constant bit-for-bit (the `a + f*(b-a)`
// form contributes exactly zero when a == b, which `a*(1-f) + b*f` does not),
// and the first and last samples are the first and last knots.
static void interpolate_custom(const std::vector<double>& k, uint32_t N, double* out)
{
    const size_t K = k.size();
    if (N == 0) return;
    if (N == 1) { out[0] = k[0]; return; }

    const double span  = (double)(K - 1);
    const double denom = (double)(N - 1);
    for (uint32_t i = 0; i + 1 < N; ++i) {
        const double x = (double)i * span / denom;   // < K-1 for i < N-1
        size_t j = (size_t)x;
        if (j + 1 >= K) j = K - 2;                   // defensive; x < K-1 already
        const double frac = x - (double)j;
        out[i] = k[j] + frac * (k[j + 1] - k[j]);
    }
    out[N - 1] = k[K - 1];
}

static void compute_window_coeffs(WindowType wt, uint32_t N, double* out)
{
    if (window_is_custom(wt)) {
        const CustomWindow& cw = g_custom[custom_index(wt)];
        // Unregistered slots cannot reach any window list, but a flat window
        // keeps this total rather than leaving the buffer undefined.
        if (!cw.loaded) { for (uint32_t i = 0; i < N; ++i) out[i] = 1.0; return; }
        interpolate_custom(cw.knots, N, out);
        return;
    }

    auto fN = (double)(N - 1);
    auto fNh = fN / 2.0;

    if (wt == WindowType::DPSS_2 || wt == WindowType::DPSS_3 ||
        wt == WindowType::DPSS_4) {
        compute_dpss(N, (wt == WindowType::DPSS_2) ? 2.0
                      : (wt == WindowType::DPSS_3) ? 3.0 : 4.0, out);
        return;
    }

    for (uint32_t i = 0; i < N; ++i) {
        double w = 1.0;
        double fi = (double)i;

        switch (wt) {
        case WindowType::RECTANGULAR:
            w = 1.0; break;

        case WindowType::BARTLETT:
            if (N & 1) w = (fi <= fNh) ? (2.0*fi/fN) : (2.0 - 2.0*fi/fN);
            else       w = (fi <= (double)(N/2-1)) ? (2.0*fi/fN) : (2.0 - 2.0*fi/fN);
            break;

        case WindowType::BARTLETT_HANN:
            w = 0.62 - 0.48*std::abs(fi/fN - 0.5) - 0.38*std::cos(2.0*M_PI*fi/fN);
            break;

        case WindowType::BLACKMAN:
            w = 0.42 - 0.5*std::cos(2.0*M_PI*fi/fN) + 0.08*std::cos(4.0*M_PI*fi/fN);
            break;

        case WindowType::BLACKMAN_HARRIS_4TERM_92DB:
            w = 0.35875 - 0.48829*std::cos(2.0*M_PI*fi/fN)
                        + 0.14128*std::cos(4.0*M_PI*fi/fN)
                        - 0.01168*std::cos(6.0*M_PI*fi/fN);
            break;

        case WindowType::CONNES: {
            double k = (fi - fNh) / fNh;
            k = 1.0 - k*k;
            w = k*k;
            break;
        }

        case WindowType::FLATTOP:
            w = 0.21557895
              - 0.41663158*std::cos(2.0*M_PI*fi/fN)
              + 0.277263158*std::cos(4.0*M_PI*fi/fN)
              - 0.083578947*std::cos(6.0*M_PI*fi/fN)
              + 0.006947368*std::cos(8.0*M_PI*fi/fN);
            break;

        case WindowType::GAUSS_025: {
            double k = (fi - fNh) / (0.25 * fNh);
            w = std::exp(-0.5*k*k);
            break;
        }
        case WindowType::GAUSS_0125: {
            double k = (fi - fNh) / (0.125 * fNh);
            w = std::exp(-0.5*k*k);
            break;
        }

        case WindowType::HAMMING:
            w = 0.54 - 0.46*std::cos(2.0*M_PI*fi/fN);
            break;

        case WindowType::HANN:
            w = 0.5 - 0.5*std::cos(2.0*M_PI*fi/fN);
            break;

        case WindowType::KAISER_BESSEL:
            w = 0.402 - 0.498*std::cos(2.0*M_PI*fi/fN)
                      + 0.098*std::cos(4.0*M_PI*fi/fN)
                      - 0.001*std::cos(6.0*M_PI*fi/fN);
            break;

        case WindowType::NUTTALL:
            w = 0.3635819 - 0.4891775*std::cos(2.0*M_PI*fi/fN)
                          + 0.1365995*std::cos(4.0*M_PI*fi/fN)
                          - 0.0106411*std::cos(6.0*M_PI*fi/fN);
            break;

        case WindowType::TRIANGLE:
            if (N & 1) {
                w = (fi < (double)((N+1)/2))
                    ? (2.0*(fi+1.0) / ((double)N+1.0))
                    : (2.0*((double)N-fi) / ((double)N+1.0));
            } else {
                w = (fi < (double)(N/2))
                    ? (2.0*(fi+1.0) / ((double)N+1.0))
                    : (2.0*((double)N-fi) / ((double)N+1.0));
            }
            break;

        case WindowType::WELCH: {
            double k = (fi - fNh) / fNh;
            w = 1.0 - k*k;
            break;
        }

        // Tukey variants: taper the first and last p/2 of the window with a Hann ramp
        case WindowType::TUKEY_005: case WindowType::TUKEY_010:
        case WindowType::TUKEY_020: case WindowType::TUKEY_050:
        case WindowType::TUKEY_075: case WindowType::TUKEY_090: {
            double p;
            switch (wt) {
                case WindowType::TUKEY_005: p = 0.05; break;
                case WindowType::TUKEY_010: p = 0.10; break;
                case WindowType::TUKEY_020: p = 0.20; break;
                case WindowType::TUKEY_050: p = 0.50; break;
                case WindowType::TUKEY_075: p = 0.75; break;
                default:                    p = 0.90; break;
            }
            int Np = (int)(p / 2.0 * N) - 1;
            if (Np > 0 && (fi <= Np || fi >= N-Np-1)) {
                double idx = (fi <= Np) ? fi : (double)(N-1) - fi;
                w = 0.5 - 0.5*std::cos(M_PI * idx / Np);
            } else {
                w = 1.0;
            }
            break;
        }

        // Partial Tukey: sub-window covering part of the block. The _2 set
        // spans 0.5; the experimental house _3H set spans 0.33 (-w only).
        case WindowType::PARTIAL_TUKEY_2_000:
        case WindowType::PARTIAL_TUKEY_2_033:
        case WindowType::PARTIAL_TUKEY_2_067:
        case WindowType::PARTIAL_TUKEY_3H_000:
        case WindowType::PARTIAL_TUKEY_3H_033:
        case WindowType::PARTIAL_TUKEY_3H_067: {
            double start = (wt == WindowType::PARTIAL_TUKEY_2_000 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_000) ? 0.0
                         : (wt == WindowType::PARTIAL_TUKEY_2_033 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_033) ? 0.33 : 0.67;
            double span  = (wt == WindowType::PARTIAL_TUKEY_3H_000 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_033 ||
                            wt == WindowType::PARTIAL_TUKEY_3H_067) ? 0.33 : 0.5;
            double end   = start + span;
            end   = std::min(end, 1.0);
            int sn = (int)(start * N), en = (int)(end * N);
            int Ng = en - sn;
            double p = 0.5;
            int Np = (int)(p / 2.0 * Ng);
            if ((int)i < sn || (int)i >= en) { w = 0.0; }
            else {
                int li = (int)i - sn;
                if (Np > 0 && (li < Np || li >= Ng-Np)) {
                    double idx = (li < Np) ? (double)li : (double)(Ng-1-li);
                    w = 0.5 - 0.5*std::cos(M_PI * idx / Np);
                } else { w = 1.0; }
            }
            break;
        }

        case WindowType::PUNCHOUT_TUKEY_2_000:
        case WindowType::PUNCHOUT_TUKEY_2_033:
        case WindowType::PUNCHOUT_TUKEY_2_067:
        case WindowType::PUNCHOUT_TUKEY_3H_025:
        case WindowType::PUNCHOUT_TUKEY_3H_050: {
            // Punchout Tukey: full window with a "hole" punched out. The _2
            // set punches 0.33; the experimental house _3H set 0.25 (-w only).
            double start = (wt == WindowType::PUNCHOUT_TUKEY_2_000) ? 0.00
                         : (wt == WindowType::PUNCHOUT_TUKEY_2_033) ? 0.33
                         : (wt == WindowType::PUNCHOUT_TUKEY_2_067) ? 0.67
                         : (wt == WindowType::PUNCHOUT_TUKEY_3H_025) ? 0.25 : 0.50;
            double hole  = (wt == WindowType::PUNCHOUT_TUKEY_3H_025 ||
                            wt == WindowType::PUNCHOUT_TUKEY_3H_050) ? 0.25 : 0.33;
            double end   = start + hole;
            int sn = (int)(start * N), en = (int)(end * N);
            double p = 0.5;
            int Ns = (int)(p / 2.0 * sn);
            int Ne = (int)(p / 2.0 * ((int)N - en));
            if ((int)i < Ns) {
                w = 0.5 - 0.5*std::cos(M_PI*(fi+1.0)/(Ns > 0 ? Ns : 1));
            } else if ((int)i < sn-Ns) {
                w = 1.0;
            } else if ((int)i < sn) {
                w = 0.5 - 0.5*std::cos(M_PI*((double)(sn-(int)i))/(Ns > 0 ? Ns : 1));
            } else if ((int)i < en) {
                w = 0.0;
            } else if ((int)i < en+Ne) {
                w = 0.5 - 0.5*std::cos(M_PI*(fi-(double)en)/(Ne > 0 ? Ne : 1));
            } else if ((int)i < (int)N-Ne) {
                w = 1.0;
            } else {
                w = 0.5 - 0.5*std::cos(M_PI*((double)N-fi)/(Ne > 0 ? Ne : 1));
            }
            break;
        }

        case WindowType::LANCZOS: {
            // sinc(2i/(N-1) - 1); not in libFLAC. Experimental (-w only).
            double x = 2.0*fi/fN - 1.0;
            w = (x == 0.0) ? 1.0 : std::sin(M_PI*x) / (M_PI*x);
            break;
        }

        case WindowType::BOHMAN: {
            // (1-|x|)cos(pi|x|) + sin(pi|x|)/pi, x in [-1,1]. Experimental.
            double x = std::abs(2.0*fi/fN - 1.0);
            w = (1.0 - x)*std::cos(M_PI*x) + std::sin(M_PI*x)/M_PI;
            break;
        }

        case WindowType::PARZEN: {
            // Piecewise cubic B-spline (scipy form), x = |i - (N-1)/2| / (N/2).
            // Experimental.
            double x = std::abs(fi - fNh) / ((double)N / 2.0);
            if (x <= 0.5) w = 1.0 - 6.0*x*x*(1.0 - x);
            else { double k = 1.0 - x; w = 2.0*k*k*k; }
            break;
        }

        case WindowType::PLANCKTAPER_010:
        case WindowType::PLANCKTAPER_025: {
            // Flat center, C-infinity edges: w = 1/(e^Z + 1) over the first
            // and last eps*N samples, Z = eps*N*(1/j + 1/(j - eps*N)), with
            // w(0) = w(N-1) = 0. Experimental.
            double eps = (wt == WindowType::PLANCKTAPER_010) ? 0.10 : 0.25;
            double t = eps * (double)N;
            double j = std::min(fi, fN - fi);
            if (j <= 0.0)     w = 0.0;
            else if (j < t) { double Z = t*(1.0/j + 1.0/(j - t));
                              w = 1.0 / (std::exp(Z) + 1.0); }
            else              w = 1.0;
            break;
        }

        // libFLAC-faithful partial_tukey(3): parts at thirds, 10% overlap,
        // taper p = 0.2 — the geometry `flac -A partial_tukey(3)` builds
        // (stream_encoder.c overlap_units math, window.c ramp indexing),
        // re-derived per-index in doubles. Experimental (-w only).
        case WindowType::PARTIAL_TUKEY_3_1:
        case WindowType::PARTIAL_TUKEY_3_2:
        case WindowType::PARTIAL_TUKEY_3_3: {
            double m = (wt == WindowType::PARTIAL_TUKEY_3_1) ? 0.0
                     : (wt == WindowType::PARTIAL_TUKEY_3_2) ? 1.0 : 2.0;
            double u = 1.0/(1.0 - 0.1) - 1.0;              // overlap_units
            int sn = (int)(m / (3.0 + u) * N);
            int en = (int)((m + 1.0 + u) / (3.0 + u) * N);
            int Np = (int)(0.2 / 2.0 * (en - sn));
            int n = (int)i;
            if (n < sn || n >= en)               w = 0.0;
            else if (Np > 0 && n < sn + Np)      w = 0.5 - 0.5*std::cos(M_PI*(double)(n - sn + 1)/Np);
            else if (Np > 0 && n >= en - Np)     w = 0.5 - 0.5*std::cos(M_PI*(double)(en - n)/Np);
            else                                 w = 1.0;
            break;
        }

        // libFLAC-faithful punchout_tukey(3): holes at thirds, 20% overlap,
        // taper p = 0.2. Experimental (-w only).
        case WindowType::PUNCHOUT_TUKEY_3_1:
        case WindowType::PUNCHOUT_TUKEY_3_2:
        case WindowType::PUNCHOUT_TUKEY_3_3: {
            double m = (wt == WindowType::PUNCHOUT_TUKEY_3_1) ? 0.0
                     : (wt == WindowType::PUNCHOUT_TUKEY_3_2) ? 1.0 : 2.0;
            double u = 1.0/(1.0 - 0.2) - 1.0;              // overlap_units
            int sn = (int)(m / (3.0 + u) * N);
            int en = (int)((m + 1.0 + u) / (3.0 + u) * N);
            int Ns = (int)(0.2 / 2.0 * sn);
            int Ne = (int)(0.2 / 2.0 * ((int)N - en));
            int n = (int)i;
            if (n >= sn && n < en)                            w = 0.0;
            else if (Ns > 0 && n < Ns)                        w = 0.5 - 0.5*std::cos(M_PI*(double)(n + 1)/Ns);
            else if (Ns > 0 && n >= sn - Ns && n < sn)        w = 0.5 - 0.5*std::cos(M_PI*(double)(sn - n)/Ns);
            else if (Ne > 0 && n >= en && n < en + Ne)        w = 0.5 - 0.5*std::cos(M_PI*(double)(n - en + 1)/Ne);
            else if (Ne > 0 && n >= (int)N - Ne)              w = 0.5 - 0.5*std::cos(M_PI*(double)((int)N - n)/Ne);
            else                                              w = 1.0;
            break;
        }

        // Asymmetric exponentials: e^(-k·i/(N-1)) and its mirror. Not in
        // libFLAC. Experimental (-w only).
        case WindowType::EXPDECAY_2:  w = std::exp(-2.0*fi/fN);        break;
        case WindowType::EXPDECAY_4:  w = std::exp(-4.0*fi/fN);        break;
        case WindowType::EXPATTACK_2: w = std::exp(-2.0*(fN - fi)/fN); break;
        case WindowType::EXPATTACK_4: w = std::exp(-4.0*(fN - fi)/fN); break;

        // Attack-decay: exponential rise (rate 6, normalized to reach 1) over
        // the first f·N samples, half-cosine decay to 0 after. Experimental
        // (-w only).
        case WindowType::ATTACKDECAY_005:
        case WindowType::ATTACKDECAY_010:
        case WindowType::ATTACKDECAY_020: {
            double f = (wt == WindowType::ATTACKDECAY_005) ? 0.05
                     : (wt == WindowType::ATTACKDECAY_010) ? 0.10 : 0.20;
            double R = f * (double)N;
            if (fi < R) {
                double t = std::min((fi + 1.0) / R, 1.0);
                w = (1.0 - std::exp(-6.0*t)) / (1.0 - std::exp(-6.0));
            } else {
                w = std::cos(0.5*M_PI*(fi - R)/(fN - R));
            }
            break;
        }

        default:
            w = 1.0; break;
        }

        out[i] = w;
    }
}

// Public wrapper: the pure-GPU encoder uploads a coefficient table to the device
// once and never applies a window on the host, so it needs the shapes without
// going through apply_window's per-block table lookup.
void window_coefficients(WindowType wt, uint32_t N, double* out)
{
    compute_window_coeffs(wt, N, out);
}


static int candidate_slot(uint32_t N)
{
    for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c)
        if (DP_CANDIDATES[c] == N) return (int)c;
    return -1;
}


#ifdef FLACOUT_DUMP_BLOCKCOST
// Offline sink for the block-cost estimator's error: one row per (node, block
// size) the DP considered, carrying the granule features available at
// estimation time, the estimate, and the exact encoded cost. The point is to
// find out whether the estimator's error is *systematic* — a per-block-size
// bias a five-parameter recalibration could absorb — or content-dependent, in
// which case it needs a model. Compiles to nothing unless defined; path from
// FLACOUT_DUMP_PATH, else stderr.
#include <cstdio>
namespace {
struct BlockCostDump {
    std::FILE* fh = nullptr;
    std::mutex mu;
    BlockCostDump() {
        const char* path = std::getenv("FLACOUT_DUMP_PATH");
        fh = path ? std::fopen(path, "w") : stderr;
        if (!fh) fh = stderr;
        std::fprintf(fh, "node\tstart\tbsize\tch\tbps\test\texact\t"
                         "energy\ttilt1\ttilt2\tcv2\tmode\testL\testR\testM\testS\n");
    }
    ~BlockCostDump() { if (fh && fh != stderr) std::fclose(fh); }
    void row(size_t node, uint64_t start, uint32_t bsize, uint32_t ch, uint32_t bps,
             uint32_t est, uint32_t exact, double energy, double t1, double t2,
             double cv2, int mode, double eL, double eR, double eM, double eS) {
        std::lock_guard<std::mutex> lk(mu);
        std::fprintf(fh, "%zu\t%llu\t%u\t%u\t%u\t%u\t%u\t%.6g\t%.6g\t%.6g\t%.6g"
                         "\t%d\t%.6g\t%.6g\t%.6g\t%.6g\n",
                     node, (unsigned long long)start, bsize, ch, bps, est, exact,
                     energy, t1, t2, cv2, mode, eL, eR, eM, eS);
    }
};
BlockCostDump g_block_dump;
} // namespace
#endif

#ifdef FLACOUT_DUMP_CANDIDATES
// Offline training/analysis sink: one row per candidate the ranked scan
// actually evaluated, carrying the features available at ranking time and the
// exact bit cost that was only knowable afterwards. Enabled at build time and
// pointed at a file by FLACOUT_DUMP_PATH; compiles to nothing otherwise.
//
// Run it through the ranked path with a candidate limit large enough that
// nothing is cut (-c 100000 -p 0) so every subframe dumps its whole ladder.
#include <cstdio>
namespace {
struct CandidateDump {
    std::FILE* fh = nullptr;
    std::mutex mu;
    std::atomic<uint64_t> subframe{0};
    CandidateDump() {
        const char* path = std::getenv("FLACOUT_DUMP_PATH");
        fh = path ? std::fopen(path, "w") : stderr;
        if (!fh) fh = stderr;
        std::fprintf(fh, "sf\twin\tord\tbsize\tbps\tlderr0\tlderr_ord\twsq\tzf\t"
                         "model_bps\traw_bps\tvar_raw\tscore\tcost\t"
                         "lderr_prev\tlderr_next\tlderr_1\tlderr_max\tmax_ord\n");
    }
    ~CandidateDump() { if (fh && fh != stderr) std::fclose(fh); }
};
CandidateDump g_dump;
} // namespace
#endif

#ifdef FLACOUT_DUMP_LATTICE
// Sink for the coefficient-lattice refinement (-Q): one row per +-1
// perturbation costed, carrying the signed cost delta against the rounded
// point the quantizer chose. Answers whether that point is an axis-local
// minimum of the *exact* cost, and if not, by how much it misses.
namespace {
struct LatticeDump {
    std::FILE* fh = nullptr;
    std::mutex mu;
    LatticeDump() {
        const char* path = std::getenv("FLACOUT_LATTICE_DUMP_PATH");
        fh = path ? std::fopen(path, "w") : stderr;
        if (!fh) fh = stderr;
        std::fprintf(fh, "bsize\tord\tprec\ttap\tdelta\tdcost"
                         "\tbps\twasted\tshift\tbase\tcoef\n");
    }
    ~LatticeDump() { if (fh && fh != stderr) std::fclose(fh); }
};
LatticeDump g_ldump;
} // namespace
#endif

#ifdef FLACOUT_DUMP_PRECISION
// Second offline sink, one row per *precision rung* of every candidate the
// scan evaluated. The candidate dump above asks whether the (window, order)
// ordering is right; this one asks whether the precision ladder — 8 full
// residual+Rice passes per candidate, never pruned in practice — can be
// priced analytically instead of run.
//
// The analytic claim: for predictor c, the windowed residual energy is
// E(c) = E_a + (c-a)'R(c-a), where a is the Levinson solution and R the
// windowed autocorrelation. Both are already in hand, so a rung's cost is
// an O(order^2) quadratic form rather than an O(bsize*order) pass. Each row
// carries `ea` and `drd` so the offline model can be scored against `cost`,
// which is what the rung actually cost.
#include <cstdio>
namespace {
struct PrecisionDump {
    std::FILE* fh = nullptr;
    std::mutex mu;
    PrecisionDump() {
        const char* path = std::getenv("FLACOUT_PREC_DUMP_PATH");
        fh = path ? std::fopen(path, "w") : stderr;
        if (!fh) fh = stderr;
        std::fprintf(fh, "sf\twin\tord\tbsize\tbps\tprec\tshift\tclamped\t"
                         "cost\tea\tdrd\twsq\tr0\tsd2\tdrd1\tdrd4\n");
    }
    ~PrecisionDump() { if (fh && fh != stderr) std::fclose(fh); }
};
PrecisionDump g_pdump;
std::atomic<uint64_t> g_pdump_sf{0};
} // namespace
#endif

#ifdef FLACOUT_DUMP_GOLDEN
// Golden test vectors for the GPU sweep kernel. Captures one subframe's worth
// of (shifted samples, quantized candidates, exact Rice cost) so an
// out-of-tree kernel can be held to exact agreement with calculate_rice_cost
// rather than to "looks about right". The cost model is a contract (see
// CLAUDE.md); a GPU reimplementation of it needs the same regression net the
// CPU side has.
//
// Binary, little-endian, native int32:
//   char[8] "FLGOLD1"
//   u32 bsize, u32 eff_bps, u32 hdr_fixed, u32 ncand
//   i32 shifted[bsize]
//   ncand x { i32 ord, i32 shift, i32 qc[32], u32 rice, u32 cost }
namespace {
struct GoldenDump {
    std::FILE*        fh = nullptr;
    std::mutex        mu;
    std::atomic<bool> claimed{false};
    uint32_t          want_bs = 1024;
    uint32_t          ncand = 0;
    long              ncand_pos = 0;
    GoldenDump() {
        if (const char* p = std::getenv("FLACOUT_GOLDEN_PATH")) fh = std::fopen(p, "wb");
        if (const char* b = std::getenv("FLACOUT_GOLDEN_BS")) want_bs = (uint32_t)std::atoi(b);
    }
    ~GoldenDump() {
        if (!fh) return;
        std::fseek(fh, ncand_pos, SEEK_SET);
        std::fwrite(&ncand, 4, 1, fh);   // patch the count now that it is known
        std::fclose(fh);
        std::fprintf(stderr, "GOLDEN wrote %u candidates at bsize=%u\n", ncand, want_bs);
    }
};
GoldenDump g_golden;
} // namespace
#endif

#ifdef FLACOUT_DUMP_FP32RANK
// Third offline sink, and the one that prices a GPU port. A GPU can evaluate
// candidates in fp32 roughly 2.2x faster than in exact integer (measured on an
// M4 Max: 3963 vs 1835 GMAC/s), but fp32 cannot produce the emitted residual —
// a 15-bit coefficient times a 16-bit sample carries up to 30 significant bits
// against fp32's 24-bit mantissa, so the products are inexact before any
// accumulation. The workable split is FLACCL's: sweep every candidate in fp32,
// then exactly evaluate only the best K.
//
// That is only sound if the fp32 ordering keeps the true winner near the top.
// This sink measures exactly that. Per subframe it costs every candidate twice
// — once exactly, once with the residual computed in fp32 the way a GPU kernel
// would — then records where the exact winner lands in the fp32 ordering, and
// what keeping only the top K would have cost in real bits.
//
// The excess-bits columns are the decision metric, not the rank histogram:
// rank only matters through the bits it costs.
namespace {
struct Fp32RankDump {
    static constexpr int NK = 6;
    static constexpr int KS[NK] = {1, 2, 4, 8, 16, 32};
    std::mutex mu;
    std::vector<uint64_t> rank_hist;      // exact winner's index in fp32 order
    uint64_t subframes = 0;
    uint64_t candidates = 0;
    uint64_t exact_bits = 0;              // sum of the true minimum cost
    uint64_t excess[NK] = {};             // extra bits if only top-K kept

    void record(std::vector<std::pair<uint32_t,uint32_t>>& cc) {
        if (cc.empty()) return;
        // order by fp32 cost, tie-broken by index so the result is stable
        std::vector<uint32_t> idx(cc.size());
        for (uint32_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b){
            if (cc[a].second != cc[b].second) return cc[a].second < cc[b].second;
            return a < b;
        });
        uint32_t truemin = UINT32_MAX, truearg = 0;
        for (uint32_t i = 0; i < cc.size(); ++i)
            if (cc[i].first < truemin) { truemin = cc[i].first; truearg = i; }
        size_t rank = 0;
        while (rank < idx.size() && idx[rank] != truearg) ++rank;

        std::lock_guard<std::mutex> lk(mu);
        if (rank_hist.size() <= rank) rank_hist.resize(rank + 1, 0);
        rank_hist[rank]++;
        subframes++;
        candidates += cc.size();
        exact_bits += truemin;
        for (int k = 0; k < NK; ++k) {
            uint32_t best = UINT32_MAX;
            for (size_t i = 0; i < idx.size() && i < (size_t)KS[k]; ++i)
                if (cc[idx[i]].first < best) best = cc[idx[i]].first;
            excess[k] += (uint64_t)(best - truemin);
        }
    }

    ~Fp32RankDump() {
        if (!subframes) return;
        std::fprintf(stderr,
            "\nFP32RANK  subframes=%llu  candidates=%llu (%.1f per subframe)\n",
            (unsigned long long)subframes, (unsigned long long)candidates,
            (double)candidates / (double)subframes);
        uint64_t cum = 0;
        std::fprintf(stderr, "  exact winner's rank in the fp32 ordering:\n");
        for (size_t r = 0; r < rank_hist.size() && r < 16; ++r) {
            cum += rank_hist[r];
            std::fprintf(stderr, "    rank %-3zu %8llu  (%.2f%% cumulative)\n",
                r, (unsigned long long)rank_hist[r],
                100.0 * (double)cum / (double)subframes);
        }
        if (rank_hist.size() > 16) {
            uint64_t tail = 0;
            for (size_t r = 16; r < rank_hist.size(); ++r) tail += rank_hist[r];
            std::fprintf(stderr, "    rank >=16 %7llu  (worst %zu)\n",
                (unsigned long long)tail, rank_hist.size() - 1);
        }
        std::fprintf(stderr, "  cost of keeping only the top K by fp32 cost:\n");
        for (int k = 0; k < NK; ++k)
            std::fprintf(stderr, "    K=%-3d  +%llu bits  (%+.4f%% of subframe bits)\n",
                KS[k], (unsigned long long)excess[k],
                100.0 * (double)excess[k] / (double)exact_bits);
    }
};
constexpr int Fp32RankDump::KS[];
Fp32RankDump g_fp32rank;
} // namespace
#endif

// How small a window coefficient must be, relative to the window's peak, to
// count the sample as one the window cannot see. 0.0 means "exactly zero",
// which is what the blind-region term originally tested; see the calibration
// note at the ranked scorer for why that is worth a threshold.
// Highest LPC coefficient precision whose winner -Q still tries to refine.
// 15 disables the gate (every winner refined). See the gate in
// optimize_subframe for the measured frontier behind the default.
#ifndef FLACOUT_LATTICE_MAX_PREC
#define FLACOUT_LATTICE_MAX_PREC 10
#endif

#ifndef FLACOUT_BLIND_EPS
#define FLACOUT_BLIND_EPS 0.0
#endif

// Fraction of a window's coefficients that are blind by the above rule.
// At FLACOUT_BLIND_EPS == 0 this is |w[i]| <= 0, i.e. exactly zero.
static double blind_fraction(const double* w, uint32_t n)
{
    double peak = 0.0;
    for (uint32_t i = 0; i < n; ++i) peak = std::max(peak, std::abs(w[i]));
    const double cut = (double)FLACOUT_BLIND_EPS * peak;
    uint32_t z = 0;
    for (uint32_t i = 0; i < n; ++i)
        if (std::abs(w[i]) <= cut) ++z;
    return (double)z / (double)n;
}

// All compiled-in windows (incl. experimental) x 5 candidate sizes, built once
// on first use (~254 KB per window; ~13 MB at 51 windows,
// thread-safe magic static — worker threads block on the first builder and
// read lock-free forever after). The values are computed by the exact code
// that used to run inline per block, so the output is bit-identical.
//
// Runtime-loaded (custom:) windows are deliberately not in here: this table is
// built lazily during encoding, whereas custom shapes are registered before
// it, so folding them in would trade a zero-cost lookup for an
// order-of-initialisation trap. They carry equivalent per-entry tables built
// at registration instead.
static const double* window_table(WindowType wt, int slot)
{
    if (window_is_custom(wt)) {
        const CustomWindow& cw = g_custom[custom_index(wt)];
        return &cw.tables[cw.offsets[(size_t)slot]];
    }

    static const std::vector<double> tables = [] {
        size_t total = 0;
        for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) total += DP_CANDIDATES[c];
        std::vector<double> t((size_t)WindowType::CUSTOM_BEGIN * total);
        size_t off = 0;
        for (int w = 0; w < (int)WindowType::CUSTOM_BEGIN; ++w)
            for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
                compute_window_coeffs((WindowType)w, DP_CANDIDATES[c], &t[off]);
                off += DP_CANDIDATES[c];
            }
        return t;
    }();

    // offset of (wt, slot): wt strides the whole size set, then sizes before slot
    size_t size_sum = 0, prefix = 0;
    for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) size_sum += DP_CANDIDATES[c];
    for (int c = 0; c < slot; ++c) prefix += DP_CANDIDATES[c];
    return &tables[(size_t)wt * size_sum + prefix];
}

// Load a knot file into a free custom slot. Everything the encoder will later
// need from the shape — coefficients, energy, zero fraction, at each DP block
// size — is computed here, while we are still single-threaded.
WindowType register_custom_window(const std::string& path, std::string* error)
{
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return WindowType::COUNT;
    };

    for (int i = 0; i < MAX_CUSTOM_WINDOWS; ++i)
        if (g_custom[i].loaded && g_custom[i].path == path)
            return (WindowType)((int)WindowType::CUSTOM_BEGIN + i);

    int slot = -1;
    for (int i = 0; i < MAX_CUSTOM_WINDOWS; ++i)
        if (!g_custom[i].loaded) { slot = i; break; }
    if (slot < 0)
        return fail("no free custom window slots (limit " +
                    std::to_string(MAX_CUSTOM_WINDOWS) + ")");

    std::ifstream in(path);
    if (!in) return fail("cannot open file");

    std::vector<double> knots;
    std::string line;
    size_t lineno = 0;
    constexpr size_t MAX_KNOTS = 1u << 20;
    while (std::getline(in, line)) {
        ++lineno;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.erase(hash);
        for (char& c : line) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ls(line);
        std::string tok;
        while (ls >> tok) {
            size_t used = 0;
            double v;
            try {
                v = std::stod(tok, &used);
            } catch (const std::exception&) {
                return fail("line " + std::to_string(lineno) +
                            ": not a number: '" + tok + "'");
            }
            if (used != tok.size())
                return fail("line " + std::to_string(lineno) +
                            ": trailing junk after number: '" + tok + "'");
            if (!std::isfinite(v))
                return fail("line " + std::to_string(lineno) +
                            ": value is not finite: '" + tok + "'");
            if (knots.size() >= MAX_KNOTS)
                return fail("more than " + std::to_string(MAX_KNOTS) + " knots");
            knots.push_back(v);
        }
    }
    if (knots.size() < 2)
        return fail("need at least 2 knots, found " +
                    std::to_string(knots.size()));

    bool all_zero = true;
    for (double v : knots) if (v != 0.0) { all_zero = false; break; }
    if (all_zero) return fail("window is identically zero");

    CustomWindow& cw = g_custom[slot];
    cw.path  = path;
    cw.knots = std::move(knots);

    size_t total = 0;
    cw.offsets.resize(NUM_DP_CANDIDATES);
    for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
        cw.offsets[c] = total;
        total += DP_CANDIDATES[c];
    }
    cw.tables.resize(total);
    cw.energy.resize(NUM_DP_CANDIDATES);
    cw.zero_frac.resize(NUM_DP_CANDIDATES);
    for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
        double* t = &cw.tables[cw.offsets[c]];
        interpolate_custom(cw.knots, DP_CANDIDATES[c], t);
        double e = 0.0;
        for (uint32_t i = 0; i < DP_CANDIDATES[c]; ++i) e += t[i] * t[i];
        cw.energy[c]    = e;
        cw.zero_frac[c] = blind_fraction(t, DP_CANDIDATES[c]);
    }

    // Last: nothing may observe a half-built entry.
    cw.loaded = true;
    return (WindowType)((int)WindowType::CUSTOM_BEGIN + slot);
}

// Sum of squared window coefficients, the normalizer that turns a windowed
// Levinson error into an absolute residual-variance estimate (see the ranked
// scoring in optimize_subframe). Cached for the precomputed table sizes —
// COUNT x 5 doubles, built lazily from the tables themselves — and computed on
// the fly for the rare other sizes (remainder block, short-stream path).
static double window_energy(WindowType wt, uint32_t N)
{
    const int slot = candidate_slot(N);
    if (window_is_custom(wt)) {
        const CustomWindow& cw = g_custom[custom_index(wt)];
        if (slot >= 0) return cw.energy[(size_t)slot];
    } else if (slot >= 0) {
        static const std::vector<double> energies = [] {
            std::vector<double> e((size_t)WindowType::CUSTOM_BEGIN * NUM_DP_CANDIDATES);
            for (int w = 0; w < (int)WindowType::CUSTOM_BEGIN; ++w)
                for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
                    const double* t = window_table((WindowType)w, (int)c);
                    double s = 0.0;
                    for (uint32_t i = 0; i < DP_CANDIDATES[c]; ++i) s += t[i] * t[i];
                    e[(size_t)w * NUM_DP_CANDIDATES + c] = s;
                }
            return e;
        }();
        return energies[(size_t)wt * NUM_DP_CANDIDATES + (size_t)slot];
    }
    std::vector<double> tmp(N);
    compute_window_coeffs(wt, N, tmp.data());
    double s = 0.0;
    for (uint32_t i = 0; i < N; ++i) s += tmp[i] * tmp[i];
    return s;
}

// Fraction of exactly-zero coefficients in a window — the block region the
// window is blind to. Non-zero only for partial/punchout shapes (and the
// single zero endpoint samples of e.g. Planck windows, which round to ~0
// fraction). The ranked scorer uses it to price the blind region at the raw
// signal's variance instead of pretending it is modeled. Cached like
// window_energy for the table sizes.
static double window_zero_frac(WindowType wt, uint32_t N)
{
    const int slot = candidate_slot(N);
    if (window_is_custom(wt)) {
        const CustomWindow& cw = g_custom[custom_index(wt)];
        if (slot >= 0) return cw.zero_frac[(size_t)slot];
    } else if (slot >= 0) {
        static const std::vector<double> fracs = [] {
            std::vector<double> f((size_t)WindowType::CUSTOM_BEGIN * NUM_DP_CANDIDATES);
            for (int w = 0; w < (int)WindowType::CUSTOM_BEGIN; ++w)
                for (size_t c = 0; c < NUM_DP_CANDIDATES; ++c) {
                    const double* t = window_table((WindowType)w, (int)c);
                    f[(size_t)w * NUM_DP_CANDIDATES + c] =
                        blind_fraction(t, DP_CANDIDATES[c]);
                }
            return f;
        }();
        return fracs[(size_t)wt * NUM_DP_CANDIDATES + (size_t)slot];
    }
    std::vector<double> tmp(N);
    compute_window_coeffs(wt, N, tmp.data());
    return blind_fraction(tmp.data(), N);
}


// How much of the search the GPU actually absorbed. Worth printing because it
// is the number that explains -G's speedup, and it is not guessable: workers
// take the queue with try_lock, so the share depends on how dispatch latency
// happens to interleave with CPU encode time, not on anything configured.
void Optimizer::report_gpu(uint64_t total_candidates) const
{
    if (!m_gpu || !m_verbose) return;
    uint64_t gc = 0; double gs = 0.0;
    m_gpu->stats(&gc, &gs);
    // Not gated on available(): a device that was lost mid-encode still did
    // work, and its absence at the end is the single most useful thing to say
    // about a run that suddenly got slower.
    if (!m_gpu->available())
        std::cout << (m_gpu->gave_up()
                          ? "GPU: throttle switched the encode to the CPU; the "
                            "device was slower than the host\n"
                          : "GPU: device was lost during the encode; the rest "
                            "ran on the CPU\n");
    if (!gc) { std::cout << "GPU: no batches dispatched\n"; return; }
    // Candidates per second is the portable figure: it does not depend on the
    // fixture's length or on how the CPU/GPU split happened to fall, so it is
    // what to compare when running this on a different device.
    char buf[160];
    std::snprintf(buf, sizeof buf,
                  "GPU: %llu candidates in %.2fs (%.3g candidates/s)\n",
                  (unsigned long long)gc, gs,
                  gs > 0.0 ? (double)gc / gs : 0.0);
    std::cout << buf;
    // MACs, not candidates: a candidate at bsize 16384 is sixteen times one at
    // 1024, so a count-based share badly overstates what the device absorbed
    // if it skews toward short blocks -- which it does, since a long dispatch
    // holds its slot while short ones cycle through.
    std::snprintf(buf, sizeof buf, "GPU: %llu MACs\n",
                  (unsigned long long)m_gpu->macs());
    std::cout << buf;
    // What the throttle decided and why. Printing the two rates it compared is
    // the only way to tell "the device was declined because it is slower" from
    // "the device was declined because of a bug in the throttle", and the ratio
    // is the number to quote when reporting a device's usefulness.
    double gmps = 0.0, cmps = 0.0, acc = 0.0;
    m_gpu->throttle_stats(&gmps, &cmps, &acc);
    if (gmps > 0.0 || cmps > 0.0) {
        std::snprintf(buf, sizeof buf,
                      "GPU: throttle %s -- device %.3g vs one CPU thread %.3g "
                      "MACs/s (%.2fx), accepted %.0f%% of offers\n",
                      m_gpu->duty() ? "pinned" : "adaptive",
                      gmps, cmps, cmps > 0.0 ? gmps / cmps : 0.0, acc * 100.0);
        std::cout << buf;
    }
    (void)total_candidates;
}

void Optimizer::apply_window(
    const int32_t* samples, uint32_t N, int wasted_bits,
    WindowType wt, double* out)
{
    const int slot = candidate_slot(N);
    const double* w;
    std::vector<double> scratch;
    if (slot >= 0) {
        w = window_table(wt, slot);
    } else {
        scratch.resize(N);
        compute_window_coeffs(wt, N, scratch.data());
        w = scratch.data();
    }
    // One multiply per sample, same arithmetic as the fused version, and
    // -ffp-contract=off means nothing can fuse into it: bit-exact.
    for (uint32_t i = 0; i < N; ++i)
        out[i] = (double)(samples[i] >> wasted_bits) * w[i];
}

// ============================================================
// Levinson-Durbin — all orders in one pass
// ============================================================

void Optimizer::compute_lpc_all_orders(
    const double* autoc, float out_coeffs[][32], int max_order, double* out_err)
{
    // Negative marks "recursion never got here"; callers must skip those.
    if (out_err) for (int i = 0; i <= max_order; ++i) out_err[i] = -1.0;
    // ld_a[i][j] = coefficient j for predictor of order i (1-indexed in both).
    // We use a stack-allocated array — 33×33 doubles = ~8 KB, safe.
    double ld_a[33][33];
    double ld_e[33];
    std::memset(ld_a, 0, sizeof(ld_a));

    ld_e[0] = autoc[0];
    if (out_err) out_err[0] = ld_e[0];
    if (ld_e[0] <= 0.0) {
        for (int i = 0; i < max_order; ++i)
            for (int j = 0; j < 32; ++j) out_coeffs[i][j] = 0.0f;
        return;
    }

    for (int ord = 1; ord <= max_order; ++ord) {
        double lambda = autoc[ord];
        for (int j = 1; j < ord; ++j)
            lambda -= ld_a[ord-1][j] * autoc[ord-j];

        ld_a[ord][ord] = lambda / ld_e[ord-1];

        // If PARCOR magnitude >= 1 the filter is unstable; zero out remaining orders.
        if (std::abs(ld_a[ord][ord]) >= 1.0) {
            for (int i = ord-1; i < max_order; ++i)
                for (int j = 0; j < 32; ++j) out_coeffs[i][j] = 0.0f;
            return;
        }

        ld_e[ord] = ld_e[ord-1] * (1.0 - ld_a[ord][ord]*ld_a[ord][ord]);
        if (ld_e[ord] <= 0.0) ld_e[ord] = 1e-10;
        if (out_err) out_err[ord] = ld_e[ord];

        for (int j = 1; j < ord; ++j)
            ld_a[ord][j] = ld_a[ord-1][j] - ld_a[ord][ord] * ld_a[ord-1][ord-j];

        // Save coefficients for this order
        for (int j = 0; j < ord; ++j)
            out_coeffs[ord-1][j] = (float)ld_a[ord][j+1];
    }
}

// Legacy single-order wrapper (used by the DP fast path).
void Optimizer::compute_lpc_coefficients(const double* autoc, float* out, int order) {
    float tmp[32][32];
    compute_lpc_all_orders(autoc, tmp, order);
    for (int i = 0; i < order; ++i) out[i] = tmp[order-1][i];
}

// ============================================================
// Rice cost
// ============================================================

// Four-lane unsigned vector for the Rice sum accumulators. GCC/clang lower the
// generic vector extension to whatever the target has (NEON usra, SSE2
// psrld+paddd); MSVC has no equivalent, so it takes the scalar array instead.
// Both spellings compute identical sums — this only changes instruction
// selection, never results.
#if defined(__clang__) || defined(__GNUC__)
#  define FLACOUT_HAVE_VECEXT 1
typedef uint32_t u32x4 __attribute__((vector_size(16)));

// Pairwise horizontal add: lane i of the result is the sum of lanes 2i, 2i+1
// of (a ++ b). Both compilers lower this shuffle+add pattern to a single
// pairwise-add instruction where one exists (NEON addp, SSSE3 phaddd).
static inline u32x4 hadd_pairs(u32x4 a, u32x4 b) {
#if defined(__clang__)
    return __builtin_shufflevector(a, b, 0, 2, 4, 6)
         + __builtin_shufflevector(a, b, 1, 3, 5, 7);
#else
    return __builtin_shuffle(a, b, (u32x4){0, 2, 4, 6})
         + __builtin_shuffle(a, b, (u32x4){1, 3, 5, 7});
#endif
}
#else
#  define FLACOUT_HAVE_VECEXT 0
#endif

// Fold a signed residual into the unsigned value Rice coding actually emits.
// Done through uint32_t because left-shifting a negative int32_t is UB.
static inline uint32_t zigzag(int32_t r) {
    return ((uint32_t)r << 1) ^ (uint32_t)(r >> 31);
}

// Quantized-LPC residuals for one candidate, over already-wasted-bit-shifted
// samples.
//
// Register-blocked: a fixed batch of consecutive output samples is held in
// accumulators while the tap loop runs over all `ord` coefficients, so the
// partial sums never reach memory at all.
//
// The obvious alternatives both lose. One `pred` per sample accumulated over j
// walks the history backwards into a serial int64 chain that will not
// vectorize. Hoisting the taps outside and streaming a `pred[]` array
// vectorizes, but then every tap reloads and rewrites that whole array — with
// ord up to 32 the array moves through the load/store units 32 times, and the
// profile showed exactly that: 90 memory ops in the inner loop and not a single
// widening multiply-accumulate.
//
// Blocking the output dimension gets both properties: BLOCK independent
// accumulators give the vectorizer something to work with, and they stay in
// registers across the whole tap loop, so per tap the only memory traffic is
// reading the sample history.
//
// Integer addition is associative, so this is exact — same partial products,
// same sum, and in fact the same order as the original scalar version.
//
// autoc[lag] = sum over j of w[j]*w[j+lag], for lag in [0, max_lag].
//
// Computed a band of lags at a time rather than one lag per pass. The
// straightforward version walks the whole block once per lag — 33 passes over
// `bsize` doubles — and each pass is a reduction, so loads dominate and the
// serial accumulator chain gives the vectorizer nothing to work with. Handling
// LAG_BAND lags together gets LAG_BAND multiply-accumulates out of every loaded
// w[j] and cuts the number of passes by the same factor. At 33 the whole
// max_lag=32 range is one pass; measured in isolation that is 2.0x over the
// original band of 8 (register-pressure sweet spots are not monotonic — 16
// measured no better than 8, so widen all the way or not at all).
//
// This is floating point, so unlike the integer loops the summation order is
// *not* free to change — reassociating would perturb the coefficients and with
// them the output. It is preserved exactly: the parallelism is across lags, each
// keeping its own accumulator summed over ascending j, and each band's ragged
// tail (where the highest lag in the band would run off the end) is finished per
// lag afterwards, which is still ascending j for that lag.
// Order and coefficient precision of the estimator's predictor. The order is
// the estimate's main remaining bias — the real search reaches 32 — and the
// precision is nearly free either way (order*prec is ~120 bits against a
// payload of thousands), so it sits at the top of the range for predictor
// fidelity rather than to save bits.
#ifndef FLACOUT_EST_ORDER
#define FLACOUT_EST_ORDER 8
#endif
#ifndef FLACOUT_EST_PREC
#define FLACOUT_EST_PREC 15
#endif
// Also price a Fixed order-2 subframe and take the cheaper. Worth -0.0226% of
// the album corpus on top of the LPC-only estimate (-0.2072% against -0.1846%),
// and it is what pulls the two regressing tracks back: +1.015% -> +0.860% and
// +0.039% -> +0.012%. It costs the biggest winner a little (-3.99% -> -3.83%),
// so this is a net call, not a free one.
#ifndef FLACOUT_EST_FIXED
#define FLACOUT_EST_FIXED 1
#endif

// One band of W lags. W must be a compile-time constant or a[] is indexed
// dynamically and spills, which is the whole point of banding (see above).
// Safe whenever lag0 + W <= n, which every caller below guarantees via
// max_lag <= n - 1.
template <int W>
static inline void autocorr_band(
    const double* w, uint32_t n, int lag0, double* autoc)
{
    double a[W] = {};
    // Range of j where every lag in this band is still in bounds.
    const uint32_t jend = n - (uint32_t)(lag0 + W - 1);
    for (uint32_t j = 0; j < jend; ++j) {
        const double x = w[j];
        for (int l = 0; l < W; ++l) a[l] += x * w[j + lag0 + l];
    }
    // Ragged tail: lag0+l runs off the end later for smaller l.
    for (int l = 0; l < W; ++l) {
        const uint32_t lim = n - (uint32_t)(lag0 + l);
        for (uint32_t j = jend; j < lim; ++j) a[l] += w[j] * w[j + lag0 + l];
    }
    for (int l = 0; l < W; ++l) autoc[lag0 + l] = a[l];
}

static void autocorrelation(
    const double* w, uint32_t n, int max_lag, double* autoc)
{
    constexpr int LAG_BAND = 33;

    int lag0 = 0;
    for (; lag0 + LAG_BAND <= max_lag + 1; lag0 += LAG_BAND)
        autocorr_band<LAG_BAND>(w, n, lag0, autoc);

    // Deliberately *not* generalised to narrower bands here. The DP estimator
    // also wants a band (of FLACOUT_EST_ORDER + 1 lags) but it calls
    // autocorr_band directly, because giving this function a second caller made
    // the compiler stop inlining it into analyse_window — which is the one thing
    // the note above says must not happen. Measured: bare -e on music_3s went
    // 1.056x with two callers, and back to parity with one.

    // Leftover lags that do not fill a band.
    for (int lag = lag0; lag <= max_lag; ++lag) {
        double s = 0.0;
        for (uint32_t j = 0; j < n - (uint32_t)lag; ++j) s += w[j] * w[j + lag];
        autoc[lag] = s;
    }
}

// The blocked main loop, parameterised on accumulator width. Acc=int64_t is
// the always-safe form; Acc=int32_t is selected by the caller's bound below
// and is the same arithmetic in half the register width — twice the SIMD
// lanes per multiply-accumulate, and no widening.
template <typename Acc, uint32_t BLOCK>
static inline uint32_t lpc_residual_blocked(
    const int32_t* shifted, uint32_t bsize,
    const int32_t* qc, int ord, int shift, int32_t* residuals,
    int64_t* pred_out)
{
    uint32_t i = (uint32_t)ord;
    for (; i + BLOCK <= bsize; i += BLOCK) {
        Acc acc[BLOCK] = {};
        for (int j = 0; j < ord; ++j) {
            const Acc      c   = (Acc)qc[j];
            const int32_t* src = shifted + i - 1 - j;
            // Must be unrolled, or acc[] is indexed dynamically and spills.
#if defined(__clang__)
#  pragma unroll
#elif defined(__GNUC__)
#  pragma GCC unroll 16
#endif
            for (uint32_t k = 0; k < BLOCK; ++k) acc[k] += c * (Acc)src[k];
        }
        if (pred_out)
            for (uint32_t k = 0; k < BLOCK; ++k) pred_out[i + k] = acc[k];
        for (uint32_t k = 0; k < BLOCK; ++k)
            residuals[i + k] = shifted[i + k] - (int32_t)(acc[k] >> shift);
    }
    return i;
}

// `max_sum_abs_qc` is INT32_MAX >> (eff_bps - 1), the same bound the
// precision-ladder delta already applies to its correction taps: every
// partial sum of the dot product is bounded by sum|qc| * max|sample|, and
// max|sample| is 2^(eff_bps-1), so when sum|qc| clears the bound the whole
// accumulation fits in int32. Results are bit-identical either way —
// identical partial products in identical order, and an arithmetic right
// shift of an in-range value is the same at both widths. 0 disables it.
//
// It is worth branching on because the bound is 256x looser at 16 bits than
// at 24: measured on the master mix (16-bit), 94.9% of all multiply-
// accumulates take the narrow path; on 24-bit music_5s, 0.7%.
static void compute_lpc_residuals(
    const int32_t* shifted, uint32_t bsize,
    const int32_t* qc, int ord, int shift, int32_t* residuals,
    int64_t* pred_out = nullptr, int64_t max_sum_abs_qc = 0)
{
    assert(ord >= 1);

    // 16 int64 accumulators = 8 128-bit registers, which leaves plenty spare
    // for the sample history and coefficients on both NEON (32) and SSE2 (16).
    constexpr uint32_t BLOCK = 16;

    bool narrow = false;
    if (max_sum_abs_qc > 0) {
        int64_t s = 0;
        for (int j = 0; j < ord; ++j) s += std::abs((int64_t)qc[j]);
        narrow = s <= max_sum_abs_qc;
    }

    uint32_t i = narrow
        ? lpc_residual_blocked<int32_t, BLOCK>(shifted, bsize, qc, ord, shift,
                                               residuals, pred_out)
        : lpc_residual_blocked<int64_t, BLOCK>(shifted, bsize, qc, ord, shift,
                                               residuals, pred_out);

    // Tail: fewer than BLOCK samples left.
    for (; i < bsize; ++i) {
        int64_t p = 0;
        for (int j = 0; j < ord; ++j)
            p += (int64_t)qc[j] * (int64_t)shifted[i - 1 - j];
        if (pred_out) pred_out[i] = p;
        residuals[i] = shifted[i] - (int32_t)(p >> shift);
    }

    // Warm-up samples are stored verbatim.
    for (int k = 0; k < ord; ++k) residuals[k] = shifted[k];
}

// Precision-ladder delta update. When the ladder steps prec -> prec+1 the
// quantization target doubles exactly (v = lpc[j] * 2^shift, and shift grows
// by one), so the new coefficients are qc'[j] = 2*qc[j] + d[j] where d[j] is
// a small integer near zero (exactly {-1,0,1} under independent rounding;
// error feedback spreads it slightly wider) — many of them zero. The new
// prediction is therefore
//
//     pred'[i] = 2*pred[i] + sum over nonzero d of d[j]*shifted[i-1-j]
//
// which replaces a full ord-tap widening multiply-accumulate with int32
// multiplies over the nonzero taps (twice the SIMD width of the int64 loop):
// every partial sum of the correction is bounded by sum|d| * max|sample|,
// and the caller only takes this path when that is < 2^31. pred itself is
// the exact integer dot product of the new coefficients — no error
// accumulates across steps — so the residuals are bit-identical to a full
// recompute.
static void update_lpc_residuals_delta(
    const int32_t* shifted, uint32_t bsize,
    const int32_t* d, int ord, int shift,
    int64_t* pred, int32_t* residuals)
{
    // Gather the nonzero taps once; the sample loops only touch those.
    int     tap_off [32];
    int32_t tap_sign[32];
    int nnz = 0;
    for (int j = 0; j < ord; ++j)
        if (d[j]) { tap_off[nnz] = j; tap_sign[nnz] = d[j]; ++nnz; }

    // Wider than the int64 loop's 16: int32 accumulators are half the size, so
    // 32 still fits in 8 vector registers, and the extra independent chains
    // cover the multiply-accumulate latency the serial per-lane adds expose.
    constexpr uint32_t BLOCK = 32;

    uint32_t i = (uint32_t)ord;
    for (; i + BLOCK <= bsize; i += BLOCK) {
        int32_t corr[BLOCK] = {};
        for (int t = 0; t < nnz; ++t) {
            const int32_t  c   = tap_sign[t];
            const int32_t* src = shifted + i - 1 - tap_off[t];
#if defined(__clang__)
#  pragma unroll
#elif defined(__GNUC__)
#  pragma GCC unroll 32
#endif
            for (uint32_t k = 0; k < BLOCK; ++k) corr[k] += c * src[k];
        }
        for (uint32_t k = 0; k < BLOCK; ++k) {
            const int64_t p = 2 * pred[i + k] + (int64_t)corr[k];
            pred[i + k] = p;
            residuals[i + k] = shifted[i + k] - (int32_t)(p >> shift);
        }
    }

    for (; i < bsize; ++i) {
        int64_t p = 2 * pred[i];
        for (int t = 0; t < nnz; ++t)
            p += (int64_t)tap_sign[t] * (int64_t)shifted[i - 1 - tap_off[t]];
        pred[i] = p;
        residuals[i] = shifted[i] - (int32_t)(p >> shift);
    }
    // Warm-up residuals are already in place from the full compute.
}

// sum(u_i >> k) is additive over disjoint ranges, so sums are computed once at the finest partition order and folded upward instead of rescanning per order.
uint32_t Optimizer::calculate_rice_cost(
    const int32_t* residuals, uint32_t block_size,
    uint32_t order, SubframeParams* out_params)
{
    static constexpr int NUM_K = 15;
    static constexpr int MAX_PARTS = 256; // 1 << 8

    // Valid partition orders form a contiguous prefix [0, max_p_order]
    // since 2^p | block_size implies 2^(p-1) | block_size. Also cap so the
    // finest partition is never smaller than order: per the FLAC format,
    // only partition 0 may have fewer than p_size residuals (its count is
    // p_size - order); every other partition is always exactly p_size long.
    // A decoder never re-derives "warm-up spilled into partition 1" — that's
    // purely a bitstream desync if we choose a partition order that fine.
    int max_p_order = 0;
    for (int p = 1; p <= 8; ++p) {
        if (block_size % (1u << p) != 0) break;
        if ((block_size >> p) < order) break;
        max_p_order = p;
    }

    uint32_t num_parts = 1u << max_p_order;
    uint32_t p_size    = block_size / num_parts;

    // High-k extension for RICE2 (coding method 1, 5-bit parameters):
    // Σ(u>>k) for k = 15..30 equals Σ(v>>(k-15)) with v = u>>15, so a second
    // 16-lane pass over the shifted residuals gives the exact sums. It only
    // runs for partitions whose folded residuals actually reach 2^15 —
    // 16-bit content never does, so its cost and output are untouched.
    static constexpr int NUM_K2 = 16; // k = 15..30
    uint64_t sums[MAX_PARTS][NUM_K];
    uint64_t sums2[MAX_PARTS][NUM_K2];
    bool     any_high = false;
    uint32_t n_res[MAX_PARTS];
    uint32_t max_abs_arr[MAX_PARTS];

    INSTR(g_instr.rice_calls.fetch_add(1, std::memory_order_relaxed));
    INSTR(g_instr.rice_scan_samples.fetch_add(block_size - order, std::memory_order_relaxed));
    INSTR(g_instr.rice_k_ops.fetch_add((uint64_t)(block_size - order) * NUM_K, std::memory_order_relaxed));
    INSTR(g_instr.rice_fold_ops.fetch_add((uint64_t)(num_parts - 1) * NUM_K, std::memory_order_relaxed));

    for (uint32_t p = 0; p < num_parts; ++p) {
        uint32_t start = p * p_size;
        uint32_t end   = start + p_size;
        uint32_t first = std::max(start, order); // skip warm-up in partition 0

        uint64_t* s = sums[p];
        // The chunked path below accumulates into s[] and needs it zeroed; the
        // single-chunk fast path writes each s[k] exactly once and does not.

        // One pass computing both the magnitude bound and the 15 partial sums.
        //
        // The bound is an OR of every folded residual, not a maximum. Only the
        // position of its highest set bit is ever used — `< (1u << 30)` and the
        // escape-code bit-width search below both depend on nothing else — and
        // OR has the same highest set bit as max, because every u contributes
        // its bits. OR is also what makes the overflow test below cheap: since
        // each u's bits are a subset of the OR, u <= or_all, so `count * or_all`
        // bounds the accumulated sum. Folded u is 2*|r| for r>=0 and 2*|r|-1
        // for r<0, so the |r| the escape path wants is u >> 1.
        //
        // Accumulating in 32-bit lanes lets each (s[k] += u >> k) become a
        // single shift-right-and-accumulate over four residuals at a time,
        // rather than 15 dependent 64-bit adds per residual. 32-bit lanes hold
        // at most 0xFFFFFFFF, so the range is walked in fixed chunks and widened
        // into the 64-bit sums between them. The chunk bound cannot be computed
        // up front without a separate maximum pass, so instead each chunk is
        // accumulated optimistically and re-run in 64-bit on the rare occasion
        // the OR proves it could have wrapped.
        static constexpr uint32_t CHUNK = 1024;
        uint32_t or_all = 0;

#if FLACOUT_HAVE_VECEXT
        // Single-chunk fast path. For power-of-two block sizes — every DP
        // candidate — the finest partition is at most 64 residuals, so this is
        // the path every partition of every candidate actually takes; the
        // chunked loop below only ever runs for the odd-sized remainder block.
        // With so few residuals per partition the bookkeeping around the scan
        // costs as much as the scan, so this path strips it down: no zeroed
        // u64 sums to read-modify-write, no part[] staging, and the fifteen
        // per-lane reductions collapse into a pairwise-add tree whose u64
        // widening lands directly in sums[p].
        if (end - first <= CHUNK) {
            const uint32_t cnt = end - first;
            u32x4 acc[16] = {}; // 16th stays zero: pads the reduction tree
            u32x4 orv = {};
            uint32_t i = first;
            for (; i + 4 <= end; i += 4) {
                const u32x4 u = { zigzag(residuals[i]),     zigzag(residuals[i + 1]),
                                  zigzag(residuals[i + 2]), zigzag(residuals[i + 3]) };
                orv |= u;
#  if defined(__clang__)
#    pragma unroll
#  elif defined(__GNUC__)
#    pragma GCC unroll 15
#  endif
                for (int k = 0; k < NUM_K; ++k) acc[k] += u >> k;
            }
            uint32_t tail[16] = {};
            uint32_t or_tail = 0;
            for (; i < end; ++i) {
                const uint32_t u = zigzag(residuals[i]);
                or_tail |= u;
                for (int k = 0; k < NUM_K; ++k) tail[k] += u >> k;
            }
            or_all = or_tail | orv[0] | orv[1] | orv[2] | orv[3];

            if ((uint64_t)cnt * or_all <= 0xFFFFFFFFull) {
                INSTR(g_instr.rice_chunk_fast.fetch_add(1, std::memory_order_relaxed));
                for (int k = 0; k < NUM_K; k += 4) {
                    // r = [sum(acc[k]), sum(acc[k+1]), sum(acc[k+2]), sum(acc[k+3])]
                    const u32x4 r = hadd_pairs(hadd_pairs(acc[k],     acc[k + 1]),
                                               hadd_pairs(acc[k + 2], acc[k + 3]));
                    for (int j = 0; j < 4 && k + j < NUM_K; ++j)
                        s[k + j] = (uint64_t)r[j] + tail[k + j];
                }
            } else {
                INSTR(g_instr.rice_chunk_slow.fetch_add(1, std::memory_order_relaxed));
                for (int k = 0; k < NUM_K; ++k) s[k] = 0;
                for (uint32_t j = first; j < end; ++j) {
                    const uint32_t u = zigzag(residuals[j]);
                    for (int k = 0; k < NUM_K; ++k) s[k] += (u >> k);
                }
            }

            n_res[p]       = cnt;
            max_abs_arr[p] = or_all >> 1;
            continue;
        }
#endif
        for (int k = 0; k < NUM_K; ++k) s[k] = 0;
        for (uint32_t base = first; base < end; ) {
            const uint32_t stop = std::min(base + CHUNK, end);
            const uint32_t cnt  = stop - base;
            uint32_t or_chunk = 0;
            uint32_t part[NUM_K];
            uint32_t i = base;
#if FLACOUT_HAVE_VECEXT
            u32x4 acc[NUM_K] = {};
            u32x4 orv = {};
            for (; i + 4 <= stop; i += 4) {
                const u32x4 u = { zigzag(residuals[i]),     zigzag(residuals[i + 1]),
                                  zigzag(residuals[i + 2]), zigzag(residuals[i + 3]) };
                orv |= u;
                // Must be fully unrolled: `k` has to be a constant for each
                // shift to fold into the accumulate.
#  if defined(__clang__)
#    pragma unroll
#  elif defined(__GNUC__)
#    pragma GCC unroll 15
#  endif
                for (int k = 0; k < NUM_K; ++k) acc[k] += u >> k;
            }
            uint32_t tail[NUM_K] = {};
            for (; i < stop; ++i) {
                const uint32_t u = zigzag(residuals[i]);
                or_chunk |= u;
                for (int k = 0; k < NUM_K; ++k) tail[k] += u >> k;
            }
            or_chunk |= orv[0] | orv[1] | orv[2] | orv[3];
            for (int k = 0; k < NUM_K; ++k)
                part[k] = acc[k][0] + acc[k][1] + acc[k][2] + acc[k][3] + tail[k];
#else
            for (int k = 0; k < NUM_K; ++k) part[k] = 0;
            for (; i < stop; ++i) {
                const uint32_t u = zigzag(residuals[i]);
                or_chunk |= u;
                for (int k = 0; k < NUM_K; ++k) part[k] += u >> k;
            }
#endif
            // part[] is only trustworthy if no lane could have wrapped. part[0]
            // is the largest of the 15, and is bounded by cnt * or_chunk.
            if ((uint64_t)cnt * or_chunk <= 0xFFFFFFFFull) {
                INSTR(g_instr.rice_chunk_fast.fetch_add(1, std::memory_order_relaxed));
                for (int k = 0; k < NUM_K; ++k) s[k] += part[k];
            } else {
                INSTR(g_instr.rice_chunk_slow.fetch_add(1, std::memory_order_relaxed));
                for (uint32_t j = base; j < stop; ++j) {
                    const uint32_t u = zigzag(residuals[j]);
                    for (int k = 0; k < NUM_K; ++k) s[k] += (u >> k);
                }
            }
            or_all |= or_chunk;
            base = stop;
        }
        const uint32_t mabs = or_all >> 1;

        n_res[p]        = (first < end) ? (end - first) : 0;
        max_abs_arr[p]  = mabs;
    }

    // Second pass for the high-k sums, only where they can be non-zero.
    for (uint32_t p = 0; p < num_parts; ++p) {
        if ((max_abs_arr[p] >> 14) == 0) continue; // all u < 2^15 → sums2 ≡ 0
        if (!any_high) {
            any_high = true;
            std::memset(sums2, 0, sizeof(uint64_t) * num_parts * NUM_K2);
        }
        const uint32_t start = p * p_size;
        const uint32_t end   = start + p_size;
        const uint32_t first = std::max(start, order);
        INSTR(g_instr.rice_sums2_parts.fetch_add(1, std::memory_order_relaxed));
        INSTR(g_instr.rice_sums2_samples.fetch_add(end - first, std::memory_order_relaxed));
        uint64_t* s2 = sums2[p];
        // Vectorized the same way as the 15-lane low-k scan above, and for the
        // same reason: this is not the rare path the original comment assumed.
        // 24-bit content rescans 41-73% of all residuals here (measured,
        // music_3s and s24_2s under -e), against 0.18% at 16 bits -- so a
        // scalar loop doing 16 shift-adds per residual was costing roughly
        // what the whole vectorized low-k pass costs, on exactly the content
        // where RICE2 exists to help.
        //
        // 32-bit lanes need the same wrap guard the low-k scan uses: v is
        // bounded by 2^17 and a partition by 65535 residuals, whose product
        // exceeds 32 bits. The OR of every v bounds each lane's total, so one
        // compare decides whether the fast accumulation held.
#if FLACOUT_HAVE_VECEXT
        {
            u32x4 acc2[NUM_K2] = {};
            u32x4 orv2 = {};
            uint32_t i = first;
            for (; i + 4 <= end; i += 4) {
                const u32x4 v = { zigzag(residuals[i])     >> 15,
                                  zigzag(residuals[i + 1]) >> 15,
                                  zigzag(residuals[i + 2]) >> 15,
                                  zigzag(residuals[i + 3]) >> 15 };
                orv2 |= v;
#  if defined(__clang__)
#    pragma unroll
#  elif defined(__GNUC__)
#    pragma GCC unroll 16
#  endif
                for (int k = 0; k < NUM_K2; ++k) acc2[k] += v >> k;
            }
            uint32_t tail2[NUM_K2] = {};
            uint32_t or_tail2 = 0;
            for (; i < end; ++i) {
                const uint32_t v = zigzag(residuals[i]) >> 15;
                or_tail2 |= v;
                for (int k = 0; k < NUM_K2; ++k) tail2[k] += v >> k;
            }
            const uint32_t or_all2 =
                or_tail2 | orv2[0] | orv2[1] | orv2[2] | orv2[3];
            if ((uint64_t)(end - first) * or_all2 <= 0xFFFFFFFFull) {
                for (int k = 0; k < NUM_K2; ++k)
                    s2[k] = (uint64_t)acc2[k][0] + acc2[k][1] + acc2[k][2]
                          + acc2[k][3] + tail2[k];
            } else {
                for (int k = 0; k < NUM_K2; ++k) s2[k] = 0;
                for (uint32_t j = first; j < end; ++j) {
                    const uint32_t v = zigzag(residuals[j]) >> 15;
                    for (int k = 0; k < NUM_K2; ++k) s2[k] += v >> k;
                }
            }
        }
#else
        for (uint32_t i = first; i < end; ++i) {
            const uint32_t v = zigzag(residuals[i]) >> 15;
            for (int k = 0; k < NUM_K2; ++k) s2[k] += v >> k;
        }
#endif
    }
    uint64_t best_total = std::numeric_limits<uint64_t>::max();
    int      best_porder = 0;
    int      best_method = 0;
    // Only tracked when the caller wants parameters back. During the candidate
    // search it does not, and skipping this drops a 1 KB zero-init plus a memcpy
    // per improving partition order from a call made millions of times.
    int      best_ks[MAX_PARTS];
    if (out_params) std::memset(best_ks, 0, sizeof(best_ks));

    uint32_t cur_num_parts = num_parts;
    for (int p_order = max_p_order; p_order >= 0; --p_order) {
        uint64_t total0 = 4 * cur_num_parts; // method 0: 4-bit rice param per partition
        uint64_t total1 = 5 * cur_num_parts; // method 1 (RICE2): 5-bit params, k up to 30
        int      ks0[MAX_PARTS];
        int      ks1[MAX_PARTS];

        for (uint32_t p = 0; p < cur_num_parts; ++p) {
            uint32_t   n = n_res[p];
            uint64_t*  s = sums[p];

            uint64_t best0_bits = std::numeric_limits<uint64_t>::max();
            uint64_t best1_bits = std::numeric_limits<uint64_t>::max();
            int      best0_k = 0, best1_k = 0;

            // --- Try Rice parameters, k ascending ---
            // bits(k) is exactly convex in k: s[k] = 2*s[k+1] + (count of
            // residuals with bit k set), so the forward difference
            // bits(k+1) - bits(k) = n - s[k+1] - o_k is nondecreasing in k.
            // Scanning ascending, the running best is bits(k-1) until the
            // minimum is passed, so the first k that fails to improve proves
            // every later k is no better — stop there. Strict '<' keeps the
            // smallest k on ties, exactly like the full scan did. Convexity
            // holds straight through the k=14/15 boundary (the sums2 rows are
            // exact continuations), so when the scan is still improving at
            // k=14 it carries on into the RICE2-only range.
            for (int k = 0; k < NUM_K; ++k) {
                uint64_t bits = (uint64_t)n * (1 + k) + s[k];
                if (bits < best0_bits) { best0_bits = bits; best0_k = k; }
                else break;
            }
            best1_bits = best0_bits;
            best1_k    = best0_k;
            if (any_high && best0_k == NUM_K - 1) {
                uint64_t* s2 = sums2[p];
                for (int k = NUM_K; k <= 30; ++k) {
                    uint64_t bits = (uint64_t)n * (1 + k) + s2[k - NUM_K];
                    if (bits < best1_bits) { best1_bits = bits; best1_k = k; }
                    else break;
                }
            }

            // --- Try the escape code: verbatim residuals ---
            // marker + 5-bit bps + bps bits per residual (marker cost is the
            // per-partition param cost already counted above, same as a k).
            // The bps field is 5 bits, so residuals wider than 31 bits cannot
            // be represented by escape at all; skip it so normal Rice (which
            // has no such limit) is chosen instead.
            if (max_abs_arr[p] < (1u << 30)) {
                uint32_t max_abs = max_abs_arr[p];
                // Smallest width whose sign bit clears max_abs: 2 + floor(log2)
                // for nonzero values, 1 for zero — same result the old
                // increment loop produced, in one bit-scan.
#if defined(_MSC_VER) && !defined(__clang__)
                unsigned long hi;
                int escape_bps = _BitScanReverse(&hi, max_abs) ? (int)hi + 2 : 1;
#else
                int escape_bps = max_abs ? 33 - __builtin_clz(max_abs) : 1;
#endif

                uint64_t escape_bits = 5ull + (uint64_t)escape_bps * n;
                if (escape_bits < best0_bits) {
                    best0_bits = escape_bits;
                    best0_k = 15 + (escape_bps << 8); // escape: marker in low byte, bps above
                }
                if (escape_bits < best1_bits) {
                    best1_bits = escape_bits;
                    best1_k = 31 + (escape_bps << 8);
                }
            }

            total0 += best0_bits;
            ks0[p]  = best0_k;
            total1 += best1_bits;
            ks1[p]  = best1_k;
        }

        // '<=' since we iterate p_order descending: keeps the smallest p_order
        // on a tie, same as before. Method 0 is evaluated first and method 1
        // must win strictly, so 16-bit content (where the methods tie at best)
        // keeps its method-0 bitstream unchanged.
        if (total0 <= best_total) {
            best_total  = total0;
            best_porder = p_order;
            best_method = 0;
            if (out_params) std::memcpy(best_ks, ks0, cur_num_parts * sizeof(int));
        }
        if (any_high && total1 < best_total) {
            best_total  = total1;
            best_porder = p_order;
            best_method = 1;
            if (out_params) std::memcpy(best_ks, ks1, cur_num_parts * sizeof(int));
        }

        if (p_order == 0) break;

        // Fold pairs of partitions up to the next coarser order.
        uint32_t next_num_parts = cur_num_parts / 2;
        for (uint32_t p = 0; p < next_num_parts; ++p) {
            uint32_t left = 2 * p, right = 2 * p + 1;
            n_res[p]       = n_res[left] + n_res[right];
            max_abs_arr[p] = std::max(max_abs_arr[left], max_abs_arr[right]);
            for (int k = 0; k < NUM_K; ++k)
                sums[p][k] = sums[left][k] + sums[right][k];
            if (any_high)
                for (int k = 0; k < NUM_K2; ++k)
                    sums2[p][k] = sums2[left][k] + sums2[right][k];
        }
        cur_num_parts = next_num_parts;
    }

    if (out_params) {
        out_params->rice_partition_order = best_porder;
        out_params->rice_method          = best_method;
        std::memcpy(out_params->rice_k, best_ks, (1u << best_porder) * sizeof(int));
    }
    return best_total > std::numeric_limits<uint32_t>::max()
         ? std::numeric_limits<uint32_t>::max()
         : (uint32_t)best_total;
}


// ============================================================
// Coefficient quantization
// ============================================================

// Quantize LPC coefficients to `precision` bits (sign included), mirroring
// FLAC__lpc_quantize_coefficients in third_party/libflac/src/libFLAC/lpc.c.
// The two must stay in agreement or we build different predictors than the
// format's reference encoder for the same (window, order, precision).
//
// Error feedback: each tap's rounding shortfall is carried into the next tap,
// so the running sum of quantization errors stays within half an LSB. That
// shapes the coefficient noise high-pass (a (1 - z^-1) character), where audio
// has the least energy, instead of leaving it white — measurably better
// predictors than rounding each tap in isolation.
//
// Out-of-range taps are clamped into [qmin, qmax], not treated as failure: a
// clamped predictor is legal and may still be the best available at this
// order. A negative shift is not representable in the format, so that branch
// scales the coefficients down instead and emits shift = 0.
//
// Returns false only when no usable quantization exists (all-zero
// coefficients, or shift below the 5-bit field's minimum). `*clamped` reports
// whether any tap was clamped, for instrumentation.
// floor(log2(max|lpc[j]|)), or INT32_MIN when the coefficients are all zero.
// Computed via frexp so it may be negative when every coefficient is below 1
// in magnitude — which *raises* the shift and quantizes more finely. Split
// out of quantize_lpc_coeffs because it is precision-invariant: the ladder
// sweeps 8 precisions per candidate and only this part can be hoisted.
static int lpc_log2cmax(const float* lpc, int order)
{
    double cmax = 0.0;
    for (int j = 0; j < order; ++j)
        cmax = std::max(cmax, std::abs((double)lpc[j]));
    if (cmax <= 0.0) return std::numeric_limits<int32_t>::min();
    int e;
    (void)std::frexp(cmax, &e);
    return e - 1;
}

static bool quantize_lpc_coeffs(const float* lpc, int order, int precision,
                                int log2cmax,
                                int32_t* qc, int* out_shift, bool* clamped)
{
    if (log2cmax == std::numeric_limits<int32_t>::min())
        return false; // all-zero coefficients

    const int32_t qmax = (1 << (precision - 1)) - 1;
    const int32_t qmin = -(1 << (precision - 1));

    // 5-bit signed shift field in the subframe header.
    constexpr int max_shiftlimit = 15;
    constexpr int min_shiftlimit = -max_shiftlimit - 1;

    int shift = (precision - 1) - log2cmax - 1;

    if (shift > max_shiftlimit)
        shift = max_shiftlimit;
    else if (shift < min_shiftlimit)
        return false;

    *clamped = false;
    double error = 0.0;
    if (shift >= 0) {
        for (int j = 0; j < order; ++j) {
            error += (double)lpc[j] * (double)(1 << shift);
            int32_t q = (int32_t)std::lround(error);
            if (q > qmax)      { q = qmax; *clamped = true; }
            else if (q < qmin) { q = qmin; *clamped = true; }
            error -= q;
            qc[j] = q;
        }
    } else {
        const int nshift = -shift;
        for (int j = 0; j < order; ++j) {
            error += (double)lpc[j] / (double)(1 << nshift);
            int32_t q = (int32_t)std::lround(error);
            if (q > qmax)      { q = qmax; *clamped = true; }
            else if (q < qmin) { q = qmin; *clamped = true; }
            error -= q;
            qc[j] = q;
        }
        shift = 0;
    }
    *out_shift = shift;
    return true;
}

// ============================================================
// Subframe cost (fast path — single window, used by old DP shim)
// ============================================================

uint32_t Optimizer::estimate_subframe_cost(
    const int32_t* samples, uint32_t bsize,
    int mode, int order, int precision, int wasted, int bps,
    SubframeParams* out)
{
    if (out) {
        out->mode = mode; out->order = order;
        out->lpc_precision = precision; out->wasted_bits = wasted;
        out->lpc_shift = (precision > 0) ? precision - 1 : 0;
    }

    uint32_t header = 8u + (wasted ? (uint32_t)(1 + wasted) : 0u);

    if (mode == 0) {
        for (uint32_t i = 1; i < bsize; ++i)
            if (samples[i] != samples[0]) return std::numeric_limits<uint32_t>::max();
        return header + (uint32_t)(bps - wasted);
    }
    if (mode == 1) return header + (uint32_t)(bps - wasted) * bsize;

    std::vector<int32_t> residuals(bsize);

    if (mode == 2) {
        // Warm-up verbatim, then one loop per order with the shift hoisted.
        //
        // This used to read and shift four history samples for every output
        // regardless of order, and switch on `order` inside the loop: 5 shifts
        // and a jump per sample to do work that at order 2 needs 2 shifts and
        // no branch. It did not matter while this only priced the Fixed modes
        // of optimize_subframe, but estimate_lpc_bits_fast now calls it for
        // every (node, candidate) in the DP, so it is on the estimated path's
        // hot loop. Same arithmetic in the same order — bit-identical.
        for (uint32_t i = 0; i < (uint32_t)order && i < bsize; ++i)
            residuals[i] = samples[i] >> wasted;
        const uint32_t i0 = (uint32_t)order;
        switch (order) {
            case 0:
                for (uint32_t i = i0; i < bsize; ++i)
                    residuals[i] = samples[i] >> wasted;
                break;
            case 1:
                for (uint32_t i = i0; i < bsize; ++i)
                    residuals[i] = (samples[i] >> wasted) - (samples[i-1] >> wasted);
                break;
            case 2:
                for (uint32_t i = i0; i < bsize; ++i)
                    residuals[i] = (samples[i] >> wasted)
                                 - 2 * (samples[i-1] >> wasted)
                                 +     (samples[i-2] >> wasted);
                break;
            case 3:
                for (uint32_t i = i0; i < bsize; ++i)
                    residuals[i] = (samples[i] >> wasted)
                                 - 3 * (samples[i-1] >> wasted)
                                 + 3 * (samples[i-2] >> wasted)
                                 -     (samples[i-3] >> wasted);
                break;
            case 4:
                for (uint32_t i = i0; i < bsize; ++i)
                    residuals[i] = (samples[i] >> wasted)
                                 - 4 * (samples[i-1] >> wasted)
                                 + 6 * (samples[i-2] >> wasted)
                                 - 4 * (samples[i-3] >> wasted)
                                 +     (samples[i-4] >> wasted);
                break;
        }
    } else {
        // LPC (rectangular window — fast path only)
        //
        // `shifted` is materialised rather than re-deriving `samples[i] >> wasted`
        // per use, because compute_lpc_residuals below wants it as an array —
        // see the note there.
        std::vector<int32_t> shifted(bsize);
        for (uint32_t i = 0; i < bsize; ++i) shifted[i] = samples[i] >> wasted;
        std::vector<double> f(bsize);
        for (uint32_t i = 0; i < bsize; ++i) f[i] = (double)shifted[i];
        // One banded pass, not order+1 separate reduction passes. Identical
        // arithmetic in identical order — a band accumulates each lag over
        // ascending j exactly as the per-lag loop did — and this was the largest
        // single self-time item in default mode (36% of worker self time) once
        // estimate_lpc_bits_fast began calling it for every (node, candidate).
        //
        // autocorr_band directly rather than autocorrelation(): that function
        // must keep exactly one caller so it stays inlined into analyse_window,
        // see the note there.
        double autoc[33] = {};
        if (order == FLACOUT_EST_ORDER && bsize > (uint32_t)order) {
            autocorr_band<FLACOUT_EST_ORDER + 1>(f.data(), bsize, 0, autoc);
        } else {
            for (int lag = 0; lag <= order && (uint32_t)lag < bsize; ++lag)
                for (uint32_t j = 0; j < bsize - (uint32_t)lag; ++j)
                    autoc[lag] += f[j] * f[j+lag];
        }

        float lpc[32];
        compute_lpc_coefficients(autoc, lpc, order);

        int32_t qc[32];
        int  shift   = 0;
        bool clamped = false;
        if (!quantize_lpc_coeffs(lpc, order, precision, lpc_log2cmax(lpc, order),
                                 qc, &shift, &clamped)) {
            // Degenerate coefficients — a zero predictor keeps the old
            // behaviour (residual == signal) without a special-cased return.
            std::memset(qc, 0, (size_t)order * sizeof(int32_t));
            shift = 0;
        }
        if (out) {
            out->lpc_shift = shift;
            std::memcpy(out->q_coeffs, qc, order * sizeof(int32_t));
        }

        // The hand-vectorized kernel, not a scalar dot product. It computes the
        // same partial products in the same order, and its narrow-accumulator
        // path is documented bit-identical, so this is a pure speedup — but it
        // is the one that matters here: the loop this replaced re-shifted every
        // history sample *inside* the tap loop, so at order 8 it paid 8 shifts
        // and 8 scalar multiply-adds per output where the kernel keeps 16
        // outputs in registers and walks the taps outside.
        const uint32_t eff_bps = (uint32_t)(bps - wasted);
        const int64_t max_sum_abs_qc =
            (eff_bps >= 1 && eff_bps <= 31) ? (int64_t)(INT32_MAX >> (eff_bps - 1)) : 0;
        if (order >= 1)
            compute_lpc_residuals(shifted.data(), bsize, qc, order, shift,
                                  residuals.data(), nullptr, max_sum_abs_qc);
        else
            for (uint32_t i = 0; i < bsize; ++i) residuals[i] = shifted[i];
        header += 4u + 5u + (uint32_t)(order * precision);

    }

    header += (uint32_t)order * (uint32_t)(bps - wasted); // warm-up samples
    // +6: residual block header (2-bit coding method + 4-bit partition order),
    // written once per subframe by write_residual but not part of the
    // per-partition cost calculate_rice_cost returns.
    return header + 6u + calculate_rice_cost(residuals.data(), bsize, order, out);
}

// ============================================================
// Exhaustive multi-window subframe optimisation
// ============================================================


SubframeParams Optimizer::optimize_subframe(
    const int32_t* samples, uint32_t bsize, uint32_t bps,
    const std::vector<WindowType>& windows,
    unsigned max_candidates, unsigned patience, unsigned precision_rungs,
    unsigned lattice_sweeps,
    GpuEvaluator* gpu)
{
    SubframeParams best{};
    best.bits_cost = std::numeric_limits<uint32_t>::max();
#ifdef FLACOUT_INSTRUMENT
    int instr_best_win = -1;
    struct InstrOnExit {
        const SubframeParams& b; const int& w;
        ~InstrOnExit() {
            if (b.mode == 3) {
                g_instr.best_order_hist[b.order].fetch_add(1, std::memory_order_relaxed);
                g_instr.best_prec_hist[b.lpc_precision].fetch_add(1, std::memory_order_relaxed);
                if (w >= 0) g_instr.best_window_hist[w].fetch_add(1, std::memory_order_relaxed);
            }
        }
    } instr_on_exit{best, instr_best_win};
#endif

    // ---- Wasted-bits detection ----
    int wasted = 0;
    int32_t mask = 0;
    for (uint32_t i = 0; i < bsize; ++i) mask |= samples[i];
    if (mask != 0)
        while ((mask & 1) == 0) { mask >>= 1; ++wasted; }
    const uint32_t eff_bps = bps - (uint32_t)wasted;

    auto try_update = [&](SubframeParams& cur, uint32_t cost) {
        if (cost < best.bits_cost) { best = cur; best.bits_cost = cost; }
    };

    // ---- Mode 0: Constant ----
    {
        SubframeParams cur{};
        uint32_t cost = estimate_subframe_cost(samples, bsize, 0, 0, 0, wasted, (int)bps, &cur);
        try_update(cur, cost);
    }

    // ---- Mode 1: Verbatim ----
    {
        SubframeParams cur{};
        uint32_t cost = estimate_subframe_cost(samples, bsize, 1, 0, 0, wasted, (int)bps, &cur);
        try_update(cur, cost);
    }

    // ---- Mode 2: Fixed (orders 0–4) ----
    for (int ord = 0; ord <= 4; ++ord) {
        if ((uint32_t)ord >= bsize) break;
        SubframeParams cur{};
        uint32_t cost = estimate_subframe_cost(samples, bsize, 2, ord, 0, wasted, (int)bps, &cur);
        try_update(cur, cost);
    }

    // ---- Mode 3: LPC with multi-window exhaustive search ----
    if (bsize >= 2) {
        std::vector<double> windowed(bsize);
        std::vector<int32_t> residuals(bsize);
        std::vector<int64_t> pred(bsize); // unshifted predictions, for the precision-ladder delta
        float all_lpc[32][32]; // all_lpc[order-1][coeff]

        // hoisted: samples[i] >> wasted was re-read for every window/order/precision combo below
        std::vector<int32_t> shifted(bsize);
        for (uint32_t i = 0; i < bsize; ++i) shifted[i] = samples[i] >> wasted;

        // precision set is constant for the whole subframe; build once
        std::vector<int> precisions;
        for (int p = 8; p <= 15; ++p) precisions.push_back(p);
        const uint32_t min_prec  = (uint32_t)precisions.front();

#ifdef FLACOUT_DUMP_FP32RANK
        // (exact cost, fp32 cost) for every candidate this subframe evaluates.
        std::vector<std::pair<uint32_t,uint32_t>> fp32_costs;
        std::vector<int32_t> fp32_res;
#endif

        // Rice costs memoised by predictor. Different windows routinely
        // quantize to the SAME (order, shift, coefficients) at a given order,
        // and an identical predictor has an identical residual and an
        // identical Rice cost -- only the header differs, and that is
        // recomputed per candidate anyway. Measured on real subframes (golden
        // vectors): 4.2% of 6656 candidates at bsize 1024, 8.4% at 4096, none
        // of them adjacent, so a backward peek finds none of it.
        //
        // Keyed by a hash into an arena of coefficients so a hit costs one
        // lookup and one memcmp against ~100k multiply-accumulates avoided.
        // Only worth it on the full sweep. Duplicates are cross-window, so a
        // ranked search evaluating a handful of (window, order) pairs almost
        // never hits one and just pays the hash: measured -2% at -c 8 against
        // +4 to +16% at -c 0.
        const bool use_memo = (max_candidates == 0);
        struct MemoEnt { int ord, shift; uint32_t qoff; uint32_t rice; };
        std::vector<MemoEnt>  memo;
        std::vector<int32_t>  memo_qc;
        std::unordered_map<uint64_t, std::vector<uint32_t>> memo_ix;

        // Narrow-accumulator bound for compute_lpc_residuals; see its comment.
        // Same expression the ladder delta applies to its correction taps.
        const int64_t max_sum_abs_qc =
            (eff_bps >= 1 && eff_bps <= 31) ? (int64_t)(INT32_MAX >> (eff_bps - 1)) : 0;
        const uint32_t hdr_fixed = 8u + (wasted ? (uint32_t)(1 + wasted) : 0u) + 4u + 5u;

#ifdef FLACOUT_DUMP_GOLDEN
        // First subframe of the requested block size wins the capture; the
        // rest run untouched, so this cannot perturb the search it observes.
        bool golden_active = false;
        if (g_golden.fh && bsize == g_golden.want_bs &&
            !g_golden.claimed.exchange(true, std::memory_order_acq_rel)) {
            std::lock_guard<std::mutex> lk(g_golden.mu);
            golden_active = true;
            std::fwrite("FLGOLD1", 1, 8, g_golden.fh);
            uint32_t v = bsize;                 std::fwrite(&v, 4, 1, g_golden.fh);
            v = eff_bps;                        std::fwrite(&v, 4, 1, g_golden.fh);
            v = hdr_fixed;                      std::fwrite(&v, 4, 1, g_golden.fh);
            g_golden.ncand_pos = std::ftell(g_golden.fh);
            v = 0;                              std::fwrite(&v, 4, 1, g_golden.fh);
            std::fwrite(shifted.data(), 4, bsize, g_golden.fh);
        }
#endif

        // Winner tracked as a bare description rather than a filled-in
        // SubframeParams; see the cost-only call below. Seeded from the best
        // non-LPC mode so it doubles as the pruning bound, exactly as
        // best.bits_cost did — it starts at the same value and decreases at the
        // same points, so pruning decisions are unchanged.
        uint32_t best_lpc_cost = best.bits_cost;
        int      bl_ord = 0, bl_prec = 0, bl_shift = 0;
        int32_t  bl_qc[32] = {};

        INSTR(g_instr.subframes.fetch_add(1, std::memory_order_relaxed));

        // The windowed autocorrelation of whichever window eval_candidate is
        // currently pricing — the R of the analytic ladder model below.
        // analyse_window keeps `autoc` local, and in the ranked path the
        // candidates are evaluated long after their window was analysed, so
        // the ranked driver stashes a copy per window and points cand_autoc at
        // the right one before each call.
        double        cur_autoc[33] = {};
        const double* cand_autoc = cur_autoc;
#ifdef FLACOUT_DUMP_PRECISION
        const uint64_t pdump_sf = g_pdump_sf.fetch_add(1, std::memory_order_relaxed);
#endif

        const int max_order = (int)std::min((uint32_t)32, bsize - 1);

        // Evaluate one candidate — a coefficient set at one order — across
        // every precision, updating the winner. Shared by both drivers below so
        // ranked and exhaustive search cost a candidate identically; they differ
        // only in which candidates they hand it.
        auto eval_candidate = [&](const float* lpc, int ord, WindowType wt) -> uint32_t {
            uint32_t cand_best = std::numeric_limits<uint32_t>::max();
            (void)wt;
            INSTR(g_instr.order_iters.fetch_add(1, std::memory_order_relaxed));
            INSTR(g_instr.win_order_hist[ord].fetch_add(1, std::memory_order_relaxed));

            // Precision-invariant: hoisted out of the ladder below.
            const int log2cmax = lpc_log2cmax(lpc, ord);

            // Precision-ladder delta state: prev_qc/prev_shift describe the
            // coefficients pred[] currently holds the dot products of.
            // The int32 correction accumulator needs every partial sum within
            // sum|d| * max|sample| < 2^31, with |sample| <= 2^(eff_bps-1) —
            // checked per step below once sum|d| is known.
            const int64_t max_sum_abs_d =
                (eff_bps <= 31) ? (int64_t)(INT32_MAX >> (eff_bps - 1)) : 0;
            // pred[] exists only to seed the ladder delta, and the delta needs
            // two adjacent rungs. At -L 1 -- the default effort level, and the
            // recommended companion to -e -- exactly one rung is ever encoded,
            // so every candidate was writing a whole block of int64 predictions
            // that nothing could ever read. Skip the store entirely there.
            int64_t* const pred_ptr =
                (precision_rungs == 1) ? nullptr : pred.data();
            bool       have_pred = false;
            int        prev_shift = 0;
            int32_t    prev_qc[32];

            // ---- Analytic precision ladder (-L) ----
            // Every rung below costs a full residual + Rice pass, and the
            // header bound almost never cuts one: measured 8 rungs entered per
            // candidate with zero pruned. But a rung's cost is predictable
            // without encoding it. For a predictor c the windowed residual
            // energy is E(c) = E_a + (c-a)'R(c-a), with a the Levinson
            // solution and R the windowed autocorrelation — both already
            // computed for this candidate — so quantizing (O(order)) and
            // evaluating the quadratic form (O(order^2)) prices the whole
            // ladder for a few percent of one rung. Only the best `m_ladder`
            // rungs are then encoded for real.
            //
            // Measured against encoding all 8, as a fraction of subframe bits:
            // 1 rung 0.016%, 2 rungs 0.007-0.008%, 3 rungs 0.004%, alike on 16-
            // and 24-bit material. See PRECISION_LADDER_PLAN.md.
            //
            // Cheaper models of the same quantity were tried and are much
            // worse: banding the quadratic form to |i-j| <= 4 gives up 2.5-12x
            // more, the diagonal alone more still, and treating the
            // quantization error as uniform white noise — which would need no
            // quantize call at all — is worst. quantize_lpc_coeffs uses error
            // feedback, which shapes the error and shrinks (c-a)'R(c-a) by a
            // median factor of 2^-4.15; the white-noise model does not know
            // that, so the quantizer has to actually run. Only the residual
            // pass can be predicted away.
            bool rung_sel[16];
            const bool use_ladder =
                precision_rungs > 0 && precision_rungs < precisions.size();
            if (use_ladder) {
                const double* ac = cand_autoc;
                double ea = ac[0];
                for (int j = 0; j < ord; ++j) ea -= (double)lpc[j] * ac[j + 1];
                if (!(ea > 0.0)) ea = 0.0;

                double score[16];
                for (size_t i = 0; i < precisions.size(); ++i)
                    score[i] = std::numeric_limits<double>::max();

                for (size_t i = 0; i < precisions.size(); ++i) {
                    const int prec = precisions[i];
                    // Same bound the encoding loop uses, so the ladder never
                    // scores a rung that loop would have refused to enter.
                    if (hdr_fixed + (uint32_t)ord * (eff_bps + (uint32_t)prec)
                            >= best_lpc_cost)
                        break;
                    int32_t qc[32];
                    int     shift   = 0;
                    bool    clamped = false;
                    if (!quantize_lpc_coeffs(lpc, ord, prec, log2cmax, qc, &shift,
                                             &clamped))
                        continue;
                    const double scale = 1.0 / (double)((int64_t)1 << shift);
                    double d[32];
                    for (int j = 0; j < ord; ++j)
                        d[j] = (double)qc[j] * scale - (double)lpc[j];
                    double drd = 0.0;
                    for (int a = 0; a < ord; ++a) {
                        double row = 0.0;
                        for (int b = 0; b < ord; ++b)
                            row += d[b] * ac[std::abs(a - b)];
                        drd += d[a] * row;
                    }
                    const double e = ea + drd;
                    // order*prec is the exact coefficient cost; the rest is the
                    // Gaussian entropy of the predicted residual. Terms that do
                    // not vary across the ladder (the header, the window-energy
                    // normalisation) are dropped — only the ordering matters.
                    score[i] = (double)ord * (double)prec
                             + 0.5 * (double)(bsize - (uint32_t)ord)
                               * std::log2(e > 0.0 ? e : 1e-300);
                }

                for (size_t i = 0; i < precisions.size(); ++i) rung_sel[i] = false;
                for (unsigned k = 0; k < precision_rungs; ++k) {
                    size_t best_i = precisions.size();
                    double best_s = std::numeric_limits<double>::max();
                    for (size_t i = 0; i < precisions.size(); ++i)
                        if (!rung_sel[i] && score[i] < best_s) {
                            best_s = score[i];
                            best_i = i;
                        }
                    if (best_i == precisions.size()) break;
                    rung_sel[best_i] = true;
                }
            }

#ifdef FLACOUT_INSTRUMENT
            int prec_idx = 0;
#endif
            for (size_t pi = 0; pi < precisions.size(); ++pi) {
                const int prec = precisions[pi];
                INSTR(++prec_idx);
                // fixed cost is a lower bound on this candidate (rice >= 0);
                // grows with precision, so break once it can't beat best.
                uint32_t hdr = hdr_fixed + (uint32_t)ord * (eff_bps + (uint32_t)prec);
                if (hdr >= best_lpc_cost) {
                    INSTR(g_instr.prec_pruned_break.fetch_add(precisions.size() - prec_idx + 1, std::memory_order_relaxed));
                    break;
                }
                // Ascending order is kept even when most rungs are skipped: the
                // delta update below needs shift == prev_shift + 1, so two
                // adjacent selected rungs still avoid a full recompute.
                if (use_ladder && !rung_sel[pi]) continue;
                INSTR(g_instr.prec_iters.fetch_add(1, std::memory_order_relaxed));

                // Quantize with error feedback (mirrors libFLAC; see
                // quantize_lpc_coeffs). Out-of-range taps are clamped, not
                // discarded — a clamped predictor may still win this order.
                int32_t qc[32];
                int     shift   = 0;
                bool    clamped = false;
                if (!quantize_lpc_coeffs(lpc, ord, prec, log2cmax, qc, &shift, &clamped))
                    continue; // degenerate coefficients, no usable quantization
                if (clamped) { INSTR(g_instr.overflow_skips.fetch_add(1, std::memory_order_relaxed)); }


                // Have we already priced this exact predictor?
                bool memo_hit = false;
                uint64_t mh = 0;
                if (use_memo) {
                    mh = 1469598103934665603ull ^ (uint64_t)ord;
                    mh = mh * 1099511628211ull ^ (uint64_t)(uint32_t)shift;
                    for (int j = 0; j < ord; ++j)
                        mh = (mh ^ (uint64_t)(uint32_t)qc[j]) * 1099511628211ull;
                    auto it = memo_ix.find(mh);
                    if (it != memo_ix.end()) {
                        for (uint32_t mi : it->second) {
                            const MemoEnt& e = memo[mi];
                            if (e.ord != ord || e.shift != shift) continue;
                            if (std::memcmp(&memo_qc[e.qoff], qc,
                                            (size_t)ord * sizeof(int32_t)) != 0) continue;
                            const uint32_t cost2 = hdr + 6u + e.rice;
                            if (cost2 < cand_best) cand_best = cost2;
                            if (cost2 < best_lpc_cost) {
                                best_lpc_cost = cost2;
                                bl_ord = ord; bl_prec = prec; bl_shift = shift;
                                std::memcpy(bl_qc, qc, (size_t)ord * sizeof(int32_t));
                                INSTR(instr_best_win = (int)wt);
                            }
                            memo_hit = true;
                            break;
                        }
                    }
                }
                if (memo_hit) continue;

                INSTR(g_instr.residual_calls.fetch_add(1, std::memory_order_relaxed));

                // One ladder step up from the coefficients pred[] was built
                // for: apply the small integer correction d[j] = qc[j] -
                // 2*prev_qc[j] instead of recomputing. With error feedback in
                // the quantizer the per-tap errors can exceed half an LSB, so
                // d[j] lands in a small range around zero rather than strictly
                // {-1,0,1}; the kernel multiplies by d[j] so any magnitude is
                // exact — the only gate is the int32 accumulator bound, which
                // caps sum|d| (a clamped tap can blow d up arbitrarily).
                bool delta_done = false;
                if (have_pred && shift == prev_shift + 1) {
                    int32_t d[32];
                    int64_t sum_abs_d = 0;
                    for (int j = 0; j < ord; ++j) {
                        d[j] = qc[j] - 2 * prev_qc[j];
                        sum_abs_d += std::abs((int64_t)d[j]);
                    }
                    if (sum_abs_d <= max_sum_abs_d) {
                        INSTR(g_instr.residual_delta.fetch_add(1, std::memory_order_relaxed));
                        update_lpc_residuals_delta(shifted.data(), bsize, d, ord,
                                                   shift, pred.data(), residuals.data());
                        delta_done = true;
                    }
                }
                if (!delta_done) {
                    INSTR(g_instr.residual_macs.fetch_add((uint64_t)(bsize - ord) * ord, std::memory_order_relaxed));
                    compute_lpc_residuals(shifted.data(), bsize, qc, ord, shift,
                                          residuals.data(), pred_ptr,
                                          max_sum_abs_qc);
                }
                std::memcpy(prev_qc, qc, (size_t)ord * sizeof(int32_t));
                prev_shift = shift;
                have_pred  = (pred_ptr != nullptr);

                // Cost only — no out-params. Filling in a SubframeParams
                // here would zero and populate ~1.3 KB (rice_k[256] alone is
                // 1 KB) for every candidate, and all but one of them is
                // discarded. Only the winner's identity is kept; its
                // parameters are reconstructed once, after the search.
                uint32_t rice = calculate_rice_cost(residuals.data(), bsize,
                                                    (uint32_t)ord, nullptr);
                // +6: residual block header (2-bit coding method + 4-bit
                // partition order), see estimate_subframe_cost for detail.
                const uint32_t cost = hdr + 6u + rice;
#ifdef FLACOUT_DUMP_GOLDEN
                if (golden_active) {
                    std::lock_guard<std::mutex> lk(g_golden.mu);
                    int32_t rec_ord = ord, rec_shift = shift;
                    int32_t rec_qc[32] = {};
                    for (int j = 0; j < ord; ++j) rec_qc[j] = qc[j];
                    std::fwrite(&rec_ord,   4,  1, g_golden.fh);
                    std::fwrite(&rec_shift, 4,  1, g_golden.fh);
                    std::fwrite(rec_qc,     4, 32, g_golden.fh);
                    std::fwrite(&rice,      4,  1, g_golden.fh);
                    std::fwrite(&cost,      4,  1, g_golden.fh);
                    g_golden.ncand++;
                }
#endif
#ifdef FLACOUT_DUMP_FP32RANK
                {
                    // The same candidate priced the way a GPU fp32 kernel
                    // would. Coefficients are pre-scaled by 2^-shift, which is
                    // exact (power-of-two divisor), so every bit of error here
                    // comes from the products and their accumulation — the
                    // thing fp32 genuinely cannot represent.
                    //
                    // floor(), not rint(): the integer path is an arithmetic
                    // right shift, so flooring isolates precision loss instead
                    // of confounding it with a different rounding rule.
                    //
                    // Conservative against a real GPU in one respect: this is
                    // built with -ffp-contract=off, so no FMA. A GPU kernel
                    // would contract and be slightly *more* accurate.
                    fp32_res.resize(bsize);
                    const float inv = 1.0f / (float)((int64_t)1 << shift);
                    float fc[32];
                    for (int j = 0; j < ord; ++j) fc[j] = (float)qc[j] * inv;
                    for (int j = 0; j < ord; ++j) fp32_res[j] = shifted[j];
                    for (uint32_t i2 = (uint32_t)ord; i2 < bsize; ++i2) {
                        float sum = 0.0f;
                        for (int j = 0; j < ord; ++j)
                            sum += fc[j] * (float)shifted[i2 - 1 - j];
                        fp32_res[i2] = shifted[i2] - (int32_t)std::floor(sum);
                    }
                    const uint32_t frice = calculate_rice_cost(
                        fp32_res.data(), bsize, (uint32_t)ord, nullptr);
                    fp32_costs.emplace_back(cost, hdr + 6u + frice);
                }
#endif
#ifdef FLACOUT_DUMP_PRECISION
                {
                    // E_a = r0 - a'r, the Levinson residual energy at this
                    // order; drd = (aq - a)'R(aq - a), the extra energy this
                    // rung's quantization introduces. Predicted rung energy is
                    // their sum — everything else in the cost is either exact
                    // (ord*prec) or constant across the ladder.
                    const double* ac = cand_autoc;
                    double ea = ac[0];
                    for (int j = 0; j < ord; ++j) ea -= (double)lpc[j] * ac[j+1];
                    const double scale = 1.0 / (double)((int64_t)1 << shift);
                    double d[32];
                    for (int j = 0; j < ord; ++j)
                        d[j] = (double)qc[j] * scale - (double)lpc[j];
                    // Full form and three cheaper approximations of it: the
                    // diagonal alone (O(ord)) and bands of width 1 and 4
                    // (O(ord*L)). The full form is O(ord^2), which at order 32
                    // is a sixth of a residual pass -- worth knowing whether a
                    // band buys the same ranking for less.
                    double drd = 0.0, drd1 = 0.0, drd4 = 0.0, sd2 = 0.0;
                    for (int i = 0; i < ord; ++i) {
                        sd2 += d[i] * d[i];
                        for (int j = 0; j < ord; ++j) {
                            const int lag = std::abs(i - j);
                            const double t = d[i] * d[j] * ac[lag];
                            drd += t;
                            if (lag <= 1) drd1 += t;
                            if (lag <= 4) drd4 += t;
                        }
                    }
                    std::lock_guard<std::mutex> lk(g_pdump.mu);
                    std::fprintf(g_pdump.fh,
                        "%llu\t%d\t%d\t%u\t%u\t%d\t%d\t%d\t%u\t%.10g\t%.10g\t%.10g\t%.10g"
                        "\t%.10g\t%.10g\t%.10g\n",
                        (unsigned long long)pdump_sf, (int)wt, ord, bsize, eff_bps,
                        prec, shift, clamped ? 1 : 0, cost, ea, drd,
                        window_energy(wt, bsize), ac[0], sd2, drd1, drd4);
                }
#endif
                if (use_memo) {
                    MemoEnt e{ord, shift, (uint32_t)memo_qc.size(), rice};
                    memo_qc.insert(memo_qc.end(), qc, qc + ord);
                    memo_ix[mh].push_back((uint32_t)memo.size());
                    memo.push_back(e);
                }

                if (cost < cand_best) cand_best = cost;
                if (cost < best_lpc_cost) {
                    best_lpc_cost = cost;
                    bl_ord = ord; bl_prec = prec; bl_shift = shift;
                    std::memcpy(bl_qc, qc, (size_t)ord * sizeof(int32_t));
                    INSTR(instr_best_win = (int)wt);
                }
            }
            return cand_best;
        };

        // Autocorrelation + Levinson-Durbin for one window. Returns false if the
        // windowed signal has no energy and the window should be skipped.
        auto analyse_window = [&](WindowType wt, double* out_err) -> bool {
            INSTR(g_instr.windows_run.fetch_add(1, std::memory_order_relaxed));
            INSTR(g_instr.window_samples.fetch_add(bsize, std::memory_order_relaxed));
            apply_window(samples, bsize, wasted, wt, windowed.data());

            // Autocorrelation for all lags 0..min(32, bsize-1)
            double autoc[33] = {};
            int max_lag = (int)std::min((uint32_t)32, bsize - 1);
#ifdef FLACOUT_INSTRUMENT
            for (int lag = 0; lag <= max_lag; ++lag)
                g_instr.autoc_macs.fetch_add(bsize - (uint32_t)lag, std::memory_order_relaxed);
#endif
            autocorrelation(windowed.data(), bsize, max_lag, autoc);
            std::memcpy(cur_autoc, autoc, sizeof(autoc));
            if (autoc[0] <= 0.0) return false;

            std::memset(all_lpc, 0, sizeof(all_lpc));
            compute_lpc_all_orders(autoc, all_lpc, max_order, out_err);
            return true;
        };

        bool gpu_done = false;
        // One timed CPU candidate per subframe, which is what the adaptive
        // throttle compares the device against (GpuEvaluator::note_cpu). It has
        // to be measured here rather than modelled: the rate that matters is one
        // worker thread's, under whatever load the rest of the pool is putting on
        // the machine right now, and that is not knowable from device properties.
        //
        // One sample per subframe, not per candidate, so the clock pair is noise
        // next to the candidate it times; and none of it is compiled unless
        // Vulkan is, nor entered unless a device came up.
#ifdef FLACOUT_HAVE_VULKAN
        bool cpu_timed = !(gpu && gpu->available());
#else
        bool cpu_timed = true;
#endif
        auto eval_candidate_timed = [&](const float* lpc, int ord,
                                        WindowType wt) -> uint32_t {
            if (cpu_timed) return eval_candidate(lpc, ord, wt);
            cpu_timed = true;
#ifdef FLACOUT_HAVE_VULKAN
            const auto t0 = std::chrono::steady_clock::now();
            const uint32_t r = eval_candidate(lpc, ord, wt);
            const double dt =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count();
            // eval_candidate covers the whole precision ladder for one (window,
            // order), where a GPU candidate is one (window, order, precision) --
            // so scale to the same nominal MAC count the device reports.
            gpu->note_cpu((uint64_t)(bsize - (uint32_t)ord) * (uint64_t)ord *
                              (uint64_t)precisions.size(), dt);
            return r;
#else
            return eval_candidate(lpc, ord, wt);
#endif
        };

#ifdef FLACOUT_HAVE_VULKAN
        // ---- GPU: one batch per subframe (-G) ----
        //
        // Only the exhaustive path offloads. The ranked driver's patience rule
        // is a sequential stop condition -- it decides whether to look at the
        // next candidate from the cost of the last -- so there is no batch to
        // form. That is not a limitation worth working around: `-G` exists to
        // make the unlimited sweep affordable, which is the search the ranking
        // was invented to avoid.
        //
        // The GPU skips the header-bound pruning the CPU loop does. Pruned
        // candidates have hdr >= best_lpc_cost and rice >= 0, so they could
        // never have won; evaluating them anyway costs time and changes
        // nothing. The winner, and therefore the output, is identical.
        if (gpu && gpu->available() && (bsize % 32u) == 0u &&
            max_candidates == 0 && precision_rungs == 0 &&
            gpu->would_accept()) {
            struct Meta { int ord, prec, shift; };
            std::vector<GpuEvaluator::Candidate> gcands;
            std::vector<Meta> meta;
            gcands.reserve(windows.size() * (size_t)max_order * precisions.size());
            meta.reserve(gcands.capacity());

            for (WindowType wt : windows) {
                if (!analyse_window(wt, nullptr)) continue;
                for (int ord = 1; ord <= max_order; ++ord) {
                    const float* lpc = all_lpc[ord - 1];
                    const int log2cmax = lpc_log2cmax(lpc, ord);
                    for (size_t pi = 0; pi < precisions.size(); ++pi) {
                        const int prec = precisions[pi];
                        GpuEvaluator::Candidate cd{};
                        int  shift = 0;
                        bool clamped = false;
                        if (!quantize_lpc_coeffs(lpc, ord, prec, log2cmax,
                                                 cd.qc, &shift, &clamped))
                            continue;
                        cd.order = ord;
                        cd.shift = shift;
                        gcands.push_back(cd);
                        meta.push_back({ord, prec, shift});
                    }
                }
            }

            std::vector<uint32_t> gcosts;
            if (gcands.size() >= gpu->min_batch() &&
                gpu->evaluate(shifted.data(), bsize, gcands, gcosts)) {
                for (size_t i = 0; i < gcosts.size(); ++i) {
                    const uint32_t hdr = hdr_fixed +
                        (uint32_t)meta[i].ord * (eff_bps + (uint32_t)meta[i].prec);
                    const uint32_t cost = hdr + 6u + gcosts[i];
                    if (cost < best_lpc_cost) {
                        best_lpc_cost = cost;
                        bl_ord   = meta[i].ord;
                        bl_prec  = meta[i].prec;
                        bl_shift = meta[i].shift;
                        std::memcpy(bl_qc, gcands[i].qc, sizeof(bl_qc));
                    }
                }
                gpu_done = true;
            }
        }
#endif

        if (max_candidates == 0 && !gpu_done) {
            // ---- Exhaustive: every (window, order) pair ----
            for (WindowType wt : windows) {
                if (!analyse_window(wt, nullptr)) continue;
                for (int ord = 1; ord <= max_order; ++ord) {
                    // sound lower bound: fixed cost (header + warm-up + coeffs) alone,
                    // at the cheapest precision, since rice cost >= 0. it grows with
                    // order, so once it can't beat best neither can any higher order.
                    uint32_t hdr_min = hdr_fixed + (uint32_t)ord * (eff_bps + min_prec);
                    if (hdr_min >= best_lpc_cost) {
                        INSTR(g_instr.order_pruned_break.fetch_add(max_order - ord + 1, std::memory_order_relaxed));
                        break;
                    }
                    eval_candidate_timed(all_lpc[ord - 1], ord, wt);
                }
            }
        } else if (!gpu_done) {
            // ---- Ranked: score every (window, order) pair, then fully
            //      evaluate only the most promising `max_candidates` of them.
            //
            // Levinson-Durbin already produces the residual energy at each
            // order as a by-product; the exhaustive path just discards it. For
            // a residual that is roughly stationary across the block, that
            // windowed-domain energy is ~ var_e x sum(w^2), so dividing by the
            // window's energy estimates the absolute residual variance, which
            // the Gaussian entropy 0.5*log2(2*pi*e*var) (the same model
            // estimate_lpc_bits_fast uses) turns into estimated residual bits.
            // Absolute variance is what makes scores comparable across windows:
            // each window scales the signal differently, so raw err[ord]
            // values are in different units.
            //
            // Scoring by the err[ord]/err[0] *ratio* instead (energy fraction
            // removed) was tried first and measures ~0.02-0.26% worse across
            // real music and synthetics: it normalizes by the signal power
            // seen *through the window*, which flatters windows aimed at
            // quiet parts of the block. Also tried and rejected: scoring by
            // predicted residual energy on the RAW signal via the quadratic
            // form a'Ra over the unwindowed autocorrelation — the
            // autocorrelation method's zero-extension edge bias exceeds the
            // true residual energy by orders of magnitude on predictable
            // content, which is the very pathology windowing exists to kill.
            //
            // The *peephole* problem needs one more term: partial/punchout
            // windows genuinely predict the slice of block they look at, so
            // their windowed error says nothing about the samples they zero
            // out — pricing those as free floods the top-N with slice-local
            // candidates (measured: all-26 at N=8 lost to tukey050 alone).
            // Fix: charge the blind fraction of the block above what the
            // model claims, toward the raw signal's variance. Dense windows
            // have zero_frac == 0 and are unaffected either way.
            //
            // Calibration (FLACOUT_BLIND_BETA). Charging the blind region the
            // *full* raw variance, as this first did, treats the predictor as
            // worthless outside the window; it is not, because audio is
            // locally stationary. Measured too pessimistic by about 4x: it
            // priced sparse windows out of the ranked top-N even where exact
            // costing shows them winning. On a 188-track corpus, moving beta
            // 1.0 -> 0.25 leaves the dense-window default alone (+12 B) and
            // takes 13,082 B (-0.078%) off -a adaptive mode, whose whole
            // purpose is routing transient blocks to these windows. Offered
            // punchouttukey2_033 as a 5th window on held-out excerpts, its
            // gain goes from -2711 B to about -5600 B against a -9068 B
            // exact-pricing ceiling. beta = 0 (no penalty) is not the answer:
            // it re-floods the pool, costing +5762 B when all 26 windows are
            // offered at -c 8.
            //
            // FLACOUT_BLIND_EPS exists because the blind test is an equality
            // against zero, which a shape can dodge by sitting just above it
            // — a CMA-ES window search (bench/window_search.py) found exactly
            // that exploit, worth 1257 B of a 5332 B gain. A relative
            // threshold closes it and buys a little more besides, but it also
            // counts dense windows' taper skirts as blind, which cost +3442 B
            // on the 3-minute tonal fixture at eps = 1e-2 for a further
            // -558 B on the corpus. Left at 0 (exact zero) on that evidence;
            // with beta at 0.25 the exploit is worth a quarter of what it was.
            struct Cand {
                double score; int wi; int ord;
#ifdef FLACOUT_DUMP_CANDIDATES
                // Everything the scorer saw, carried so the dump can ask
                // offline whether a better function of these would have
                // ranked the eventual winner higher. The neighbouring
                // Levinson errors are what the scorer throws away: E_m/E_{m-1}
                // is the reflection coefficient at this order (|k|^2 = 1 -
                // E_m/E_{m-1}), E_{m+1} says whether the next order is still
                // paying, and E_max says how much is left to win at all.
                double lderr0, lderr_ord, wsq, zf, model_bps;
                double lderr_prev, lderr_next, lderr_1, lderr_max;
                int    max_ord;
#endif
            };
            std::vector<Cand> cands;
            cands.reserve(windows.size() * (size_t)max_order);


// How much of the gap between the model's claim and the raw signal's cost a
// blind sample is charged. 1.0 — the original — assumes the predictor has no
// power at all outside the window, which measured 4x too pessimistic: audio
// is locally stationary, so coefficients fit on the visible slice do predict
// the rest. See the calibration note below.
#ifndef FLACOUT_BLIND_BETA
#define FLACOUT_BLIND_BETA 0.25
#endif

            // Entropy terms are clamped at zero: the Gaussian differential
            // entropy goes negative once var < 1/(2πe), but Rice cannot code
            // below ~0 bits/sample, and letting a peephole window's
            // tiny-slice variance contribute large *negative* bits vaulted
            // partial windows over every dense candidate on pure-tone
            // content (+4523 B on the 3-min synthetic tonal fixture).
            double var_raw = 0.0;
            for (uint32_t i = 0; i < bsize; ++i)
                var_raw += (double)shifted[i] * (double)shifted[i];
            var_raw /= (double)bsize;
            const double raw_bits_per_sample = (var_raw > 0.0)
                ? std::max(0.0, 0.5 * std::log2(2.0 * M_PI * M_E * var_raw)) : 0.0;

            // all_lpc for every window, so the winning candidates do not have to
            // re-run apply_window + autocorrelation to get their coefficients
            // back. 4 KB per window.
            constexpr size_t LPC_STRIDE = 32 * 32;
            std::vector<float> lpc_store(windows.size() * LPC_STRIDE, 0.0f);
            // One autocorrelation per window, kept for the ladder model: the
            // scan evaluates candidates in ranked order, long after the loop
            // below has moved on from their window.
            std::vector<double> autoc_store(windows.size() * 33, 0.0);

            for (size_t wi = 0; wi < windows.size(); ++wi) {
                double lderr[33];
                if (!analyse_window(windows[wi], lderr)) continue;
                std::memcpy(&lpc_store[wi * LPC_STRIDE], all_lpc, sizeof(all_lpc));
                std::memcpy(&autoc_store[wi * 33], cur_autoc, sizeof(cur_autoc));
                if (lderr[0] <= 0.0) continue;
                const double wsq = window_energy(windows[wi], bsize);
                if (!(wsq > 0.0)) continue;
                const double zf = window_zero_frac(windows[wi], bsize);
                for (int ord = 1; ord <= max_order; ++ord) {
                    if (lderr[ord] <= 0.0) continue; // recursion stopped short of this order
                    const double var_e = lderr[ord] / wsq;
                    const double model_bits_per_sample =
                        std::max(0.0, 0.5 * std::log2(2.0 * M_PI * M_E * var_e));
                    // Blind samples are priced between what the model claims
                    // and what the raw signal costs. FLACOUT_BLIND_BETA == 1
                    // charges them the full raw entropy — no predictive power
                    // at all outside the window — which is the pessimistic
                    // end; the coefficients are fit on the visible slice but
                    // audio is locally stationary, so they do predict the rest
                    // somewhat.
                    const double blind_bits_per_sample =
                        model_bits_per_sample
                        + (double)FLACOUT_BLIND_BETA
                          * (raw_bits_per_sample - model_bits_per_sample);
                    const double resid_bits =
                        (model_bits_per_sample * (1.0 - zf)
                         + blind_bits_per_sample * zf) * (double)(bsize - ord);
                    const double coef_bits  = (double)ord * (double)(eff_bps + min_prec);
#ifdef FLACOUT_DUMP_CANDIDATES
                    // -1 marks orders the recursion stopped short of; fall
                    // back to this order's own error so ratios stay finite.
                    auto lde = [&](int k) {
                        if (k < 0 || k > max_order) return lderr[ord];
                        return lderr[k] > 0.0 ? lderr[k] : lderr[ord];
                    };

                    cands.push_back({ resid_bits + coef_bits, (int)wi, ord,
                                      lderr[0], lderr[ord], wsq, zf,
                                      model_bits_per_sample,
                                      lde(ord - 1), lde(ord + 1), lde(1),
                                      lde(max_order), max_order });
#else
                    cands.push_back({ resid_bits + coef_bits, (int)wi, ord });
#endif
                }
            }

#ifdef FLACOUT_DUMP_CANDIDATES
            const uint64_t dump_sf =
                g_dump.subframe.fetch_add(1, std::memory_order_relaxed);
#endif
            auto by_score = [](const Cand& a, const Cand& b) { return a.score < b.score; };
            size_t keep = std::min((size_t)max_candidates, cands.size());
#ifdef FLACOUT_HAVE_VULKAN
            // ---- GPU pre-pass for the ranked driver ----
            //
            // The ranked scan looks sequential, but its only state is
            // best_lpc_cost, and the one thing that reads it -- eval_candidate's
            // precision pruning -- can only skip precisions whose header alone
            // already exceeds it. Those can never lower best_lpc_cost, so
            // pricing them anyway changes nothing. The whole ranked list is
            // therefore computable in one batch, and the scan replayed against
            // it lands on the same winner, in the same order, with the same
            // patience stop. Byte-identical, not merely equivalent.
            //
            // Not usable with -L: the ladder's rung scoring *selects* on
            // best_lpc_cost rather than merely skipping losers, so batching
            // would change which rungs are encoded and therefore the output.
            std::vector<uint32_t> gcost;      // per ranked candidate: best cost
            std::vector<int>      gprec, gshift;
            std::vector<std::array<int32_t,32>> gqc;
            bool gpu_ranked = false;
            auto gpu_prepass = [&]() {
                if (!gpu || !gpu->available() || (bsize % 32u) != 0u) return;
                if (precision_rungs != 0) return;
                if (!gpu->would_accept()) return;
                // Batch only as far down the ranked list as the scan can
                // plausibly reach. Pricing the whole list is speculative work,
                // and at small -c almost all of it is thrown away: with
                // -e -c 8 the list is 26x32x8 = 6656 candidates and patience
                // consumes ~50, so the GPU did 100x the CPU's work and ran
                // 0.94x. The scan cannot pass `keep` until it has seen
                // `patience` consecutive non-improvements, so keep + a few
                // patience windows covers it in all but pathological cases --
                // and anything past the batch simply falls back to
                // eval_candidate, which is the same answer either way.
                const size_t reach = std::min(cands.size(),
                    (size_t)keep + 4u * (size_t)patience + 16u);
                // Decline before building, not after. The build below runs
                // quantize_lpc_coeffs over every (candidate, precision) pair in
                // `reach`, and until this check moved up here a min_batch
                // rejection threw all of it away and eval_candidate quantized
                // the same pairs again -- so the knob meant to keep small work
                // on the CPU was buying pure overhead instead.
                // `reach * precisions.size()` is the exact upper bound on what
                // the loop can append, so this never declines a batch the old
                // order would have accepted.
                //
                // Justified structurally, not by a stopwatch: the only machine
                // with a discrete GPU available had a 16-23% run-to-run noise
                // floor (measured, 8 interleaved repeats), which swamps a
                // change this size. Do not quote a speedup for it without a
                // quiet machine.
                if (reach * precisions.size() < gpu->min_batch()) return;
                std::vector<GpuEvaluator::Candidate> batch;
                std::vector<std::pair<uint32_t,int>> owner;   // (ranked index, prec)
                batch.reserve(reach * precisions.size());
                owner.reserve(batch.capacity());
                for (size_t c = 0; c < reach; ++c) {
                    const Cand& cd = cands[c];
                    const float* lpc =
                        &lpc_store[cd.wi * LPC_STRIDE + (size_t)(cd.ord - 1) * 32];
                    const int log2cmax = lpc_log2cmax(lpc, cd.ord);
                    for (size_t pi = 0; pi < precisions.size(); ++pi) {
                        GpuEvaluator::Candidate gc{};
                        int  sh = 0; bool cl = false;
                        if (!quantize_lpc_coeffs(lpc, cd.ord, precisions[pi],
                                                 log2cmax, gc.qc, &sh, &cl)) continue;
                        gc.order = cd.ord; gc.shift = sh;
                        batch.push_back(gc);
                        owner.emplace_back((uint32_t)c, precisions[pi]);
                    }
                }
                std::vector<uint32_t> costs;
                if (batch.size() < gpu->min_batch() ||
                    !gpu->evaluate(shifted.data(), bsize, batch, costs))
                    return;
                gcost.assign(reach, std::numeric_limits<uint32_t>::max());
                gprec.assign(reach, 0);
                gshift.assign(reach, 0);
                gqc.resize(reach);
                for (size_t i = 0; i < costs.size(); ++i) {
                    const uint32_t c = owner[i].first;
                    const uint32_t hdr = hdr_fixed +
                        (uint32_t)batch[i].order * (eff_bps + (uint32_t)owner[i].second);
                    const uint32_t tot = hdr + 6u + costs[i];
                    if (tot < gcost[c]) {
                        gcost[c]  = tot;
                        gprec[c]  = owner[i].second;
                        gshift[c] = batch[i].shift;
                        std::memcpy(gqc[c].data(), batch[i].qc, 32 * sizeof(int32_t));
                    }
                }
                gpu_ranked = true;
            };

            // Same contract as eval_candidate: returns the candidate's best
            // cost and folds it into the running winner.
            auto eval_ranked_gpu = [&](size_t c) -> uint32_t {
                const uint32_t cc = gcost[c];
                if (cc < best_lpc_cost) {
                    best_lpc_cost = cc;
                    bl_ord   = cands[c].ord;
                    bl_prec  = gprec[c];
                    bl_shift = gshift[c];
                    std::memcpy(bl_qc, gqc[c].data(), sizeof(bl_qc));
                }
                return cc;
            };
#endif

            std::partial_sort(cands.begin(), cands.begin() + keep, cands.end(), by_score);
#ifdef FLACOUT_HAVE_VULKAN
            if (patience == 0) gpu_prepass();
#endif

            // ---- Patience ----
            // The winner's rank is heavy-tailed, not tied: measured over all
            // candidates, rank 0 wins 56% of sine24 subframes but the tail
            // reaches rank 59, and on music only ~51% of winners fall inside
            // rank 7. A fixed cut at N therefore misses a long tail that a
            // wider tolerance band around rank 0 cannot reach.
            //
            // Use the exact costs already being computed as the stopping
            // signal instead: descend the ranked list and keep going while it
            // is still yielding improvements, stopping only after `patience`
            // consecutive candidates fail to beat the best. Where the model
            // ordered well, the first few confirm it and the scan stops near
            // N; where it ordered badly, the scan follows the improvements
            // down. Cost is paid only on the subframes that need it.
            if (patience > 0) {
                std::sort(cands.begin(), cands.end(), by_score);
#ifdef FLACOUT_HAVE_VULKAN
                gpu_prepass();
#endif
                uint32_t prev  = best_lpc_cost;
                size_t   since = 0;
                for (size_t c = 0; c < cands.size(); ++c) {
                    if (c >= keep && since >= patience) break;
                    const Cand& cd = cands[c];
                    uint32_t hdr_min = hdr_fixed + (uint32_t)cd.ord * (eff_bps + min_prec);
                    if (hdr_min >= best_lpc_cost) { ++since; continue; }
                    cand_autoc = &autoc_store[(size_t)cd.wi * 33];
                    uint32_t cc;
#ifdef FLACOUT_HAVE_VULKAN
                    if (gpu_ranked && c < gcost.size()) cc = eval_ranked_gpu(c); else
#endif
                    cc = eval_candidate_timed(
                        &lpc_store[cd.wi * LPC_STRIDE + (size_t)(cd.ord - 1) * 32],
                        cd.ord, windows[cd.wi]);
#ifdef FLACOUT_DUMP_CANDIDATES
                    if (cc != std::numeric_limits<uint32_t>::max()) {
                        std::lock_guard<std::mutex> lk(g_dump.mu);
                        std::fprintf(g_dump.fh,
                            "%llu\t%d\t%d\t%u\t%u\t%.10g\t%.10g\t%.10g\t%.6g\t"
                            "%.10g\t%.10g\t%.10g\t%.10g\t%u\t"
                            "%.10g\t%.10g\t%.10g\t%.10g\t%d\n",
                            (unsigned long long)dump_sf, (int)windows[cd.wi], cd.ord,
                            bsize, eff_bps, cd.lderr0, cd.lderr_ord, cd.wsq, cd.zf,
                            cd.model_bps, raw_bits_per_sample, var_raw, cd.score, cc,
                            cd.lderr_prev, cd.lderr_next, cd.lderr_1, cd.lderr_max,
                            cd.max_ord);
                    }
#else
                    (void)cc;
#endif
                    if (best_lpc_cost < prev) { prev = best_lpc_cost; since = 0; }
                    else ++since;
                }
                keep = 0;  // scan already done
            }

            // Best-first, so the exact cost of a strong candidate tightens the
            // pruning bound before the weaker ones are tried.
            for (size_t c = 0; c < keep; ++c) {
                const Cand& cd = cands[c];
                uint32_t hdr_min = hdr_fixed + (uint32_t)cd.ord * (eff_bps + min_prec);
                if (hdr_min >= best_lpc_cost) continue;
                cand_autoc = &autoc_store[(size_t)cd.wi * 33];
                uint32_t cc;
#ifdef FLACOUT_HAVE_VULKAN
                if (gpu_ranked && c < gcost.size()) cc = eval_ranked_gpu(c); else
#endif
                cc = eval_candidate_timed(
                    &lpc_store[cd.wi * LPC_STRIDE + (size_t)(cd.ord - 1) * 32],
                    cd.ord, windows[cd.wi]);
#ifdef FLACOUT_DUMP_CANDIDATES
                if (cc != std::numeric_limits<uint32_t>::max()) {
                    std::lock_guard<std::mutex> lk(g_dump.mu);
                    std::fprintf(g_dump.fh,
                        "%llu\t%d\t%d\t%u\t%u\t%.10g\t%.10g\t%.10g\t%.6g\t"
                        "%.10g\t%.10g\t%.10g\t%.10g\t%u\t"
                        "%.10g\t%.10g\t%.10g\t%.10g\t%d\n",
                        (unsigned long long)dump_sf, (int)windows[cd.wi], cd.ord,
                        bsize, eff_bps, cd.lderr0, cd.lderr_ord, cd.wsq, cd.zf,
                        cd.model_bps, raw_bits_per_sample, var_raw, cd.score, cc,
                        cd.lderr_prev, cd.lderr_next, cd.lderr_1, cd.lderr_max,
                        cd.max_ord);
                }
#else
                (void)cc;
#endif
            }
        }

        // ---- Coefficient-lattice refinement (-Q) ----
        // Every candidate above got its integer coefficients from one fixed
        // rule: round the Levinson solution with error feedback. That rule
        // minimizes the quantization error's quadratic form d'Rd, which is not
        // the cost we are actually paying — Rice bits are a step function of
        // the residual magnitudes, so the cheapest lattice point near the
        // real-valued optimum need not be the rounded one. Nothing in the
        // search ever revisits it.
        //
        // Coordinate descent fixes that for the winner: try each tap at +-1,
        // keep any perturbation that lowers the *exact* cost, sweep until a
        // full pass finds nothing. FFmpeg's multi_dim_quant (flacenc.c) does
        // the same search by enumerating all 3^order corners with at most 8
        // taps differing, which is exponential and therefore unusable above
        // order ~12 — it is off by default there. A coordinate sweep is
        // 2*order evaluations and finds the same axis-local minimum.
        //
        // This runs once per subframe on the winner alone, so it does not
        // multiply the search: 2*order*sweeps residual+Rice passes against the
        // thousands the ranked scan already ran. It cannot lose bits — a
        // perturbation is adopted only when the exact cost strictly drops.
        // Precision gate. The lattice step is 2^-shift, so a low-precision
        // winner was rounded far from the cost-optimal lattice point and has
        // room to search, while a prec-15 one is already essentially on it —
        // measured over every perturbation, 97.1% of the improvements on a
        // gaining fixture sit at prec <= 9, and a fixture whose winners are
        // mostly prec 11-15 yields nothing at all.
        //
        // Gating strictly improves the frontier (master mix, -Q 2, against
        // -Q 0 at 0.80 s / 16827941 B, interleaved best-of-4 on an idle
        // machine -- see trap 9, an earlier version of this table was measured
        // against a second encoder and read ~0.015x high throughout):
        //
        //   gate  time   x       bytes
        //   <=8   -      -       -3020   (50.5% of ungated -- too aggressive)
        //   <=9   0.84   1.050   -5626   (94.1%)
        //   <=10  0.85   1.062   -5851   (97.8%)  <- default
        //   <=11  0.86   1.075   -5901   (98.7%)
        //   <=12  0.86   1.075   -5934   (99.2%)
        //   none  0.89   1.113   -5981   (100%)
        //
        // Read that table honestly: 10, 11 and 12 are inside timing noise of
        // each other (0.85/0.86/0.86), so the real finding is "gate somewhere
        // in 10-12", and 10 is picked for holding the most gain per unit time
        // over the <=9 step (+225 B for +0.012x, against +83 B for the same
        // 0.013x from 10 to 12). Only the ends are firm: <=8 throws away half
        // the gain, and ungated pays 0.051x more than <=10 for 2.2% more.
        // The gated point dominates ungated -Q 1 outright on both axes
        // (0.85 s / -5851 B against 0.86 s / -5596 B), so this is not a
        // speed-for-size trade.
        //
        // Under exact DP the gate still captures the gains (94.4% of the
        // improvements and 92.9% of the headroom bits are at prec <= 10,
        // measured over 660492 perturbations at -e -c 8 -L 1), but -Q is far
        // more expensive there -- 1.24-1.80x rather than 1.06x -- because the
        // exact DP prices every (position, block size, stereo mode) with a
        // full optimize_subframe call, so the refinement runs on ~31x more
        // subframes than end up in the file.
        if (lattice_sweeps > 0 && bl_ord > 0 &&
            bl_prec <= FLACOUT_LATTICE_MAX_PREC) {
            const uint32_t hdr =
                hdr_fixed + (uint32_t)bl_ord * (eff_bps + (uint32_t)bl_prec);
            const int32_t qmax = (1 << (bl_prec - 1)) - 1;
            const int32_t qmin = -(1 << (bl_prec - 1));
            int32_t try_qc[32];
            std::memcpy(try_qc, bl_qc, (size_t)bl_ord * sizeof(int32_t));

            for (unsigned sweep = 0; sweep < lattice_sweeps; ++sweep) {
                bool improved = false;
                for (int j = 0; j < bl_ord; ++j) {
                    const int32_t base = try_qc[j];
                    bool accepted = false;
                    for (int delta = -1; delta <= 1; delta += 2) {
                        const int64_t v = (int64_t)base + delta;
                        if (v < qmin || v > qmax) continue;
                        try_qc[j] = (int32_t)v;
                        INSTR(g_instr.lattice_evals.fetch_add(1, std::memory_order_relaxed));
                        compute_lpc_residuals(shifted.data(), bsize, try_qc,
                                              bl_ord, bl_shift, residuals.data(),
                                              nullptr, max_sum_abs_qc);
                        const uint32_t cost =
                            hdr + 6u + calculate_rice_cost(residuals.data(), bsize,
                                                           (uint32_t)bl_ord, nullptr);
#ifdef FLACOUT_DUMP_LATTICE
                        // Signed cost delta per perturbation. Reading the
                        // *distribution* is the point: all-positive says the
                        // rounded point really is an axis-local minimum, while
                        // a large constant offset would mean the cost
                        // expression here disagrees with the search's.
                        {
                            std::lock_guard<std::mutex> lk(g_ldump.mu);
                            // base = the winner's exact cost, so dcost/base is
                            // the *relative* steepness of the cost surface and
                            // base/bsize its bits/sample. shift is the one that
                            // matters: the lattice step is 2^-shift, and that
                            // is what the precision gate above is really a
                            // proxy for. coef is the tap's own magnitude.
                            std::fprintf(g_ldump.fh,
                                         "%u\t%d\t%d\t%d\t%d\t%lld"
                                         "\t%u\t%d\t%d\t%u\t%d\n",
                                         bsize, bl_ord, bl_prec, j, delta,
                                         (long long)cost - (long long)best_lpc_cost,
                                         eff_bps, wasted, bl_shift, best_lpc_cost,
                                         base);
                        }
#endif
                        if (cost < best_lpc_cost) {
                            best_lpc_cost = cost;
                            std::memcpy(bl_qc, try_qc,
                                        (size_t)bl_ord * sizeof(int32_t));
                            INSTR(g_instr.lattice_accepts.fetch_add(1, std::memory_order_relaxed));
                            accepted = true;
                            break; // this tap moved; go on to the next one
                        }
                    }
                    if (!accepted) try_qc[j] = base;
                    improved |= accepted;
                }
                if (!improved) break; // a clean sweep: this is an axis-local min
            }
        }

#ifdef FLACOUT_DUMP_FP32RANK
        g_fp32rank.record(fp32_costs);
#endif

        // Materialize the winning LPC candidate, if it beat the non-LPC modes.
        // Re-deriving its residuals costs one more pass out of the millions the
        // search just ran, and in exchange every candidate above skipped the
        // per-candidate parameter bookkeeping.
        if (bl_ord > 0) {
            compute_lpc_residuals(shifted.data(), bsize, bl_qc, bl_ord, bl_shift,
                                  residuals.data(), nullptr, max_sum_abs_qc);
            SubframeParams cur{};
            cur.mode          = 3;
            cur.order         = bl_ord;
            cur.lpc_precision = bl_prec;
            cur.lpc_shift     = bl_shift;
            cur.wasted_bits   = wasted;
            std::memcpy(cur.q_coeffs, bl_qc, (size_t)bl_ord * sizeof(int32_t));
            uint32_t rice = calculate_rice_cost(residuals.data(), bsize,
                                                (uint32_t)bl_ord, &cur);
            const uint32_t hdr = hdr_fixed + (uint32_t)bl_ord * (eff_bps + (uint32_t)bl_prec);
            // A capped GPU partition search returns costs that are upper
            // bounds, so the exact re-pricing here can come in lower. The
            // encoder uses this exact value either way; only the equality
            // pinning the two together stops holding.
            assert(hdr + 6u + rice == best_lpc_cost ||
                   (gpu && gpu->partition_cap() < 8));
            try_update(cur, hdr + 6u + rice);
        }
    }

    return best;
}

// ============================================================
// DP fast-path: granule precomputation + cost estimation
// ============================================================

void Optimizer::precompute_granules(
    const std::vector<std::vector<int32_t>>& pcm_data)
{
    size_t num_g = pcm_data[0].size() / 16;
    m_granules.assign(m_channels, std::vector<Granule>(num_g));

    for (uint32_t c = 0; c < m_channels; ++c)
        for (size_t g = 0; g < num_g; ++g) {
            const int32_t* src = &pcm_data[c][g * 16];
            for (int i = 0; i <= 8; ++i) {
                double s = 0;
                for (int j = 0; j < 16 - i; ++j) s += (double)src[j] * src[j+i];
                m_granules[c][g].autoc[i] = s;
            }
        }

}

uint32_t Optimizer::estimate_lpc_bits_fast(
    const std::vector<std::vector<int32_t>>& pcm_data,
    int channel, uint32_t n_start, uint32_t n_end, int bps) const
{
    // Real Rice bits off a cheap predictor, not the entropy of its residual
    // energy.
    //
    // This used to price a block as 0.5*log2(2*pi*e*var) bits per sample from
    // the order-8 Levinson error — a Gaussian differential entropy standing in
    // for what Rice coding actually charges. That proxy is what made the
    // estimate *disperse*: per block against the real cost, est/exact ran p10
    // 1.075 to p90 1.515 (cv 12.5%), and replaying the DP on both cost columns
    // showed the dispersion costs 0.45-0.65% of the file in partition regret
    // alone — most of what -e is worth. Recalibrating the median recovers at
    // most 6% of that, because the error is spread rather than bias, so the
    // model itself had to go. See "What the estimated DP's gap actually is".
    //
    // estimate_subframe_cost already prices a subframe the way the encoder
    // does — rectangular autocorrelation, Levinson, quantize, residuals,
    // calculate_rice_cost — so this is a call, not new arithmetic, and the
    // remaining error is only what the real search does *better* (up to order
    // 32, ten windows, eight precisions, four stereo modes) rather than a model
    // of the coder.
    //
    // It no longer reads the granule cache. That is not a saving: the cache was
    // O(bsize/16) per call and this is several O(bsize) passes. What it buys is
    // accuracy, and the wall-clock is in the commit message.
    const int32_t* smp   = &pcm_data[(size_t)channel][(size_t)n_start * 16];
    const uint32_t bsize = (n_end - n_start) * 16;

    // Wasted-bits detection, as optimize_subframe does it: the real encoder
    // will strip these, so an estimate that ignored them would price
    // reduced-depth content (a common master-tape artefact) far too high.
    int wasted = 0;
    int32_t mask = 0;
    for (uint32_t i = 0; i < bsize; ++i) mask |= smp[i];
    // Digital silence: a CONSTANT subframe, header plus one value. Reproduces
    // what the old model's `err <= 0` branch returned, and keeps degenerate
    // all-zero autocorrelations out of the Levinson recursion below.
    if (mask == 0) return 8u + (uint32_t)bps;
    while ((mask & 1) == 0) { mask >>= 1; ++wasted; }

    // Frame-level overhead is priced by the DP itself via
    // FrameWriter::frame_bits, so it must not be baked in here too;
    // estimate_subframe_cost returns subframe bits only, which is the same
    // footing the exact path's SubframeParams::bits_cost is on.
    const uint32_t lpc = estimate_subframe_cost(smp, bsize, 3, FLACOUT_EST_ORDER,
                                                FLACOUT_EST_PREC, wasted, bps);
#if FLACOUT_EST_FIXED
    // The encoder picks the cheapest of Constant / Verbatim / Fixed / LPC, so an
    // LPC-only estimate overcharges every block that ends up Fixed by the
    // coefficients and warm-up an LPC subframe pays and a Fixed one does not
    // (order*prec + order*bps — ~250 bits at order 8). That is noise against a
    // loud block's payload and can dominate a quiet one, which biases the
    // partition exactly where the payload is smallest. A Fixed order-2 pass is
    // the cheap half of the comparison: no autocorrelation, no quantisation.
    const uint32_t fixed = estimate_subframe_cost(smp, bsize, 2, 2, 0, wasted, bps);
    return std::min(lpc, fixed);
#else
    return lpc;
#endif
}

// ============================================================
// Main entry point
// ============================================================
//
// Strategy:
//   1. Try a small set of candidate block sizes (all powers-of-two and common FLAC sizes).
//   2. For each candidate, estimate the total encoded bits using the fast granule-based LPC model.
//   3. Pick the winning block size.
//   4. Run exhaustive multi-window, multi-mode subframe optimization in parallel on each block.
//
// This avoids the O(n × MAX_GRANULES) DP that becomes prohibitive for long files,
// while still exploring the full block-size search space at a coarse level.

std::vector<BlockParams> Optimizer::find_optimal_block_partitioning(
    const std::vector<std::vector<int32_t>>& pcm_data)
{
    // Only the estimated path reads the granule cache, and since
    // estimate_lpc_bits_fast started counting real Rice bits from PCM, the only
    // reader left is select_windows (which already falls back to the configured
    // set on an empty cache), gated on m_adaptive && !full_search() in
    // compute_block. `-a` is on at every effort level, so the cache still earns
    // its keep in estimated mode — but if it ever stops being the default,
    // this guard should tighten to `!full_search() && m_adaptive`. Under -e the
    // cache was built and then never touched, at 4.5 bytes per sample per
    // channel — 91.5 MB on a 10.2M-sample stereo track, a fifth of that run's
    // 466 MB peak — plus a single-threaded pass over the whole stream before
    // any worker starts.
    //
    // Nothing reads what this skips, so exact-DP output is unchanged by
    // construction rather than by measurement.
    //
    // FLACOUT_DUMP_BLOCKCOST is the exception: its phase-1 block *does* read
    // the cache under full_search(), because comparing the estimator against
    // exact costs is the entire point of that build.
#ifdef FLACOUT_DUMP_BLOCKCOST
    precompute_granules(pcm_data);
#else
    if (!full_search()) precompute_granules(pcm_data);
#endif

    const size_t total_samples  = pcm_data[0].size();

    // -----------------------------------------------------------------
    // Short-stream fast path
    // -----------------------------------------------------------------
    // If the audio is shorter than the smallest candidate block size
    // (1024 samples), skip block-size selection entirely and emit one
    // block covering all samples (FLAC allows block sizes from 1 to 65535).
    if (total_samples < 1024) {
        if (m_verbose)
            std::cout << "Short stream (" << total_samples << " samples): "
                      << "using single block.\n";
        BlockParams bp{};
        bp.block_size = (uint32_t)total_samples;
        if (m_channels == 1) {
            bp.stereo_mode  = 0;
            bp.subframes[0] = optimize_subframe(pcm_data[0].data(),
                                                (uint32_t)total_samples, m_bps, m_windows, m_max_candidates, m_patience, m_precision_rungs, m_lattice_sweeps, m_gpu.get());
        } else {
            uint32_t best_bits = std::numeric_limits<uint32_t>::max();
            for (int mode : {0, 8, 9, 10}) {
                std::vector<int32_t> ch0(total_samples), ch1(total_samples);
                for (size_t k = 0; k < total_samples; ++k) {
                    int32_t L = pcm_data[0][k], R = pcm_data[1][k];
                    if (mode == 0)  { ch0[k] = L;        ch1[k] = R; }
                    else if (mode == 8)  { ch0[k] = L;        ch1[k] = L - R; }
                    else if (mode == 9)  { ch0[k] = L - R;    ch1[k] = R; }
                    else             { ch0[k] = (L+R)>>1; ch1[k] = L - R; }
                }
                // mode 9 = right+side: ch0 is side (needs +1 bit), ch1 is right
                uint32_t bps0 = (mode == 9) ? m_bps + 1 : m_bps;
                uint32_t bps1 = (mode == 9) ? m_bps     : (mode == 0 ? m_bps : m_bps + 1);
                SubframeParams s0 = optimize_subframe(ch0.data(), (uint32_t)total_samples, bps0, m_windows, m_max_candidates, m_patience, m_precision_rungs, m_lattice_sweeps, m_gpu.get());
                SubframeParams s1 = optimize_subframe(ch1.data(), (uint32_t)total_samples, bps1, m_windows, m_max_candidates, m_patience, m_precision_rungs, m_lattice_sweeps, m_gpu.get());
                if (s0.bits_cost + s1.bits_cost < best_bits) {
                    best_bits = s0.bits_cost + s1.bits_cost;
                    bp.stereo_mode  = mode;
                    bp.subframes[0] = s0;
                    bp.subframes[1] = s1;
                }
            }
        }
        bp.total_bits = bp.subframes[0].bits_cost
                      + (m_channels > 1 ? bp.subframes[1].bits_cost : 0);
        return { bp };
    }


    // -----------------------------------------------------------------
    // Variable-block-size DP
    // -----------------------------------------------------------------
    //
    // Candidates: the -b ladder, default {1024..16384}, with STEP = its GCD.
    // Node i = position i*STEP in the audio.
    // Edge (i→j) = one FLAC frame covering [i*STEP, j*STEP).
    // Cost = FrameWriter::frame_bits: the exact encoded frame size, header
    // (including the UTF-8 sample number, which grows with stream position),
    // subframe payload, byte-alignment pad, and CRCs.
    //
    // Phase 1: build all N×K work items, evaluate ALL in parallel with a
    //          flat thread pool.  All compute_block() calls are independent
    //          (different sample offsets, no shared writes), so 100% of
    //          threads stay busy for the full Phase 1 duration.
    // Phase 2: sequential DP over precomputed cost table — O(N×K), instant.
    // Phase 3: back-trace to recover the optimal frame sequence.

    // Shared with the window-table cache, which precomputes coefficients for
    // exactly these sizes (see DP_CANDIDATES at file scope).
    // Configurable via -b; defaults to DP_CANDIDATES, for which the window
    // coefficient tables are precomputed. STEP is the ladder's GCD, so every
    // candidate walks from one node to another.
    const std::vector<uint32_t>& CANDIDATES = m_dp_candidates;
    const size_t   NUM_CANDS = CANDIDATES.size();
    const uint32_t STEP      = m_dp_step;

    const size_t   num_nodes = total_samples / STEP;
    const uint32_t remainder = (uint32_t)(total_samples % STEP);

    // Phase 1: parallel precomputation -----------------------------------
    struct WorkItem { size_t node; size_t ci; };
    std::vector<WorkItem> work;
    work.reserve(num_nodes * NUM_CANDS);
    for (size_t n = 0; n < num_nodes; ++n)
        for (size_t c = 0; c < NUM_CANDS; ++c)
            if ((uint64_t)n * STEP + CANDIDATES[c] <= total_samples)
                work.push_back({n, c});

    // Longest-processing-time-first. Workers pull from one shared counter, so
    // whatever order this vector is in is the order work is handed out. Built
    // node-major, the 16384-sample blocks — 16x the cost of a 1024 — are spread
    // evenly through the queue, and some land near the end where there is no
    // remaining work to overlap them with. Handing out the expensive ones first
    // leaves only cheap blocks to fill the tail. Purely a scheduling change:
    // cost_table is indexed by (node, candidate), not by completion order.
    std::stable_sort(work.begin(), work.end(),
                     [&CANDIDATES](const WorkItem& a, const WorkItem& b) {
                         return CANDIDATES[a.ci] > CANDIDATES[b.ci];
                     });

    std::vector<BlockParams> cost_table(num_nodes * NUM_CANDS);

    unsigned nthreads = std::max(1u, static_cast<unsigned>(std::thread::hardware_concurrency()));
    // -t is absolute, not a cap. Clamping it to hardware_concurrency made it
    // impossible to oversubscribe, which is exactly what -G wants: a worker
    // blocked on a GPU fence is not runnable, so a pool sized to the core
    // count leaves cores idle while the device works.
    if (m_max_threads > 0) nthreads = (unsigned)m_max_threads;

    if (m_verbose)
        std::cout << "DP: " << num_nodes << " nodes × " << NUM_CANDS
                  << " candidates = " << work.size() << " blocks on "
                  << nthreads << " threads\n";

    {
        std::atomic<size_t> next{0}, done{0};
        std::mutex           cout_mtx;
        std::vector<std::thread> threads;
        for (int t = 0; t < nthreads; ++t) {
            threads.emplace_back([&]() {
                for (;;) {
                    size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= work.size()) break;
                    auto [node, ci] = work[idx];
                    if (full_search()) {
                        cost_table[node * NUM_CANDS + ci] =
                            compute_block(pcm_data, (uint64_t)node * STEP, CANDIDATES[ci]);
                    } else {
                        uint32_t bits = 0;
                        // granules are 16 samples each; nodes are STEP samples
                        // each, and -b validation keeps STEP a multiple of 16
                        static constexpr uint32_t GRANULE_SIZE = 16u;
                        uint32_t g_start = (uint32_t)(node * (STEP / GRANULE_SIZE));
                        uint32_t g_end   = g_start + CANDIDATES[ci] / GRANULE_SIZE;
                        if (m_channels == 1) {
                            bits = estimate_lpc_bits_fast(pcm_data, 0, g_start, g_end, m_bps);
                        } else {
                            bits = estimate_lpc_bits_fast(pcm_data, 0, g_start, g_end, m_bps) +
                                   estimate_lpc_bits_fast(pcm_data, 1, g_start, g_end, m_bps);
                        }
                        BlockParams bp{};
                        bp.block_size = CANDIDATES[ci];
                        bp.total_bits = bits;
                        cost_table[node * NUM_CANDS + ci] = bp;
                    }
#ifdef FLACOUT_DUMP_BLOCKCOST
                    // Phase-0 dataset: what the DP's estimator claimed for this
                    // (node, block size) against what encoding it actually
                    // costs. Both numbers are subframe payload bits on the same
                    // footing — frame_bits() adds the header afterwards to
                    // whichever the DP used — so they are directly comparable.
                    //
                    // Cost is why this is a separate build: it evaluates every
                    // candidate exactly *and* estimates it, so a dump run does
                    // the work of -e plus the estimator.
                    {
                        static constexpr uint32_t GS = 16u;
                        const uint32_t bsize = CANDIDATES[ci];
                        const uint32_t g0 = (uint32_t)(node * (STEP / GS));
                        const uint32_t g1 = g0 + bsize / GS;
                        uint32_t est = 0;
                        for (uint32_t ch = 0; ch < std::min(m_channels, 2u); ++ch)
                            est += estimate_lpc_bits_fast(pcm_data, (int)ch, g0, g1, m_bps);
                        const BlockParams exact = full_search()
                            ? cost_table[node * NUM_CANDS + ci]
                            : compute_block(pcm_data, (uint64_t)node * STEP, bsize);

                        // Span features, all from the granule cache — no PCM
                        // pass. cv2 of granule energy is the transient measure
                        // -a already uses; the lag ratios are spectral tilt.
                        double e_sum = 0.0, e_sq = 0.0, a1 = 0.0, a2 = 0.0, a0 = 0.0;
                        for (uint32_t g = g0; g < g1; ++g) {
                            const double e = m_granules[0][g].autoc[0];
                            e_sum += e; e_sq += e * e;
                            a0 += e; a1 += m_granules[0][g].autoc[1];
                            a2 += m_granules[0][g].autoc[2];
                        }
                        const double n  = (double)(g1 - g0);
                        const double mu = e_sum / n;
                        const double cv2 = (mu > 0.0) ? (e_sq / n - mu * mu) / (mu * mu) : 0.0;
                        // Per-mode estimates, computed straight from PCM so
                        // the shipping estimator is untouched. This is what
                        // says *which* term is wrong, rather than only that
                        // the total is: two attempts at a stereo term were
                        // reverted for want of exactly this breakdown.
                        double eL = 0, eR = 0, eM = 0, eS = 0;
                        if (m_channels >= 2) {
                            const uint64_t s0 = (uint64_t)node * STEP;
                            auto est_of = [&](int which) {
                                double ac[9] = {};
                                for (uint32_t i = 0; i + 8 < bsize; ++i) {
                                    const int32_t L = pcm_data[0][s0 + i];
                                    const int32_t R = pcm_data[1][s0 + i];
                                    const double v = which == 0 ? L : which == 1 ? R
                                                   : which == 2 ? ((L + R) >> 1) : (L - R);
                                    for (int k = 0; k <= 8; ++k) {
                                        const int32_t L2 = pcm_data[0][s0 + i + k];
                                        const int32_t R2 = pcm_data[1][s0 + i + k];
                                        const double w = which == 0 ? L2 : which == 1 ? R2
                                                       : which == 2 ? ((L2 + R2) >> 1) : (L2 - R2);
                                        ac[k] += v * w;
                                    }
                                }
                                float c[32];
                                compute_lpc_coefficients(ac, c, 8);
                                double e = ac[0];
                                for (int k = 0; k < 8; ++k) e -= (double)c[k] * ac[k+1];
                                if (e <= 0) return 8.0 + (double)m_bps;
                                double bp = 0.5 * std::log2(2.0*M_PI*M_E*(e/bsize));
                                if (bp < 1.0) bp = 1.0;
                                return 8.0 + bsize * bp;
                            };
                            eL = est_of(0); eR = est_of(1); eM = est_of(2); eS = est_of(3);
                        }
                        g_block_dump.row(node, (uint64_t)node * STEP, bsize,
                                         m_channels, m_bps, est, exact.total_bits,
                                         a0, (a0 > 0 ? a1 / a0 : 0.0),
                                         (a0 > 0 ? a2 / a0 : 0.0), cv2,
                                         exact.stereo_mode, eL, eR, eM, eS);
                    }
#endif
                    size_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (m_verbose && (d % 10 == 0 || d == work.size())) {
                        std::lock_guard<std::mutex> lk(cout_mtx);
                        std::cout << "\r  " << d << "/" << work.size()
                                  << " (" << d * 100 / work.size() << "%)" << std::flush;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        if (m_verbose) std::cout << "\n";
    }

    // Remainder block
    BlockParams remainder_bp{};
    if (remainder > 0)
        remainder_bp = compute_block(pcm_data, (uint64_t)num_nodes * STEP, remainder);

    // ---- Irregular-node DP: frame reuse with off-grid input frames ------
    // The regular DP below only accepts reuse edges whose boundaries lie on
    // the 1024 grid. When any usable input frame is off-grid (a -b 4608
    // stream, or the input's final remainder frame), switch to a node set of
    // grid positions ∪ input-frame boundaries ∪ stream end. Input frames
    // chain between their own boundaries; small exactly-encoded "bridge"
    // blocks connect each irregular node to the grid, so the path can enter
    // and leave an input-frame run anywhere. Everything stays exact-cost, so
    // the DP remains optimal over the union of both partitions. This branch
    // is bypassed entirely unless -e -r meets an off-grid frame, keeping the
    // regular path byte-identical.
    bool irregular = false;
    if (full_search()) {
        for (const auto& e : m_reuse_edges) {
            if (e.start_sample % STEP != 0 || e.block_size % STEP != 0 ||
                e.start_sample + e.block_size > (uint64_t)num_nodes * STEP) {
                irregular = true;
                break;
            }
        }
    }
    if (irregular) {
        const uint64_t grid_end = (uint64_t)num_nodes * STEP;

        std::vector<uint64_t> pos;
        pos.reserve(num_nodes + 2 + m_reuse_edges.size() * 2);
        for (size_t n = 0; n <= num_nodes; ++n) pos.push_back((uint64_t)n * STEP);
        pos.push_back(total_samples);
        for (const auto& e : m_reuse_edges) {
            pos.push_back(e.start_sample);
            pos.push_back(e.start_sample + e.block_size);
        }
        std::sort(pos.begin(), pos.end());
        pos.erase(std::unique(pos.begin(), pos.end()), pos.end());
        const size_t NN = pos.size();
        auto node_of = [&](uint64_t p) -> size_t {
            return (size_t)(std::lower_bound(pos.begin(), pos.end(), p) - pos.begin());
        };
        const size_t terminal = node_of(total_samples);

        // Bridge spans (deduplicated), then encoded in parallel like phase 1.
        struct Bridge { uint64_t start; uint32_t size; BlockParams bp; };
        std::vector<std::pair<uint64_t, uint32_t>> spans;
        auto add_span = [&](uint64_t s, uint64_t e2) {
            if (e2 <= s || e2 - s < 16 || e2 - s > 65535) return;
            spans.emplace_back(s, (uint32_t)(e2 - s));
        };
        for (size_t idx = 0; idx < NN; ++idx) {
            const uint64_t p = pos[idx];
            if (p % STEP == 0 && p <= grid_end) continue; // grid node
            if (p == total_samples) continue;             // terminal
            uint64_t g = (p / STEP) * STEP;               // entering bridge
            if (p - g < 16 && g >= STEP) g -= STEP;
            add_span(std::min(g, grid_end), p);
            uint64_t q = ((p / STEP) + 1) * STEP;         // leaving bridge
            if (q - p < 16) q += STEP;
            if (q > grid_end) q = total_samples;
            add_span(p, q);
        }
        std::sort(spans.begin(), spans.end());
        spans.erase(std::unique(spans.begin(), spans.end()), spans.end());

        std::vector<Bridge> bridges(spans.size());
        {
            std::atomic<size_t> next{0};
            std::vector<std::thread> threads;
            for (unsigned t = 0; t < nthreads; ++t) {
                threads.emplace_back([&]() {
                    for (;;) {
                        size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= spans.size()) break;
                        auto [s, sz] = spans[idx];
                        bridges[idx] = Bridge{ s, sz, compute_block(pcm_data, s, sz) };
                    }
                });
            }
            for (auto& th : threads) th.join();
        }

        if (m_verbose)
            std::cout << "DP: irregular reuse graph — " << NN << " nodes, "
                      << m_reuse_edges.size() << " input-frame edges, "
                      << bridges.size() << " bridge blocks\n";

        // Per-node edge lists for the non-grid edge types.
        std::vector<std::vector<uint32_t>> reuse_at2(NN), bridge_at(NN);
        for (size_t k = 0; k < m_reuse_edges.size(); ++k)
            reuse_at2[node_of(m_reuse_edges[k].start_sample)].push_back((uint32_t)k);
        for (size_t b = 0; b < bridges.size(); ++b)
            bridge_at[node_of(bridges[b].start)].push_back((uint32_t)b);

        // DP over node indices. Edge tags: 0..NUM_CANDS-1 grid candidate,
        // then reuse edges, then bridges, then the remainder block.
        const uint64_t INF = std::numeric_limits<uint64_t>::max();
        const uint32_t TAG_REUSE  = (uint32_t)NUM_CANDS;
        const uint32_t TAG_BRIDGE = TAG_REUSE + (uint32_t)m_reuse_edges.size();
        const uint32_t TAG_REM    = TAG_BRIDGE + (uint32_t)bridges.size();
        std::vector<uint64_t> dp2(NN, INF);
        std::vector<int64_t>  par2(NN, -1);
        std::vector<uint32_t> tag2(NN, 0);
        dp2[0] = 0;
        auto relax = [&](size_t i, size_t j, uint64_t w, uint32_t tag) {
            if (dp2[i] == INF) return;
            if (dp2[i] + w < dp2[j]) {
                dp2[j] = dp2[i] + w;
                par2[j] = (int64_t)i;
                tag2[j] = tag;
            }
        };
        for (size_t i = 0; i < NN; ++i) {
            if (dp2[i] == INF) continue;
            const uint64_t p = pos[i];
            if (p % STEP == 0 && p < grid_end) {
                const size_t g = (size_t)(p / STEP);
                for (size_t c = 0; c < NUM_CANDS; ++c) {
                    const uint64_t e2 = p + CANDIDATES[c];
                    if (e2 > grid_end) continue;
                    relax(i, node_of(e2),
                          FrameWriter::frame_bits(p, CANDIDATES[c], m_sample_rate,
                                                  cost_table[g * NUM_CANDS + c].total_bits),
                          (uint32_t)c);
                }
            }
            if (p == grid_end && remainder > 0)
                relax(i, terminal,
                      FrameWriter::frame_bits(p, remainder, m_sample_rate,
                                              remainder_bp.total_bits),
                      TAG_REM);
            for (uint32_t k : reuse_at2[i]) {
                const auto& e = m_reuse_edges[k];
                relax(i, node_of(e.start_sample + e.block_size),
                      (uint64_t)e.frame_bytes * 8u, TAG_REUSE + k);
            }
            for (uint32_t b : bridge_at[i]) {
                const auto& br = bridges[b];
                relax(i, node_of(br.start + br.size),
                      FrameWriter::frame_bits(br.start, br.size, m_sample_rate,
                                              br.bp.total_bits),
                      TAG_BRIDGE + b);
            }
        }

        // Back-trace and assemble. The path is guaranteed to exist: the grid
        // candidates plus the remainder block alone already span the stream.
        std::vector<BlockParams> result2;
        std::vector<std::pair<size_t, uint32_t>> rpath;
        for (size_t cur = terminal; cur != 0; ) {
            if (par2[cur] < 0) { rpath.clear(); break; }
            rpath.emplace_back((size_t)par2[cur], tag2[cur]);
            cur = (size_t)par2[cur];
        }
        std::reverse(rpath.begin(), rpath.end());
        for (auto [i, tag] : rpath) {
            if (tag < (uint32_t)NUM_CANDS) {
                result2.push_back(cost_table[(size_t)(pos[i] / STEP) * NUM_CANDS + tag]);
            } else if (tag < TAG_BRIDGE) {
                const auto& e = m_reuse_edges[tag - TAG_REUSE];
                BlockParams bp{};
                bp.block_size  = e.block_size;
                bp.total_bits  = e.frame_bytes * 8u;
                bp.reuse_index = (int32_t)e.input_index;
                result2.push_back(bp);
            } else if (tag < TAG_REM) {
                result2.push_back(bridges[tag - TAG_BRIDGE].bp);
            } else {
                result2.push_back(remainder_bp);
            }
        }
        if (!rpath.empty()) {
            if (m_verbose) {
                std::map<uint32_t, int> bs_hist;
                for (const auto& bp : result2) ++bs_hist[bp.block_size];
                std::cout << "DP done. Distribution:";
                for (const auto& [bs, cnt] : bs_hist)
                    std::cout << "  bs=" << bs << "×" << cnt;
                std::cout << "\n";
            }
            return result2;
        }
        // Unreachable terminal would mean a malformed graph — fall through
        // to the regular DP rather than fail.
    }

    // Reuse edges: input frames as exact-cost alternatives, bucketed by
    // start node. Exact-DP only — both edge types are then exact bit counts
    // (frame_bits for ours, the rewritten frame's size for reuse), so the DP
    // is optimal over the union of both partitions and can mix them. On a
    // tie the re-encoded frame wins (strict <), which keeps a re-encode of
    // our own output byte-stable.
    std::vector<std::vector<size_t>> reuse_at;
    if (full_search() && !m_reuse_edges.empty()) {
        reuse_at.resize(num_nodes);
        for (size_t k = 0; k < m_reuse_edges.size(); ++k) {
            const auto& e = m_reuse_edges[k];
            if (e.start_sample % STEP != 0 || e.block_size % STEP != 0) continue;
            const uint64_t end = e.start_sample + e.block_size;
            if (end > (uint64_t)num_nodes * STEP) continue; // remainder region
            reuse_at[(size_t)(e.start_sample / STEP)].push_back(k);
        }
    }

    // Phase 2: DP --------------------------------------------------------
    std::vector<uint64_t> dp       (num_nodes + 1, std::numeric_limits<uint64_t>::max());
    std::vector<int>      dp_parent(num_nodes + 1, -1);
    std::vector<int>      dp_cand  (num_nodes + 1, -1);
    dp[0] = 0;

    for (size_t i = 0; i < num_nodes; ++i) {
        if (dp[i] == std::numeric_limits<uint64_t>::max()) continue;
        for (size_t c = 0; c < NUM_CANDS; ++c) {
            size_t j = i + CANDIDATES[c] / STEP;
            if (j > num_nodes) continue;
            if ((uint64_t)i * STEP + CANDIDATES[c] > total_samples) continue;
            uint64_t cost = dp[i] + FrameWriter::frame_bits(
                                        (uint64_t)i * STEP, CANDIDATES[c],
                                        m_sample_rate,
                                        cost_table[i * NUM_CANDS + c].total_bits);
            if (cost < dp[j]) {
                dp[j]        = cost;
                dp_parent[j] = (int)i;
                dp_cand  [j] = (int)c;
            }
        }
        if (!reuse_at.empty()) {
            for (size_t k : reuse_at[i]) {
                const auto& e = m_reuse_edges[k];
                size_t j = i + e.block_size / STEP;
                uint64_t cost = dp[i] + (uint64_t)e.frame_bytes * 8u;
                if (cost < dp[j]) {
                    dp[j]        = cost;
                    dp_parent[j] = (int)i;
                    dp_cand  [j] = (int)(NUM_CANDS + k); // reuse edge marker
                }
            }
        }
    }

    // Phase 3: back-trace ------------------------------------------------
    std::vector<std::pair<size_t,size_t>> path;
    for (size_t cur = num_nodes; cur > 0; ) {
        int par = dp_parent[cur], cand = dp_cand[cur];
        if (par < 0) break;
        path.emplace_back((size_t)par, (size_t)cand);
        cur = (size_t)par;
    }
    std::reverse(path.begin(), path.end());

    std::vector<BlockParams> result(path.size() + (remainder > 0 ? 1 : 0));
    if (!full_search()) {
        std::atomic<size_t> next{0}, done{0};
        std::mutex           cout_mtx;
        std::vector<std::thread> threads;
        if (m_verbose) std::cout << "Optimizing " << path.size() << " selected blocks...\n";
        for (int t = 0; t < nthreads; ++t) {
            threads.emplace_back([&]() {
                for (;;) {
                    size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= path.size()) break;
                    auto [node, ci] = path[idx];
                    result[idx] = compute_block(pcm_data, (uint64_t)node * STEP, CANDIDATES[ci]);
                    size_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (m_verbose && (d % 10 == 0 || d == path.size())) {
                        std::lock_guard<std::mutex> lk(cout_mtx);
                        std::cout << "\r  " << d << "/" << path.size() << std::flush;
                    }
                }
            });
        }
        for (auto& th : threads) th.join();
        if (m_verbose) std::cout << "\n";
    } else {
        for (size_t idx = 0; idx < path.size(); ++idx) {
            auto [node, ci] = path[idx];
            if (ci >= NUM_CANDS) {
                // Reuse edge: no encoded parameters — the caller emits the
                // rewritten input frame identified by reuse_index.
                const auto& e = m_reuse_edges[ci - NUM_CANDS];
                BlockParams bp{};
                bp.block_size  = e.block_size;
                bp.total_bits  = e.frame_bytes * 8u;
                bp.reuse_index = (int32_t)e.input_index;
                result[idx] = bp;
            } else {
                result[idx] = cost_table[node * NUM_CANDS + ci];
            }
        }
    }

    if (remainder > 0)
        result.back() = remainder_bp;

    if (m_verbose) {
        std::map<uint32_t,int> bs_hist;
        for (const auto& bp : result) ++bs_hist[bp.block_size];
        std::cout << "DP done. Distribution:";
        for (const auto& [bs, cnt] : bs_hist)
            std::cout << "  bs=" << bs << "×" << cnt;
        std::cout << "\n";
    }
        report_gpu(0);

    return result;
}



// ============================================================
// compute_block: fully-optimised BlockParams for one frame
// ============================================================

// Adaptive window selection: a hand stat→set map over the free granule
// statistics (WINDOWS_PLAN.md "Adaptive ICI window selection"). Every set has
// exactly 4 windows so the analysis cost per encoded block matches the fixed
// shortlist; only membership adapts. Thresholds are first-cut hand picks —
// calibrate against measurement before trusting them.
// Energy-dispersion threshold above which a block is treated as transient and
// offered peak-aligned sparse windows. See the sweep note at the gate below.
#ifndef FLACOUT_TRANSIENT_CV2
#define FLACOUT_TRANSIENT_CV2 0.5
#endif

std::vector<WindowType> Optimizer::select_windows(
    uint64_t sample_start, uint32_t block_size) const
{
    // Falling back means "no better idea than the configured set", which is
    // the shortlist itself — not a hardcoded copy of what it used to be.
    const std::vector<WindowType>& def = m_windows;
    if (m_granules.empty() || m_granules[0].empty()) return def;
    const size_t g0 = (size_t)(sample_start / 16);
    const size_t g1 = std::min((size_t)((sample_start + block_size) / 16),
                               m_granules[0].size());
    if (g1 <= g0 + 8) return def; // too few granules for meaningful stats

    const size_t n = g1 - g0;
    double sumE = 0.0, sumE2 = 0.0, sumL1 = 0.0, maxE = -1.0;
    size_t argmax = g0;
    for (size_t g = g0; g < g1; ++g) {
        double E = 0.0, L1 = 0.0;
        for (uint32_t c = 0; c < m_channels; ++c) {
            E  += m_granules[c][g].autoc[0];
            L1 += m_granules[c][g].autoc[1];
        }
        sumE += E; sumE2 += E * E; sumL1 += L1;
        if (E > maxE) { maxE = E; argmax = g; }
    }
    if (sumE <= 0.0) return def; // digital silence

    const double mean = sumE / (double)n;
    const double var  = std::max(0.0, sumE2 / (double)n - mean * mean);
    const double cv2  = var / (mean * mean); // energy dispersion: high ⇒ transient
    const double tilt = sumL1 / sumE;        // lag1/lag0: ~1 tonal, ~0 noisy
    const double pos  = (double)(argmax - g0) / (double)n;

    // Transient routing gate: 0.5 saturates (0.25 measured identical). Both
    // search modes route: -c 0 prices partial/punchout windows exactly
    // (music_20s −1991 B vs fixed-4, 3-min synthetic percussion −19983 B),
    // and the ranked scorer's zero-fraction term (see optimize_subframe)
    // prices their blind region honestly — before that term existed, routing
    // under -c 8 lost (+2489 B music / +4151 B percussion); with it, it wins
    // (−104 B / −17992 B).
    // Additive, not replacing. The selector used to substitute its own set for
    // the shortlist, which made sense while the shortlist was four dense
    // tapers and the sparse windows lived nowhere else. Now that the
    // partial/punchout pair is standard, substituting *removed* more windows
    // than it added and cost bytes outright. Start from the configured set and
    // add only what the statistics argue for; the patience scaling in
    // compute_block widens the candidate budget to match.
    //
    // Adding globally is not an option — measured, each of these costs bytes
    // when handed to every block (partialtukey2 at offset 0 alone is +253 B on
    // the master mix), because at a fixed budget an extra window crowds the
    // ranked top-N everywhere to pay off somewhere. Gating is the whole point:
    // spend the slot only where the content predicts a winner.
    std::vector<WindowType> out = def;
    auto add = [&out](WindowType w) {
        if (std::find(out.begin(), out.end(), w) == out.end()) out.push_back(w);
    };

    if (cv2 > (double)FLACOUT_TRANSIENT_CV2) {
        // Transient content: offer the partial/punchout pair that isolates
        // the energy peak, plus general-purpose tapers. rect stays in the set
        // because the gate has a false-positive mode — amplitude-modulated
        // tonal content (beats) has high energy dispersion too, and dropping
        // rect there cost +4523 B on the 3-min synthetic tonal fixture. A
        // routed block therefore analyses 5 windows instead of 4; routed
        // blocks are a minority, so the extra analysis is marginal.
        // The shortlist covers peaks at 1/3 and 2/3; a peak in the first third
        // is the offset it has nothing for, in either polarity — the partial
        // that isolates the peak, and the punchout that excises it so the
        // predictor can fit the calmer remainder.
        if (pos < 0.33) { add(WindowType::PARTIAL_TUKEY_2_000);
                          add(WindowType::PUNCHOUT_TUKEY_2_000); }
        else if (pos < 0.67) { add(WindowType::PARTIAL_TUKEY_2_033);
                               add(WindowType::PUNCHOUT_TUKEY_2_033); }
        else               { add(WindowType::PARTIAL_TUKEY_2_067);
                             add(WindowType::PUNCHOUT_TUKEY_2_067); }
        return out;
    }
    if (tilt > 0.95) { // stationary tonal: barely-tapered windows do best
        add(WindowType::TUKEY_005);
        add(WindowType::TUKEY_010);
        return out;
    }
    if (tilt < 0.60) { // noisy: heavy tapers
        add(WindowType::TUKEY_075);
        add(WindowType::TUKEY_090);
        return out;
    }
    return def;
}

BlockParams Optimizer::compute_block(
    const std::vector<std::vector<int32_t>>& pcm_data,
    uint64_t sample_start, uint32_t block_size) const
{
    BlockParams bp{};
    bp.block_size = block_size;

    // Adaptive selection replaces the fixed window list per block. Estimated
    // DP only: under -e the wide set is already offered, so a selector has
    // nothing to add (and the DP's phase-1 blocks would pay it N×K times).
    std::vector<WindowType> selected;
    const std::vector<WindowType>* win_set = &m_windows;
    if (m_adaptive && !full_search()) {
        selected = select_windows(sample_start, block_size);
        win_set = &selected;
    }
    const std::vector<WindowType>& wins = *win_set;

    // Patience counts *consecutive* candidates that fail to improve, so it is
    // a budget denominated in candidates, and the candidate pool is windows x
    // orders. Handing a block a wider window set without widening that budget
    // spends the same scan over a bigger pool and simply pushes good dense
    // candidates past the cut. Measured: a transient block gets ~7 windows
    // where the configured set has 4, and at unscaled patience one album
    // regressed +0.0217% with 9 of 17 tracks worse — the worst track alone
    // (+52613 B) flipped to -4619 B once patience was raised by hand. Scaling
    // it with the pool restores that: same album -0.0712%, no track worse.
    // Ratio 1 when the set is not replaced, so this is inert without -a.
    // Only ever widens: a specialised set can be *smaller* than the
    // configured one, and shrinking the budget there would be a second way to
    // lose good candidates rather than a saving.
    unsigned patience = m_patience;
    if (!m_windows.empty() && wins.size() > m_windows.size())
        patience = (unsigned)((m_patience * wins.size() + m_windows.size() / 2)
                              / m_windows.size());

    if (m_channels == 1) {
        bp.stereo_mode  = 0;
        bp.subframes[0] = optimize_subframe(
            &pcm_data[0][sample_start], block_size, m_bps, wins, m_max_candidates, patience, m_precision_rungs, m_lattice_sweeps, m_gpu.get());
    } else {
        uint32_t best_bits = std::numeric_limits<uint32_t>::max();

        // Independent stereo (mode 0)
        std::vector<int> modes_to_test = {0, 8, 9, 10};
        if (!full_search()) {
            // Fast estimation using Fixed Predictor (order 2)
            auto estimate_ch = [&](const int32_t* smp, int bits_per_sample) -> uint32_t {
                int wasted = 0;
                int32_t mask = 0;
                for (uint32_t i = 0; i < block_size; ++i) mask |= smp[i];
                if (mask != 0) while ((mask & 1) == 0) { mask >>= 1; ++wasted; }
                return estimate_subframe_cost(smp, block_size, 2, 2, 0, wasted, bits_per_sample);
            };
            
            uint32_t best_est = std::numeric_limits<uint32_t>::max();
            int best_mode = 0;
            for (int mode : modes_to_test) {
                uint32_t cost = 0;
                if (mode == 0) {
                    cost = estimate_ch(&pcm_data[0][sample_start], m_bps) +
                           estimate_ch(&pcm_data[1][sample_start], m_bps);
                } else {
                    std::vector<int32_t> ch0(block_size), ch1(block_size);
                    for (uint32_t k = 0; k < block_size; ++k) {
                        int32_t L = pcm_data[0][sample_start + k], R = pcm_data[1][sample_start + k];
                        if      (mode == 8)  { ch0[k] = L;        ch1[k] = L - R; }
                        else if (mode == 9)  { ch0[k] = L - R;    ch1[k] = R;     }
                        else                 { ch0[k] = (L+R)>>1; ch1[k] = L - R; }
                    }
                    cost = estimate_ch(ch0.data(), m_bps) + estimate_ch(ch1.data(), m_bps + 1);
                }
                if (cost < best_est) {
                    best_est = cost;
                    best_mode = mode;
                }
            }
            modes_to_test = {best_mode};
        }

        // the 4 stereo modes draw from only 4 distinct channel signals; S=L-R
        // alone appears in modes 8/9/10, so optimize each signal at most once.
        enum { SIG_L = 0, SIG_R, SIG_S, SIG_M };
        bool           have[4] = { false, false, false, false };
        SubframeParams cache[4];

        auto get_sig = [&](int sig) -> const SubframeParams& {
            if (have[sig]) return cache[sig];
            if (sig == SIG_L) {
                cache[sig] = optimize_subframe(&pcm_data[0][sample_start], block_size, m_bps, wins, m_max_candidates, patience, m_precision_rungs, m_lattice_sweeps, m_gpu.get());
            } else if (sig == SIG_R) {
                cache[sig] = optimize_subframe(&pcm_data[1][sample_start], block_size, m_bps, wins, m_max_candidates, patience, m_precision_rungs, m_lattice_sweeps, m_gpu.get());
            } else {
                std::vector<int32_t> ch(block_size);
                uint32_t bps_s;
                if (sig == SIG_S) {
                    for (uint32_t k = 0; k < block_size; ++k)
                        ch[k] = pcm_data[0][sample_start + k] - pcm_data[1][sample_start + k];
                    bps_s = m_bps + 1;
                } else { // SIG_M
                    for (uint32_t k = 0; k < block_size; ++k)
                        ch[k] = (pcm_data[0][sample_start + k] + pcm_data[1][sample_start + k]) >> 1;
                    bps_s = m_bps;
                }
                cache[sig] = optimize_subframe(ch.data(), block_size, bps_s, wins, m_max_candidates, patience, m_precision_rungs, m_lattice_sweeps, m_gpu.get());
            }
            have[sig] = true;
            return cache[sig];
        };

        // (ch0, ch1) signal pair per stereo mode
        auto mode_sigs = [](int mode) -> std::pair<int,int> {
            switch (mode) {
                case 0:  return { SIG_L, SIG_R };
                case 8:  return { SIG_L, SIG_S };
                case 9:  return { SIG_S, SIG_R };
                default: return { SIG_M, SIG_S }; // mode 10
            }
        };

        for (int mode : modes_to_test) {
            auto [sig0, sig1] = mode_sigs(mode);
            const SubframeParams& s0 = get_sig(sig0);
            const SubframeParams& s1 = get_sig(sig1);
            uint32_t cost = s0.bits_cost + s1.bits_cost;
            if (cost < best_bits) {
                best_bits       = cost;
                bp.stereo_mode  = mode;
                bp.subframes[0] = s0;
                bp.subframes[1] = s1;
            }
        }
    }

    bp.total_bits = bp.subframes[0].bits_cost
                  + (m_channels > 1 ? bp.subframes[1].bits_cost : 0);
    return bp;
}

} // namespace flacoutcpp
