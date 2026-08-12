#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>
#include <sstream>
#include "flacoutcpp.hpp"
#include "gpu_encoder.hpp"

using namespace flacoutcpp;

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options] <input.flac> [output.flac]\n"
        << "       " << prog << " [options] <input-dir> <output-dir>\n"
        << "\n"
        << "When the input is a directory, every .flac under it is encoded into\n"
        << "the output directory, mirroring the subdirectory structure. Files are\n"
        << "processed one at a time -- each already uses every core (or the GPU)\n"
        << "-- and a failure is reported and skipped rather than stopping the run.\n"
        << "With -P this is also the fast path for many files: the device context\n"
        << "is built once for the batch instead of once per file (~28 ms each).\n"
        << "Options:\n"
        << "  -e, --exhaustive     Exact search: fully encode every block-size and\n"
        << "                       stereo-mode choice instead of estimating them,\n"
        << "                       and offer all windows (extremely slow).\n"
        << "                       Worth far more than any -E level: 0.6% (24-bit)\n"
        << "                       to 2.3% (16-bit) on real music, against ~0.2%\n"
        << "                       across the whole dial. Rarely worth it bare:\n"
        << "                       '-e -L 1' is ~5x faster for 99.5% of the gain,\n"
        << "                       and '-e -E 0' ~80x faster for ~76% of it.\n"
        << "  -c, --candidates N   Fully evaluate only the N most promising\n"
        << "                       (window, order) pairs per subframe, ranked by\n"
        << "                       Levinson-Durbin prediction error. 0 = no limit.\n"
        << "                       Default: 24 (effort level 3), or 0 when -e is\n"
        << "                       given without -c/-L/-E. Composes with -e\n"
        << "                       (e.g. -e -c 8). Larger N is slower and\n"
        << "                       compresses better.\n"
        << "  -p, --patience N     Keep scanning past -c N while candidates are\n"
        << "                       still improving; stop after N consecutive that\n"
        << "                       are not. Makes -c a floor, not a ceiling.\n"
        << "                       Default: 2x -c. 0 disables it (plain top-N cut).\n"
        << "  -E, --effort N       Effort 0-12: one dial along the measured\n"
        << "                       size/time frontier, setting -c, -L, -a and\n"
        << "                       — from level 10 — exact DP, together (they\n"
        << "                       are not independent; which mix is efficient\n"
        << "                       shifts with the budget). 0 fastest, 9 = every\n"
        << "                       candidate and every rung under estimated DP,\n"
        << "                       10-12 = exact DP at increasing depth. Level 3\n"
        << "                       is the default. Against it, on a 188-track\n"
        << "                       mix: 0 is +0.15% at 0.8x the time, 6 is\n"
        << "                       -0.03% at 1.5x, 9 is -0.05% at 5.7x, 10 is\n"
        << "                       -0.52% at 7.7x, 12 is -0.61% at 24x. Level 10\n"
        << "                       is the value corner — ten times level 9's\n"
        << "                       compression for 20% more time — because exact\n"
        << "                       pricing is worth far more than search depth.\n"
        << "                       An explicit -c/-p/-L/-a wins; the level's -a\n"
        << "                       yields to -e/-w rather than erroring.\n"
        << "  -L, --rungs N        Encode only the N most promising of the 8 LPC\n"
        << "                       coefficient precisions per candidate, chosen by\n"
        << "                       an analytic model of the quantization error\n"
        << "                       instead of by encoding all of them.\n"
        << "                       Default: 1 (effort level 3); 0 under a bare\n"
        << "                       -e, which prices the whole ladder. Against\n"
        << "                       all 8 rungs: 1 costs 0.019% for 1.31x, 2\n"
        << "                       costs 0.009% for 1.25x, 3 costs 0.005%.\n"
        << "                       -c and -L are not independent —\n"
        << "                       prefer -E, which pairs them along the measured\n"
        << "                       frontier, unless you know which pair you want.\n"
        << "  -G, --gpu            Evaluate LPC candidates on the GPU (Vulkan).\n"
        << "                       Bit-exact with the CPU path, so the output is\n"
        << "                       byte-identical; only the search runs elsewhere.\n"
        << "                       Composes with -c/-p/-e; prices the whole\n"
        << "                       precision ladder, so it does not combine\n"
        << "                       with -L. Needs a\n"
        << "      --gpu-min-batch N  Smallest candidate batch worth dispatching\n"
        << "                       (default 0 = dispatch everything, which\n"
        << "                       measured fastest here). Raise it if a device\n"
        << "                       has a long submit path; small batches then\n"
        << "                       stay on the CPU.\n"
        << "      --gpu-slots N      Dispatches in flight (1-16). Default is\n"
        << "                       auto: 3 normally, 1 on a device running the\n"
        << "                       compat kernel. Each parks a worker on a\n"
        << "                       fence, so more is not better unless the GPU\n"
        << "                       outruns the CPU -- on an old iGPU 3 slots\n"
        << "                       cost 3.7x against 1.\n"
        << "      --gpu-duty N       Percent of offered subframes the GPU takes\n"
        << "                       (1-100), or 0 = adaptive, the default: the\n"
        << "                       device's measured throughput is compared\n"
        << "                       against one CPU thread's and it takes work\n"
        << "                       only while it is winning. Pin it to A/B the\n"
        << "                       throttle itself; no fixed share is right on\n"
        << "                       every device, or even at every bit depth.\n"
        << "      --gpu-partition-cap N\n"
        << "                       Cap the GPU's partition-order search (1-8).\n"
        << "                       8 (default) reproduces the CPU exactly. Lower\n"
        << "                       is faster but may rank a candidate wrong; the\n"
        << "                       winner is still priced exactly, so output\n"
        << "                       stays lossless and its size never mis-stated.\n"
        << "                       build with -DFLACOUT_VULKAN=ON and a device\n"
        << "                       with 32-lane subgroups and shaderInt64.\n"
        << "  -P, --pure-gpu       Encode entirely on the GPU: windowing,\n"
        << "                       autocorrelation, Levinson, quantization, the\n"
        << "                       Rice sweep, mode choice, bit packing and both\n"
        << "                       CRCs all run as compute dispatches. A DIFFERENT\n"
        << "                       encoder, not a backend for -G: fixed block size,\n"
        << "                       no DP, no reuse, fp32 analysis. Output is\n"
        << "                       lossless but not byte-identical to the CPU path,\n"
        << "                       so verify by decoding, not with cmp. Ignores\n"
        << "                       -c/-p/-L/-e/-E/-Q/-b/-a. MD5 stays on a host\n"
        << "                       thread (it cannot be parallelized) and overlaps\n"
        << "                       the encode, so it is free.\n"
        << "      --pg-block N       Frame size for -P: multiple of 256, <= 16384\n"
        << "                       (default 4096, which measured best on real\n"
        << "                       music). Stationary content wants more: on a\n"
        << "                       synthetic tonal mix 16384 is -1.9%, while on\n"
        << "                       real music it is +0.4%.\n"
        << "      --pg-prec L        Comma-separated LPC precisions to sweep\n"
        << "                       (1-4 of them, each 5-15; default 15).\n"
        << "      --pg-orders N      LPC orders swept per (block, signal, window)\n"
        << "                       out of 32, ranked by Levinson error (default 2;\n"
        << "                       32 = all). Windows are worth far more than\n"
        << "                       orders, which is why the default pairs 2 orders\n"
        << "                       with the full 10-window shortlist: 8x2 beats 6x6\n"
        << "                       on both size and time. The pair moves together --\n"
        << "                       2 orders with fewer windows costs size.\n"
        << "      --pg-pcap N        Cap the sweep's Rice partition-order search\n"
        << "                       (1-8, default 4). Ranking only -- the winner is\n"
        << "                       re-priced with the full search. The kernel is\n"
        << "                       dominated by cross-lane partition closes, so\n"
        << "                       this is its biggest knob: 4 is 3.4x for +0.029%.\n"
        << "      --pg-chunk N       Frames per device chunk (default 256). Bounds\n"
        << "                       peak device memory; frames are independent at a\n"
        << "                       fixed block size, so this costs only memory.\n"
        << "  -Q, --lattice N      Refine the winning subframe's quantized LPC\n"
        << "                       coefficients by coordinate descent: try each\n"
        << "                       tap at +-1, keep what lowers the exact cost,\n"
        << "                       up to N sweeps (0 = off, the default).\n"
        << "                       Experimental. Never grows a subframe.\n"
        << "  -b, --blocks <list>  Comma-separated block sizes the DP may choose\n"
        << "                       from (default: 1024,2048,4096,8192,16384).\n"
        << "                       Each must be a multiple of 16 in [16, 65520],\n"
        << "                       and every size must be a multiple of the\n"
        << "                       smallest, or the DP cannot reach the stream's\n"
        << "                       end. FLAC's own limits are 16 and 65535, but\n"
        << "                       65535 is odd, so no usable grid reaches it;\n"
        << "                       65520 is the largest attainable size, and\n"
        << "                       needs a smallest size that divides it (e.g.\n"
        << "                       16 or 5040, not 1024). Cost scales with\n"
        << "                       sum(sizes)/gcd(sizes): the default is 31 block\n"
        << "                       -samples of work per input sample, and\n"
        << "                       16,...,32768 is 4095 — about 130x. Best paired\n"
        << "                       with -e, which prices every choice exactly.\n"
        << "  -n, --no-metadata    Do not copy metadata from input to output\n"
        << "  -a, --adaptive-windows  Add windows chosen from each block's signal\n"
        << "                       statistics to the shortlist. On by default;\n"
        << "                       estimated-DP only, so it yields silently to\n"
        << "                       -e/-w and is an error only when named there.\n"
        << "  -A, --no-adaptive-windows  Turn that off.\n"
        << "  -R, --no-reuse       Disable input-frame reuse. By default, input\n"
        << "                       frames that beat the re-encoded ones are spliced\n"
        << "                       into the output (and the input is copied through\n"
        << "                       if the output would still be larger), so\n"
        << "                       re-encoding never grows a file. -R measures the\n"
        << "                       raw search alone — mainly for testing\n"
        << "  -W, --warn-superior  Warn on stderr when the input's own frames beat\n"
        << "                       the re-encode (i.e. frame reuse fired), naming\n"
        << "                       the input's encoder when its metadata says.\n"
        << "                       Prints even with -q; incompatible with -R\n"
        << "  -q, --quiet          Suppress all progress output\n"
        << "  -t, --threads N      Limit parallel worker threads (default: all CPUs)\n"
        << "  -w, --windows <list> Comma-separated list of apodization windows to use\n"
        << "                       (default: all 26 with a bare -e, else\n"
        << "                       tukey005,tukey020,\n"
        << "                       tukey050,hann,welch,rect and the partial/punchout\n"
        << "                       tukey pair at .33/.67)\n"
        << "                       An entry of the form custom:<file> loads a window\n"
        << "                       shape from a knot file (up to 4 per run); see\n"
        << "                       bench/windows/example_taper.txt for the format\n"
        << "Available window names:\n"
        << "  rect, bartlett, bartletthann, blackman, blackmanharris, connes, flattop,\n"
        << "  gauss025, gauss0125, hamming, hann, kaiserbessel, nuttall, triangle, welch,\n"
        << "  tukey005, tukey010, tukey020, tukey050, tukey075, tukey090,\n"
        << "  partialtukey2, partialtukey2_033, partialtukey2_067,\n"
        << "  punchouttukey2_033, punchouttukey2_067\n"
        << "Experimental windows (never in a default set; explicit -w only):\n"
        << "  lanczos, bohman, parzen, plancktaper010, plancktaper025,\n"
        << "  partialtukey3_{1,2,3}, punchouttukey3_{1,2,3},\n"
        << "  partialtukey3h_{000,033,067}, punchouttukey3h_{025,050},\n"
        << "  punchouttukey2_000,\n"
        << "  expdecay{2,4}, expattack{2,4}, attackdecay{005,010,020},\n"
        << "  dpss{2,3,4}\n";
}

// Split a comma-separated string into tokens.
static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

namespace fs = std::filesystem;

// Every .flac under `root`, sorted, so a batch is reproducible and its progress
// output is readable.
static std::vector<fs::path> find_flacs(const fs::path& root, std::error_code& ec) {
    std::vector<fs::path> out;
    for (auto it = fs::recursive_directory_iterator(
             root, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        std::string ext = it->path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        if (ext == ".flac") out.push_back(it->path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::string human(uintmax_t bytes) {
    static const char* u[] = {"B", "KiB", "MiB", "GiB"};
    double v = (double)bytes; int i = 0;
    while (v >= 1024.0 && i < 3) { v /= 1024.0; ++i; }
    std::ostringstream os;
    os << std::fixed << std::setprecision(v < 10 && i ? 2 : 1) << v << " " << u[i];
    return os.str();
}

// Encode a tree. Returns the number of files that failed.
static int run_batch(const fs::path& in_dir, const fs::path& out_dir,
                     flacoutcpp::Config cfg) {
    std::error_code ec;
    const auto files = find_flacs(in_dir, ec);
    if (ec) {
        std::cerr << "Error: cannot read " << in_dir << ": " << ec.message() << "\n";
        return 1;
    }
    if (files.empty()) {
        std::cerr << "Error: no .flac files under " << in_dir << ".\n";
        return 1;
    }
    const bool progress = cfg.verbose;
    // Per-file banners would be one screen per track. The batch prints one line
    // per file instead, and the encoder stays quiet.
    cfg.verbose = false;

    if (progress)
        std::cout << files.size() << " files under " << in_dir << " -> " << out_dir << "\n";

    uintmax_t in_bytes = 0, out_bytes = 0;
    int failed = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < files.size(); ++i) {
        const fs::path rel = fs::relative(files[i], in_dir, ec);
        const fs::path dst = out_dir / (ec ? files[i].filename() : rel);
        fs::create_directories(dst.parent_path(), ec);
        if (ec) {
            std::cerr << "Error: cannot create " << dst.parent_path() << ": "
                      << ec.message() << "\n";
            ++failed;
            continue;
        }
        // Refuse to write over the file being read. Same-directory in/out is an
        // easy typo and would destroy the input mid-encode.
        if (fs::equivalent(files[i], dst, ec)) {
            std::cerr << "Error: " << files[i] << " would overwrite itself; skipped.\n";
            ++failed;
            continue;
        }
        ec.clear();

        const auto ft = std::chrono::steady_clock::now();
        const bool ok = flacoutcpp::optimise(files[i].string(), dst.string(), cfg);
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ft).count();
        if (!ok) {
            std::cerr << "Failed: " << files[i].string() << "\n";
            ++failed;
            continue;
        }
        const uintmax_t ib = fs::file_size(files[i], ec);
        const uintmax_t ob = fs::file_size(dst, ec);
        if (!ec) { in_bytes += ib; out_bytes += ob; }
        if (progress) {
            std::ostringstream line;
            line << "[" << (i + 1) << "/" << files.size() << "] "
                 << rel.string() << "  " << std::fixed << std::setprecision(2)
                 << secs << " s";
            if (!ec && ib)
                line << "  " << std::showpos << std::setprecision(2)
                     << 100.0 * ((double)ob - (double)ib) / (double)ib << "%"
                     << std::noshowpos;
            std::cout << line.str() << "\n" << std::flush;
        }
    }

    // The device context outlives a single file by design; a batch is where it
    // gets used, and this is where it stops being needed.
    flacoutcpp::release_shared_pure_gpu_encoder();

    if (progress) {
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "\n" << (files.size() - (size_t)failed) << "/" << files.size()
                  << " files in " << std::fixed << std::setprecision(2) << secs << " s\n"
                  << human(in_bytes) << " -> " << human(out_bytes);
        if (in_bytes)
            std::cout << "  (" << std::showpos << std::setprecision(3)
                      << 100.0 * ((double)out_bytes - (double)in_bytes) / (double)in_bytes
                      << "%" << std::noshowpos << ")";
        std::cout << "\n";
        if (failed) std::cout << failed << " failed\n";
    }
    return failed;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return EXIT_FAILURE; }

    flacoutcpp::Config cfg;
    std::vector<std::string> positional;
    bool candidates_given = false;
    // Effort is a preset for -c/-L, so an explicit knob must win regardless of
    // the order the two appear in. Record what was named and re-apply it after
    // the level, rather than depending on argv order.
    bool patience_given = false, rungs_given = false;
    bool adaptive_given = false, adaptive_off = false;
    int  effort = -1;
    unsigned given_candidates = 0, given_rungs = 0;
    int      given_patience = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--version") {
            std::cout << FLACOUTCPP_VERSION << "\n";
            return EXIT_SUCCESS;

        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;

        } else if (arg == "-e" || arg == "--exhaustive") {
            cfg.exhaustive = true;

        } else if (arg == "-c" || arg == "--candidates") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -c requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                // stoul accepts a leading '-' by wrapping; reject it explicitly.
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                given_candidates = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -c requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            candidates_given = true;

        } else if (arg == "-p" || arg == "--patience") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -p requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                given_patience = static_cast<int>(std::stoul(argv[i]));
                patience_given = true;
            } catch (const std::exception&) {
                std::cerr << "Error: -p requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-E" || arg == "--effort") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -E requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                effort = static_cast<int>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -E requires an effort level 0-12, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-L" || arg == "--rungs") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -L requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                given_rungs = static_cast<unsigned>(std::stoul(argv[i]));
                rungs_given = true;
            } catch (const std::exception&) {
                std::cerr << "Error: -L requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--gpu-duty") {
            if (i + 1 >= argc) { std::cerr << "Error: --gpu-duty requires a number.\n"; return EXIT_FAILURE; }
            ++i;
            try {
                unsigned v = static_cast<unsigned>(std::stoul(argv[i]));
                if (v > 100) throw std::invalid_argument("range");
                cfg.gpu_duty = v;   // 0 = adaptive
            } catch (const std::exception&) {
                std::cerr << "Error: --gpu-duty takes 0-100, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--gpu-slots") {
            if (i + 1 >= argc) { std::cerr << "Error: --gpu-slots requires a number.\n"; return EXIT_FAILURE; }
            ++i;
            try {
                unsigned v = static_cast<unsigned>(std::stoul(argv[i]));
                if (v < 1 || v > 16) throw std::invalid_argument("range");
                cfg.gpu_slots = v;
            } catch (const std::exception&) {
                std::cerr << "Error: --gpu-slots takes 1-16, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--gpu-partition-cap") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --gpu-partition-cap requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                unsigned v = static_cast<unsigned>(std::stoul(argv[i]));
                if (v < 1 || v > 8) throw std::invalid_argument("range");
                cfg.gpu_partition_cap = v;
            } catch (const std::exception&) {
                std::cerr << "Error: --gpu-partition-cap takes 1-8, got '"
                          << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--gpu-min-batch") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --gpu-min-batch requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                cfg.gpu_min_batch = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: --gpu-min-batch requires a non-negative "
                             "integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-G" || arg == "--gpu") {
            // The search defaults are applied below, alongside -e's: the GPU
            // batches a whole subframe's candidates in one dispatch, which
            // only the unlimited sweep produces, and the precision ladder
            // exists to dodge CPU cost that the GPU does not pay.
            cfg.use_gpu = true;

        } else if (arg == "-P" || arg == "--pure-gpu") {
            cfg.pure_gpu = true;

        } else if (arg == "--pg-block") {
            if (i + 1 >= argc) { std::cerr << "Error: --pg-block requires a number.\n"; return EXIT_FAILURE; }
            ++i;
            try {
                cfg.pg_block_size = (uint32_t)std::stoul(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Error: --pg-block requires a number, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            if (cfg.pg_block_size % 256 != 0 || cfg.pg_block_size < 256 ||
                cfg.pg_block_size > 16384) {
                std::cerr << "Error: --pg-block must be a multiple of 256 in [256, 16384].\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--pg-orders") {
            if (i + 1 >= argc) { std::cerr << "Error: --pg-orders requires a number.\n"; return EXIT_FAILURE; }
            ++i;
            try {
                cfg.pg_orders = (uint32_t)std::stoul(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Error: --pg-orders requires a number, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            if (cfg.pg_orders < 1 || cfg.pg_orders > 32) {
                std::cerr << "Error: --pg-orders takes 1-32.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--pg-pcap") {
            if (i + 1 >= argc) { std::cerr << "Error: --pg-pcap requires a number.\n"; return EXIT_FAILURE; }
            ++i;
            try {
                cfg.pg_partition_cap = (uint32_t)std::stoul(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Error: --pg-pcap requires a number, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            if (cfg.pg_partition_cap < 1 || cfg.pg_partition_cap > 8) {
                std::cerr << "Error: --pg-pcap takes 1-8.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--pg-chunk") {
            if (i + 1 >= argc) { std::cerr << "Error: --pg-chunk requires a number.\n"; return EXIT_FAILURE; }
            ++i;
            try {
                cfg.pg_blocks_per_chunk = (uint32_t)std::stoul(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Error: --pg-chunk requires a number, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }
            if (cfg.pg_blocks_per_chunk < 1 || cfg.pg_blocks_per_chunk > 1024) {
                std::cerr << "Error: --pg-chunk takes 1-1024.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "--pg-prec") {
            if (i + 1 >= argc) { std::cerr << "Error: --pg-prec requires a list.\n"; return EXIT_FAILURE; }
            ++i;
            cfg.pg_precisions.clear();
            std::string spec(argv[i]);
            size_t pos = 0;
            while (pos <= spec.size()) {
                const size_t comma = spec.find(',', pos);
                const std::string tok = spec.substr(pos, comma == std::string::npos
                                                        ? std::string::npos : comma - pos);
                if (!tok.empty()) {
                    try {
                        const int p = std::stoi(tok);
                        if (p < 5 || p > 15) throw std::out_of_range("range");
                        cfg.pg_precisions.push_back(p);
                    } catch (const std::exception&) {
                        std::cerr << "Error: --pg-prec entries must be 5-15, got '"
                                  << tok << "'.\n";
                        return EXIT_FAILURE;
                    }
                }
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (cfg.pg_precisions.empty() || cfg.pg_precisions.size() > 4) {
                std::cerr << "Error: --pg-prec takes 1-4 precisions.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-Q" || arg == "--lattice") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -Q requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                if (argv[i][0] == '-') throw std::invalid_argument("negative");
                cfg.lattice_sweeps = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -Q requires a non-negative integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-a" || arg == "--adaptive-windows") {
            cfg.adaptive_windows = true;
            adaptive_given = true;

        } else if (arg == "-A" || arg == "--no-adaptive-windows") {
            cfg.adaptive_windows = false;
            adaptive_off = true;

        } else if (arg == "-R" || arg == "--no-reuse") {
            cfg.reuse_frames = false;

        } else if (arg == "-W" || arg == "--warn-superior") {
            cfg.warn_superior = true;

        } else if (arg == "-q" || arg == "--quiet") {
            cfg.verbose = false;

        } else if (arg == "-n" || arg == "--no-metadata") {
            cfg.copy_metadata = false;

        } else if (arg == "-w" || arg == "--windows") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -w requires an argument.\n";
                return EXIT_FAILURE;
            }
            ++i;
            for (const auto& name : split_csv(argv[i])) {
                // A custom: entry names a file, so a bad one is a mistake worth
                // stopping for — silently dropping it would quietly change the
                // window set a measurement run was built around.
                if (name.rfind("custom:", 0) == 0) {
                    std::string err;
                    auto wt = register_custom_window(name.substr(7), &err);
                    if (wt == WindowType::COUNT) {
                        std::cerr << "Error: -w " << name << ": " << err << "\n";
                        return EXIT_FAILURE;
                    }
                    cfg.windows.push_back(wt);
                    continue;
                }
                auto wt = window_from_name(name);
                if (wt == WindowType::COUNT) {
                    std::cerr << "Warning: unrecognised window '" << name << "' — skipped.\n";
                } else {
                    cfg.windows.push_back(wt);
                }
            }

        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 >= argc) {
                std::cerr << "Error: -t requires a number.\n";
                return EXIT_FAILURE;
            }
            ++i;
            try {
                cfg.max_threads = static_cast<unsigned>(std::stoul(argv[i]));
            } catch (const std::exception&) {
                std::cerr << "Error: -t requires a positive integer, got '" << argv[i] << "'.\n";
                return EXIT_FAILURE;
            }

        } else if (arg == "-b" || arg == "--blocks") {
            // Block-size ladder for the DP. Rejecting bad ladders here keeps
            // the optimizer free of the checks: it may assume the list is
            // non-empty and that its GCD is a usable node spacing.
            if (i + 1 >= argc) {
                std::cerr << "Error: -b requires a comma-separated block-size list.\n";
                return EXIT_FAILURE;
            }
            ++i;
            std::vector<uint32_t> sizes;
            for (const auto& tok : split_csv(argv[i])) {
                unsigned long v = 0;
                try {
                    size_t used = 0;
                    v = std::stoul(tok, &used);
                    if (used != tok.size()) throw std::invalid_argument("junk");
                } catch (const std::exception&) {
                    std::cerr << "Error: -b: not a block size: '" << tok << "'.\n";
                    return EXIT_FAILURE;
                }
                // 16 is FLAC's minimum; 65535 is the maximum a 16-bit
                // STREAMINFO field can hold, but the DP needs multiples of 16
                // (its nodes sit on a grid whose spacing divides every
                // candidate, and the estimated path indexes 16-sample
                // granules), so 65520 is the largest reachable size.
                if (v < 16 || v > 65520) {
                    std::cerr << "Error: -b: block size " << v
                              << " out of range [16, 65520].\n";
                    return EXIT_FAILURE;
                }
                if (v % 16 != 0) {
                    std::cerr << "Error: -b: block size " << v
                              << " is not a multiple of 16.\n";
                    return EXIT_FAILURE;
                }
                sizes.push_back((uint32_t)v);
            }
            if (sizes.empty()) {
                std::cerr << "Error: -b: empty block-size list.\n";
                return EXIT_FAILURE;
            }
            // Every size must be a multiple of the smallest. The DP's nodes
            // sit every gcd(sizes) samples, but the positions actually
            // *reachable* from the start are the sums of candidates. If the
            // smallest candidate is larger than the gcd, most nodes — quite
            // possibly the final one — cannot be reached at all, and the DP
            // finds no path and emits an empty stream. (Caught the hard way:
            // 1024,...,65520 gives gcd 16 with a smallest step of 1024, and
            // produced a 99-byte file that failed `flac -t`.)
            std::sort(sizes.begin(), sizes.end());
            sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
            for (uint32_t v : sizes) {
                if (v % sizes.front() != 0) {
                    std::cerr << "Error: -b: " << v << " is not a multiple of the "
                              << "smallest size " << sizes.front()
                              << ". Every block size must be a multiple of the "
                              << "smallest, or the DP cannot reach the end of "
                              << "the stream.\n";
                    return EXIT_FAILURE;
                }
            }
            cfg.dp_candidates = std::move(sizes);

        } else if (arg.substr(0, 2) == "--" || arg.substr(0, 1) == "-") {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;

        } else {
            positional.push_back(arg);
        }
    }

    if (positional.empty()) {
        std::cerr << "Error: no input file specified.\n";
        return EXIT_FAILURE;
    }

    const std::string input  = positional[0];
    std::error_code dir_ec;
    // A directory input is a batch. The output must be named: there is no
    // sensible default that is not either in-place or a surprise.
    const bool batch = fs::is_directory(input, dir_ec);
    if (batch && positional.size() < 2) {
        std::cerr << "Error: a directory input needs an output directory.\n";
        return EXIT_FAILURE;
    }
    const std::string output = (positional.size() >= 2)
                                 ? positional[1]
                                 : input + ".optimized.flac";

    // Effort first, then whatever was named explicitly — so `-E 7 -L 0` means
    // "level 7's depth, but price the whole ladder".
    if (effort >= 0 && !flacoutcpp::apply_effort(cfg, effort)) {
        std::cerr << "Error: -E takes an effort level 0-12, got " << effort << ".\n";
        return EXIT_FAILURE;
    }
    if (candidates_given)  cfg.max_candidates  = given_candidates;
    if (patience_given)    cfg.patience        = given_patience;
    if (rungs_given)       cfg.precision_rungs = given_rungs;
    // An explicit -a still means -a, including the hard error below when it is
    // combined with -e or -w. A level's adaptive setting is a preference, not
    // a request, so apply_effort already dropped it in those cases.
    if (adaptive_given)    cfg.adaptive_windows = true;
    if (adaptive_off)      cfg.adaptive_windows = false;

    // -e alone means the classic unlimited sweep, and the same for the other
    // two estimated-DP defaults: bare -e must still mean "exhaustive", not
    // "exhaustive with the default search bolted on". Any of -c/-L/-E is a
    // deliberate statement about the search and is left alone — `-e -E 5` means
    // level 5's depth under exact DP, which used to be silently widened back to
    // a full sweep.
    if (cfg.exhaustive && effort < 0) {
        if (!candidates_given) cfg.max_candidates  = 0;
        if (!rungs_given)      cfg.precision_rungs = 0;
    }

    // -G composes with whatever search is configured -- the ranked driver's
    // costs are batchable, since its only sequential state is best_lpc_cost
    // and the pruning that reads it can only skip candidates that could not
    // have won. The precision ladder is the exception: it *selects* rungs on
    // that bound rather than merely skipping losers, so batching it would
    // change which rungs are encoded. The ladder exists to dodge CPU cost the
    // GPU does not pay, so -G prices the whole ladder instead, unless -L was
    // asked for explicitly -- in which case those subframes stay on the CPU.
    // Only the bare default is overridden, never an explicit -E or -L: -G is
    // a statement about where the search runs, not what it searches, and that
    // invariant is what makes "byte-identical to the CPU build" mean anything.
    // Measured on the master mix, silently turning -E 0's -L 1 into -L 0 moved
    // both size (-3944 B) and time (+0.07 s) -- a different search wearing the
    // same flag. Same treatment -e already gets a few lines up.
    if (cfg.use_gpu && effort < 0 && !rungs_given) cfg.precision_rungs = 0;
    if (cfg.use_gpu && cfg.precision_rungs != 0) {
        std::cerr << "Note: -L and -G do not combine; those subframes run "
                     "on the CPU.\n";
    }
    // -e and -w each define their own window set. The adaptive *default* yields
    // to them silently; only an explicit -a is a contradiction worth rejecting.
    if (!adaptive_given && (cfg.exhaustive || !cfg.windows.empty()))
        cfg.adaptive_windows = false;

    // -W reads the reuse comparison's results, so it needs reuse enabled.
    if (cfg.warn_superior && !cfg.reuse_frames) {
        std::cerr << "Error: -W detects superior input frames via the reuse "
                     "comparison; it cannot be combined with -R.\n";
        return EXIT_FAILURE;
    }

    // Adaptive selection chooses the window set itself; -e and -w each define
    // their own set, so the combinations are contradictory rather than merely
    // redundant — reject them.
    if (cfg.adaptive_windows && (cfg.exhaustive || !cfg.windows.empty())) {
        std::cerr << "Error: -a is estimated-DP only and picks its own windows; "
                     "it cannot be combined with -e or -w.\n";
        return EXIT_FAILURE;
    }

    // Patience defaults to twice the candidate budget; resolve it here so the
    // rest of the program sees a concrete number.
    if (cfg.patience < 0)
        cfg.patience = static_cast<int>(cfg.max_candidates) * 2;

    if (cfg.verbose) {
        if (cfg.max_candidates > 0)
            std::cout << "Ranked search: " << cfg.max_candidates
                      << " candidates/subframe, patience "
                      << (cfg.patience > 0 ? std::to_string(cfg.patience) : std::string("off"))
                      << "\n";
        else
            std::cout << "Ranked search: unlimited (full sweep)\n";
        if (cfg.windows.empty()) {
            if (cfg.exhaustive)
                std::cout << "Windows: all (" << all_window_types().size() << " functions)\n";
            else {
                // Printed from default_shortlist(), never from a copy of it:
                // this line named four windows long after the list grew to ten.
                const auto sl = default_shortlist();
                std::cout << "Windows: default short list (" << sl.size() << "): ";
                for (size_t i = 0; i < sl.size(); ++i) {
                    if (i) std::cout << ", ";
                    std::cout << window_to_name(sl[i]);
                }
                if (cfg.adaptive_windows)
                    std::cout << " (+ adaptive per-block additions)";
                std::cout << "\n";
            }
        }
        else {
            std::cout << "Windows: ";
            for (size_t i = 0; i < cfg.windows.size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << window_to_name(cfg.windows[i]);
            }
            std::cout << "\n";
        }
        if (!batch)
            std::cout << "Optimising: " << input << " -> " << output << "\n";
        if (!cfg.copy_metadata) std::cout << "Metadata copying disabled.\n";
    }

    if (batch) {
        const int failed = run_batch(input, output, cfg);
        return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    if (!flacoutcpp::optimise(input, output, cfg)) {
        std::cerr << "Optimisation failed.\n";
        return EXIT_FAILURE;
    }
    flacoutcpp::release_shared_pure_gpu_encoder();

    if (cfg.verbose) std::cout << "Done.\n";
    return EXIT_SUCCESS;
}
