#include "processor.hpp"
#include "optimizer.hpp"
#include "frame_writer.hpp"
#include "md5.hpp"
#include "gpu_encoder.hpp"
#include "FLAC/stream_decoder.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>
#include <cstring>

namespace flacoutcpp {

// Per-worker decode range, for the parallel streaming decode used by -P. These
// are thread_local rather than members because every worker shares one Processor
// as its libFLAC client_data, and the alternative -- a per-worker context struct
// -- would change all three callback signatures for the CPU path's benefit too.
static thread_local uint64_t t_range_lo = 0;
static thread_local uint64_t t_range_hi = UINT64_MAX;
/// End sample of the last frame this worker decoded, so its loop knows when it
/// has covered its range.
static thread_local uint64_t t_frame_end = 0;

// Frame-reuse bookkeeping under a *parallel* decode. Each worker records the
// frames it decodes with their byte ranges; the ranges are merged and sorted
// once every worker has joined. Ownership of a boundary frame is exactly one
// worker's: the seeking worker drops the frame the seek landed on (t_skip_rec)
// and the previous worker decodes one frame past its range to cover it, so the
// byte offsets chain unbroken across the join. The merge still dedups, as a
// guard rather than an expected case.
static thread_local std::vector<Processor::PendingFrameRec> t_frames;
/// Byte offset just past the previous frame this worker decoded; seeded from the
/// decoder's position right after the seek.
static thread_local uint64_t t_prev_end = 0;
/// Cleared per worker: a position query that fails once makes the whole map
/// unusable, and the merge must see that.
static thread_local bool t_pos_ok = true;
/// Set across a seek. FLAC__stream_decoder_seek_absolute delivers the frame it
/// lands on through the write callback *before* it returns, which is before
/// t_prev_end has been seeded -- so that one frame's byte_start is not knowable
/// here and the record must be dropped. The previous worker supplies it instead;
/// see the overlap in decode_parallel.
static thread_local bool t_skip_rec = false;

// ============================================================
// Constructor / destructor
// ============================================================

Processor::Processor(const std::string& in, const std::string& out,
                     ProcessorConfig config)
    : m_input(in), m_output(out), m_config(std::move(config)) {}

Processor::~Processor() = default;

// ============================================================
// Raw metadata extraction
// Reads non-STREAMINFO metadata blocks verbatim from the FLAC
// container so we can splice them unchanged into the output.
// ============================================================

bool Processor::read_extra_metadata_blocks(
    std::vector<std::vector<uint8_t>>& out_blocks) const
{
    std::ifstream f(m_input, std::ios::binary);
    if (!f) return false;

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "fLaC") return false;

    bool is_last = false;
    while (!is_last && f) {
        uint8_t hdr[4];
        f.read(reinterpret_cast<char*>(hdr), 4);
        if (!f || f.gcount() < 4) break;

        is_last      = (hdr[0] & 0x80u) != 0;
        uint8_t type = hdr[0] & 0x7Fu;
        uint32_t len = ((uint32_t)hdr[1] << 16)
                     | ((uint32_t)hdr[2] <<  8)
                     |  (uint32_t)hdr[3];

        std::vector<uint8_t> data(len);
        f.read(reinterpret_cast<char*>(data.data()), len);

        if (type != 0) { // skip STREAMINFO (we re-generate it)
            // Store: 4-byte header (is_last cleared for now) + payload
            std::vector<uint8_t> block;
            block.reserve(4 + len);
            block.push_back(type & 0x7Fu); // is_last will be set by the caller
            block.push_back(hdr[1]);
            block.push_back(hdr[2]);
            block.push_back(hdr[3]);
            block.insert(block.end(), data.begin(), data.end());
            out_blocks.push_back(std::move(block));
        }
    }
    return true;
}

// Vendor identity from a VORBIS_COMMENT block (type 4): the payload begins
// with a 32-bit little-endian length followed by the UTF-8 vendor string —
// which is where encoders identify themselves ("reference libFLAC x.y.z",
// "Lavf...", ...). Blocks are stored header+payload by
// read_extra_metadata_blocks, so the payload starts at offset 4.
static std::string vendor_from_blocks(
    const std::vector<std::vector<uint8_t>>& blocks)
{
    for (const auto& b : blocks) {
        if (b.size() < 8 || (b[0] & 0x7Fu) != 4u) continue;
        const uint32_t len = (uint32_t)b[4] | ((uint32_t)b[5] << 8)
                           | ((uint32_t)b[6] << 16) | ((uint32_t)b[7] << 24);
        if (8ull + len > b.size()) continue;
        std::string v(reinterpret_cast<const char*>(&b[8]), len);
        for (char& c : v)
            if ((unsigned char)c < 0x20) c = ' '; // keep the warning one-line
        return v;
    }
    return {};
}

// ------------------------------------------------------------
// SEEKTABLE (block type 3)
//
// The input's seek points describe the *input's* partition, and this encoder
// rewrites the stream with a different one -- so copying the block through
// verbatim ships offsets that point into the middle of our frames. libFLAC's
// own seek fails outright on such a file, and a player that trusts it lands in
// the wrong place. The block is therefore rebuilt from the frames actually
// emitted, in place: it keeps its size, so it can be patched after the audio
// is written the same way STREAMINFO is.
//
// A seek point is 18 bytes, all big-endian: target sample number (8), byte
// offset of that frame's header from the first frame header (8), and the
// frame's sample count (2). Sample number 0xFFFF'FFFF'FFFF'FFFF marks a
// placeholder point, which the format allows and requires to sort last.
// ------------------------------------------------------------
namespace {

constexpr size_t SEEKPOINT_BYTES = 18;
constexpr uint64_t SEEKPOINT_PLACEHOLDER = ~(uint64_t)0;

struct EmittedFrame {
    uint64_t first_sample;
    uint64_t offset;      // from the first byte of the first frame header
    uint32_t block_size;
};

uint64_t rd_be64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}
void wr_be(uint8_t* p, uint64_t v, int n) {
    for (int i = n - 1; i >= 0; --i) { p[i] = (uint8_t)(v & 0xFFu); v >>= 8; }
}
void wr_placeholder(uint8_t* p) {
    wr_be(p, SEEKPOINT_PLACEHOLDER, 8);
    wr_be(p + 8, 0, 8);
    wr_be(p + 16, 0, 2);
}

/// Rewrite `payload` (a SEEKTABLE block's payload, header excluded) so every
/// point names a frame in `frames`. Each surviving input target is snapped to
/// the frame containing it; targets that collapse onto the same frame merge,
/// since two points may not name one frame, and the freed slots become
/// placeholders. Point count -- and therefore block size -- never changes.
void rebuild_seektable(std::vector<uint8_t>& payload,
                       const std::vector<EmittedFrame>& frames)
{
    const size_t n = payload.size() / SEEKPOINT_BYTES;

    // Map each real input target onto the frame that contains it.
    std::vector<size_t> hits;
    hits.reserve(n);
    if (!frames.empty()) {
        for (size_t i = 0; i < n; ++i) {
            const uint64_t want = rd_be64(&payload[i * SEEKPOINT_BYTES]);
            if (want == SEEKPOINT_PLACEHOLDER) continue;
            const auto it = std::upper_bound(
                frames.begin(), frames.end(), want,
                [](uint64_t s, const EmittedFrame& f) { return s < f.first_sample; });
            if (it == frames.begin()) continue;             // before the stream
            const size_t k = (size_t)(it - frames.begin()) - 1;
            if (want >= frames[k].first_sample + frames[k].block_size) continue;  // past the end
            hits.push_back(k);
        }
        std::sort(hits.begin(), hits.end());
        hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
    }

    for (size_t i = 0; i < n; ++i) {
        uint8_t* p = &payload[i * SEEKPOINT_BYTES];
        if (i < hits.size()) {
            const EmittedFrame& f = frames[hits[i]];
            wr_be(p, f.first_sample, 8);
            wr_be(p + 8, f.offset, 8);
            wr_be(p + 16, f.block_size, 2);
        } else {
            wr_placeholder(p);
        }
    }
}

}  // namespace

// ============================================================
// Main pipeline
// ============================================================

bool Processor::process() {
    // --- Step 1: collect raw extra metadata blocks ----
    std::vector<std::vector<uint8_t>> extra_blocks;
    if (m_config.copy_metadata) {
        if (!read_extra_metadata_blocks(extra_blocks))
            std::cerr << "Warning: could not copy metadata from " << m_input << "\n";
    }

    // -P diverges before the decode, not after: it drives its own streaming
    // decode so the device can start on chunk 0 while the rest of the file is
    // still being decoded. It shares the metadata handling and the file plumbing
    // with the CPU path and nothing else.
    if (m_config.pure_gpu)
        return process_pure_gpu(extra_blocks);

    // --- Step 2: decode PCM with libFLAC ----
    FLAC__StreamDecoder* decoder = FLAC__stream_decoder_new();
    if (!decoder) return false;

    FLAC__stream_decoder_set_metadata_respond(decoder, FLAC__METADATA_TYPE_STREAMINFO);
    if (FLAC__stream_decoder_init_file(decoder, m_input.c_str(),
                                       write_callback, metadata_callback,
                                       error_callback, this)
        != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
        FLAC__stream_decoder_delete(decoder);
        return false;
    }

    // For frame reuse we need each input frame's byte range. Decode metadata
    // first so the position query below lands on the first audio frame -- and,
    // for the parallel path, so STREAMINFO's sample count is known before the
    // buffers are sized.
    bool ok = FLAC__stream_decoder_process_until_end_of_metadata(decoder);
    if (ok && m_config.reuse_frames &&
        !FLAC__stream_decoder_get_decode_position(decoder, &m_prev_frame_end))
        m_frame_pos_ok = false;

    // Parallel decode, ported from -P. libFLAC is single-threaded and its decode
    // is a fixed cost the search cannot hide: measured at 0.24 s of a 1.05 s run
    // on a 4.7-minute track at the leanest search settings, i.e. a quarter of it,
    // against 0.024 s for eight workers.
    //
    // Frames are independent once you know where they start, and a seek is how a
    // worker finds out. Needs a sample count to size the buffers, so a STREAMINFO
    // without one keeps the serial path -- as do short files, where the seeks and
    // the extra decoder instances cost more than they save.
    const uint64_t align = (m_max_blocksize > 0) ? m_max_blocksize : 4096;
    unsigned nthr = ok && m_total_samples > 0
                  ? decode_thread_count(m_total_samples, align) : 1;
    if (nthr > 1) {
        m_pcm_data.assign(m_channels, std::vector<int32_t>());
        for (auto& ch : m_pcm_data) ch.resize((size_t)m_total_samples);
        m_stream_mode = true;
        m_decoded.store(0, std::memory_order_relaxed);
        m_decode_done.store(false, std::memory_order_relaxed);
        m_decode_failed.store(false, std::memory_order_relaxed);

        uint64_t rlen = (m_total_samples + nthr - 1) / nthr;
        rlen = ((rlen + align - 1) / align) * align;
        nthr = (unsigned)((m_total_samples + rlen - 1) / rlen);
        ok = decode_parallel(decoder, nthr, rlen);
        m_stream_mode = false;
    } else {
        ok = ok && FLAC__stream_decoder_process_until_end_of_stream(decoder);
    }
    FLAC__stream_decoder_delete(decoder);

    if (!ok || m_pcm_data.empty() || m_total_samples == 0) return false;

    if (m_config.verbose)
        std::cout << "Decoded " << m_total_samples << " samples ("
                  << m_channels << " ch, " << m_bps << " bps, "
                  << m_sample_rate << " Hz)\n";

    // --- Step 2b: MD5 over interleaved little-endian PCM, on its own thread ----
    //
    // The FLAC spec mandates MD5 over the raw audio (channel-interleaved,
    // little-endian, ceil(bps/8) bytes per sample). Nothing downstream needs the
    // digest until STREAMINFO is patched at the very end, and the hash depends
    // only on the decoded samples -- which the optimiser only reads. So it runs
    // beside the search instead of in front of it.
    //
    // Two changes, both ported from -P, where they were worth 0.221 -> 0.170 s:
    //
    //  - **Batched.** update() was being called once per *sample* with a 4-6 byte
    //    buffer, which is mostly call overhead; 4096 samples per call is not.
    //  - **Threaded.** MD5 is compute-bound at a few hundred MB/s and cannot be
    //    parallelised internally (it chains 64-byte blocks with no combine
    //    operator), so the only way to hide it is to overlap it with something
    //    else. Here that is the whole search.
    //
    // It costs one thread against the optimiser's pool for ~100 ms on a 5-minute
    // track, and buys all of it back on any configuration where the search is
    // shorter than the hash.
    const int bytes_per_sample = (m_bps + 7) / 8;
    detail::MD5 md5;
    // Joined by the guard below on *every* exit from here on. Without it an early
    // `return false` (a failed optimise, a failed open) would run ~std::thread on
    // a joinable thread, which is std::terminate rather than an error path.
    struct Joiner {
        std::thread& t;
        ~Joiner() { if (t.joinable()) t.join(); }
    };
    std::thread md5_thread([this, bytes_per_sample, &md5]() {
        const uint32_t stride = m_channels * (uint32_t)bytes_per_sample;
        constexpr uint64_t BATCH = 4096;
        std::vector<uint8_t> buf((size_t)BATCH * stride);
        for (uint64_t at = 0; at < m_total_samples; ) {
            const uint64_t n = std::min<uint64_t>(BATCH, m_total_samples - at);
            uint8_t* w = buf.data();
            for (uint64_t i = 0; i < n; ++i)
                for (uint32_t c = 0; c < m_channels; ++c) {
                    const int32_t v = m_pcm_data[c][at + i];
                    for (int b = 0; b < bytes_per_sample; ++b)
                        *w++ = (uint8_t)(v >> (b * 8));
                }
            md5.update(buf.data(), (size_t)n * stride);
            at += n;
        }
    });
    Joiner md5_joiner{md5_thread};

    // --- Step 2c: frame-reuse preparation ----
    // Validate that the recorded input frames tile the stream and load the
    // input bytes for payload splicing. Under exact-DP mode the grid-aligned
    // frames are also offered to the partitioning DP as exact-cost edges, so
    // the DP can mix the input's partition with its own.
    bool reuse = m_config.reuse_frames && m_frame_pos_ok && !m_input_frames.empty();
    if (const char* p = std::getenv("FLACOUT_DUMP_FRAMES")) {
        if (FILE* df = fopen(p, "w")) {
            for (const auto& x : m_input_frames)
                fprintf(df, "%llu %u %llu %llu\n", (unsigned long long)x.first_sample,
                        x.block_size, (unsigned long long)x.byte_start,
                        (unsigned long long)x.byte_end);
            fclose(df);
        }
    }
    std::vector<uint8_t> input_bytes;
    if (reuse) {
        uint64_t expect = 0;
        for (const auto& f : m_input_frames) {
            if (f.first_sample != expect || f.byte_end <= f.byte_start) { reuse = false; break; }
            expect += f.block_size;
        }
        if (expect != m_total_samples) reuse = false;
    }
    if (reuse) {
        std::ifstream inf(m_input, std::ios::binary);
        inf.seekg(0, std::ios::end);
        input_bytes.resize((size_t)inf.tellg());
        inf.seekg(0, std::ios::beg);
        inf.read(reinterpret_cast<char*>(input_bytes.data()),
                 (std::streamsize)input_bytes.size());
        if (!inf || input_bytes.size() < m_input_frames.back().byte_end)
            reuse = false;
    }

    std::vector<ReuseEdge> reuse_edges;
    if (reuse && m_config.exhaustive) {
        for (size_t i = 0; i < m_input_frames.size(); ++i) {
            const auto& f = m_input_frames[i];
            // Off-grid frames are usable too: the DP grows nodes at input
            // frame boundaries and bridges them to its grid.
            auto fb = FrameWriter::rewrite_frame(
                &input_bytes[f.byte_start], (size_t)(f.byte_end - f.byte_start),
                f.first_sample, f.block_size, m_sample_rate);
            if (fb.empty()) continue;
            reuse_edges.push_back(ReuseEdge{ f.first_sample, f.block_size,
                                             (uint32_t)fb.size(), (uint32_t)i });
        }
    }

    // --- Step 3: run optimiser ----
    // Resolve the patience default here rather than only in the CLI, so a
    // library caller that never touches Config::patience gets the same search
    // as `flacoutcpp` with no flags.
    const unsigned resolved_patience =
        m_config.patience < 0 ? m_config.max_candidates * 2
                              : (unsigned)m_config.patience;
    Optimizer opt(m_channels, m_bps, m_sample_rate, m_config.windows, m_config.max_threads,
                  m_config.exhaustive, m_config.verbose, m_config.max_candidates,
                  m_config.adaptive_windows, resolved_patience,
                  m_config.precision_rungs, m_config.dp_candidates,
                  m_config.lattice_sweeps,
                  m_config.use_gpu,
                  m_config.gpu_min_batch,
                  m_config.gpu_partition_cap,
                  m_config.gpu_slots,
                  m_config.gpu_duty);
    if (!reuse_edges.empty())
        opt.set_reuse_edges(std::move(reuse_edges));
    std::vector<BlockParams> blocks = opt.find_optimal_block_partitioning(m_pcm_data);

    if (blocks.empty()) {
        std::cerr << "Error: optimizer produced no blocks.\n";
        return false;
    }

    // Compute min/max block size (known before writing)
    uint32_t min_bs = std::numeric_limits<uint32_t>::max(), max_bs = 0;
    for (const auto& b : blocks) {
        min_bs = std::min(min_bs, b.block_size);
        max_bs = std::max(max_bs, b.block_size);
    }

    // --- Step 4: open a temporary output file and write header ----
    // Written to m_output + ".partial" and renamed into place only after a
    // fully successful write, so a failure or crash partway through never
    // leaves a corrupt/truncated file at m_output (nor clobbers a pre-existing
    // file there before we know we can replace it).
    // We need seekp() later to update STREAMINFO, so use fstream.
    const std::string tmp_output = m_output + ".partial";
    std::fstream out(tmp_output,
                     std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Error: cannot open output file: " << tmp_output << "\n";
        return false;
    }

    // 4-byte FLAC magic
    out.write("fLaC", 4);

    // STREAMINFO block (is_last = true only if there are no extra blocks)
    bool si_is_last = extra_blocks.empty();
    auto si_block = FrameWriter::make_streaminfo_block(
        si_is_last,
        min_bs, max_bs,
        0, 0,   // min/max framesize unknown at this point
        m_sample_rate, m_channels, m_bps, m_total_samples);
    out.write(reinterpret_cast<const char*>(si_block.data()),
              (std::streamsize)si_block.size());

    // Extra metadata blocks. A SEEKTABLE among them is written out as-is here
    // and patched once the frames it must describe exist; note where its
    // payload landed. Only the first is rebuilt — the format permits one.
    std::streamoff       seektable_at = -1;
    std::vector<uint8_t> seektable_pts;
    for (size_t i = 0; i < extra_blocks.size(); ++i) {
        bool is_last = (i == extra_blocks.size() - 1);
        auto& blk = extra_blocks[i];
        // Set/clear the is_last bit in the stored header byte
        if (is_last) blk[0] |= 0x80u;
        else         blk[0] &= 0x7Fu;
        if ((blk[0] & 0x7Fu) == 3u && blk.size() > 4 && seektable_at < 0) {
            seektable_at = (std::streamoff)out.tellp() + 4;   // skip the block header
            seektable_pts.assign(blk.begin() + 4, blk.end());
        }
        out.write(reinterpret_cast<const char*>(blk.data()),
                  (std::streamsize)blk.size());
    }
    const bool want_seektable = seektable_at >= 0
                             && seektable_pts.size() >= SEEKPOINT_BYTES;

    // --- Step 5: encode and write frames ----
    FrameWriter fw;
    uint32_t min_frm = std::numeric_limits<uint32_t>::max();
    uint32_t max_frm = 0;
    uint32_t emit_min_bs = std::numeric_limits<uint32_t>::max();
    uint32_t emit_max_bs = 0;
    size_t   total_written = 0;
    size_t   frames_reused = 0;
    size_t   frames_emitted = 0;

    // Only collected when a SEEKTABLE has to be rebuilt against it.
    std::vector<EmittedFrame> emitted;
    uint64_t emit_sample = 0;

    auto emit = [&](const std::vector<uint8_t>& fb, uint32_t bs) {
        ++frames_emitted;
        if (want_seektable)
            emitted.push_back(EmittedFrame{emit_sample, total_written, bs});
        emit_sample += bs;
        out.write(reinterpret_cast<const char*>(fb.data()),
                  (std::streamsize)fb.size());
        min_frm = std::min(min_frm, (uint32_t)fb.size());
        max_frm = std::max(max_frm, (uint32_t)fb.size());
        emit_min_bs = std::min(emit_min_bs, bs);
        emit_max_bs = std::max(emit_max_bs, bs);
        total_written += fb.size();
    };

    // Build one frame from a DP block: either a re-encode, or — when the DP
    // chose a reuse edge — the rewritten input frame it identified.
    auto build_frame = [&](const BlockParams& block,
                           uint64_t sample_number) -> std::vector<uint8_t> {
        if (block.reuse_index >= 0) {
            const auto& f = m_input_frames[(size_t)block.reuse_index];
            return FrameWriter::rewrite_frame(
                &input_bytes[f.byte_start], (size_t)(f.byte_end - f.byte_start),
                f.first_sample, f.block_size, m_sample_rate);
        }
        auto fb = fw.write_frame(block, m_pcm_data, sample_number,
                                 m_sample_rate, m_bps);
        // Debug builds: the optimizer's predicted frame size must equal what
        // the writer actually emitted, or the DP is optimizing a cost the
        // stream does not pay. (block.total_bits is exact here under -e;
        // heuristic blocks also come from compute_block.)
        assert(fb.size() * 8 ==
               FrameWriter::frame_bits(sample_number, block.block_size,
                                       m_sample_rate, block.total_bits));
        return fb;
    };

    if (!reuse) {
        uint64_t sample_number = 0;
        for (const auto& block : blocks) {
            emit(build_frame(block, sample_number), block.block_size);
            sample_number += block.block_size;
        }
    } else {
        // Two-pointer walk over both partitions. A segment closes whenever
        // they hit a common sample boundary; whichever side spent fewer
        // bytes over the segment is emitted. Ties keep the re-encoded
        // frames, so a second pass over our own output is byte-stable.
        size_t bi = 0, ij = 0;
        uint64_t our_end = 0, in_end = 0;
        std::vector<std::pair<std::vector<uint8_t>, uint32_t>> ours, theirs;
        size_t our_sz = 0, in_sz = 0, our_reused = 0;
        bool   seg_ok = true; // false if any input frame failed to rewrite

        while (bi < blocks.size() || ij < m_input_frames.size()) {
            if (our_end <= in_end && bi < blocks.size()) {
                const auto& block = blocks[bi++];
                if (block.reuse_index >= 0) our_reused++;
                auto fb = build_frame(block, our_end);
                our_sz += fb.size();
                our_end += block.block_size;
                ours.emplace_back(std::move(fb), block.block_size);
            } else if (ij < m_input_frames.size()) {
                const auto& f = m_input_frames[ij++];
                auto fb = FrameWriter::rewrite_frame(
                    &input_bytes[f.byte_start],
                    (size_t)(f.byte_end - f.byte_start),
                    f.first_sample, f.block_size, m_sample_rate);
                if (fb.empty()) seg_ok = false;
                else in_sz += fb.size();
                in_end += f.block_size;
                theirs.emplace_back(std::move(fb), f.block_size);
            } else {
                break; // partitions disagree on total length; validated above
            }

            if (our_end == in_end && !ours.empty() && !theirs.empty()) {
                const bool take_input = seg_ok && in_sz < our_sz;
                auto& chosen = take_input ? theirs : ours;
                frames_reused += take_input ? theirs.size() : our_reused;
                for (auto& [fb, bs] : chosen) emit(fb, bs);
                ours.clear(); theirs.clear();
                our_sz = in_sz = 0; our_reused = 0;
                seg_ok = true;
            }
        }
    }

    // The hash has been running beside the search; this is the first point that
    // needs its result. (The guard would join at scope exit anyway, but the
    // digest cannot be read before the thread is done.)
    if (md5_thread.joinable()) md5_thread.join();
    const auto md5_digest = md5.digest();

    // --- Step 6: seek back and update STREAMINFO with frame sizes + MD5 ----
    // Block sizes come from the emitted frames — with reuse they can differ
    // from the DP's blocks.
    // Position: fLaC(4) + STREAMINFO block header(4) = byte 8
    out.seekp(8, std::ios::beg);
    auto si_updated = FrameWriter::make_streaminfo_block(
        si_is_last, emit_min_bs, emit_max_bs,
        min_frm, max_frm,
        m_sample_rate, m_channels, m_bps, m_total_samples,
        md5_digest.data());
    // Write only the 34-byte payload (skip the 4-byte block header)
    out.write(reinterpret_cast<const char*>(si_updated.data() + 4), 34);

    // --- Step 6b: rebuild SEEKTABLE against the frames actually emitted ----
    if (want_seektable) {
        rebuild_seektable(seektable_pts, emitted);
        out.seekp(seektable_at, std::ios::beg);
        out.write(reinterpret_cast<const char*>(seektable_pts.data()),
                  (std::streamsize)seektable_pts.size());
    }

    out.flush();
    if (!out) {
        std::cerr << "Error: write failed.\n";
        out.close();
        std::remove(tmp_output.c_str());
        return false;
    }
    out.close();

    // Whole-file guarantee: if the finished file is still larger than what
    // shipping the input's own frames would cost, do that instead.
    //
    // Per-segment reuse alone does not get us here. It picks min(ours,
    // theirs), but "theirs" is the input's frames *rewritten* — and
    // rewrite_frame emits variable blocking strategy with a sample number
    // where libFLAC's fixed-blocksize output carried a frame number, which is
    // 1-2 UTF-8 bytes longer per frame. A segment can also fall back to our
    // side regardless of size if any input frame fails to rewrite. So reuse
    // bounds the output against what we could emit, not against the input,
    // and this fallback is what closes the gap.
    bool copied_through = false;
    if (m_config.reuse_frames) {
        std::ifstream a(tmp_output, std::ios::binary | std::ios::ate);
        const std::streamoff out_sz = a ? (std::streamoff)a.tellg() : -1;

        if (m_config.copy_metadata) {
            std::ifstream b(m_input, std::ios::binary | std::ios::ate);
            if (a && b && out_sz > (std::streamoff)b.tellg()) {
                b.seekg(0, std::ios::beg);
                std::ofstream repl(tmp_output, std::ios::binary | std::ios::trunc);
                repl << b.rdbuf();
                if (!repl) {
                    std::cerr << "Error: copy-through write failed.\n";
                    std::remove(tmp_output.c_str());
                    return false;
                }
                copied_through = true;
            }
        } else if (reuse && !m_input_frames.empty() && out_sz >= 0) {
            // -n asked for the metadata to be dropped, not for the guarantee
            // to be dropped with it. Ship the input's audio frames verbatim
            // under a fresh header carrying STREAMINFO alone: the input's own
            // STREAMINFO already describes exactly these frames (framing,
            // sample count, MD5), so it transfers unchanged apart from having
            // to become the last metadata block.
            constexpr uint64_t HDR = 4 + 4 + 34;  // "fLaC" + block header + payload
            const uint64_t audio_start = m_input_frames.front().byte_start;
            const uint64_t audio_end   = m_input_frames.back().byte_end;
            if (audio_end > audio_start && (uint64_t)out_sz > HDR + (audio_end - audio_start)) {
                std::ifstream b(m_input, std::ios::binary);
                std::vector<char> head(HDR);
                if (b && b.read(head.data(), (std::streamsize)HDR)
                      && std::memcmp(head.data(), "fLaC", 4) == 0
                      && (head[4] & 0x7F) == 0                       // STREAMINFO
                      && ((uint8_t)head[5] << 16 | (uint8_t)head[6] << 8
                          | (uint8_t)head[7]) == 34) {
                    head[4] = (char)0x80;  // STREAMINFO, and now the last block
                    std::ofstream repl(tmp_output, std::ios::binary | std::ios::trunc);
                    repl.write(head.data(), (std::streamsize)HDR);
                    b.seekg((std::streamoff)audio_start, std::ios::beg);
                    std::vector<char> buf(1 << 16);
                    uint64_t left = audio_end - audio_start;
                    while (left > 0 && b && repl) {
                        const std::streamsize n =
                            (std::streamsize)std::min<uint64_t>(left, buf.size());
                        b.read(buf.data(), n);
                        repl.write(buf.data(), b.gcount());
                        left -= (uint64_t)b.gcount();
                        if (b.gcount() == 0) break;
                    }
                    if (!repl || left != 0) {
                        std::cerr << "Error: copy-through write failed.\n";
                        std::remove(tmp_output.c_str());
                        return false;
                    }
                    copied_through = true;
                }
            }
        }
    }

    // -W: the reuse comparison doubles as a detector for "the input encoder
    // beat this search somewhere" — surface that, with the culprit's vendor
    // identity when its metadata carries one.
    if (m_config.warn_superior && (frames_reused > 0 || copied_through)) {
        std::vector<std::vector<uint8_t>> fresh;
        const auto* blocks_src = &extra_blocks;
        if (!m_config.copy_metadata) {  // -n skipped the metadata read; do it now
            read_extra_metadata_blocks(fresh);
            blocks_src = &fresh;
        }
        const std::string vendor = vendor_from_blocks(*blocks_src);
        std::cerr << "Warning: ";
        if (copied_through)
            std::cerr << "the entire input was copied through — "
                         "the re-encode would have been larger";
        else
            std::cerr << "input frames beat the re-encode for " << frames_reused
                      << " of " << frames_emitted << " frames";
        std::cerr << " (input encoder: "
                  << (vendor.empty() ? std::string("unknown")
                                     : "\"" + vendor + "\"")
                  << ").\n";
    }

    if (std::rename(tmp_output.c_str(), m_output.c_str()) != 0) {
        std::cerr << "Error: could not finalize output file: " << m_output << "\n";
        std::remove(tmp_output.c_str());
        return false;
    }

    if (m_config.verbose) {
        if (copied_through)
            std::cout << "Output would exceed the input; copied the input through unchanged.\n";
        else
            std::cout << "Wrote " << total_written << " bytes of audio data ("
                      << blocks.size() << " frames, min=" << emit_min_bs
                      << " max=" << emit_max_bs << " samples/frame"
                      << (frames_reused
                          ? ", " + std::to_string(frames_reused) + " input frames reused"
                          : std::string())
                      << ").\n";
    }
    return true;
}

// ============================================================
// The -P pipeline
// ============================================================

unsigned Processor::decode_thread_count(uint64_t total_samples, uint64_t align) const
{
    unsigned n = std::thread::hardware_concurrency();
    n = std::max(1u, std::min(n ? n / 2 : 1u, 8u));
    if (const char* e = std::getenv("FLACOUT_DECODE_THREADS"))
        n = (unsigned)std::max(1, std::min(32, atoi(e)));
    // Below a few ranges the seeks and extra decoder instances cost more than
    // they save, and a range shorter than one frame cannot be split at all.
    if (align == 0 || total_samples < 4 * align) n = 1;
    return n;
}

void Processor::merge_input_frames(std::vector<std::vector<PendingFrameRec>>& per_worker)
{
    m_input_frames.clear();
    std::vector<PendingFrameRec> all;
    for (auto& v : per_worker) {
        if (v.empty() && !m_frame_pos_ok) return;
        all.insert(all.end(), v.begin(), v.end());
    }
    std::sort(all.begin(), all.end(),
              [](const PendingFrameRec& a, const PendingFrameRec& b) {
                  return a.first_sample < b.first_sample;
              });
    // Guard: workers are supposed to own boundary frames exclusively, but a
    // duplicate is harmless as long as only one copy is kept.
    uint64_t expect = 0;
    for (const auto& f : all) {
        if (!m_input_frames.empty() &&
            f.first_sample == m_input_frames.back().first_sample) continue;
        if (f.first_sample != expect || f.byte_end <= f.byte_start) {
            m_frame_pos_ok = false;
            m_input_frames.clear();
            return;
        }
        m_input_frames.push_back(InputFrame{f.first_sample, f.block_size,
                                            f.byte_start, f.byte_end});
        expect += f.block_size;
    }
    if (expect != m_total_samples) { m_frame_pos_ok = false; m_input_frames.clear(); }
}

bool Processor::decode_parallel(FLAC__StreamDecoder* first, unsigned nthr,
                                uint64_t range_len)
{
    m_range_len = range_len;
    m_range_done.assign(nthr, 0);
    m_map_frames = m_config.reuse_frames;

    std::vector<std::vector<PendingFrameRec>> per_worker(nthr);
    std::vector<uint8_t> pos_ok(nthr, 1);
    std::vector<std::thread> workers;
    std::atomic<unsigned> live{nthr};

    for (unsigned i = 0; i < nthr; ++i) {
        workers.emplace_back([this, i, range_len, first, &live, &per_worker, &pos_ok]() {
            const uint64_t lo = (uint64_t)i * range_len;
            const uint64_t hi = std::min(lo + range_len, m_total_samples);
            t_range_lo = lo;
            t_range_hi = hi;
            t_frames.clear();
            t_pos_ok = m_config.reuse_frames;
            t_prev_end = 0;
            t_skip_rec = false;

            auto fail = [&]() {
                m_decode_failed.store(true, std::memory_order_release);
                m_decode_cv.notify_all();
                if (--live == 0) {
                    m_decode_done.store(true, std::memory_order_release);
                    m_decode_cv.notify_all();
                }
            };

            FLAC__StreamDecoder* d = first;
            bool own = false;
            if (i != 0) {
                d = FLAC__stream_decoder_new();
                if (!d || FLAC__stream_decoder_init_file(
                              d, m_input.c_str(), write_callback, nullptr,
                              error_callback, this)
                          != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
                    if (d) FLAC__stream_decoder_delete(d);
                    fail();
                    return;
                }
                own = true;
                // The seek hands the landing frame to the write callback before
                // it returns; that record is the previous worker's to make.
                t_skip_rec = true;
                const bool seek_ok = FLAC__stream_decoder_seek_absolute(d, lo);
                t_skip_rec = false;
                if (!seek_ok) {
                    FLAC__stream_decoder_delete(d);
                    fail();
                    return;
                }
            }
            // The byte offset the first frame starts at: after the seek for a
            // worker that seeked, after the metadata for worker 0 (which the
            // caller has already positioned).
            if (t_pos_ok && !FLAC__stream_decoder_get_decode_position(d, &t_prev_end))
                t_pos_ok = false;

            // Reuse needs one frame of overlap past the range end. The next
            // worker seeks to `hi` and cannot record the frame it lands on, so
            // this worker decodes it: `hi + 1` keeps going through a frame that
            // ends exactly at the boundary and stops on one that crosses it
            // (already the next worker's landing frame, hence already covered).
            // The extra samples are clipped away by range, so only the byte map
            // grows. Without reuse there is nothing to record and no overlap.
            const uint64_t cover = (m_config.reuse_frames && hi < m_total_samples)
                                 ? hi + 1 : hi;

            bool ok = true;
            for (;;) {
                const FLAC__StreamDecoderState st = FLAC__stream_decoder_get_state(d);
                if (st == FLAC__STREAM_DECODER_END_OF_STREAM) break;
                if (st == FLAC__STREAM_DECODER_ABORTED ||
                    st == FLAC__STREAM_DECODER_OGG_ERROR ||
                    st == FLAC__STREAM_DECODER_SEEK_ERROR ||
                    st == FLAC__STREAM_DECODER_MEMORY_ALLOCATION_ERROR) { ok = false; break; }
                if (!FLAC__stream_decoder_process_single(d)) { ok = false; break; }
                if (t_frame_end >= cover) break;
            }
            if (own) FLAC__stream_decoder_delete(d);
            if (!ok) m_decode_failed.store(true, std::memory_order_release);

            per_worker[i] = std::move(t_frames);
            pos_ok[i] = t_pos_ok ? 1 : 0;

            advance_decoded(i);
            if (--live == 0) {
                m_decode_done.store(true, std::memory_order_release);
                m_decode_cv.notify_all();
            }
        });
    }
    for (auto& t : workers) if (t.joinable()) t.join();

    if (m_config.reuse_frames) {
        for (unsigned i = 0; i < nthr; ++i)
            if (!pos_ok[i]) { m_frame_pos_ok = false; break; }
        if (m_frame_pos_ok) merge_input_frames(per_worker);
        else                m_input_frames.clear();
    }
    return !m_decode_failed.load(std::memory_order_acquire);
}

void Processor::advance_decoded(size_t idx)
{
    std::lock_guard<std::mutex> lk(m_decode_mu);
    if (idx < m_range_done.size()) m_range_done[idx] = 1;
    // Push the published count to the end of the contiguous completed prefix.
    // Ranges finish out of order, so a single completion may unlock several.
    uint64_t at = m_decoded.load(std::memory_order_relaxed);
    size_t next = (size_t)(at / m_range_len);
    while (next < m_range_done.size() && m_range_done[next]) {
        at = std::min((uint64_t)(next + 1) * m_range_len, m_total_samples);
        ++next;
    }
    m_decoded.store(at, std::memory_order_release);
    m_decode_cv.notify_all();
}

bool Processor::process_pure_gpu(std::vector<std::vector<uint8_t>>& extra_blocks)
{
    // ---- decode metadata on this thread, so the stream shape is known --------
    // FLACOUT_PG_WALL: wall-clock phases, to see what is left outside the
    // overlapped decode/encode window.
    const bool wt = std::getenv("FLACOUT_PG_WALL") != nullptr;
    auto wt0 = std::chrono::steady_clock::now();
    const auto t_origin = wt0;
    auto wlap = [&](const char* what) {
        if (!wt) return;
        const auto now = std::chrono::steady_clock::now();
        std::printf("  wall %-24s %7.2f ms\n", what,
                    1e3 * std::chrono::duration<double>(now - wt0).count());
        wt0 = now;
    };

    FLAC__StreamDecoder* decoder = FLAC__stream_decoder_new();
    if (!decoder) return false;
    FLAC__stream_decoder_set_metadata_respond(decoder, FLAC__METADATA_TYPE_STREAMINFO);
    if (FLAC__stream_decoder_init_file(decoder, m_input.c_str(),
                                       write_callback, metadata_callback,
                                       error_callback, this)
        != FLAC__STREAM_DECODER_INIT_STATUS_OK) {
        FLAC__stream_decoder_delete(decoder);
        return false;
    }
    if (!FLAC__stream_decoder_process_until_end_of_metadata(decoder) ||
        m_channels == 0 || m_total_samples == 0) {
        // A STREAMINFO without a sample count cannot be preallocated, and
        // streaming needs a fixed destination. Rare enough to just refuse: the
        // CPU path handles those files.
        FLAC__stream_decoder_delete(decoder);
        std::cerr << "Error: -P needs a STREAMINFO with a known sample count.\n";
        return false;
    }

    // ---- preallocate, then decode on a background thread ---------------------
    wlap("metadata decode");
    m_pcm_data.assign(m_channels, std::vector<int32_t>());
    for (auto& ch : m_pcm_data) ch.resize((size_t)m_total_samples);
    wlap("PCM buffer alloc");
    m_stream_mode = true;
    m_decoded.store(0, std::memory_order_relaxed);
    m_decode_done.store(false, std::memory_order_relaxed);
    m_decode_failed.store(false, std::memory_order_relaxed);

    // MD5 gets its own thread, following the same published counter as the
    // encoder. It was briefly folded into the decode thread on the theory that
    // the samples are already hot -- but MD5 is compute-bound at a few hundred
    // MB/s, not memory-bound, so that serialised ~66 ms of hashing behind a
    // 109 ms decode and threw away most of the overlap this restructure exists
    // for (master mix 0.239 s -> 0.221 s instead of the predicted ~0.13 s).
    // Three consumers of one producer, all overlapped, is the right shape.
    std::thread md5_thread([this]() {
        const uint32_t bpsb = (m_bps + 7) / 8;
        const uint32_t stride = m_channels * bpsb;
        // Batch rather than calling update() per sample: at 6 bytes a call the
        // per-call overhead dominates the hashing.
        constexpr uint32_t BATCH = 4096;
        std::vector<uint8_t> buf((size_t)BATCH * stride);
        uint64_t at = 0;
        while (at < m_total_samples) {
            uint64_t have;
            {
                std::unique_lock<std::mutex> lk(m_decode_mu);
                m_decode_cv.wait(lk, [&] {
                    return m_decoded.load(std::memory_order_acquire) > at ||
                           m_decode_done.load(std::memory_order_acquire) ||
                           m_decode_failed.load(std::memory_order_acquire);
                });
                have = m_decoded.load(std::memory_order_acquire);
            }
            if (m_decode_failed.load(std::memory_order_acquire)) return;
            if (have <= at) return;   // producer finished short
            while (at < have) {
                const uint32_t n = (uint32_t)std::min<uint64_t>(BATCH, have - at);
                uint8_t* w = buf.data();
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t c = 0; c < m_channels; ++c) {
                        const int32_t v = m_pcm_data[c][at + i];
                        for (uint32_t b = 0; b < bpsb; ++b)
                            *w++ = (uint8_t)(v >> (b * 8));
                    }
                m_md5.update(buf.data(), (size_t)n * stride);
                at += n;
            }
        }
    });

    // Parallel decode. libFLAC is single-threaded and its decode was 109 ms of a
    // 148 ms wall clock against ~99 ms of device time, so the two were co-bound
    // and cutting either alone bought nothing. Frames are independent once you
    // know where they start, and each worker learns that by seeking: the stream
    // is split into contiguous ranges, worker i seeks to its start and decodes
    // until it passes its end. Each frame is placed by its own sample number and
    // clipped to the worker's range, so there is exactly one writer per sample.
    //
    // The first worker reuses the already-open decoder (it starts at 0 and needs
    // no seek); the rest open their own.
    unsigned nthr = std::thread::hardware_concurrency();
    nthr = std::max(1u, std::min(nthr ? nthr / 2 : 1u, 8u));
    if (const char* e = std::getenv("FLACOUT_PG_DECODE_THREADS"))
        nthr = std::max(1, std::min(32, atoi(e)));
    if (m_total_samples < 4ull * m_config.pg_block_size) nthr = 1;

    // Ranges are block-aligned so the published prefix always lands on a frame
    // boundary the encoder can consume.
    const uint64_t blk = std::max(1u, m_config.pg_block_size);
    uint64_t rlen = (m_total_samples + nthr - 1) / nthr;
    rlen = ((rlen + blk - 1) / blk) * blk;
    nthr = (unsigned)((m_total_samples + rlen - 1) / rlen);
    m_range_len = rlen;
    m_range_done.assign(nthr, 0);

    std::vector<std::thread> decoders;
    std::atomic<unsigned> live{nthr};
    for (unsigned i = 0; i < nthr; ++i) {
        decoders.emplace_back([this, i, rlen, decoder, &live, t_origin]() {
            const uint64_t lo = (uint64_t)i * rlen;
            const uint64_t hi = std::min(lo + rlen, m_total_samples);
            t_range_lo = lo;
            t_range_hi = hi;

            FLAC__StreamDecoder* d = decoder;
            bool own = false;
            if (i != 0) {
                d = FLAC__stream_decoder_new();
                if (d && FLAC__stream_decoder_init_file(
                             d, m_input.c_str(), write_callback, nullptr,
                             error_callback, this)
                         == FLAC__STREAM_DECODER_INIT_STATUS_OK) {
                    own = true;
                } else {
                    if (d) FLAC__stream_decoder_delete(d);
                    m_decode_failed.store(true, std::memory_order_release);
                    m_decode_cv.notify_all();
                    if (--live == 0) {
                        m_decode_done.store(true, std::memory_order_release);
                        m_decode_cv.notify_all();
                    }
                    return;
                }
                if (!FLAC__stream_decoder_seek_absolute(d, lo)) {
                    FLAC__stream_decoder_delete(d);
                    m_decode_failed.store(true, std::memory_order_release);
                    m_decode_cv.notify_all();
                    if (--live == 0) {
                        m_decode_done.store(true, std::memory_order_release);
                        m_decode_cv.notify_all();
                    }
                    return;
                }
            }

            // Decode frame by frame until this range is covered. The seek already
            // positioned inside it, and process_single stops at end of stream.
            bool ok = true;
            for (;;) {
                const FLAC__StreamDecoderState st = FLAC__stream_decoder_get_state(d);
                if (st == FLAC__STREAM_DECODER_END_OF_STREAM) break;
                if (st == FLAC__STREAM_DECODER_ABORTED ||
                    st == FLAC__STREAM_DECODER_OGG_ERROR ||
                    st == FLAC__STREAM_DECODER_SEEK_ERROR ||
                    st == FLAC__STREAM_DECODER_MEMORY_ALLOCATION_ERROR) { ok = false; break; }
                if (!FLAC__stream_decoder_process_single(d)) { ok = false; break; }
                if (t_frame_end >= hi) break;
            }
            if (own) FLAC__stream_decoder_delete(d);
            if (!ok) m_decode_failed.store(true, std::memory_order_release);

            advance_decoded(i);
            if (--live == 0) {
                m_decode_done.store(true, std::memory_order_release);
                m_decode_cv.notify_all();
                if (std::getenv("FLACOUT_PG_WALL"))
                    std::printf("  wall %-24s %7.2f ms\n", "[decode complete at]",
                                1e3 * std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t_origin).count());
            }
        });
    }

    // The barrier the encoder blocks on. Returns false rather than deadlocking if
    // the decoder dies or ends short.
    auto wait_for = [this](uint64_t n) -> bool {
        std::unique_lock<std::mutex> lk(m_decode_mu);
        m_decode_cv.wait(lk, [&] {
            return m_decoded.load(std::memory_order_acquire) >= n ||
                   m_decode_done.load(std::memory_order_acquire) ||
                   m_decode_failed.load(std::memory_order_acquire);
        });
        return m_decoded.load(std::memory_order_acquire) >= n &&
               !m_decode_failed.load(std::memory_order_acquire);
    };

    auto finish_decode = [&]() {
        for (auto& th : decoders) if (th.joinable()) th.join();
        // The decode thread's exit notifies the cv, so the MD5 consumer cannot
        // still be waiting once that join returns.
        if (md5_thread.joinable()) md5_thread.join();
        FLAC__stream_decoder_delete(decoder);
    };

    // Vulkan comes up *after* the decode is already running. Instance creation is
    // ~17 ms on MoltenVK and cannot be cached (the pipeline cache handles the
    // other ~10 ms), but nothing about it depends on the audio -- so it belongs
    // beside the decode rather than in front of it.
    PureGpuEncoder::Config gcfg;
    gcfg.block_size       = m_config.pg_block_size;
    gcfg.windows          = m_config.windows;
    gcfg.blocks_per_chunk = m_config.pg_blocks_per_chunk;
    gcfg.orders           = m_config.pg_orders;
    gcfg.partition_cap    = m_config.pg_partition_cap;
    gcfg.verbose          = m_config.verbose;
    if (!m_config.pg_precisions.empty()) gcfg.precisions = m_config.pg_precisions;

    // Shared across files rather than per-file: Vulkan init and the pipeline
    // load are ~20 ms and depend on the stream's shape, not its samples, so a
    // batch of same-shaped tracks pays them once. A single run is unaffected.
    std::string gwhy;
    PureGpuEncoder* enc = shared_pure_gpu_encoder(m_channels, m_bps,
                                                  m_sample_rate, gcfg, &gwhy);
    if (!enc) {
        finish_decode();
        std::cerr << "Error: -P unavailable: " << gwhy << "\n";
        return false;
    }
    if (m_config.verbose)
        std::cout << "Pure GPU: " << gwhy << "\n";

    wlap("Vulkan init (overlapped)");
    const std::string tmp_output = m_output + ".partial";
    std::fstream out(tmp_output,
                     std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        finish_decode();
        std::cerr << "Error: cannot open output file: " << tmp_output << "\n";
        return false;
    }

    // A copied-through SEEKTABLE describes the input's partition, not ours, and
    // this path emits fixed pg_block_size frames through a chunk sink that does
    // not expose per-frame byte offsets -- so unlike process(), it cannot
    // rebuild the block. Drop it: no seek table beats a wrong one, which
    // libFLAC's own seek fails on outright.
    extra_blocks.erase(
        std::remove_if(extra_blocks.begin(), extra_blocks.end(),
                       [](const std::vector<uint8_t>& b) {
                           return !b.empty() && (b[0] & 0x7Fu) == 3u;
                       }),
        extra_blocks.end());

    out.write("fLaC", 4);
    const bool si_is_last = extra_blocks.empty();
    auto si_block = FrameWriter::make_streaminfo_block(
        si_is_last, m_config.pg_block_size, m_config.pg_block_size, 0, 0,
        m_sample_rate, m_channels, m_bps, m_total_samples);
    out.write(reinterpret_cast<const char*>(si_block.data()),
              (std::streamsize)si_block.size());
    for (size_t i = 0; i < extra_blocks.size(); ++i) {
        auto& blk = extra_blocks[i];
        if (i == extra_blocks.size() - 1) blk[0] |= 0x80u;
        else                              blk[0] &= 0x7Fu;
        out.write(reinterpret_cast<const char*>(blk.data()),
                  (std::streamsize)blk.size());
    }

    wlap("open + header");
    PureGpuEncoder::Stats st{};
    bool wrote_ok = true;
    const bool enc_ok = enc->encode(m_pcm_data,
        [&](const uint8_t* p, size_t n) {
            out.write(reinterpret_cast<const char*>(p), (std::streamsize)n);
            if (!out) { wrote_ok = false; return false; }
            return true;
        }, &st, wait_for);

    wlap("encode (overlapped decode)");
    finish_decode();
    wlap("join decode/md5");

    if (!enc_ok || !wrote_ok || m_decode_failed.load(std::memory_order_acquire)) {
        std::cerr << "Error: "
                  << (!wrote_ok ? "write failed"
                     : m_decode_failed.load(std::memory_order_acquire)
                       ? "decode failed" : "GPU encode failed")
                  << ".\n";
        out.close();
        std::remove(tmp_output.c_str());
        return false;
    }

    // Accumulated on the MD5 thread, which finish_decode() has now joined.
    auto md5_digest = m_md5.digest();

    // Patch STREAMINFO: fLaC(4) + block header(4) = byte 8, then the 34-byte
    // payload only.
    out.seekp(8, std::ios::beg);
    auto si_updated = FrameWriter::make_streaminfo_block(
        si_is_last, st.min_block, st.max_block, st.min_frame, st.max_frame,
        m_sample_rate, m_channels, m_bps, m_total_samples, md5_digest.data());
    out.write(reinterpret_cast<const char*>(si_updated.data() + 4), 34);
    out.flush();
    if (!out) {
        std::cerr << "Error: write failed.\n";
        out.close();
        std::remove(tmp_output.c_str());
        return false;
    }
    out.close();

    wlap("streaminfo patch + close");
    if (std::rename(tmp_output.c_str(), m_output.c_str()) != 0) {
        std::cerr << "Error: could not finalize output file: " << m_output << "\n";
        std::remove(tmp_output.c_str());
        return false;
    }

    if (m_config.verbose) {
        // This is now decode-and-encode overlapped, so it is wall clock for the
        // pipeline rather than device time.
        const double secs = st.device_secs;
        std::cout << "Wrote " << st.bytes << " bytes of audio data ("
                  << st.frames << " frames, min=" << st.min_block
                  << " max=" << st.max_block << " samples/frame) in "
                  << secs << " s";
        if (secs > 0.0)
            std::cout << " (" << (double)m_total_samples / secs / 1e6
                      << " Msample/s)";
        std::cout << ".\n";
    }
    return true;
}

// ============================================================
// libFLAC decoder callbacks
// ============================================================

FLAC__StreamDecoderWriteStatus Processor::write_callback(
    const FLAC__StreamDecoder* decoder, const FLAC__Frame* frame,
    const FLAC__int32* const buffer[], void* client_data)
{
    auto* self = static_cast<Processor*>(client_data);
    uint32_t nch   = frame->header.channels;
    uint32_t bsize = frame->header.blocksize;

    if (self->m_stream_mode) {
        // The frame carries its own position, so a thread that seeked into the
        // middle of the stream writes to the right place and ranges may complete
        // in any order.
        uint64_t at;
        if (frame->header.number_type == FLAC__FRAME_NUMBER_TYPE_SAMPLE_NUMBER)
            at = frame->header.number.sample_number;
        else
            at = (uint64_t)frame->header.number.frame_number * bsize;

        // Clip to this worker's range. A seek lands on the frame *containing* the
        // target sample, so the first frame usually starts before it -- those
        // samples belong to the previous range and are written by its owner.
        // Clipping keeps exactly one writer per sample rather than relying on two
        // writers agreeing.
        const uint64_t lo = std::max(at, t_range_lo);
        const uint64_t hi = std::min(at + bsize,
                                     std::min(t_range_hi, self->m_total_samples));
        t_frame_end = at + bsize;
        if (hi > lo)
            for (uint32_t c = 0; c < nch; ++c)
                std::memcpy(self->m_pcm_data[c].data() + lo,
                            buffer[c] + (lo - at),
                            (size_t)(hi - lo) * sizeof(int32_t));

        // Reuse needs every input frame's byte range. Recorded per worker and
        // merged after the join -- unlike the sample data, this cannot be placed
        // by position, because the byte offsets only chain within one decoder.
        if (t_skip_rec) {
            // Delivered by the seek, not by this worker's decode loop. Its
            // byte_start would be a stale t_prev_end, and at a mid-frame landing
            // its header is a synthetic partial (libFLAC shifts off the samples
            // before the target and shortens blocksize to match), which is not a
            // frame that exists in the file at all. Either way the previous
            // worker records this frame properly.
            t_skip_rec = false;
        } else if (self->m_map_frames && t_pos_ok) {
            uint64_t end = 0;
            if (FLAC__stream_decoder_get_decode_position(decoder, &end)) {
                t_frames.push_back(Processor::PendingFrameRec{at, bsize, t_prev_end, end});
                t_prev_end = end;
            } else {
                t_pos_ok = false;
                t_frames.clear();
            }
        }

        return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
    }

    if (self->m_pcm_data.empty())
        self->m_pcm_data.resize(nch);

    // Frame-reuse bookkeeping: the decode position after a frame is that
    // frame's end; its start is the previous frame's end. Sample position
    // comes from the running decode count, so it is right for both fixed-
    // and variable-blocksize inputs.
    if (self->m_config.reuse_frames && self->m_frame_pos_ok) {
        uint64_t end = 0;
        if (FLAC__stream_decoder_get_decode_position(decoder, &end)) {
            self->m_input_frames.push_back(InputFrame{
                (uint64_t)self->m_pcm_data[0].size(), bsize,
                self->m_prev_frame_end, end });
            self->m_prev_frame_end = end;
        } else {
            self->m_frame_pos_ok = false;
            self->m_input_frames.clear();
        }
    }

    for (uint32_t c = 0; c < nch; ++c)
        self->m_pcm_data[c].insert(
            self->m_pcm_data[c].end(), buffer[c], buffer[c] + bsize);

    return FLAC__STREAM_DECODER_WRITE_STATUS_CONTINUE;
}

void Processor::error_callback(
    const FLAC__StreamDecoder*, FLAC__StreamDecoderErrorStatus status, void*)
{
    std::cerr << "Decoder error: " << FLAC__StreamDecoderErrorStatusString[status] << "\n";
}

void Processor::metadata_callback(
    const FLAC__StreamDecoder*, const FLAC__StreamMetadata* metadata, void* client_data)
{
    auto* self = static_cast<Processor*>(client_data);
    if (metadata->type == FLAC__METADATA_TYPE_STREAMINFO) {
        self->m_sample_rate   = metadata->data.stream_info.sample_rate;
        self->m_channels      = metadata->data.stream_info.channels;
        self->m_bps           = metadata->data.stream_info.bits_per_sample;
        self->m_total_samples = metadata->data.stream_info.total_samples;
        self->m_max_blocksize = metadata->data.stream_info.max_blocksize;
    }
}

} // namespace flacoutcpp
