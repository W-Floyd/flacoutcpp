/**
 * @file processor.hpp
 * @brief Top-level pipeline coordinator for decoding, optimizing, and encoding.
 */

#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>
#include "md5.hpp"
#include "FLAC/stream_decoder.h"
#include "optimizer.hpp"

namespace flacoutcpp {

/**
 * @brief Configuration parameters for the Processor pipeline.
 */
struct ProcessorConfig {
    /**
     * @brief Whether to copy non-audio metadata blocks from the input file.
     * 
     * When true, VORBIS_COMMENT, PICTURE, and other metadata blocks are copied
     * directly into the optimized output FLAC.
     */
    bool copy_metadata = true;

    /**
     * @brief Apodization windows to test during the exhaustive LPC search.
     * 
     * If empty, all 26 standard windows are evaluated to maximize compression
     * (experimental windows must be named explicitly).
     */
    std::vector<WindowType> windows;

    /// DP block-size ladder (`-b`); empty uses the built-in default.
    /// See Config::dp_candidates for the multiple-of-16 constraint.
    std::vector<uint32_t> dp_candidates;

    /**
     * @brief Maximum number of threads to spawn for dynamic programming evaluation.
     * 
     * A value of 0 indicates that the hardware concurrency limit should be used.
     */
    unsigned max_threads = 0;

    /**
     * @brief Exact-search mode: fully encode every block-partitioning choice
     * (and every stereo mode) instead of pricing them by estimate. Orthogonal
     * to `max_candidates`.
     */
    bool exhaustive = false;

    /**
     * @brief Ranked-search budget: (window, order) pairs fully evaluated per
     * subframe, ranked by Levinson-Durbin prediction error. 0 = no limit
     * (exhaustive sweep). Changing it changes the output — it trades
     * compression for speed.
     */
    unsigned max_candidates = 8;

    /**
     * @brief Consecutive non-improving candidates before the ranked scan
     * stops, making `max_candidates` a floor rather than a ceiling.
     * -1 = 2 x `max_candidates`; 0 = plain top-N cut.
     * See flacoutcpp::Config::patience.
     */
    int patience = -1;

    /**
     * @brief LPC precision rungs encoded per candidate; 0 = all 8.
     * See flacoutcpp::Config::precision_rungs.
     */
    unsigned precision_rungs = 0;

    /**
     * @brief Coefficient-lattice refinement sweeps; 0 = off.
     * See flacoutcpp::Config::lattice_sweeps.
     */
    unsigned lattice_sweeps = 0;

    /**
     * @brief Evaluate LPC candidates on the GPU; see flacoutcpp::Config::use_gpu.
     */
    bool use_gpu = false;

    /// See flacoutcpp::Config::gpu_min_batch.
    unsigned gpu_min_batch = 0;

    /// See flacoutcpp::Config::gpu_partition_cap.
    unsigned gpu_partition_cap = 8;

    /// See flacoutcpp::Config::gpu_slots.
    unsigned gpu_slots = 3;

    /// See flacoutcpp::Config::gpu_duty.
    unsigned gpu_duty = 100;

    /// Fully GPU-resident encoder. See flacoutcpp::Config::pure_gpu — and note
    /// it makes every other search field here irrelevant.
    bool pure_gpu = false;

    /// See flacoutcpp::Config::pg_block_size.
    uint32_t pg_block_size = 4096;

    /// See flacoutcpp::Config::pg_precisions.
    std::vector<int> pg_precisions;

    /// See flacoutcpp::Config::pg_orders.
    uint32_t pg_orders = 3;

    /// See flacoutcpp::Config::pg_partition_cap.
    uint32_t pg_partition_cap = 4;

    /// See flacoutcpp::Config::pg_blocks_per_chunk.
    uint32_t pg_blocks_per_chunk = 256;

    /**
     * @brief Adaptive per-subframe window selection (experimental,
     * estimated-DP modes only). See flacoutcpp::Config::adaptive_windows.
     */
    bool adaptive_windows = false;

    /**
     * @brief Splice input frames that beat the re-encoded ones, and copy the
     * whole input through if the output would still be larger. On by
     * default; see flacoutcpp::Config::reuse_frames.
     */
    bool reuse_frames = true;

    /**
     * @brief Warn on stderr when input frames beat the re-encode. See
     * flacoutcpp::Config::warn_superior.
     */
    bool warn_superior = false;

    /**
     * @brief If false, suppresses progress/statistics stdout output.
     * Errors always go to stderr regardless of this setting.
     */
    bool verbose = true;
};

/**
 * @brief Main engine that coordinates FLAC decoding, DP optimization, and encoding.
 * 
 * The Processor wraps libFLAC to decode the entire input audio file into memory.
 * It then invokes the Optimizer to find the best variable block-size partition
 * and precise encoding parameters. Finally, it uses FrameWriter to serialize the
 * optimized FLAC bitstream and compute the MD5 checksum.
 */
class Processor {
public:
    /**
     * @brief Construct a new Processor object.
     * 
     * @param input_file Path to the input FLAC file.
     * @param output_file Path to write the optimized output FLAC file.
     * @param config Pipeline configuration options.
     */
    Processor(const std::string& input_file,
              const std::string& output_file,
              ProcessorConfig    config = {});

    /**
     * @brief Destroy the Processor object.
     */
    ~Processor();

    /**
     * @brief Execute the end-to-end optimization pipeline.
     * 
     * Decoding, DP optimization, serialization, and MD5 computation happen here.
     * 
     * @return true if the file was successfully optimized and written.
     * @return false if an error occurred during decoding, optimization, or writing.
     */
    bool process();

private:
    /// @cond INTERNAL

    // --- libFLAC decoder callbacks (PCM extraction) ----
    static FLAC__StreamDecoderWriteStatus write_callback(
        const FLAC__StreamDecoder* decoder, const FLAC__Frame* frame,
        const FLAC__int32* const buffer[], void* client_data);
    static void error_callback(
        const FLAC__StreamDecoder* decoder, FLAC__StreamDecoderErrorStatus status, void* client_data);
    static void metadata_callback(
        const FLAC__StreamDecoder* decoder, const FLAC__StreamMetadata* metadata, void* client_data);

    // Raw byte copy of non-STREAMINFO metadata blocks from input file.
    bool read_extra_metadata_blocks(std::vector<std::vector<uint8_t>>& out_blocks) const;

    /// The `-P` pipeline: everything from decoded PCM to the finished file, with
    /// the encode done entirely on the device. Kept separate from process()
    /// rather than branching inside it, because it shares only the decode and
    /// the file plumbing and none of the search, reuse or DP machinery — a
    /// branch through all of that would put the CPU path at risk for nothing.
    /// Called after decode; assumes m_pcm_data and the stream fields are set.
    bool process_pure_gpu(std::vector<std::vector<uint8_t>>& extra_blocks);

    // --- Member state ----
    std::string     m_input;
    std::string     m_output;
    ProcessorConfig m_config;

    // Input frame map for frame reuse: sample span and byte range of every
    // frame in the input file, recorded during decode (reuse_frames only).
public:
    /// What a worker records per frame; merged into InputFrame after the join.
    /// Public only because the decode workers keep it in a file-scope
    /// thread_local -- one per worker, since byte offsets chain within a single
    /// decoder and cannot be placed by sample position the way the PCM can.
    struct PendingFrameRec {
        uint64_t first_sample;
        uint32_t block_size;
        uint64_t byte_start;
        uint64_t byte_end;
    };

private:
    struct InputFrame {
        uint64_t first_sample;
        uint32_t block_size;
        uint64_t byte_start;
        uint64_t byte_end;
    };
    std::vector<InputFrame> m_input_frames;
    uint64_t m_prev_frame_end = 0;   // rolling byte offset during decode
    /// STREAMINFO's max blocksize, used to align parallel-decode ranges onto
    /// frame boundaries so a worker's first frame is usually its own.
    uint32_t m_max_blocksize = 0;
    bool     m_frame_pos_ok   = true; // false if the decoder can't report positions

    // --- streaming decode, used by -P only ----
    //
    // The CPU path decodes the whole file, then encodes. -P cannot afford that:
    // libFLAC's single-threaded decode was 0.109 s of the master mix's 0.239 s
    // against 0.106 s of device time, so the two are near-equal and serialising
    // them wastes half the wall clock.
    //
    // In streaming mode write_callback writes into a *preallocated* m_pcm_data at
    // m_decoded and then publishes the new count with release ordering, so the
    // encoder may read anything below what it has acquired. That is the whole
    // synchronisation: no lock on the sample data itself.
    bool                  m_stream_mode = false;
    /// Highest sample count that is complete *contiguously* from zero. With a
    /// parallel decode the ranges finish out of order, so this advances only as a
    /// prefix of them completes -- which is what lets readers keep the simple
    /// "anything below this is readable" rule.
    std::atomic<uint64_t> m_decoded{0};
    /// One entry per decode range; see advance_decoded().
    std::vector<uint8_t>  m_range_done;
    uint64_t              m_range_len = 0;
    /// Mark range `idx` complete and push m_decoded as far as the contiguous
    /// prefix now reaches. Takes m_decode_mu.
    void advance_decoded(size_t idx);

    /// Decode the whole stream with `nthr` workers into preallocated m_pcm_data,
    /// each seeking to its own block-aligned range. Used by both -P (which also
    /// consumes m_decoded as it fills) and the CPU path (which just waits).
    ///
    /// `first` is an already-initialised decoder positioned past the metadata;
    /// worker 0 reuses it, since it starts at sample 0 and needs no seek. The
    /// caller owns and deletes it.
    ///
    /// Collects the frame byte-range map when reuse is enabled, merging the
    /// per-worker records afterwards; sets m_frame_pos_ok false if any worker
    /// could not report positions or the merged map does not tile the stream.
    bool decode_parallel(FLAC__StreamDecoder* first, unsigned nthr,
                         uint64_t range_len);

    /// Threads per parallel decode: half the cores, capped, overridable with
    /// FLACOUT_DECODE_THREADS. 1 disables the parallel path entirely.
    unsigned decode_thread_count(uint64_t total_samples, uint64_t align) const;

    /// Merge and validate what the workers recorded; see decode_parallel().
    void merge_input_frames(std::vector<std::vector<PendingFrameRec>>& per_worker);
    std::atomic<bool>     m_decode_done{false};
    std::atomic<bool>     m_decode_failed{false};
    std::mutex              m_decode_mu;
    std::condition_variable m_decode_cv;
    /// MD5 runs on the decode thread, folded into the same pass. It is the one
    /// stage that cannot be parallelised (no combine operator, unlike CRC), and
    /// doing it here means it costs nothing: the samples are already hot, and the
    /// thread would otherwise be waiting on I/O.
    detail::MD5 m_md5;

    // Decoded PCM (per-channel, arrays of channel samples)
    std::vector<std::vector<int32_t>> m_pcm_data;
    uint32_t m_sample_rate   = 0;
    uint32_t m_channels      = 0;
    uint32_t m_bps           = 0;
    uint64_t m_total_samples = 0;

    /// @endcond
};

} // namespace flacoutcpp

#endif // PROCESSOR_HPP
