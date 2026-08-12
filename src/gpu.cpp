#include "gpu.hpp"

#ifndef FLACOUT_HAVE_VULKAN

// Built without Vulkan. The stub keeps every call site compiling unchanged and
// makes `-G` a clear diagnostic instead of a link error.
namespace flacoutcpp {
struct GpuEvaluator::Impl { std::string why = "built without Vulkan support "
    "(configure with -DFLACOUT_VULKAN=ON)"; };
GpuEvaluator::GpuEvaluator() : m_impl(new Impl) {}
GpuEvaluator::~GpuEvaluator() = default;
bool GpuEvaluator::available() const { return false; }
const std::string& GpuEvaluator::why() const { return m_impl->why; }
bool GpuEvaluator::evaluate(const int32_t*, uint32_t,
                            const std::vector<Candidate>&,
                            std::vector<uint32_t>&) { return false; }
void GpuEvaluator::stats(uint64_t* c, double* s) const { if(c)*c=0; if(s)*s=0.0; }
void GpuEvaluator::set_min_batch(size_t) {}
size_t GpuEvaluator::min_batch() const { return 0; }
void GpuEvaluator::set_partition_cap(int) {}
int GpuEvaluator::partition_cap() const { return 8; }
void GpuEvaluator::set_slots(int) {}
int GpuEvaluator::slots() const { return 0; }
void GpuEvaluator::set_duty(int) {}
int GpuEvaluator::duty() const { return 100; }
void GpuEvaluator::note_cpu(uint64_t, double) {}
bool GpuEvaluator::gave_up() const { return false; }
void GpuEvaluator::throttle_stats(double* g, double* c, double* a) const {
    if (g) *g = 0.0; if (c) *c = 0.0; if (a) *a = 0.0;
}
bool GpuEvaluator::would_accept() const { return false; }
uint64_t GpuEvaluator::macs() const { return 0; }
} // namespace flacoutcpp

#else

#include <vulkan/vulkan.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "sweep_spv.h"            // generated at build time from shaders/sweep.comp
// Fallback variants of the same source, one per missing-feature combination.
#include "sweep_noarith_spv.h"     // -DSWEEP_NO_ARITH
#include "sweep_noint64_spv.h"     // -DSWEEP_NO_INT64
#include "sweep_compat_spv.h"      // both
#include "sweep_slm_spv.h"         // -DSWEEP_SLM_STATE
#include "sweep_compat_slm_spv.h"  // all three

namespace flacoutcpp {

namespace {

// Growable device-local, host-visible buffer. Apple silicon and every
// integrated part expose one such heap, and on discrete parts this still works
// (BAR-visible memory) at the cost of PCIe reads — which is acceptable because
// the transfers here are tiny next to the compute.
struct Buffer {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    void*          map  = nullptr;
    VkDeviceSize   size = 0;
};

struct PushConsts {
    int32_t ncand;
    int32_t bsize;
    int32_t flags;
    int32_t maxPOrder;
};

} // namespace

struct GpuEvaluator::Impl {
    std::string why;
    // Atomic because it is no longer write-once: a lost device clears it from a
    // worker thread while other workers are reading it in would_accept().
    std::atomic<bool> ok{false};

    VkInstance       inst   = VK_NULL_HANDLE;
    VkPhysicalDevice phys   = VK_NULL_HANDLE;
    VkDevice         dev    = VK_NULL_HANDLE;
    VkQueue          queue  = VK_NULL_HANDLE;
    uint32_t         qfam   = 0;
    VkPhysicalDeviceMemoryProperties memprops{};

    VkDescriptorSetLayout dsl   = VK_NULL_HANDLE;
    VkDescriptorPool      pool  = VK_NULL_HANDLE;
    VkPipelineLayout      plo   = VK_NULL_HANDLE;
    VkPipeline            pipe  = VK_NULL_HANDLE;
    VkShaderModule        shader= VK_NULL_HANDLE;
    // Independent slots, so several workers can have dispatches in flight.
    // With a single command buffer the whole submit-and-wait sat inside one
    // lock, which capped the device at one dispatch at a time: measured, the
    // GPU absorbed only ~31% of the search while 15 of 16 threads took the CPU
    // path. Only vkQueueSubmit needs serialising (the queue is externally
    // synchronised); waiting is per-slot and lock-free.
    static constexpr int NSLOT = 16;   // allocated; `nslot` is how many are used
    struct Slot {
        // One pool per slot. A VkCommandPool is externally synchronised, so
        // two threads may not record into buffers from the same pool at once;
        // sharing one pool across slots is a data race that happens to survive
        // at low slot counts and produced differing output at 16.
        VkCommandPool   pool  = VK_NULL_HANDLE;
        VkCommandBuffer cmd   = VK_NULL_HANDLE;
        VkFence         fence = VK_NULL_HANDLE;
        VkDescriptorSet dset  = VK_NULL_HANDLE;
        Buffer bSamples, bCands, bCosts;
        std::atomic<bool> busy{false};
    };
    Slot slots[NSLOT];
    std::atomic<int> nslot{3};

    std::mutex submit_mu;                // vkQueueSubmit only
    std::atomic<uint64_t> n_cands{0};
    std::atomic<uint64_t> n_macs{0};   // (bsize-ord)*ord, comparable to the CPU's
    // Per-call elapsed cannot be summed once dispatches overlap -- with six
    // slots in flight that counts the same wall time up to six times. Track
    // the span from the first dispatch's start to the last one's end instead,
    // which is the window the device was actually working in.
    std::atomic<uint64_t> t_first{UINT64_MAX};
    std::atomic<uint64_t> t_last{0};
    std::chrono::steady_clock::time_point t_origin = std::chrono::steady_clock::now();
    std::atomic<size_t>   min_batch{0};
    std::atomic<int>      pcap{8};
    // Batches the device declined to price (see the sentinel check in
    // evaluate). Warn once, then keep counting: a device that does this for
    // every batch has silently become a CPU run, which the user should know.
    std::atomic<uint64_t> n_invalid{0};
    std::atomic<bool>     warned_invalid{false};
    // VK_EXT_subgroup_size_control present, usable, and able to pin 32.
    bool size_ctl = false;
    // Which fallbacks the kernel is running with, because the device lacks the
    // feature or because FLACOUT_GPU_COMPAT forced it (see shaders/sweep.comp).
    // Same costs either way, more instructions; the point is that constrained
    // parts get a GPU path at all.
    bool emu_arith = false;
    bool emu_int64 = false;
    // Fold state in shared memory instead of registers, for parts whose register
    // file cannot hold it (see shaders/sweep.comp).
    bool slm_state = false;
    // Share throttle. Work is claimed greedily -- a subframe goes to the GPU
    // whenever a slot is free -- which over-commits a device slower than the
    // host: the CPU finishes its share early and idles at the DP's barrier
    // while the GPU is still working. Accepting only `duty` percent of offers
    // hands the surplus back. Deterministic (a counter, not a coin) so a run
    // stays reproducible; the output is invariant to the split either way.
    std::atomic<int>      duty{0};   // 0 = adaptive, see the note below
    std::atomic<uint64_t> offers{0};
    std::atomic<uint64_t> taken{0};

    // ---- adaptive throttle -------------------------------------------------
    //
    // A fixed duty (or a fixed slot count) cannot be right, and that is
    // measured, twice: on a Haswell iGPU three slots cost 3.7x against one,
    // while on a Tesla P4 three slots are the optimum and one loses 10%; and on
    // a single UHD 630 the best slot count *flips with bit depth*, because 24-bit
    // candidates are heavier and park a worker for longer. What decides it is
    // whether the device prices work faster than the one CPU thread that has to
    // block waiting for it -- so measure exactly that and compare.
    //
    // Both rates are in MACs/second, which is the unit that makes batches of
    // different block size and order comparable; the device's rate already
    // includes its own dispatch overhead amortised over the batch, so a device
    // with a long submit path prices small batches badly and is declined for
    // them without needing a separate min_batch heuristic.
    //
    // EMAs rather than totals, so a device that slows down (thermal, contention,
    // a fatter mix of candidates) is noticed. Stored as bit-cast doubles in
    // atomics: every worker updates them, and a torn read here would cost a
    // slightly wrong routing decision, never a wrong output -- both paths return
    // the same cost, which is the contract that makes this safe to tune at all.
    std::atomic<uint64_t> gpu_rate{0};   // MACs/s, EMA, 0 = unmeasured
    std::atomic<uint64_t> cpu_rate{0};   // MACs/s, EMA, one thread
    std::atomic<uint32_t> gpu_samples{0};
    std::atomic<uint32_t> cpu_samples{0};
    // Warm-up, until both sides have this many samples. It has to *alternate*
    // rather than accept everything: the CPU rate can only be measured on a
    // batch the CPU actually ran, so a warm-up that accepts unconditionally
    // starves itself of half its own input and never leaves warm-up. Measured
    // before the alternation was added -- a Haswell iGPU 20x slower than one
    // thread reported the ratio correctly (0.05x) and still accepted 100% of
    // offers, because `cpu_samples` sat at zero for the whole encode.
    // Three samples per side, and only one offer in WARMUP_TAKE goes to the
    // device while learning. The asymmetry is deliberate: a wrong accept on a
    // device 20x slower than a thread costs that batch's CPU time over again
    // many times, a wrong decline costs nothing at all, and the gaps this has to
    // resolve are factors of 3-20, not percentages. It also matters that files
    // are short in these units -- a 3-second fixture offers only a few dozen
    // batches, so a warm-up of 8 accepted samples *was* the whole encode.
    static constexpr uint32_t WARMUP = 3;
    static constexpr uint64_t WARMUP_TAKE = 8;
    // Re-probe a declined device occasionally, so one that was slow at the start
    // of a file (cold clocks, a small-batch phase) can win the work back --
    // backing off geometrically while it keeps losing, because on a device that
    // is 10x slower every probe costs ten batches' worth of CPU time and the
    // answer has not changed in a thousand offers. Measured need for this: at a
    // flat 1-in-64 a UHD 630 still leaked 9% of batches and gave up 18% of wall
    // clock, where the ratio it was rejected on was 0.30x.
    static constexpr uint64_t REPROBE_MIN = 64;
    static constexpr uint64_t REPROBE_MAX = 4096;
    std::atomic<uint64_t> reprobe{REPROBE_MIN};
    // And the symmetric probe: while the device is winning every offer the CPU
    // rate stops being measured and goes stale, which matters because it is the
    // rate that moves -- thread count, contention, block size mix. One batch in
    // this many goes to the CPU purely to refresh it.
    static constexpr uint64_t CPU_PROBE     = 64;
    static constexpr uint64_t CPU_PROBE_MAX = 4096;
    // Accept only if the device beats one CPU thread by this margin. It is 1.5
    // rather than ~1.0 because a saturated device also slows the rest of the
    // pool down, and that cost appears in neither rate. Measured by pinning duty
    // to 1 (device idle) and to 100 (device saturated) and reading the
    // single-thread CPU rate both ways, music_10s:
    //
    //   Haswell HD 4600 (iGPU, 8 threads)   1.70-1.75e9 -> 1.81-1.86e9  (none)
    //   Tesla P4 (discrete, 12 threads)     2.18-2.33e9 -> 1.43-2.02e9  (-12..-38%)
    //
    // So it is the *discrete* card that costs the pool -- host-visible buffer
    // traffic competing for memory bandwidth -- not the iGPU, which is the
    // opposite of the obvious guess. It changes nothing at the extremes (an
    // iGPU at 0.1x is declined either way, the P4 at 11x accepted either way);
    // the margin exists for devices that land near parity, where a 1.2x device
    // that costs the pool 20% is not worth taking.
    static constexpr double MARGIN = 1.50;
    // Below this ratio the device is not a slower helper, it is a liability, and
    // the path shuts down (see the give-up branch). Parity, not something safely
    // below it, because the ratio is measured *while the device is running* and
    // interference biases it upward: on a UHD 630 the same device reads 0.30x
    // against an idle-pool CPU rate and 0.50x against the rate it depresses
    // itself, so a threshold under parity lets a device keep itself alive by
    // slowing down its competition. Nothing measured lands between parity and
    // 9x (M4 Max, Tesla P4), so this costs no real device its slot.
    static constexpr double   GIVEUP = 1.00;
    // Enough evidence, but reachable on a short file: a 2-second fixture offers
    // only a few dozen batches, and at 64 the give-up never fired on one at all.
    static constexpr uint64_t GIVEUP_AFTER = 16;
    std::atomic<bool> gave_up{false};

    static double rd(const std::atomic<uint64_t>& a) {
        const uint64_t b = a.load(std::memory_order_relaxed);
        double d = 0.0;
        std::memcpy(&d, &b, sizeof d);
        return d;
    }
    static void ema(std::atomic<uint64_t>& a, std::atomic<uint32_t>& n, double x) {
        if (!(x > 0.0) || !std::isfinite(x)) return;
        const double prev = rd(a);
        const double next = prev > 0.0 ? prev * 0.75 + x * 0.25 : x;
        uint64_t bits = 0;
        std::memcpy(&bits, &next, sizeof bits);
        a.store(bits, std::memory_order_relaxed);
        n.fetch_add(1, std::memory_order_relaxed);
    }
    /// Should the next batch go to the device? See the note above.
    ///
    /// Called from would_accept(), i.e. *before* the caller builds the batch,
    /// and not from evaluate(). That placement is not cosmetic: building a batch
    /// costs a quantize pass over every (candidate, precision) pair, and a
    /// throttle that declined afterwards threw all of it away and made the CPU
    /// redo it. Measured with the decision inside evaluate(): a UHD 630 declined
    /// 93% of offers and still gave up 18% of wall clock, because the encoder was
    /// paying for 93% of the batches twice. The same reasoning is already
    /// written down for min_batch in optimizer.cpp.
    bool throttle_ok() {
        const int dty = duty.load(std::memory_order_relaxed);
        if (dty > 0) {   // pinned share: the old deterministic counter
            const uint64_t n = offers.fetch_add(1, std::memory_order_relaxed);
            if (dty >= 100) return true;
            return (int)((n * 100) % 10000 / 100) < dty;
        }
        const uint64_t n = offers.fetch_add(1, std::memory_order_relaxed);
        const bool need_cpu = cpu_samples.load(std::memory_order_relaxed) < WARMUP;
        const bool need_gpu = gpu_samples.load(std::memory_order_relaxed) < WARMUP;
        if (need_cpu || need_gpu)
            return (n % WARMUP_TAKE) == 0;   // learn from both, mostly on the CPU

        const double g = rd(gpu_rate), c = rd(cpu_rate);
        if (!(g > 0.0) || !(c > 0.0))
            return (n % CPU_PROBE) != 0;   // rates unknown: probe at the base rate

        // Keep the CPU rate fresh, but pay for it in proportion to what it costs.
        // A probe hands one subframe to a CPU that may be far slower per MAC, so
        // on a device winning by 68x a flat 1-in-64 probe spends ~1.6% of the
        // whole run re-measuring something that has not changed -- which is most
        // of what `-G` gave back against a pinned duty in exhaustive mode on a
        // Tesla P4 (3.011x against 3.107x). Scale the interval with the ratio and
        // that cost stays near 1.6% of one probe instead.
        const double ratio = g / c;
        uint64_t cpu_iv = (uint64_t)(CPU_PROBE * (ratio > 1.0 ? ratio : 1.0));
        if (cpu_iv > CPU_PROBE_MAX) cpu_iv = CPU_PROBE_MAX;
        if ((n % cpu_iv) == 0) return false;
        if (g > c * MARGIN) {
            reprobe.store(REPROBE_MIN, std::memory_order_relaxed);
            return true;
        }
        // Decisively losing, with enough evidence: stop using the device at all,
        // rather than trickling probes at it forever. Probing is not free even at
        // a 1-in-4096 rate -- measured on a UHD 630, pinning duty to 1 (so the
        // device is essentially idle) still cost 10% of wall clock against
        // CPU-only, because an awake iGPU takes package power and memory
        // bandwidth from the cores whatever it is doing. A device this far behind
        // is never going to catch up within one file, so the only way to recover
        // that 10% is to leave it alone.
        if (g < c * GIVEUP && n >= GIVEUP_AFTER) {
            if (!gave_up.exchange(true, std::memory_order_relaxed)) {
                ok.store(false, std::memory_order_relaxed);
                std::fprintf(stderr,
                    "GPU: device prices work at %.2fx one CPU thread; switching "
                    "the rest of this encode to the CPU.\n", g / c);
            }
            return false;
        }
        // Losing, but not hopelessly. Take one offer per `reprobe`, then double
        // the interval up to the cap; a probe that wins resets it above.
        const uint64_t iv = reprobe.load(std::memory_order_relaxed);
        if ((n % iv) != 0) return false;
        if (iv < REPROBE_MAX)
            reprobe.store(iv * 2, std::memory_order_relaxed);
        return true;
    }

    /// A device that has gone away takes the whole path down with it, and the
    /// encode finishes on the CPU. Any error is treated this way, not just
    /// DEVICE_LOST: once a submit or a wait has failed there is no reason to
    /// believe the next one, and the fallback is always correct.
    bool fail_device(VkResult r, const char* where) {
        if (!ok.exchange(false, std::memory_order_relaxed)) return false;
        std::fprintf(stderr,
            "GPU: device lost at %s (VkResult %d) -- disabling the GPU path; "
            "the rest of this encode runs on the CPU.\n", where, (int)r);
        return false;
    }

    static constexpr uint32_t WG = 128;  // 4 candidates per work group

    bool init();
    void destroy();
    bool grow(Buffer& b, VkDeviceSize need);
    uint32_t findMem(uint32_t bits, VkMemoryPropertyFlags want) const;
    void bindDescriptors(Slot& s);
};

uint32_t GpuEvaluator::Impl::findMem(uint32_t bits, VkMemoryPropertyFlags want) const {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (memprops.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

bool GpuEvaluator::Impl::grow(Buffer& b, VkDeviceSize need) {
    if (b.size >= need) return true;
    if (b.map) { vkUnmapMemory(dev, b.mem); b.map = nullptr; }
    if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
    b.buf = VK_NULL_HANDLE; b.mem = VK_NULL_HANDLE; b.size = 0;

    // Round up so a slowly growing workload does not reallocate every call.
    VkDeviceSize sz = 1 << 16;
    while (sz < need) sz <<= 1;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = sz;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bi, nullptr, &b.buf) != VK_SUCCESS) return false;

    VkMemoryRequirements mr{};
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    const uint32_t idx = findMem(mr.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (idx == UINT32_MAX) return false;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = idx;
    if (vkAllocateMemory(dev, &ai, nullptr, &b.mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(dev, b.buf, b.mem, 0) != VK_SUCCESS) return false;
    if (vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map) != VK_SUCCESS) return false;
    b.size = sz;
    return true;
}

void GpuEvaluator::Impl::bindDescriptors(Slot& s) {
    VkDescriptorBufferInfo dbi[3]{};
    dbi[0].buffer = s.bSamples.buf; dbi[0].range = VK_WHOLE_SIZE;
    dbi[1].buffer = s.bCands.buf;   dbi[1].range = VK_WHOLE_SIZE;
    dbi[2].buffer = s.bCosts.buf;   dbi[2].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wr[3]{};
    for (int i = 0; i < 3; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = s.dset;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 3, wr, 0, nullptr);
}

bool GpuEvaluator::Impl::init() {
    // ---- instance --------------------------------------------------------
    // Portability enumeration exists only on layered implementations
    // (MoltenVK); enabling it unconditionally fails a native loader.
    uint32_t nie = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, nullptr);
    std::vector<VkExtensionProperties> ie(nie);
    vkEnumerateInstanceExtensionProperties(nullptr, &nie, ie.data());
    std::vector<const char*> iexts;
    VkInstanceCreateFlags iflags = 0;
    for (const auto& e : ie) {
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_enumeration")) {
            iexts.push_back("VK_KHR_portability_enumeration");
            iflags |= 0x00000001u;  // ENUMERATE_PORTABILITY_BIT_KHR
        } else if (!std::strcmp(e.extensionName,
                                "VK_KHR_get_physical_device_properties2")) {
            iexts.push_back("VK_KHR_get_physical_device_properties2");
        }
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "flacoutcpp";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.flags = iflags;
    ici.enabledExtensionCount = (uint32_t)iexts.size();
    ici.ppEnabledExtensionNames = iexts.data();
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        why = "no Vulkan loader or instance creation failed"; return false;
    }

    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(inst, &nd, nullptr);
    if (!nd) { why = "no Vulkan devices"; return false; }
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(inst, &nd, devs.data());

    // Prefer a discrete part; the search is compute-bound, so a dedicated GPU
    // wins whenever one exists.
    int pick = 0;
    for (uint32_t i = 0; i < nd; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pick = (int)i; break; }
    }
    // FLACOUT_GPU_DEVICE overrides that: an index, or a substring of the device
    // name. Needed to measure anything on a machine with two devices, since
    // "prefer discrete" otherwise makes the integrated one untestable -- which
    // is how the interference question above nearly went unanswered.
    if (const char* sel = std::getenv("FLACOUT_GPU_DEVICE")) {
        const std::string want(sel);
        bool found = false;
        if (!want.empty() && want.find_first_not_of("0123456789") == std::string::npos) {
            const uint32_t idx = (uint32_t)std::strtoul(want.c_str(), nullptr, 10);
            if (idx < nd) { pick = (int)idx; found = true; }
        } else {
            for (uint32_t i = 0; i < nd && !found; ++i) {
                VkPhysicalDeviceProperties p{};
                vkGetPhysicalDeviceProperties(devs[i], &p);
                if (std::string(p.deviceName).find(want) != std::string::npos) {
                    pick = (int)i; found = true;
                }
            }
        }
        if (!found)
            std::fprintf(stderr,
                "GPU: FLACOUT_GPU_DEVICE='%s' matched no device; using the "
                "default choice.\n", sel);
    }
    phys = devs[pick];

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys, &props);
    vkGetPhysicalDeviceMemoryProperties(phys, &memprops);

    // VkPhysicalDeviceSubgroupProperties::subgroupSize is the device's
    // *default*, not a promise about any particular dispatch: on Intel ANV the
    // compiler picks SIMD8/16/32 per shader, so a device advertising 32 can
    // still run this kernel at 16. The kernel notices (gl_SubgroupSize) and
    // bails, which used to poison the search. Checking this property was
    // therefore validating a number that does not govern the dispatch.
    //
    // VK_EXT_subgroup_size_control fixes it properly by *pinning* the width at
    // pipeline creation. Where it is available the property is only used to
    // check that 32 is reachable; where it is not, fall back to trusting the
    // property as before and rely on the kernel's guard.
    uint32_t nde = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, nullptr);
    std::vector<VkExtensionProperties> de(nde);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &nde, de.data());
    for (const auto& e : de)
        if (!std::strcmp(e.extensionName, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
            size_ctl = true;

    VkPhysicalDeviceSubgroupProperties sgp{};
    sgp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT sgc{};
    sgc.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sgp;
    if (size_ctl) sgp.pNext = &sgc;
    vkGetPhysicalDeviceProperties2(phys, &p2);

    if (size_ctl) {
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgf{};
        sgf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &sgf;
        vkGetPhysicalDeviceFeatures2(phys, &f2);
        // Full subgroups matter as much as the size: without it a partially
        // filled subgroup leaves lanes inactive, and this kernel maps one
        // bit-plane per lane. WG (128) is a multiple of 32, so every subgroup
        // is full by construction once the size is pinned.
        if (!sgf.subgroupSizeControl || !sgf.computeFullSubgroups ||
            !(sgc.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
            sgc.minSubgroupSize > 32u || sgc.maxSubgroupSize < 32u)
            size_ctl = false;
    }

    // The kernel assigns one bit-plane per lane, so it needs exactly 32.
    if (!size_ctl && sgp.subgroupSize != 32) {
        why = std::string(props.deviceName) + ": subgroup size is " +
              std::to_string(sgp.subgroupSize) + ", the kernel requires 32" +
              " (and VK_EXT_subgroup_size_control is unavailable to pin it)";
        return false;
    }
    // Ballot and shuffle-relative are structural: the bit-plane fold IS a
    // ballot, and the weighted reverse scan is a shuffle-down. Nothing cheap
    // replaces either, so they stay hard requirements.
    const VkSubgroupFeatureFlags needsg =
        VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
        VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    if ((sgp.supportedOperations & needsg) != needsg) {
        why = std::string(props.deviceName) +
              ": missing required subgroup operations (basic/ballot/shuffle-relative)";
        return false;
    }

    VkPhysicalDeviceFeatures feat{};
    vkGetPhysicalDeviceFeatures(phys, &feat);

    // Two soft requirements, each with a fallback in the shader. Mesa hasvk on
    // Haswell misses both, which is what motivated them -- and the reason the
    // fallbacks are emulation rather than approximation is that the cost this
    // kernel returns must stay exactly the CPU's, or the byte-identical
    // contract (and bench/check.sh with it) breaks.
    const bool arith = (sgp.supportedOperations &
                        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    emu_arith = !arith;
    emu_int64 = !feat.shaderInt64;

    // FLACOUT_GPU_COMPAT forces a fallback on a device that does not need it:
    // `arith`, `int64` or `1`/`both`. That is how the emulations get measured
    // and bit-exactness-checked on hardware that can also run the fast variant
    // -- otherwise the only way to test them is to own a 2013 iGPU.
    if (const char* e = std::getenv("FLACOUT_GPU_COMPAT")) {
        const std::string w(e);
        if (w == "arith" || w == "1" || w == "both") emu_arith = true;
        if (w == "int64" || w == "1" || w == "both") emu_int64 = true;
    }
    // Fold state to shared memory on any device constrained enough to be running
    // an emulation, which is the only signal Vulkan offers for "small register
    // file" -- there is no queryable GRF budget. Measured on Haswell GT2, where
    // the register-resident arrays compile to 342:564 spills:fills and SLM is
    // worth 5.8x; and on an M4 Max, where forcing it is 0.98x, i.e. free.
    // Capable devices keep the register version, so none of this is on their
    // path at all. FLACOUT_GPU_SLM=1/0 forces it either way.
    slm_state = emu_arith || emu_int64;
    if (const char* e = std::getenv("FLACOUT_GPU_SLM"))
        slm_state = (e[0] == '1');
    if (slm_state &&
        props.limits.maxComputeSharedMemorySize < 5u * 9u * 128u * 4u)
        slm_state = false;

    // Auto slot count. One slot on a constrained device: each slot parks a CPU
    // worker on a fence, and a device slower than the host converts that into
    // idle cores -- Haswell GT2 at 3 slots is 0.26x the CPU-only wall clock, at
    // 1 slot 0.94x, same output. --gpu-slots overrides.
    if (emu_arith || emu_int64) nslot.store(1, std::memory_order_relaxed);

    // subgroupMin/Max are emulated with a shuffle-xor butterfly, so plain
    // SHUFFLE is required once ARITHMETIC is missing.
    if (emu_arith && !(sgp.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) {
        why = std::string(props.deviceName) +
              ": no subgroup arithmetic and no shuffle to emulate it with";
        return false;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qp(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qp.data());
    bool found = false;
    // A compute-only family avoids sharing the graphics queue's scheduling.
    for (uint32_t i = 0; i < nq; ++i)
        if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qfam = i; found = true; break; }
    if (!found)
        for (uint32_t i = 0; i < nq; ++i)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; found = true; break; }
    if (!found) { why = "no compute queue"; return false; }

    std::vector<const char*> dexts;
    for (const auto& e : de)
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_subset"))
            dexts.push_back("VK_KHR_portability_subset");
    if (size_ctl) dexts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures want{};
    want.shaderInt64 = feat.shaderInt64;
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT sgf_on{};
    sgf_on.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    sgf_on.subgroupSizeControl  = VK_TRUE;
    sgf_on.computeFullSubgroups = VK_TRUE;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    if (size_ctl) dci.pNext = &sgf_on;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dexts.size();
    dci.ppEnabledExtensionNames = dexts.data();
    dci.pEnabledFeatures = &want;
    if (vkCreateDevice(phys, &dci, nullptr, &dev) != VK_SUCCESS) {
        why = std::string(props.deviceName) + ": device creation failed"; return false;
    }
    vkGetDeviceQueue(dev, qfam, 0, &queue);

    // ---- pipeline --------------------------------------------------------
    VkDescriptorSetLayoutBinding bind[3]{};
    for (int i = 0; i < 3; ++i) {
        bind[i].binding = (uint32_t)i;
        bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].descriptorCount = 1;
        bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = bind;
    if (vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl) != VK_SUCCESS) {
        why = "descriptor set layout"; return false;
    }
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * NSLOT};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = NSLOT;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(dev, &dpci, nullptr, &pool) != VK_SUCCESS) {
        why = "descriptor pool"; return false;
    }
    for (int i = 0; i < NSLOT; ++i) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        if (vkAllocateDescriptorSets(dev, &dsai, &slots[i].dset) != VK_SUCCESS) {
            why = "descriptor set"; return false;
        }
    }

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(dev, &plci, nullptr, &plo) != VK_SUCCESS) {
        why = "pipeline layout"; return false;
    }

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    // Five variants, not 2^3: SLM only exists paired with the full set of
    // emulations or with none, because the parts that need one need the other.
    if (emu_arith && emu_int64 && slm_state) {
        smci.codeSize = sizeof(kSweepCompatSlmSpv); smci.pCode = kSweepCompatSlmSpv;
    } else if (emu_arith && emu_int64) {
        smci.codeSize = sizeof(kSweepCompatSpv);   smci.pCode = kSweepCompatSpv;
    } else if (emu_arith) {
        smci.codeSize = sizeof(kSweepNoArithSpv);  smci.pCode = kSweepNoArithSpv;
    } else if (emu_int64) {
        smci.codeSize = sizeof(kSweepNoInt64Spv);  smci.pCode = kSweepNoInt64Spv;
    } else if (slm_state) {
        smci.codeSize = sizeof(kSweepSlmSpv);      smci.pCode = kSweepSlmSpv;
    } else {
        smci.codeSize = sizeof(kSweepSpv);         smci.pCode = kSweepSpv;
    }
    if (vkCreateShaderModule(dev, &smci, nullptr, &shader) != VK_SUCCESS) {
        why = "shader module"; return false;
    }
    // Pin the dispatch width. Without this the driver is free to compile the
    // kernel at any supported subgroup size and the one-bit-plane-per-lane
    // mapping silently stops holding.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT rss{};
    rss.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    rss.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = shader;
    cpi.stage.pName = "main";
    if (size_ctl) {
        cpi.stage.pNext = &rss;
        cpi.stage.flags =
            VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    }
    cpi.layout = plo;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe) != VK_SUCCESS) {
        why = std::string(props.deviceName) + ": compute pipeline creation failed";
        return false;
    }

    for (int i = 0; i < NSLOT; ++i) {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = qfam;
        if (vkCreateCommandPool(dev, &cpci, nullptr, &slots[i].pool) != VK_SUCCESS) {
            why = "command pool"; return false;
        }
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = slots[i].pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev, &cbai, &slots[i].cmd) != VK_SUCCESS) {
            why = "command buffer"; return false;
        }
        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(dev, &fci, nullptr, &slots[i].fence) != VK_SUCCESS) {
            why = "fence"; return false;
        }
    }

    why = std::string(props.deviceName) + " (subgroup 32 " +
          (size_ctl ? "pinned" : "by default") +
          (emu_arith ? ", emulated subgroup min/max" : "") +
          (emu_int64 ? ", emulated int64" : ", shaderInt64") +
          (slm_state ? ", fold state in SLM" : "") + ")";
    ok.store(true, std::memory_order_relaxed);
    return true;
}

void GpuEvaluator::Impl::destroy() {
    if (dev == VK_NULL_HANDLE) { if (inst) vkDestroyInstance(inst, nullptr); return; }
    vkDeviceWaitIdle(dev);
    auto killbuf = [&](Buffer& b) {
        if (b.map) vkUnmapMemory(dev, b.mem);
        if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
        if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
        b = Buffer{};
    };
    for (int i = 0; i < NSLOT; ++i) {
        killbuf(slots[i].bSamples); killbuf(slots[i].bCands); killbuf(slots[i].bCosts);
        if (slots[i].fence) vkDestroyFence(dev, slots[i].fence, nullptr);
        if (slots[i].pool)  vkDestroyCommandPool(dev, slots[i].pool, nullptr);
    }
    if (pipe)   vkDestroyPipeline(dev, pipe, nullptr);
    if (shader) vkDestroyShaderModule(dev, shader, nullptr);
    if (plo)    vkDestroyPipelineLayout(dev, plo, nullptr);
    if (pool)   vkDestroyDescriptorPool(dev, pool, nullptr);
    if (dsl)    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyDevice(dev, nullptr);
    if (inst)   vkDestroyInstance(inst, nullptr);
}

// ---------------------------------------------------------------- public

GpuEvaluator::GpuEvaluator() : m_impl(new Impl) {
    if (!m_impl->init()) m_impl->ok.store(false, std::memory_order_relaxed);
}

GpuEvaluator::~GpuEvaluator() { m_impl->destroy(); }

bool GpuEvaluator::available() const {
    return m_impl->ok.load(std::memory_order_relaxed);
}
void GpuEvaluator::set_min_batch(size_t n) {
    m_impl->min_batch.store(n, std::memory_order_relaxed);
}
size_t GpuEvaluator::min_batch() const {
    return m_impl->min_batch.load(std::memory_order_relaxed);
}
void GpuEvaluator::set_slots(int n) {
    // 0 = auto: leave whatever init() derived from the device.
    if (n == 0) return;
    m_impl->nslot.store(n < 1 ? 1 : (n > Impl::NSLOT ? Impl::NSLOT : n),
                        std::memory_order_relaxed);
}
int GpuEvaluator::slots() const {
    return m_impl->nslot.load(std::memory_order_relaxed);
}
void GpuEvaluator::set_duty(int pct) {
    // 0 stays 0: it selects the adaptive throttle rather than a pinned share.
    m_impl->duty.store(pct <= 0 ? 0 : (pct > 100 ? 100 : pct),
                       std::memory_order_relaxed);
}
int GpuEvaluator::duty() const {
    return m_impl->duty.load(std::memory_order_relaxed);
}
bool GpuEvaluator::gave_up() const {
    return m_impl->gave_up.load(std::memory_order_relaxed);
}
void GpuEvaluator::note_cpu(uint64_t macs, double seconds) {
    Impl& I = *m_impl;
    // Recorded even when the share is pinned, and only *consulted* when it is
    // not. Measuring under a pinned duty is how the throttle itself gets A/B'd:
    // with duty 100 the device is saturated and with duty 1 it is idle, so the
    // difference in this rate is the interference the device inflicts on the CPU
    // pool -- which the per-batch rule cannot otherwise see.
    if (!macs || !(seconds > 0.0)) return;
    I.ema(I.cpu_rate, I.cpu_samples, (double)macs / seconds);
}
void GpuEvaluator::throttle_stats(double* gpu_mps, double* cpu_mps,
                                  double* accept) const {
    const Impl& I = *m_impl;
    if (gpu_mps) *gpu_mps = Impl::rd(I.gpu_rate);
    if (cpu_mps) *cpu_mps = Impl::rd(I.cpu_rate);
    if (accept) {
        const uint64_t o = I.offers.load(std::memory_order_relaxed);
        const uint64_t t = I.taken.load(std::memory_order_relaxed);
        *accept = o ? (double)t / (double)o : 0.0;
    }
}
uint64_t GpuEvaluator::macs() const {
    return m_impl->n_macs.load(std::memory_order_relaxed);
}
bool GpuEvaluator::would_accept() const {
    Impl& I = *m_impl;
    if (!I.ok.load(std::memory_order_relaxed)) return false;
    const int nsl = I.nslot.load(std::memory_order_relaxed);
    bool slot = false;
    for (int i = 0; i < nsl && !slot; ++i)
        if (!I.slots[i].busy.load(std::memory_order_relaxed)) slot = true;
    if (!slot) return false;
    // Ask the throttle only once a slot is actually free, so a busy device does
    // not burn offers -- the counters drive the warm-up alternation and the
    // re-probe interval, and inflating them with slot contention would make both
    // fire at the wrong rate.
    return I.throttle_ok();
}
void GpuEvaluator::set_partition_cap(int p) {
    m_impl->pcap.store(p < 1 ? 1 : (p > 8 ? 8 : p), std::memory_order_relaxed);
}
int GpuEvaluator::partition_cap() const {
    return m_impl->pcap.load(std::memory_order_relaxed);
}
const std::string& GpuEvaluator::why() const { return m_impl->why; }

void GpuEvaluator::stats(uint64_t* candidates, double* seconds) const {
    if (candidates) *candidates = m_impl->n_cands.load(std::memory_order_relaxed);
    if (seconds) {
        const uint64_t a = m_impl->t_first.load(std::memory_order_relaxed);
        const uint64_t b = m_impl->t_last.load(std::memory_order_relaxed);
        *seconds = (a == UINT64_MAX || b <= a) ? 0.0 : (double)(b - a) * 1e-6;
    }
}

bool GpuEvaluator::evaluate(const int32_t* shifted, uint32_t bsize,
                            const std::vector<Candidate>& cands,
                            std::vector<uint32_t>& out_costs) {
    Impl& I = *m_impl;
    if (!I.ok.load(std::memory_order_relaxed) || cands.empty()) return false;
    // The kernel walks the block in fixed 32-sample chunks.
    if (bsize % 32u != 0u) return false;
    if (cands.size() < I.min_batch.load(std::memory_order_relaxed)) return false;

    // Work this batch represents, in the same unit both throughput rates use.
    // The throttle itself ran in would_accept(), before the caller paid to build
    // this batch; see the note on Impl::throttle_ok.
    uint64_t macs = 0;
    for (const auto& cd : cands)
        macs += (uint64_t)(bsize - (uint32_t)cd.order) * (uint64_t)cd.order;

    // Claim a slot. Failing is not an error: the caller encodes on the CPU
    // instead, and both paths produce the same winner, so the output is
    // unchanged either way. Idling a worker to wait for the device is strictly
    // worse than having it do the work itself.
    int si = -1;
    const int nsl = I.nslot.load(std::memory_order_relaxed);
    for (int i = 0; i < nsl; ++i) {
        bool expect = false;
        if (I.slots[i].busy.compare_exchange_strong(expect, true,
                                                    std::memory_order_acquire)) {
            si = i; break;
        }
    }
    if (si < 0) return false;
    Impl::Slot& sl = I.slots[si];
    struct Release {
        Impl::Slot& s;
        ~Release() { s.busy.store(false, std::memory_order_release); }
    } release{sl};

    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t us0 = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        t0 - I.t_origin).count();
    uint64_t prev = I.t_first.load(std::memory_order_relaxed);
    while (us0 < prev &&
           !I.t_first.compare_exchange_weak(prev, us0, std::memory_order_relaxed)) {}

    const VkDeviceSize needS = (VkDeviceSize)bsize * sizeof(int32_t);
    const VkDeviceSize needC = (VkDeviceSize)cands.size() * 34 * sizeof(int32_t);
    const VkDeviceSize needO = (VkDeviceSize)cands.size() * sizeof(uint32_t);
    const bool grew = sl.bSamples.size < needS || sl.bCands.size < needC ||
                      sl.bCosts.size   < needO;
    if (!I.grow(sl.bSamples, needS)) return false;
    if (!I.grow(sl.bCands,   needC)) return false;
    if (!I.grow(sl.bCosts,   needO)) return false;
    if (grew) I.bindDescriptors(sl);

    std::memcpy(sl.bSamples.map, shifted, (size_t)needS);
    int32_t* cp = (int32_t*)sl.bCands.map;
    for (size_t i = 0; i < cands.size(); ++i) {
        cp[i * 34 + 0] = cands[i].order;
        cp[i * 34 + 1] = cands[i].shift;
        std::memcpy(&cp[i * 34 + 2], cands[i].qc, 32 * sizeof(int32_t));
    }

    PushConsts pcv{ (int32_t)cands.size(), (int32_t)bsize, 0,
                    (int32_t)I.pcap.load(std::memory_order_relaxed) };
    const uint32_t cpw    = Impl::WG / 32;               // candidates per group
    const uint32_t groups = ((uint32_t)cands.size() + cpw - 1) / cpw;

    VkResult vr;
    if ((vr = vkResetCommandBuffer(sl.cmd, 0)) != VK_SUCCESS)
        return I.fail_device(vr, "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((vr = vkBeginCommandBuffer(sl.cmd, &bi)) != VK_SUCCESS)
        return I.fail_device(vr, "vkBeginCommandBuffer");
    vkCmdBindPipeline(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, I.pipe);
    vkCmdBindDescriptorSets(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, I.plo, 0, 1,
                            &sl.dset, 0, nullptr);
    vkCmdPushConstants(sl.cmd, I.plo, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof pcv, &pcv);
    vkCmdDispatch(sl.cmd, groups, 1, 1);
    if ((vr = vkEndCommandBuffer(sl.cmd)) != VK_SUCCESS)
        return I.fail_device(vr, "vkEndCommandBuffer");

    // Only the submit is serialised -- a VkQueue is externally synchronised,
    // but waiting is per-fence and must stay outside the lock or the device is
    // back to one dispatch at a time.
    {
        std::lock_guard<std::mutex> lk(I.submit_mu);
        vkResetFences(I.dev, 1, &sl.fence);
        VkSubmitInfo si2{};
        si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si2.commandBufferCount = 1;
        si2.pCommandBuffers = &sl.cmd;
        if ((vr = vkQueueSubmit(I.queue, 1, &si2, sl.fence)) != VK_SUCCESS)
            return I.fail_device(vr, "vkQueueSubmit");
    }
    // A lost device shows up here in practice: measured on a host that suspended
    // to S3 mid-encode, this wait never returned and the process ignored SIGKILL
    // while stuck in the driver. A finite timeout cannot fix that case -- the
    // driver never came back -- but every failure mode it *does* report has to
    // take the path down rather than silently reroute batch after batch.
    if ((vr = vkWaitForFences(I.dev, 1, &sl.fence, VK_TRUE, UINT64_MAX)) != VK_SUCCESS)
        return I.fail_device(vr, "vkWaitForFences");

    out_costs.resize(cands.size());
    std::memcpy(out_costs.data(), sl.bCosts.map, (size_t)needO);

    // The kernel writes UINT32_MAX for any invocation it could not price (see
    // the gl_SubgroupSize guard at the end of sweep.comp). That value must
    // never reach a caller: both call sites compute `hdr + 6 + cost`, which
    // wraps in uint32 to `hdr + 5` -- the *cheapest* cost representable, so a
    // failed candidate would beat every real one and win the subframe. Measured
    // on an Arc A380: -G chose LPC order 1 where the CPU chose 32, output up to
    // 4.5% larger, and --gpu-partition-cap 1/4/8 gave byte-identical output
    // because a sentinel does not depend on the partition search. A sentinel
    // that means "no answer" must not be arithmetic.
    //
    // Rejecting the whole batch rather than the individual entries keeps the
    // fallback honest: `false` here means the caller prices these candidates on
    // the CPU, which is the same answer, so the output is unchanged rather than
    // merely less wrong.
    for (uint32_t c : out_costs) {
        if (c == UINT32_MAX) {
            if (!I.warned_invalid.exchange(true, std::memory_order_relaxed))
                std::fprintf(stderr,
                    "GPU: device returned no result for a batch (subgroup size "
                    "is not %d at dispatch); those candidates run on the CPU.\n",
                    32);
            I.n_invalid.fetch_add(1, std::memory_order_relaxed);
            out_costs.clear();
            return false;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t us1 = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - I.t_origin).count();
    uint64_t last = I.t_last.load(std::memory_order_relaxed);
    while (us1 > last &&
           !I.t_last.compare_exchange_weak(last, us1, std::memory_order_relaxed)) {}
    I.n_macs.fetch_add(macs, std::memory_order_relaxed);
    I.n_cands.fetch_add(cands.size(), std::memory_order_relaxed);
    I.taken.fetch_add(1, std::memory_order_relaxed);

    // Throughput sample for the throttle. This is the whole call as the calling
    // worker experiences it -- submit, fence wait, readback -- which is the cost
    // the comparison against one CPU thread has to be made against, not the
    // device's own execution time.
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (secs > 0.0) I.ema(I.gpu_rate, I.gpu_samples, (double)macs / secs);
    return true;
}

} // namespace flacoutcpp

#endif // FLACOUT_HAVE_VULKAN
