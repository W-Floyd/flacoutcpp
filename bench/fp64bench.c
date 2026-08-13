// fp32-vs-fp64 FMA throughput on every Vulkan device, with bench/cpu_fma2.c as
// the host-side twin of the same kernel. Built to answer one question with a
// number instead of a spec sheet: could the encoder's double-precision
// coefficient path (autocorrelation, Levinson, quantize) ever move to a GPU?
//
// Answer, measured: no, on every device tested. See CLAUDE.md, "There is no
// GPU-only mode to build".
//
//   docker run --rm --platform linux/amd64 -v "$PWD/bench":/w flacout-amd64 \
//     sh -c 'cd /w && glslangValidator -V --target-env vulkan1.1 -S comp \
//              fp64bench.comp -o fp32.spv && \
//            glslangValidator -V --target-env vulkan1.1 -S comp -DUSE_FP64 \
//              fp64bench.comp -o fp64.spv && \
//            cc -O2 -o fp64bench fp64bench.c -lvulkan'
//   gcc -O3 -march=native -o cpu_fma2 cpu_fma2.c -lpthread
//
// Reads fp32.spv / fp64.spv from the working directory. Not part of the CMake
// build; it needs only libvulkan and runs where the encoder does not build.

// Measures fp32 vs fp64 FMA throughput on every Vulkan device, to answer
// "could the double-precision coefficient path move to this GPU?" with a number.
// Loads fp32.spv / fp64.spv from the working directory.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

static uint32_t* slurp(const char* p, size_t* n) {
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* b = malloc(sz); if (fread(b, 1, sz, f) != (size_t)sz) { fclose(f); return NULL; }
    fclose(f); *n = sz; return b;
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#define GROUPS 4096
#define WG     256
#define NACC   8
#define ITER   4096

static double run(VkDevice dev, VkQueue q, uint32_t qfam, VkPhysicalDevice phys,
                  const char* spv, int* ok) {
    *ok = 0;
    size_t n = 0; uint32_t* code = slurp(spv, &n);
    if (!code) { printf("    (missing %s)\n", spv); return 0.0; }
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = n; smci.pCode = code;
    VkShaderModule sm;
    if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) { free(code); return 0.0; }

    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                      VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 1; dl.pBindings = &b;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dl, NULL, &dsl);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int)};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &dsl;
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pcr;
    VkPipelineLayout plo; vkCreatePipelineLayout(dev, &pl, NULL, &plo);
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = sm;
    cp.stage.pName = "main"; cp.layout = plo;
    VkPipeline pipe;
    if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, NULL, &pipe) != VK_SUCCESS) {
        printf("    (pipeline creation failed)\n"); free(code); return 0.0;
    }

    VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bc.size = GROUPS * sizeof(float);
    bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buf; vkCreateBuffer(dev, &bc, NULL, &buf);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { mt = i; break; }
    VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = mr.size; ma.memoryTypeIndex = mt;
    VkDeviceMemory mem; vkAllocateMemory(dev, &ma, NULL, &mem);
    vkBindBufferMemory(dev, buf, mem, 0);

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &ps;
    VkDescriptorPool pool; vkCreateDescriptorPool(dev, &dp, NULL, &pool);
    VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = pool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsa, &ds);
    VkDescriptorBufferInfo dbi = {buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = ds; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qfam;
    VkCommandPool cmdpool; vkCreateCommandPool(dev, &cpci, NULL, &cmdpool);
    VkCommandBufferAllocateInfo cba = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool = cmdpool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(dev, &cba, &cmd);
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; vkCreateFence(dev, &fci, NULL, &fence);

    int iter = ITER;
    double best = 1e30;
    for (int rep = 0; rep < 4; rep++) {
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, plo, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cmd, plo, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof iter, &iter);
        vkCmdDispatch(cmd, GROUPS, 1, 1);
        vkEndCommandBuffer(cmd);
        vkResetFences(dev, 1, &fence);
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        double t0 = now();
        vkQueueSubmit(q, 1, &si, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
        double dt = now() - t0;
        if (rep && dt < best) best = dt;   // discard the first (warm-up) rep
    }
    // 2 flops per FMA.
    double flops = 2.0 * (double)GROUPS * WG * NACC * (double)iter;
    vkDestroyFence(dev, fence, NULL); vkDestroyCommandPool(dev, cmdpool, NULL);
    vkDestroyDescriptorPool(dev, pool, NULL); vkFreeMemory(dev, mem, NULL);
    vkDestroyBuffer(dev, buf, NULL); vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyPipelineLayout(dev, plo, NULL); vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroyShaderModule(dev, sm, NULL); free(code);
    *ok = 1;
    return flops / best / 1e9;   // GFLOP/s
}

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    // MoltenVK is a layered implementation: without portability enumeration it
    // reports no devices, and enabling it unconditionally fails a native loader.
    uint32_t nie = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
    VkExtensionProperties* ie = malloc(nie * sizeof *ie);
    vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
    const char* iexts[2]; uint32_t nx = 0;
    for (uint32_t k = 0; k < nie; k++)
        if (!strcmp(ie[k].extensionName, "VK_KHR_portability_enumeration")) {
            iexts[nx++] = "VK_KHR_portability_enumeration";
            ici.flags |= 0x00000001u;
        }
    ici.enabledExtensionCount = nx; ici.ppEnabledExtensionNames = iexts;
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { puts("no instance"); return 1; }
    uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, NULL);
    VkPhysicalDevice* pd = malloc(nd * sizeof *pd);
    vkEnumeratePhysicalDevices(inst, &nd, pd);
    for (uint32_t i = 0; i < nd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[i], &p);
        VkPhysicalDeviceFeatures f; vkGetPhysicalDeviceFeatures(pd[i], &f);
        printf("\n%s  (shaderFloat64 %d)\n", p.deviceName, f.shaderFloat64);
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, NULL);
        VkQueueFamilyProperties* qp = malloc(nq * sizeof *qp);
        vkGetPhysicalDeviceQueueFamilyProperties(pd[i], &nq, qp);
        uint32_t qfam = UINT32_MAX;
        for (uint32_t k = 0; k < nq; k++)
            if (qp[k].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = k; break; }
        if (qfam == UINT32_MAX) { puts("  no compute queue"); continue; }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures want; memset(&want, 0, sizeof want);
        want.shaderFloat64 = f.shaderFloat64;
        VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        dci.pEnabledFeatures = &want;
        VkDevice dev;
        if (vkCreateDevice(pd[i], &dci, NULL, &dev) != VK_SUCCESS) { puts("  no device"); continue; }
        VkQueue q; vkGetDeviceQueue(dev, qfam, 0, &q);
        int ok32 = 0, ok64 = 0;
        double g32 = run(dev, q, qfam, pd[i], "fp32.spv", &ok32);
        double g64 = f.shaderFloat64 ? run(dev, q, qfam, pd[i], "fp64.spv", &ok64) : 0.0;
        if (ok32) printf("  fp32 %8.1f GFLOP/s\n", g32);
        if (ok64) printf("  fp64 %8.1f GFLOP/s   (1/%.1f of fp32)\n", g64, g32 / g64);
        else if (!f.shaderFloat64) printf("  fp64 unsupported\n");
        vkDestroyDevice(dev, NULL);
    }
    return 0;
}
