// Vulkan capability dump, for answering "why did -G refuse this device" on a
// host with no vulkaninfo. Prints subgroup size, supportedOperations broken out
// by name, shaderInt64/16, compute limits and the extensions that matter here.
//
// Not part of the CMake build -- it needs only libvulkan, and the point is to
// run it somewhere the encoder does not build. Cross-build it in the same image
// the amd64 binary comes from:
//
//   docker run --rm --platform linux/amd64 -v "$PWD/bench":/w flacout-amd64 \
//       sh -c 'cc -O2 -o /w/vkprobe /w/vkprobe.c -lvulkan'
//
// It is what found the Haswell gap (no subgroup ARITHMETIC, no shaderInt64);
// see GPU_PLAN.md.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { puts("vkCreateInstance failed"); return 1; }
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    printf("devices: %u\n", n);
    VkPhysicalDevice *pd = malloc(n * sizeof *pd);
    vkEnumeratePhysicalDevices(inst, &n, pd);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(pd[i], &p2);
        VkPhysicalDeviceFeatures f; vkGetPhysicalDeviceFeatures(pd[i], &f);
        printf("\n[%u] %s  api %u.%u.%u  type %d\n", i, p2.properties.deviceName,
               VK_VERSION_MAJOR(p2.properties.apiVersion), VK_VERSION_MINOR(p2.properties.apiVersion),
               VK_VERSION_PATCH(p2.properties.apiVersion), p2.properties.deviceType);
        printf("  subgroupSize %u  shaderInt64 %d  shaderInt16 %d\n", sg.subgroupSize, f.shaderInt64, f.shaderInt16);
        printf("  supportedStages 0x%x (compute=%d)\n", sg.supportedStages,
               (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0);
        printf("  ops 0x%x:", sg.supportedOperations);
        struct { VkSubgroupFeatureFlags b; const char *n; } t[] = {
            {VK_SUBGROUP_FEATURE_BASIC_BIT,"basic"},{VK_SUBGROUP_FEATURE_VOTE_BIT,"vote"},
            {VK_SUBGROUP_FEATURE_ARITHMETIC_BIT,"arithmetic"},{VK_SUBGROUP_FEATURE_BALLOT_BIT,"ballot"},
            {VK_SUBGROUP_FEATURE_SHUFFLE_BIT,"shuffle"},{VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT,"shuffle_rel"},
            {VK_SUBGROUP_FEATURE_CLUSTERED_BIT,"clustered"},{VK_SUBGROUP_FEATURE_QUAD_BIT,"quad"}};
        for (unsigned k = 0; k < sizeof t/sizeof*t; k++)
            printf(" %s=%d", t[k].n, (sg.supportedOperations & t[k].b) != 0);
        printf("\n  maxComputeWorkGroupInvocations %u  maxSharedMemory %u  maxStorageBufferRange %u\n",
               p2.properties.limits.maxComputeWorkGroupInvocations,
               p2.properties.limits.maxComputeSharedMemorySize,
               p2.properties.limits.maxStorageBufferRange);
        uint32_t en = 0; vkEnumerateDeviceExtensionProperties(pd[i], NULL, &en, NULL);
        VkExtensionProperties *ep = malloc(en * sizeof *ep);
        vkEnumerateDeviceExtensionProperties(pd[i], NULL, &en, ep);
        for (uint32_t k = 0; k < en; k++)
            if (strstr(ep[k].extensionName, "subgroup") || strstr(ep[k].extensionName, "timestamp")
                || strstr(ep[k].extensionName, "atomic") || strstr(ep[k].extensionName, "8bit")
                || strstr(ep[k].extensionName, "16bit"))
                printf("  ext: %s\n", ep[k].extensionName);
    }
    return 0;
}
