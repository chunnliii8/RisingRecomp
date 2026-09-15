#include <jni.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string Version(uint32_t value) {
    std::ostringstream out;
    out << VK_VERSION_MAJOR(value) << '.' << VK_VERSION_MINOR(value) << '.'
        << VK_VERSION_PATCH(value);
    return out.str();
}

const char* Yes(bool value) { return value ? "YES" : "NO"; }

bool HasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
        return std::string(extension.extensionName) == name;
    });
}

std::string Collect() {
    std::ostringstream out;
    utsname kernel{};
    if (uname(&kernel) == 0) {
        out << "Kernel: " << kernel.release << '\n';
        out << "Machine: " << kernel.machine << '\n';
    }
    out << "Page size: " << sysconf(_SC_PAGESIZE) << " bytes\n\n";

    uint32_t loaderVersion = VK_API_VERSION_1_0;
    const auto enumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (enumerateInstanceVersion) enumerateInstanceVersion(&loaderVersion);
    out << "Vulkan loader: " << Version(loaderVersion) << '\n';

    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "RisingRecomp Diagnostic";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.pEngineName = "RisingRecomp";
    application.apiVersion = std::min(loaderVersion, VK_API_VERSION_1_3);

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &application;
    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    out << "vkCreateInstance: " << result << '\n';
    if (result != VK_SUCCESS) return out.str();

    uint32_t deviceCount = 0;
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    out << "Physical devices: " << deviceCount << '\n';
    if (result != VK_SUCCESS || deviceCount == 0) {
        vkDestroyInstance(instance, nullptr);
        return out.str();
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    for (uint32_t index = 0; index < deviceCount; ++index) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(devices[index], &properties);

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(devices[index], nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(
                devices[index], nullptr, &extensionCount, extensions.data());

        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        const auto getFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
        if (getFeatures2) {
            features.pNext = &features12;
            features12.pNext = &features13;
            getFeatures2(devices[index], &features);
        } else {
            vkGetPhysicalDeviceFeatures(devices[index], &features.features);
        }

        out << "\n[GPU " << index << "]\n";
        out << "Name: " << properties.deviceName << '\n';
        out << "Vendor ID: 0x" << std::hex << properties.vendorID << std::dec << '\n';
        out << "Device ID: 0x" << std::hex << properties.deviceID << std::dec << '\n';
        out << "Driver version: " << properties.driverVersion << '\n';
        out << "Device Vulkan API: " << Version(properties.apiVersion) << '\n';
        out << "maxPerStageDescriptorSampledImages: "
            << properties.limits.maxPerStageDescriptorSampledImages << '\n';
        out << "maxDescriptorSetSampledImages: "
            << properties.limits.maxDescriptorSetSampledImages << '\n';
        out << "maxMemoryAllocationCount: "
            << properties.limits.maxMemoryAllocationCount << '\n';
        out << "shaderInt64: " << Yes(features.features.shaderInt64) << '\n';
        out << "independentBlend: " << Yes(features.features.independentBlend) << '\n';
        out << "textureCompressionBC: " << Yes(features.features.textureCompressionBC) << '\n';
        out << "bufferDeviceAddress: " << Yes(features12.bufferDeviceAddress) << '\n';
        out << "descriptorIndexing: " << Yes(features12.descriptorIndexing) << '\n';
        out << "runtimeDescriptorArray: " << Yes(features12.runtimeDescriptorArray) << '\n';
        out << "descriptorBindingPartiallyBound: "
            << Yes(features12.descriptorBindingPartiallyBound) << '\n';
        out << "descriptorBindingUpdateUnusedWhilePending: "
            << Yes(features12.descriptorBindingUpdateUnusedWhilePending) << '\n';
        out << "descriptorBindingSampledImageUpdateAfterBind: "
            << Yes(features12.descriptorBindingSampledImageUpdateAfterBind) << '\n';
        out << "descriptorBindingVariableDescriptorCount: "
            << Yes(features12.descriptorBindingVariableDescriptorCount) << '\n';
        out << "shaderSampledImageArrayNonUniformIndexing: "
            << Yes(features12.shaderSampledImageArrayNonUniformIndexing) << '\n';
        out << "dynamicRendering: " << Yes(features13.dynamicRendering) << '\n';
        out << "VK_KHR_swapchain: "
            << Yes(HasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) << '\n';
        out << "VK_KHR_dynamic_rendering extension: "
            << Yes(HasExtension(extensions, "VK_KHR_dynamic_rendering")) << '\n';
        out << "VK_KHR_buffer_device_address extension: "
            << Yes(HasExtension(extensions, "VK_KHR_buffer_device_address")) << '\n';

        const bool rendererReady = properties.apiVersion >= VK_API_VERSION_1_3
                && features.features.shaderInt64
                && features.features.independentBlend
                && features.features.textureCompressionBC
                && features12.bufferDeviceAddress
                && features12.descriptorIndexing
                && features12.runtimeDescriptorArray
                && features12.descriptorBindingPartiallyBound
                && features12.descriptorBindingUpdateUnusedWhilePending
                && features12.descriptorBindingSampledImageUpdateAfterBind
                && features12.descriptorBindingVariableDescriptorCount
                && features12.shaderSampledImageArrayNonUniformIndexing
                && features13.dynamicRendering;
        out << "Current renderer requirements: "
            << (rendererReady ? "PASS" : "FAIL (adaptation required)") << '\n';
    }

    vkDestroyInstance(instance, nullptr);
    return out.str();
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_risingrecomp_diagnostic_MainActivity_collectNativeDiagnostics(JNIEnv* env, jclass) {
    const std::string report = Collect();
    return env->NewStringUTF(report.c_str());
}
