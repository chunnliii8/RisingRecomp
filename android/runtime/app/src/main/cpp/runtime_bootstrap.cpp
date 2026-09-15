#include <jni.h>
#include <unistd.h>

#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "stfs_intake.h"

namespace {

constexpr char kLogTag[] = "RisingRuntime";
constexpr uint32_t kCompatDescriptorSlots = 16;
constexpr uint32_t kCompatDescriptorSets = 2;
constexpr uint32_t kVsConstBytes = 256 * 16;

std::mutex g_mutex;

struct RuntimeSession {
    ANativeWindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    void Destroy() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        if (window != nullptr) {
            ANativeWindow_release(window);
            window = nullptr;
        }
    }
};

RuntimeSession g_session;

std::string Version(uint32_t value) {
    std::ostringstream out;
    out << VK_VERSION_MAJOR(value) << '.' << VK_VERSION_MINOR(value) << '.'
        << VK_VERSION_PATCH(value);
    return out.str();
}

bool FormatSupports(VkPhysicalDevice physicalDevice, VkFormat format,
                    VkFormatFeatureFlags required) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
    return (properties.optimalTilingFeatures & required) == required;
}

bool HasCompatibilityFormats(VkPhysicalDevice physicalDevice, std::ostringstream& out) {
    const VkFormatFeatureFlags colorRequired = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const VkFormatFeatureFlags depthRequired = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if (!FormatSupports(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, colorRequired)) {
        out << "Compatibility formats: FAIL (RGBA8 attachment/sample/transfer)\n";
        return false;
    }
    for (VkFormat format : {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
        if (FormatSupports(physicalDevice, format, depthRequired)) {
            out << "Compatibility depth: "
                << (format == VK_FORMAT_D24_UNORM_S8_UINT ? "D24_UNORM_S8_UINT"
                                                           : "D32_SFLOAT_S8_UINT")
                << '\n';
            return true;
        }
    }
    out << "Compatibility formats: FAIL (sampleable depth/stencil attachment)\n";
    return false;
}

bool HasCompatibilityLimits(const VkPhysicalDeviceLimits& limits, std::ostringstream& out) {
    struct Requirement {
        const char* name;
        uint64_t actual;
        uint64_t required;
    };
    const Requirement requirements[] = {
            {"maxBoundDescriptorSets", limits.maxBoundDescriptorSets, kCompatDescriptorSets},
            {"maxPerStageDescriptorSampledImages", limits.maxPerStageDescriptorSampledImages,
             kCompatDescriptorSlots * 4},
            {"maxDescriptorSetSampledImages", limits.maxDescriptorSetSampledImages,
             kCompatDescriptorSlots * 4},
            {"maxPerStageDescriptorSamplers", limits.maxPerStageDescriptorSamplers,
             kCompatDescriptorSlots},
            {"maxDescriptorSetSamplers", limits.maxDescriptorSetSamplers,
             kCompatDescriptorSlots},
            {"maxPerStageDescriptorUniformBuffers", limits.maxPerStageDescriptorUniformBuffers,
             2},
            {"maxDescriptorSetUniformBuffers", limits.maxDescriptorSetUniformBuffers, 3},
            {"maxPerStageResources", limits.maxPerStageResources, kCompatDescriptorSlots * 5 + 2},
            {"maxUniformBufferRange", limits.maxUniformBufferRange, kVsConstBytes},
    };
    bool pass = true;
    for (const Requirement& requirement : requirements) {
        if (requirement.actual < requirement.required) {
            out << "Compatibility limit: FAIL " << requirement.name << '=' << requirement.actual
                << " (needs " << requirement.required << ")\n";
            pass = false;
        }
    }
    if (pass) out << "Compatibility limits: PASS\n";
    return pass;
}

uint32_t FindGraphicsQueue(VkPhysicalDevice physicalDevice) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, queues.data());
    for (uint32_t index = 0; index < count; ++index) {
        if ((queues[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) return index;
    }
    return UINT32_MAX;
}

std::string Start(JNIEnv* env, jobject surface) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_session.Destroy();
    std::ostringstream out;
    out << "RisingRecomp Android runtime bootstrap 0.1.0\n";
    out << "Game data: not loaded (Stage 3 owns intake)\n";

    g_session.window = ANativeWindow_fromSurface(env, surface);
    if (g_session.window == nullptr) return out.str() + "ANativeWindow: FAIL\n";
    out << "ANativeWindow: PASS " << ANativeWindow_getWidth(g_session.window) << 'x'
        << ANativeWindow_getHeight(g_session.window) << '\n';

    uint32_t loaderVersion = VK_API_VERSION_1_0;
    const auto enumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (enumerateInstanceVersion != nullptr) enumerateInstanceVersion(&loaderVersion);
    out << "Vulkan loader: " << Version(loaderVersion) << '\n';
    if (loaderVersion < VK_API_VERSION_1_1) {
        g_session.Destroy();
        return out.str() + "Runtime bootstrap: FAIL (Vulkan 1.1 required)\n";
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "RisingRecomp Android Runtime";
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = "RisingRecomp";
    app.apiVersion = std::min(loaderVersion, uint32_t(VK_API_VERSION_1_3));
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &app;
    VkResult result = vkCreateInstance(&instanceInfo, nullptr, &g_session.instance);
    if (result != VK_SUCCESS) {
        g_session.Destroy();
        return out.str() + "Vulkan instance: FAIL (" + std::to_string(result) + ")\n";
    }
    out << "Vulkan instance: PASS\n";

    uint32_t deviceCount = 0;
    result = vkEnumeratePhysicalDevices(g_session.instance, &deviceCount, nullptr);
    if (result != VK_SUCCESS || deviceCount == 0) {
        g_session.Destroy();
        return out.str() + "Physical device: FAIL\n";
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(g_session.instance, &deviceCount, devices.data());

    VkPhysicalDevice selected = VK_NULL_HANDLE;
    uint32_t queueFamily = UINT32_MAX;
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_1) continue;
        std::ostringstream candidateLog;
        if (!HasCompatibilityLimits(properties.limits, candidateLog) ||
            !HasCompatibilityFormats(candidate, candidateLog)) {
            out << "GPU " << properties.deviceName << " rejected:\n" << candidateLog.str();
            continue;
        }
        const uint32_t graphicsQueue = FindGraphicsQueue(candidate);
        if (graphicsQueue == UINT32_MAX) continue;
        selected = candidate;
        queueFamily = graphicsQueue;
        out << "GPU: " << properties.deviceName << " (Vulkan "
            << Version(properties.apiVersion) << ")\n" << candidateLog.str();
        break;
    }
    if (selected == VK_NULL_HANDLE) {
        g_session.Destroy();
        return out.str() + "Runtime bootstrap: FAIL (no Vulkan 1.1 compatibility GPU)\n";
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    result = vkCreateDevice(selected, &deviceInfo, nullptr, &g_session.device);
    if (result != VK_SUCCESS) {
        g_session.Destroy();
        return out.str() + "Vulkan 1.1 logical device: FAIL (" + std::to_string(result) + ")\n";
    }
    out << "Vulkan 1.1 logical device: PASS\n";
    out << "Selected runtime profile: VULKAN 1.1 COMPATIBILITY\n";
    out << "Runtime ready; game not loaded.\n";
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "%s", out.str().c_str());
    return out.str();
}

std::string Stop() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_session.Destroy();
    return "Runtime stopped; native resources released.\n";
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_risingrecomp_runtime_MainActivity_nativeStart(JNIEnv* env, jclass, jobject surface) {
    const std::string status = Start(env, surface);
    return env->NewStringUTF(status.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_risingrecomp_runtime_MainActivity_nativeStop(JNIEnv* env, jclass) {
    const std::string status = Stop();
    return env->NewStringUTF(status.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_risingrecomp_runtime_MainActivity_nativeInspectPackage(
        JNIEnv* env, jclass, jint fd, jstring xexPath, jstring manifestPath) {
    const char* xexChars = env->GetStringUTFChars(xexPath, nullptr);
    if (xexChars == nullptr) {
        close(fd);
        return nullptr;
    }
    const char* manifestChars = env->GetStringUTFChars(manifestPath, nullptr);
    if (manifestChars == nullptr) {
        env->ReleaseStringUTFChars(xexPath, xexChars);
        close(fd);
        return nullptr;
    }
    const std::string status = InspectCaseZeroPackage(fd, xexChars, manifestChars);
    env->ReleaseStringUTFChars(manifestPath, manifestChars);
    env->ReleaseStringUTFChars(xexPath, xexChars);
    return env->NewStringUTF(status.c_str());
}
