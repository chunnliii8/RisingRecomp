#include <jni.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "compat_shaders_spv.h"

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

constexpr uint32_t kCompatDescriptorSlots = 16;
constexpr uint32_t kCompatDescriptorSetsPerDraw = 2;
constexpr uint32_t kVsConstBytes = 256 * 16;

bool FormatSupports(VkPhysicalDevice physicalDevice, VkFormat format,
                    VkFormatFeatureFlags required) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
    return (properties.optimalTilingFeatures & required) == required;
}

VkFormat PickCompatibilityDepthFormat(VkPhysicalDevice physicalDevice) {
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    const VkFormat candidates[] = {
            VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
    for (VkFormat format : candidates) {
        if (FormatSupports(physicalDevice, format, required)) return format;
    }
    return VK_FORMAT_UNDEFINED;
}

std::vector<std::string> CompatibilityGateFailures(
        VkPhysicalDevice physicalDevice, const VkPhysicalDeviceProperties& properties,
        uint32_t loaderVersion) {
    std::vector<std::string> missing;
    const auto& limits = properties.limits;
    auto need = [&missing](bool have, const char* name, uint64_t actual, uint64_t required) {
        if (!have) missing.emplace_back(std::string(name) + "=" + std::to_string(actual) +
                                        " (needs " + std::to_string(required) + ")");
    };
    need(loaderVersion >= VK_API_VERSION_1_1, "Vulkan loader API", loaderVersion,
         VK_API_VERSION_1_1);
    need(properties.apiVersion >= VK_API_VERSION_1_1, "Device Vulkan API",
         properties.apiVersion, VK_API_VERSION_1_1);
    need(limits.maxBoundDescriptorSets >= kCompatDescriptorSetsPerDraw,
         "maxBoundDescriptorSets", limits.maxBoundDescriptorSets,
         kCompatDescriptorSetsPerDraw);
    need(limits.maxPerStageDescriptorSampledImages >= kCompatDescriptorSlots * 4,
         "maxPerStageDescriptorSampledImages", limits.maxPerStageDescriptorSampledImages,
         kCompatDescriptorSlots * 4);
    need(limits.maxDescriptorSetSampledImages >= kCompatDescriptorSlots * 4,
         "maxDescriptorSetSampledImages", limits.maxDescriptorSetSampledImages,
         kCompatDescriptorSlots * 4);
    need(limits.maxPerStageDescriptorSamplers >= kCompatDescriptorSlots,
         "maxPerStageDescriptorSamplers", limits.maxPerStageDescriptorSamplers,
         kCompatDescriptorSlots);
    need(limits.maxDescriptorSetSamplers >= kCompatDescriptorSlots,
         "maxDescriptorSetSamplers", limits.maxDescriptorSetSamplers,
         kCompatDescriptorSlots);
    need(limits.maxPerStageDescriptorUniformBuffers >= 2,
         "maxPerStageDescriptorUniformBuffers", limits.maxPerStageDescriptorUniformBuffers, 2);
    need(limits.maxDescriptorSetUniformBuffers >= 3, "maxDescriptorSetUniformBuffers",
         limits.maxDescriptorSetUniformBuffers, 3);
    need(limits.maxPerStageResources >= kCompatDescriptorSlots * 5 + 2,
         "maxPerStageResources", limits.maxPerStageResources,
         kCompatDescriptorSlots * 5 + 2);
    need(limits.maxUniformBufferRange >= kVsConstBytes, "maxUniformBufferRange",
         limits.maxUniformBufferRange, kVsConstBytes);

    const VkFormatFeatureFlags colorRequired = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if (!FormatSupports(physicalDevice, VK_FORMAT_R8G8B8A8_UNORM, colorRequired))
        missing.emplace_back("R8G8B8A8_UNORM attachment/sample/transfer support");
    if (PickCompatibilityDepthFormat(physicalDevice) == VK_FORMAT_UNDEFINED)
        missing.emplace_back("sampleable depth-stencil attachment with transfer support");
    return missing;
}

bool HasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
        return std::string(extension.extensionName) == name;
    });
}

struct ProbeObjects {
    VkDevice device = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayouts[kCompatDescriptorSetsPerDraw]{};
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkShaderModule vertexShader = VK_NULL_HANDLE;
    VkShaderModule fragmentShader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    ~ProbeObjects() {
        if (device == VK_NULL_HANDLE) return;
        vkDeviceWaitIdle(device);
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (fragmentShader) vkDestroyShaderModule(device, fragmentShader, nullptr);
        if (vertexShader) vkDestroyShaderModule(device, vertexShader, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (framebuffer) vkDestroyFramebuffer(device, framebuffer, nullptr);
        if (renderPass) vkDestroyRenderPass(device, renderPass, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        for (VkDescriptorSetLayout layout : descriptorLayouts)
            if (layout) vkDestroyDescriptorSetLayout(device, layout, nullptr);
        if (sampler) vkDestroySampler(device, sampler, nullptr);
        if (imageView) vkDestroyImageView(device, imageView, nullptr);
        if (image) vkDestroyImage(device, image, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        vkDestroyDevice(device, nullptr);
    }
};

uint32_t FindMemoryType(VkPhysicalDevice physicalDevice, uint32_t bits,
                        VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return UINT32_MAX;
}

std::string RunCompatibilityProbe(VkPhysicalDevice physicalDevice) {
    std::ostringstream out;
    out << "\n[Native Vulkan 1.1 compatibility path]\n";

    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queues.data());
    uint32_t queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueCount; ++i) {
        if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queueFamily = i;
            break;
        }
    }
    if (queueFamily == UINT32_MAX) return "No graphics queue: FAIL\n";

    ProbeObjects objects;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    VkResult result = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &objects.device);
    out << "Vulkan 1.1 logical device: " << (result == VK_SUCCESS ? "PASS" : "FAIL")
        << " (" << result << ")\n";
    if (result != VK_SUCCESS) return out.str();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {4, 4, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    result = vkCreateImage(objects.device, &imageInfo, nullptr, &objects.image);
    if (result != VK_SUCCESS) {
        out << "RGBA8 native image: FAIL (" << result << ")\n";
        return out.str();
    }
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(objects.device, objects.image, &requirements);
    const uint32_t memoryType = FindMemoryType(
            physicalDevice, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryType == UINT32_MAX) {
        out << "RGBA8 native image memory: FAIL\n";
        return out.str();
    }
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memoryType;
    result = vkAllocateMemory(objects.device, &allocation, nullptr, &objects.memory);
    if (result == VK_SUCCESS) {
        result = vkBindImageMemory(objects.device, objects.image, objects.memory, 0);
    }
    if (result != VK_SUCCESS) {
        out << "RGBA8 native image memory: FAIL (" << result << ")\n";
        return out.str();
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = objects.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    result = vkCreateImageView(objects.device, &viewInfo, nullptr, &objects.imageView);
    out << "RGBA8 native image: " << (result == VK_SUCCESS ? "PASS" : "FAIL")
        << " (" << result << ")\n";
    if (result != VK_SUCCESS) return out.str();

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 1.0f;
    result = vkCreateSampler(objects.device, &samplerInfo, nullptr, &objects.sampler);
    if (result != VK_SUCCESS) {
        out << "Fixed texture descriptor: FAIL (sampler " << result << ")\n";
        return out.str();
    }

    const VkDescriptorType textureTypes[5] = {
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER,
            VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
    VkDescriptorSetLayoutBinding textureBindings[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        textureBindings[i].binding = i;
        textureBindings[i].descriptorType = textureTypes[i];
        textureBindings[i].descriptorCount = kCompatDescriptorSlots;
        textureBindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptorInfo{};
    descriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorInfo.bindingCount = 5;
    descriptorInfo.pBindings = textureBindings;
    result = vkCreateDescriptorSetLayout(
            objects.device, &descriptorInfo, nullptr, &objects.descriptorLayouts[0]);
    if (result != VK_SUCCESS) {
        out << "Fixed two-set descriptor ABI: FAIL (texture layout " << result << ")\n";
        return out.str();
    }
    VkDescriptorSetLayoutBinding constants[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        constants[i].binding = i;
        constants[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        constants[i].descriptorCount = 1;
    }
    constants[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    constants[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    constants[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    descriptorInfo.bindingCount = 3;
    descriptorInfo.pBindings = constants;
    result = vkCreateDescriptorSetLayout(
            objects.device, &descriptorInfo, nullptr, &objects.descriptorLayouts[1]);
    if (result != VK_SUCCESS) {
        out << "Fixed two-set descriptor ABI: FAIL (constant layout " << result << ")\n";
        return out.str();
    }
    const VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompatDescriptorSlots * 4},
            {VK_DESCRIPTOR_TYPE_SAMPLER, kCompatDescriptorSlots},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3}};
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = kCompatDescriptorSetsPerDraw;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;
    result = vkCreateDescriptorPool(objects.device, &poolInfo, nullptr, &objects.descriptorPool);
    VkDescriptorSet descriptorSets[kCompatDescriptorSetsPerDraw]{};
    if (result == VK_SUCCESS) {
        VkDescriptorSetAllocateInfo setInfo{};
        setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setInfo.descriptorPool = objects.descriptorPool;
        setInfo.descriptorSetCount = kCompatDescriptorSetsPerDraw;
        setInfo.pSetLayouts = objects.descriptorLayouts;
        result = vkAllocateDescriptorSets(objects.device, &setInfo, descriptorSets);
    }
    if (result != VK_SUCCESS) {
        out << "Fixed two-set descriptor ABI: FAIL (allocate " << result << ")\n";
        return out.str();
    }
    out << "Fixed two-set descriptor ABI (64 images/16 samplers/3 UBOs): PASS\n";

    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    result = vkCreateRenderPass(objects.device, &renderPassInfo, nullptr, &objects.renderPass);
    if (result != VK_SUCCESS) {
        out << "Classic render pass: FAIL (" << result << ")\n";
        return out.str();
    }
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = objects.renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &objects.imageView;
    framebufferInfo.width = 4;
    framebufferInfo.height = 4;
    framebufferInfo.layers = 1;
    result = vkCreateFramebuffer(objects.device, &framebufferInfo, nullptr, &objects.framebuffer);
    out << "Classic render pass/framebuffer: "
        << (result == VK_SUCCESS ? "PASS" : "FAIL") << " (" << result << ")\n";
    if (result != VK_SUCCESS) return out.str();

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = kCompatDescriptorSetsPerDraw;
    layoutInfo.pSetLayouts = objects.descriptorLayouts;
    result = vkCreatePipelineLayout(objects.device, &layoutInfo, nullptr, &objects.pipelineLayout);
    if (result != VK_SUCCESS) {
        out << "SPIR-V pipeline without Int64: FAIL (layout " << result << ")\n";
        return out.str();
    }

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = sizeof(kCompatibilityVertexSpv);
    shaderInfo.pCode = kCompatibilityVertexSpv;
    result = vkCreateShaderModule(objects.device, &shaderInfo, nullptr, &objects.vertexShader);
    if (result == VK_SUCCESS) {
        shaderInfo.codeSize = sizeof(kCompatibilityFragmentSpv);
        shaderInfo.pCode = kCompatibilityFragmentSpv;
        result = vkCreateShaderModule(objects.device, &shaderInfo, nullptr, &objects.fragmentShader);
    }
    if (result != VK_SUCCESS) {
        out << "SPIR-V pipeline without Int64: FAIL (shader " << result << ")\n";
        return out.str();
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = objects.vertexShader;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = objects.fragmentShader;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport viewport{0, 0, 4, 4, 0, 1};
    VkRect2D scissor{{0, 0}, {4, 4}};
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.layout = objects.pipelineLayout;
    pipelineInfo.renderPass = objects.renderPass;
    result = vkCreateGraphicsPipelines(
            objects.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &objects.pipeline);
    out << "SPIR-V pipeline without Int64: "
        << (result == VK_SUCCESS ? "PASS" : "FAIL") << " (" << result << ")\n";
    if (result != VK_SUCCESS) return out.str();

    VkCommandPoolCreateInfo commandPoolInfo{};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = queueFamily;
    result = vkCreateCommandPool(objects.device, &commandPoolInfo, nullptr, &objects.commandPool);
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (result == VK_SUCCESS) {
        VkCommandBufferAllocateInfo commandInfo{};
        commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandInfo.commandPool = objects.commandPool;
        commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandInfo.commandBufferCount = 1;
        result = vkAllocateCommandBuffers(objects.device, &commandInfo, &commandBuffer);
    }
    if (result != VK_SUCCESS) {
        out << "Offscreen native draw: FAIL (command " << result << ")\n";
        return out.str();
    }
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkClearValue clear{};
    clear.color = {{0.08f, 0.18f, 0.35f, 1.0f}};
    VkRenderPassBeginInfo passBegin{};
    passBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    passBegin.renderPass = objects.renderPass;
    passBegin.framebuffer = objects.framebuffer;
    passBegin.renderArea.extent = {4, 4};
    passBegin.clearValueCount = 1;
    passBegin.pClearValues = &clear;
    if (result == VK_SUCCESS) {
        vkCmdBeginRenderPass(commandBuffer, &passBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, objects.pipeline);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        result = vkEndCommandBuffer(commandBuffer);
    }
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (result == VK_SUCCESS) {
        result = vkCreateFence(objects.device, &fenceInfo, nullptr, &objects.fence);
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(objects.device, queueFamily, 0, &queue);
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer;
    if (result == VK_SUCCESS) result = vkQueueSubmit(queue, 1, &submit, objects.fence);
    if (result == VK_SUCCESS) {
        result = vkWaitForFences(objects.device, 1, &objects.fence, VK_TRUE, 5000000000ULL);
    }
    out << "Offscreen native draw + fence: "
        << (result == VK_SUCCESS ? "PASS" : "FAIL") << " (" << result << ")\n";
    out << "Compatibility probe: " << (result == VK_SUCCESS ? "PASS" : "FAIL") << '\n';
    return out.str();
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
            // Do not put newer core structs in the chain on a Vulkan 1.1 driver.
            // Some old Android loaders reject or mishandle unknown pNext structures.
            if (properties.apiVersion >= VK_API_VERSION_1_2) {
                features.pNext = &features12;
                if (properties.apiVersion >= VK_API_VERSION_1_3)
                    features12.pNext = &features13;
            }
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
        const std::vector<std::string> compatibilityFailures =
                CompatibilityGateFailures(devices[index], properties, loaderVersion);
        out << "Modern Vulkan 1.3 route: " << (rendererReady ? "PASS" : "UNAVAILABLE") << '\n';
        out << "Vulkan 1.1 runtime gate: "
            << (compatibilityFailures.empty() ? "PASS" : "FAIL") << '\n';
        for (const std::string& failure : compatibilityFailures)
            out << "  missing: " << failure << '\n';
        const VkFormat depthFormat = PickCompatibilityDepthFormat(devices[index]);
        if (depthFormat != VK_FORMAT_UNDEFINED) {
            out << "Compatibility depth format: "
                << (depthFormat == VK_FORMAT_D24_UNORM_S8_UINT
                            ? "D24_UNORM_S8_UINT" : "D32_SFLOAT_S8_UINT") << '\n';
        }
        out << "Selected renderer profile: "
            << (rendererReady ? "MODERN"
                              : (compatibilityFailures.empty() ? "VULKAN 1.1 COMPATIBILITY"
                                                                : "UNSUPPORTED"))
            << '\n';
        if (compatibilityFailures.empty())
            out << RunCompatibilityProbe(devices[index]);
        else
            out << "\n[Native Vulkan 1.1 compatibility path]\n"
                   "Compatibility probe: SKIPPED (runtime gate failed)\n";
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
