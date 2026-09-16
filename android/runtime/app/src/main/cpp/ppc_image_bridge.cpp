#include "ppc_image_bridge.h"

#include <cstdint>

#include <ppc_recomp_shared.h>

std::string ValidateLinkedPpcImage() {
    constexpr uintptr_t kExpectedEntry = 0x825D9F30;
    size_t count = 0;
    bool hasEntry = false;
    uintptr_t previous = 0;
    for (const PPCFuncMapping* mapping = PPCFuncMappings;
         mapping->guest != 0 || mapping->host != nullptr; ++mapping) {
        if (mapping->host == nullptr || (count != 0 && mapping->guest <= previous))
            return "PPC image: FAIL (invalid mapping table)\n";
        previous = mapping->guest;
        hasEntry |= mapping->guest == kExpectedEntry;
        ++count;
    }
    if (count == 0 || !hasEntry)
        return "PPC image: FAIL (entry point is absent)\n";
    return "PPC image: PASS (" + std::to_string(count) +
           " linked mappings; execution disabled until Stage 5)\n";
}
