cmake_minimum_required(VERSION 3.22.1)
project(RisingRecompRuntimeBootstrap LANGUAGES CXX)

add_library(rising_runtime_bootstrap SHARED runtime_bootstrap.cpp stfs_intake.cpp)
target_compile_features(rising_runtime_bootstrap PRIVATE cxx_std_20)
target_link_libraries(rising_runtime_bootstrap PRIVATE android log vulkan)

# The generated image is private and never lives in this repository. Passing both
# paths turns on the Stage 4 integration build; an ordinary public checkout still
# builds the bootstrap without requiring proprietary output.
set(CZ_PPC_DIR "" CACHE PATH "Private XenonRecomp output directory")
set(XENON_ROOT "" CACHE PATH "XenonRecomp source root (supplies SIMDe)")
if(CZ_PPC_DIR OR XENON_ROOT)
    if(NOT EXISTS "${CZ_PPC_DIR}/ppc_func_mapping.cpp")
        message(FATAL_ERROR "CZ_PPC_DIR does not contain complete generated PPC output")
    endif()
    if(NOT EXISTS "${XENON_ROOT}/thirdparty/simde/x86/avx.h")
        message(FATAL_ERROR "XENON_ROOT does not contain XenonRecomp's SIMDe dependency")
    endif()

    file(GLOB CZ_PPC_SOURCES CONFIGURE_DEPENDS "${CZ_PPC_DIR}/*.cpp")
    list(LENGTH CZ_PPC_SOURCES CZ_PPC_TU_COUNT)
    if(NOT CZ_PPC_TU_COUNT EQUAL 228)
        message(FATAL_ERROR
            "Incomplete Case Zero PPC tree: expected 228 C++ files, found ${CZ_PPC_TU_COUNT}")
    endif()

    # Stage 5 owns executable kernel services. At this pre-boot checkpoint every
    # generated import is resolved to an honest trap so the Android linker can prove
    # that the complete image is present without pretending any service works.
    file(STRINGS "${CZ_PPC_DIR}/ppc_func_mapping.cpp" CZ_IMPORT_LINES
         REGEX "__imp__[A-Za-z0-9_]+[ ]*},")
    set(CZ_IMPORT_NAMES "")
    foreach(line IN LISTS CZ_IMPORT_LINES)
        string(REGEX MATCH "__imp__[A-Za-z0-9_]+" import_name "${line}")
        list(APPEND CZ_IMPORT_NAMES "${import_name}")
    endforeach()
    list(REMOVE_DUPLICATES CZ_IMPORT_NAMES)
    list(LENGTH CZ_IMPORT_NAMES CZ_IMPORT_COUNT)
    if(NOT CZ_IMPORT_COUNT EQUAL 244)
        message(FATAL_ERROR
            "Generated import set changed: expected 244 functions, found ${CZ_IMPORT_COUNT}")
    endif()
    set(CZ_IMPORT_STUBS "${CMAKE_CURRENT_BINARY_DIR}/stage4_import_traps.cpp")
    file(WRITE "${CZ_IMPORT_STUBS}"
        "// Build-only Stage 4 traps. Generated from the private mapping table.\n"
        "#include <ppc_config.h>\n#include <ppc_context.h>\n")
    foreach(import_name IN LISTS CZ_IMPORT_NAMES)
        file(APPEND "${CZ_IMPORT_STUBS}"
            "PPC_FUNC(${import_name}) { (void)ctx; (void)base; __builtin_trap(); }\n")
    endforeach()

    add_library(casezero_ppc_image OBJECT ${CZ_PPC_SOURCES} "${CZ_IMPORT_STUBS}")
    target_compile_features(casezero_ppc_image PRIVATE cxx_std_20)
    target_include_directories(casezero_ppc_image PRIVATE
        "${CZ_PPC_DIR}" "${XENON_ROOT}/thirdparty/simde")
    target_compile_options(casezero_ppc_image PRIVATE
        -Wno-unused-variable -Wno-unused-but-set-variable -Wno-asm-operand-widths)

    target_sources(rising_runtime_bootstrap PRIVATE
        ppc_image_bridge.cpp $<TARGET_OBJECTS:casezero_ppc_image>)
    target_include_directories(rising_runtime_bootstrap PRIVATE
        "${CZ_PPC_DIR}" "${XENON_ROOT}/thirdparty/simde")
    target_compile_definitions(rising_runtime_bootstrap PRIVATE CZ_LINKED_PPC_IMAGE=1)
    target_compile_options(rising_runtime_bootstrap PRIVATE -Wno-asm-operand-widths)
    message(STATUS
        "Stage 4 private PPC integration: ${CZ_PPC_TU_COUNT} TUs, ${CZ_IMPORT_COUNT} trapped imports")
endif()
