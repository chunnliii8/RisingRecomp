cmake_minimum_required(VERSION 3.22.1)
project(RisingRecompDiagnostic LANGUAGES CXX)

add_library(rising_diagnostic SHARED diagnostic.cpp)
target_compile_features(rising_diagnostic PRIVATE cxx_std_20)
target_link_libraries(rising_diagnostic PRIVATE android log vulkan)
