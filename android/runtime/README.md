# RisingRecomp Android runtime bootstrap

This is Stage 2 of the Android roadmap. It intentionally does not include Case Zero data,
XBLA/XEX loading, the recompiled game code or an emulator. It proves the Android-native
foundation used by later stages:

- Java Activity and Surface lifecycle;
- `ANativeWindow` hand-off to C++ through JNI;
- ARM64 native shared-library loading;
- Vulkan 1.1 compatibility device selection and creation;
- deterministic startup/shutdown logs.

Build with `gradle :app:assembleDebug` from this directory.
