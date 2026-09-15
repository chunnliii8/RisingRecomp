# RisingRecomp Android runtime bootstrap

Stages 2 and 3 of the Android roadmap live here. The APK does not bundle Case Zero data,
the recompiled game code or an emulator. It proves the Android-native foundation and
inspects a user-selected XBLA package locally:

- Java Activity and Surface lifecycle;
- `ANativeWindow` hand-off to C++ through JNI;
- ARM64 native shared-library loading;
- Vulkan 1.1 compatibility device selection and creation;
- deterministic startup/shutdown logs.
- Storage Access Framework selection with persisted read permission;
- strict Case Zero title/content/STFS validation;
- bounded file-tree mapping, private manifest generation and `default.xex` extraction;
- optional `default.xex` export selected by the user.

The XBLA package and extracted XEX remain on the Android device. Neither is uploaded by
the app or included in build artifacts.

Build with `gradle :app:assembleDebug` from this directory.
