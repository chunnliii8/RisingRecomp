# RisingRecomp Android port roadmap

This roadmap keeps the Android port incremental. Each stage has a narrow deliverable and
must be validated before the next stage starts. The Android route remains native: no Xbox
360 emulator, no translation container and no Kotlin dependency.

## Non-negotiable rules

- Preserve the existing desktop Vulkan 1.3 path.
- Keep a native Vulkan 1.1 compatibility path for older Adreno and future Mali testing.
- Target ARM64 Android with the NDK and C++/Java Android APIs.
- Do not bundle copyrighted Case Zero/XBLA files in the repository or APK.
- Do not optimize blindly. First collect a reproducible log, then change one subsystem.
- Keep every stage buildable and commit tested checkpoints to `risingRecomp`.

## Stage 1 — Vulkan compatibility foundation (complete)

Deliverables:

- Standalone diagnostic APK with log viewing and export.
- Vulkan loader/device detection without requiring Vulkan 1.2/1.3 feature chains on a
  Vulkan 1.1 device.
- Separate Modern and Compatibility renderer profiles.
- Vulkan 1.1 gate using fixed descriptors, classic render passes, portable limits and
  attachment-format checks.
- Modern Vulkan 1.3 path left unchanged.

Validated result:

- Moto G34 5G, Adreno 619, Android 15, Vulkan 1.1.128.
- Runtime gate, logical device, two-set descriptor ABI, render pass, shader pipeline and
  native offscreen draw/fence all passed.
- Mali support is designed into the compatibility contract but remains unconfirmed until
  a real Mali device runs the same diagnostic.

Exit gate: **passed on the Adreno 619 reference device**.

## Stage 2 — Native Android runtime bootstrap (next)

Goal: produce a small ARM64 APK that loads RisingRecomp's native runtime safely, without
loading or executing the game yet.

Work:

1. Add a dedicated Android runtime target instead of building the full desktop program.
2. Create a Java Activity plus JNI/C++ entry point; keep Kotlin out of the project.
3. Link the minimum reusable runtime components for Android/Bionic.
4. Add `ANativeWindow` lifecycle handling and initialize the validated Vulkan profile.
5. Add structured native logs and explicit startup checkpoints.
6. Package no XBLA/XEX data and stop cleanly at a `runtime ready; game not loaded` screen.

Exit gate:

- The APK installs and opens on ARM64 Android.
- The native library loads without unresolved symbols.
- Android window creation and Vulkan device initialization succeed.
- Closing/reopening the app does not leak or crash.
- Logs identify the selected Modern or Vulkan 1.1 Compatibility profile.

## Stage 3 — Legal game-data intake and filesystem mapping

Goal: allow the user to select their own Case Zero/XBLA dump and validate its layout.

Work:

- Use Android's Storage Access Framework; do not request broad storage access.
- Detect the supported input form and locate `default.xex` plus required content files.
- Copy/extract only into app-owned storage when needed.
- Validate hashes/sizes/required files and produce actionable errors.
- Keep all game data outside Git and outside distributed APKs.

Exit gate: the APK recognizes a valid user-provided game dump and builds a stable virtual
filesystem map, but still does not execute game code.

## Stage 4 — ARM64 recompiled-code integration

Goal: link the recompiled Case Zero code and enter it under Android without rendering.

Work:

- Build the generated/recompiled game code for Android ARM64.
- Replace desktop-only OS, threading, timing, file and memory assumptions with Android
  platform shims.
- Validate guest memory layout, endianness and page-alignment assumptions.
- Add deterministic boot checkpoints before every major subsystem.

Exit gate: execution reaches the first stable game-runtime checkpoint repeatedly, with a
useful log on failure.

## Stage 5 — First game boot APK

Goal: generate the first APK that accepts the user's files and attempts to start the game.

Work:

- Connect game-data mapping, recompiled entry point and native runtime lifecycle.
- Handle fatal errors without closing silently.
- Export complete boot logs and crash markers.

Exit gate: the APK enters game initialization consistently. A black screen is acceptable
only when the log proves the game loop is alive and identifies the next blocked subsystem.

## Stage 6 — First native frame and controls

Goal: render the first correct game frame through the native Vulkan renderer.

Work:

- Connect swapchain/presentation to `ANativeWindow`.
- Validate the Vulkan 1.1 path on the Adreno reference device without Turnip.
- Add Android touch/gamepad input mapping.
- Bring up audio after graphics timing is stable.

Exit gate: visible game output, responsive input and stable frame presentation.

## Stage 7 — Compatibility and performance hardening

Goal: turn the bring-up build into a broadly testable Android port.

Work:

- Test Adreno 600, 700 and 800 families and real Mali devices.
- Keep optional Turnip/AdrenoTools experiments separate from the system-driver baseline.
- Profile CPU, GPU, memory and shader compilation before optimizing.
- Add device-specific workarounds only when logs prove they are required.

Exit gate: repeatable gameplay tests, documented device matrix and no regressions to the
desktop Modern path.

## Current position

Stage 1 is complete on the Adreno 619 reference device. Start Stage 2 with the smallest
possible Android native runtime shell; do not compile or package the full game yet.
