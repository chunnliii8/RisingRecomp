# RisingRecomp Android diagnostic

Stage 2 is a standalone ARM64 APK. It does not contain or start the game. It records the
Android device, memory/page-size facts and the Vulkan features required by the existing
Case Zero native renderer.

The UI is implemented with the Android Java API and the probe is C++/JNI. There is no
Kotlin dependency. Logs are kept privately under `files/logs/` and can be viewed in the
app or exported to a folder selected through Android's Storage Access Framework.

Build with:

```sh
cd android/diagnostic
gradle :app:assembleDebug
```
