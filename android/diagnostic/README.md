# RisingRecomp Android diagnostic

Stage 1 is a standalone ARM64 APK. It does not contain or start the game. It records the
Android device, memory/page-size facts and checks the same Vulkan 1.1 limits and formats
used by the Case Zero native renderer compatibility gate. Its live probe also creates the
real two-set fixed descriptor ABI before executing a classic render-pass draw.

The UI is implemented with the Android Java API and the probe is C++/JNI. There is no
Kotlin dependency. Logs are kept privately under `files/logs/` and can be viewed in the
app or exported to a folder selected through Android's Storage Access Framework.

Build with:

```sh
cd android/diagnostic
gradle :app:assembleDebug
```
