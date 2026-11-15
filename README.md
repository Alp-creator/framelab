# FrameLab

**Experimental NPU-assisted frame generation for Android games.**

FrameLab captures one game's screen, uses the device's neural accelerator (NPU) to estimate the motion between two real frames, and draws a generated in-between frame on the GPU. A game running at 30 fps can be shown at up to 60 fps, at the full resolution of the screen.

This is a prototype, not a finished product. It has been developed and tested on a single device: **Lenovo Tab TB373FU (MediaTek MT8792 / mt6897, Android 16)**. Reports and contributions from other devices are very welcome.

## How it works

```
game ──MediaProjection──► SurfaceTexture (GPU)
                              │
            ┌─────────────────┴───────────────────┐
            ▼                                     ▼
  160×96 luma thumbnail                full-resolution copy
  (the only data read back)            (never leaves the GPU)
            │                                     │
            ▼                                     │
  3-level block-matching pyramid                  │
  finest level on the NPU (NNAPI)                 │
            │                                     │
            ▼                                     ▼
     20×12 motion field ──────────────► midpoint shader
                                                  │
                                                  ▼
                                 touch-transparent overlay above the game
```

1. **Capture.** `MediaProjection` mirrors a single app into a `SurfaceTexture`. Frames stay on the GPU; only a 160×96 luma thumbnail is read back to the CPU.
2. **Motion search** (`native/flow.h`). Coarse-to-fine 8×8 block matching over three pyramid levels. The cost volume of the finest level (240 blocks × 81 candidates) runs as an NNAPI graph on a MediaTek accelerator; the NNAPI CPU reference device is never used. The two small coarse levels run on the CPU, where they are cheaper than an NPU round trip. Reliable reach is roughly 10–12% of the frame width per frame.
3. **Generated frame** (`CaptureService.java`, `pMid` shader). The motion field is coarse, so around a moving object it also drags static background along. Each pixel therefore tries three candidates (the interpolated motion, its own block's motion, and no motion) and trusts whichever makes the two neighbouring frames agree. Where none agrees (newly revealed areas) it shows the newer frame instead of a ghosted blend.
4. **Presentation.** The generated frame is shown as soon as it is ready; the real frame follows half a frame interval later.

Further details:

- **Static content** such as HUDs stays still: a block only gets a motion vector if it clearly beats "no motion".
- **Scene cuts** are detected (more than 40% of blocks without a match) and no frame is generated across them.
- **Overload guard.** The code can pause generation when it would make the game itself slower (`GUARD` in `CaptureService.java`). It is **switched off** in the current build, so generation never pauses.
- **Modes.** `NPU` generates frames; `OFF` only passes the captured picture through, which is useful for comparisons. Switch in the app or with the in-game **MODE** button.
- **Metrics.** Per-second timings are written to `files/comparison.csv` in the app's private storage. No picture is ever stored.

This is **analytic block matching, not a trained interpolation network.** Expect artifacts on fast camera turns and around fast-moving objects.

## Measured results (TB373FU)

| Scenario | Source | Shown | Notes |
|---|---|---|---|
| 24 fps video | 24 | 48 | sustained |
| GTA Vice City – Definitive Edition, light scenes | 30 | 60 | motion search ≈ 3–4 ms, whole step ≈ 9–20 ms per generated frame |
| GTA Vice City – Definitive Edition, heavy scenes | 30 → ~20 | ~40 | the game already saturates the GPU; FrameLab's extra full-resolution passes slow the game down |

**The bottleneck is GPU time, not the NPU.** Capturing at full resolution, copying the frame and drawing two extra full-screen passes competes with the game for the same GPU. An experiment that moved the motion search from the NPU to a GPU compute shader made the whole step about twice as slow and was removed.

Whether `mtk-neuron_shim` (the accelerator that accepts the graph on this device) executes it on the NPU silicon itself is not visible to the app; the dedicated `mtk-mdla_shim` device rejects these operations.

## Where help is most useful

- **Lower GPU cost**: zero-copy access to the previous frame (`ImageReader` + `HardwareBuffer`) instead of a full-resolution copy.
- **Finer motion field** (for example 40×24 blocks) to shrink the halos around moving objects.
- **Other chipsets**: device selection currently accepts only NNAPI accelerators whose name contains `mtk`.
- **A trained interpolation model** that fits the time budget of a 30 fps game.

## Building

The build uses the Android SDK/NDK that ships with Unity plus a JDK; there is no Gradle project. Default paths are at the top of `build.ps1`; override them if yours differ:

```powershell
.\build.ps1 -AndroidRoot 'C:/path/to/AndroidPlayer' -JavaRoot 'C:/path/to/jdk'
```

The first build creates a local development signing key in `build/`; it is not part of the repository. Output: `build/FrameLab-0.3.apk`.

Requirements: Android 14 (API 34) or newer, arm64, and an NNAPI accelerator that supports `SUB`, `MUL` and `FULLY_CONNECTED` on float tensors.

### Device tests

`tests/flow_test.cpp` runs on the device and checks that the motion search recovers known shifts, keeps a static HUD still and detects a scene cut:

```powershell
$ndk='C:/path/to/AndroidPlayer/NDK/toolchains/llvm/prebuilt/windows-x86_64/bin'
& "$ndk/aarch64-linux-android34-clang++.cmd" -std=c++17 -O3 -static-libstdc++ tests/flow_test.cpp -lneuralnetworks -o flow-test
adb push flow-test /data/local/tmp/
adb shell "chmod 755 /data/local/tmp/flow-test && /data/local/tmp/flow-test"
```

`tests/motion_test.cpp` tests the older CPU-side `MotionEngine` in `native/engine.h`, which still backs the in-app NPU test and the demo screen.

## Using it

1. **Test the NPU** – runs the accelerator self-test.
2. **Allow the overlay** – enable the *FrameLab Experimental* accessibility service. It reads no screen content and sends no input; it only provides the overlay window.
3. **Try it in a game** – in Android's dialog choose **Share one app** and pick the game. Do not share the entire screen.

The line in the top left corner shows what is happening, for example:

```
FrameLab • NPU • captured 30/s • shown real 30 + generated 30/s • motion 3.5 ms • total 10.2 ms
```

Stop with **STOP** in the top right corner or from the notification.

Everything runs on the device. The app has no internet permission and records nothing.

## Known limitations

- Tested on one MediaTek device only; other NNAPI drivers may reject the graph.
- Games with protected (secure) surfaces are captured as black.
- Adds roughly half a frame of display latency; input latency is not improved.
- With the overload guard off, a GPU-bound game can run slower while FrameLab is active.
- Visual artifacts around fast-moving objects and on fast camera turns.

## License

MIT, see [LICENSE](LICENSE).
