# FrameLab

**Experimental NPU-assisted frame generation for Android games.** FrameLab captures one game's screen, uses the device's neural accelerator to estimate motion between two real frames, and inserts a generated in-between frame on the GPU. A 30 fps game can be shown at up to 60 fps.

> **Türkçe özet:** Android oyunlarında iki gerçek kare arasına NPU yardımıyla ara kare ekleyen deneysel bir uygulama. Hareket tahmini NPU'da, ara karenin çizimi ekran kartında yapılır. Ayrıntılar aşağıda.

This is a prototype, not a finished product. It was developed and tested on one device only: **Lenovo Tab (TB373FU), MediaTek MT8792 / mt6897, Android 16**. Contributions and reports from other devices are welcome.

## How it works

```
game ──MediaProjection──► SurfaceTexture (GPU)
                              │
            ┌─────────────────┴───────────────────┐
            ▼                                     ▼
  160×96 luma thumbnail                full-resolution copy
  (only data read back to CPU)         (stays on the GPU)
            │                                     │
            ▼                                     │
  3-level block-matching pyramid                  │
  finest level on the NPU (NNAPI)                 │
            │                                     │
            ▼                                     ▼
     20×12 motion field ──────────────► midpoint warp shader
                                                  │
                                                  ▼
                                accessibility overlay on top of the game
```

- **Motion search** (`native/flow.h`): coarse-to-fine 8×8 block matching over three pyramid levels. The finest level's cost volume (240 blocks × 81 candidates) runs as an NNAPI graph on a MediaTek accelerator; the NNAPI CPU fallback is never used. The two small coarse levels run on the CPU, where they are cheaper than an NPU round trip. Search reach is about ±28 px at 160 px width (~17% of the frame per frame).
- **Static content** such as HUDs is kept still: a block only gets a motion vector if it clearly beats "no motion".
- **Scene cuts** are detected (more than 40% of blocks without a match) and no frame is generated across them.
- **Rendering** (`CaptureService.java`): the generated frame samples both neighbours half way along the motion and averages them, then the real frame follows half an interval later.
- **Overload guard**: FrameLab first measures the game's own pace with frames passing straight through. If generating frames would make the game itself slower, generation pauses and is retried later (20 s, then 40 s, then 60 s).

This is **analytic block matching, not a trained interpolation network.** Expect ghosting or a plain blend on fast camera turns and newly revealed areas.

## Measured results (TB373FU)

| Scenario | Game | Shown | Notes |
|---|---|---|---|
| 24 fps video (VLC) | 24 | 48 | sustained |
| GTA Vice City Definitive, cool device | 30 | 60 | NPU ≈ 3–4 ms, total ≈ 9–20 ms per generated frame |
| GTA Vice City, after about a minute | 30 | 30 | overload guard active: the game's GPU load leaves no room for full-resolution capture + two extra draws |

The main limit is GPU time, not the NPU. Contributions that reduce GPU cost (fewer full-resolution passes, zero-copy capture) would help the most.

## Building

The build uses the Android SDK/NDK bundled with Unity and a JDK, with no Gradle. Default paths are at the top of `build.ps1`; override them if yours differ:

```powershell
.\build.ps1 -AndroidRoot 'C:/path/to/AndroidPlayer' -JavaRoot 'C:/path/to/jdk'
```

The first build creates a local development signing key in `build/`; it is not part of the repository. Output: `build/FrameLab-0.2.apk`. Requires Android 14 (API 34) or newer on an arm64 device with an NNAPI accelerator whose name contains `mtk`.

### Device tests

`tests/flow_test.cpp` checks on the device that the NPU pyramid recovers known shifts, keeps a static HUD still, and detects a scene cut:

```powershell
$ndk='C:/path/to/AndroidPlayer/NDK/toolchains/llvm/prebuilt/windows-x86_64/bin'
& "$ndk/aarch64-linux-android34-clang++.cmd" -std=c++17 -O3 -static-libstdc++ tests/flow_test.cpp -lneuralnetworks -o flow-test
adb push flow-test /data/local/tmp/ ; adb shell "chmod 755 /data/local/tmp/flow-test && /data/local/tmp/flow-test"
```

`tests/motion_test.cpp` tests the older CPU-side `MotionEngine` in `native/engine.h`, which still backs the in-app NPU test and the demo screen.

## Using it

1. **NPU'yu test et** – runs the accelerator self-test.
2. **Görüntü katmanına izin ver** – enable the *FrameLab Deneysel* accessibility service. It reads no screen content and sends no input; it only provides the overlay window.
3. **Bir oyunda dene** – choose **"Share one app"** and pick the game. Do not share the entire screen.

The top-left line shows `oyun X fps → ekran Y fps` while frames are generated, or `ara kare duraklatıldı` when the overload guard is active. Stop with **DURDUR** (top right) or from the notification.

Everything runs on the device. The app has no internet permission and records nothing.

## Known limitations

- Tested only on MediaTek MT8792; other NNAPI drivers may reject the graph.
- Games with protected (secure) surfaces capture as black.
- Adds about half a frame of display latency; input latency is not improved.
- Under sustained GPU load the overload guard keeps generation off.

## License

MIT, see [LICENSE](LICENSE).
