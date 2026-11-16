# FrameLab

**Experimental NPU-assisted frame generation and AI super resolution for Android games.**

FrameLab captures one game's screen, uses the device's neural accelerator (NPU) to estimate the motion between two real frames, and draws a generated in-between frame on the GPU. A game running at 30 fps can be shown at up to 60 fps, at the full resolution of the screen.

Its second mode, **SR**, is a x2 AI super resolution: the game is captured at half resolution per axis and a small trained neural network on the NPU restores the full-resolution detail (for example 1472×920 → 2944×1840, or 1920×1080 → 3840×2160 on a 4K screen). See [AI super resolution](#ai-super-resolution-sr-mode).

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
- **Modes.** `NPU` generates frames; `SR` upscales with the AI network (below); `OFF` only passes the captured picture through, which is useful for comparisons. Switch in the app or with the in-game **MODE** button, which cycles NPU → SR → OFF.
- **Metrics.** Per-second timings are written to `files/comparison.csv` in the app's private storage. No picture is ever stored.

This is **analytic block matching, not a trained interpolation network.** Expect artifacts on fast camera turns and around fast-moving objects.

## AI super resolution (SR mode)

```
game ──MediaProjection, half size──► SurfaceTexture (GPU), e.g. 1472×920
                                          │
             ┌────────────────────────────┴───────────────────────┐
             ▼                                                    │
   luma, 1 byte per pixel ──readback──► NPU: trained CNN          │ colour stays on the GPU
   (written straight into the           4 × conv 3×3, int8        │
    NPU's shared input memory)                 │                  │
                                               ▼                  ▼
                              detail for each 2×2 output pixel ──► one full-size pass:
                              (4 bytes per low-res pixel)          bilinear colour + AI detail
                                                                   e.g. 2944×1840 (4K on a 4K screen)
```

- **Network** (`native/sr_model.h`, trained by `tools/train_sr.py`). Four 3×3 convolutions, 16 channels, ReLU, about 5,400 parameters and 5,300 multiply-adds per low-res pixel. It predicts what bilinear upscaling gets wrong, for each of the four output pixels a low-res pixel covers (the "sub-pixel phases"). The GPU adds that detail to its own bilinear upscale of the colour frame, so only luma travels to the NPU and back. A smaller 8-channel model (1,500 multiply-adds per pixel) is used when the larger one needs more than 20 ms per frame.
- **On the NPU.** The graph is plain NNAPI `CONV_2D`. FrameLab tries int8 with per-channel weights first, then int8 per-tensor, float16 and relaxed float32, on each neural accelerator (MDLA first on MediaTek); the NNAPI CPU reference and the GPU are never used. Every candidate is compared with a CPU reference of the same arithmetic before it is trusted; a driver that computes something else is skipped.
- **Training** is quantization-aware: the last quarter of training runs with the exact int8 rounding of the NPU graph, so the int8 result matches the float model. No deep-learning framework is needed, only numpy and OpenCV:

  ```powershell
  python tools/train_sr.py --name sr16 --chans 1,16,16,16,4 --images "C:/frames/*.png@8"
  python tools/train_sr.py --name sr8  --chans 1,8,8,8,4   --images "C:/frames/*.png@8"
  python tools/train_sr.py --export sr16=build/sr/sr16.npz sr8=build/sr/sr8.npz
  ```

  Training data: native-resolution game frames you pass with `--images` (the right quarter of each is held out for validation), the Windows wallpapers in `C:/Windows/Web`, and synthetic game-HUD images (text, digits, rings, lines). Low-res inputs are 2×2 averages of those frames, plus some that simulate a game rendering below native resolution.
- **Where it helps.** If the game renders at native resolution, a half-size capture throws detail away and SR only gets part of it back. SR is meant for games that already render at a lower resolution (a resolution setting, an emulator's internal resolution): the half-size capture then loses little, and the GPU has less to do than in `OFF` mode at full size.
- **Compare.** Long-press **MODE** in SR mode: the left half shows plain bilinear upscaling, the right half the AI result.

Quality of the shipped weights, luma PSNR against the native-resolution frame, computed on the PC with a bit-exact simulation of the int8 NPU graph (higher is better):

| Content | Bilinear x2 | Bicubic x2 | AI x2, 16-ch | AI x2, 8-ch |
|---|---|---|---|---|
| Validation set: held-out photos, game and app frames, synthetic HUD | 36.12 dB | 37.65 dB | **40.85 dB** | 40.20 dB |
| App text (held-out quarter of a TB373FU screenshot) | 30.76 dB | – | **39.96 dB** | 38.07 dB |
| GTA Vice City frame (held-out quarter) | 42.69 dB | – | 42.03 dB | 41.84 dB |

The GTA frame is the exception: that screenshot is already soft, i.e. it was rendered or captured below the panel resolution and upscaled (only 0.06% of its spectral energy lies above half the Nyquist frequency, against 1.3% for the app screenshot). A half-size capture of it loses almost nothing, and the "native" reference is itself a soft upscale. The AI result is sharper than that reference, which PSNR counts as a difference.

Measured on the TB373FU (Android 16). The NPU itself (`mtk-mdla_shim`; `mtk-neuron_shim` gives the same times) runs the int8 per-channel graph:

| Input → output | NPU time per frame (`sr-test` / in-app test) |
|---|---|
| 1472×920 → 2944×1840 (this panel) | 4.3 / 4.9 ms |
| 1920×1080 → 3840×2160 (4K UHD) | 6.3 / 6.7 ms |
| 1920×1200 → 3840×2400 | 7.0 / 7.3 ms |

- On the device the NPU output matches the CPU reference of the same int8 arithmetic to within one step, and the PC simulation to within two steps of 255 (mean difference 0.035). The device PSNR equals the PC simulation: 28.16 → 37.13 dB on the app screenshot.
- The float16 graph also runs on the MDLA but takes twice as long (8.9 ms); `mtk-dsp_shim` accepts none of the graphs. The 8-channel model is hardly faster (3.8 ms), so the 16-channel model is always chosen here.
- In GTA Vice City – Definitive Edition at 30 fps, every game frame is upscaled (30 of 30 shown) while the tablet is cool. The NPU takes 4.8–7.6 ms, the whole step 11–25 ms. The largest share is waiting for the GPU to deliver the low-res luma: the readback takes 2.5–13 ms while the game keeps the GPU busy. Uploading the detail takes 0.5–0.9 ms.
- After a minute or two of play the tablet runs hot (thermal status 2, CPU/GPU/NPU 67 °C). The NPU then takes 8 ms and the readback 27–38 ms, so SR shows 16–24 fps instead of 30. A ten-minute session in that state ran without a stop.

`tools/sr_compare.py` makes side-by-side crops and PSNR from a native-resolution frame, either from the device output of `tests/sr_test.cpp` or from a simulation of the int8 graph on the PC.

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
- **Asynchronous readback for SR** (pixel buffer object + fence, or `ImageReader` with luma on the CPU): the NPU itself needs 5–8 ms, but waiting for the GPU's luma readback takes up to 38 ms in a busy game on a hot tablet.

## Building

The build uses the Android SDK/NDK that ships with Unity plus a JDK; there is no Gradle project. Default paths are at the top of `build.ps1`; override them if yours differ:

```powershell
.\build.ps1 -AndroidRoot 'C:/path/to/AndroidPlayer' -JavaRoot 'C:/path/to/jdk'
```

The first build creates a local development signing key in `build/`; it is not part of the repository. Output: `build/FrameLab-0.4.apk`.

Requirements: Android 14 (API 34) or newer, arm64, and an NNAPI accelerator that supports `SUB`, `MUL` and `FULLY_CONNECTED` on float tensors (frame generation) or `CONV_2D` (super resolution).

### Device tests

`tests/flow_test.cpp` runs on the device and checks that the motion search recovers known shifts, keeps a static HUD still and detects a scene cut:

```powershell
$ndk='C:/path/to/AndroidPlayer/NDK/toolchains/llvm/prebuilt/windows-x86_64/bin'
& "$ndk/aarch64-linux-android34-clang++.cmd" -std=c++17 -O3 -static-libstdc++ tests/flow_test.cpp -lneuralnetworks -o flow-test
adb push flow-test /data/local/tmp/
adb shell "chmod 755 /data/local/tmp/flow-test && /data/local/tmp/flow-test"
```

`tests/motion_test.cpp` tests the older CPU-side `MotionEngine` in `native/engine.h`, which still backs the in-app NPU test and the demo screen.

`tests/sr_test.cpp` lists which accelerator accepts which super-resolution graph, checks each against the CPU reference, times it (including 4K output sizes) and, given a native-resolution frame as raw 8-bit luma, measures how much closer the NPU result is to it than bilinear upscaling:

```powershell
& "$ndk/aarch64-linux-android34-clang++.cmd" -std=c++17 -O3 -static-libstdc++ tests/sr_test.cpp -lneuralnetworks -landroid -o sr-test
adb push sr-test frame.y /data/local/tmp/
adb shell "chmod 755 /data/local/tmp/sr-test && /data/local/tmp/sr-test /data/local/tmp/frame.y 2944 1840"
adb pull /data/local/tmp/sr-out-sr16.y
python tools/sr_compare.py frame.png --device sr-out-sr16.y --out compare.png
```

## Using it

1. **Test the NPU** – runs the accelerator self-test. **Test AI super resolution** compiles and checks the SR network for this screen and times it, also at 4K output sizes.
2. **Allow the overlay** – enable the *FrameLab Experimental* accessibility service. It reads no screen content and sends no input; it only provides the overlay window.
3. **Try it in a game** – in Android's dialog choose **Share one app** and pick the game. Do not share the entire screen.

The line in the top left corner shows what is happening, for example:

```
FrameLab • NPU • captured 30/s • shown real 30 + generated 30/s • motion 3.5 ms • total 10.2 ms
FrameLab • SR ×2 AI • 1472x920 → 2944x1840 • captured 30/s • shown 30/s • NPU 5.3 ms • total 21.7 ms
```

Tap **MODE** to switch NPU → SR → OFF; in SR, long-press it for the split-screen comparison. Stop with **STOP** in the top right corner or from the notification.

Everything runs on the device. The app has no internet permission and records nothing.

## Known limitations

- Tested on one MediaTek device only; other NNAPI drivers may reject the graph.
- Games with protected (secure) surfaces are captured as black.
- Adds roughly half a frame of display latency; input latency is not improved.
- With the overload guard off, a GPU-bound game can run slower while FrameLab is active.
- Visual artifacts around fast-moving objects and on fast camera turns.
- SR restores luma detail only; colour edges are bilinear. It cannot recover detail a half-size capture of a native-resolution game has lost, and it adds one NPU pass of display latency.

## License

MIT, see [LICENSE](LICENSE).
