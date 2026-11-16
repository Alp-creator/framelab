#!/usr/bin/env python3
"""Side-by-side crops and PSNR for FrameLab's x2 super resolution.

The reference is a native-resolution frame; the low-res input is its 2x2 average (what a half-size
capture of that frame gives). The AI result comes either from the device (raw luma written by
tests/sr_test.cpp) or from a simulation of the int8 NPU graph on the PC.

    python tools/sr_compare.py frame.png --device sr-out-sr16.y --out compare.png
    python tools/sr_compare.py frame.png --weights build/sr/sr16.npz --out compare.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sr as T  # noqa: E402


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", help="native-resolution reference frame (PNG)")
    ap.add_argument("--device", help="raw 8-bit luma result from sr-test (2W x 2H of the low-res input)")
    ap.add_argument("--weights", help="checkpoint to simulate the int8 NPU graph instead")
    ap.add_argument("--out", default="sr-compare.png")
    ap.add_argument("--crops", default="2240,120,320,240;140,120,320,240;1250,900,320,240;2060,420,320,240",
                    help="x,y,w,h;... in reference pixels")
    ap.add_argument("--holdout", type=float, default=0.75, help="report PSNR also for x >= this fraction (not used in training)")
    args = ap.parse_args()

    bgr = cv2.imread(args.frame, cv2.IMREAD_COLOR)
    w, h = bgr.shape[1] // 2 // 4 * 4, bgr.shape[0] // 2 // 2 * 2  # low-res size, as in sr-test
    bgr = bgr[: 2 * h, : 2 * w]
    luma = np.clip(np.round(bgr.astype(np.float32) @ T.LUMA), 0, 255).astype(np.uint8)
    lr = (luma.reshape(h, 2, w, 2).astype(np.int32).sum(axis=(1, 3)) + 2) // 4  # the same rounding as sr-test
    lin = T.up2(lr.astype(np.float32)[None] / 255.0)[0] * 255.0
    if args.device:
        ai = np.fromfile(args.device, np.uint8).reshape(2 * h, 2 * w).astype(np.float32)
        source = "NPU"
    else:
        net = T.load_weights(args.weights)
        x = lr.astype(np.float32)[None, ..., None] / 255.0
        rows = []
        for y0 in range(0, h, 128):  # in strips, with a margin so the receptive field is complete
            a, b = max(0, y0 - 8), min(h, y0 + 128 + 8)
            ph = net.forward_int8(x[:, a:b])
            rows.append(ph[:, y0 - a: y0 - a + min(128, h - y0)])
        detail = T.from_phases(np.concatenate(rows, axis=1))[0] * 255.0
        ai = np.clip(np.round(lin + detail), 0, 255)
        source = "int8 simulation"
    lin8 = np.clip(np.round(lin), 0, 255)
    x0 = int(2 * w * args.holdout) // 2 * 2
    print("luma PSNR vs native, whole frame: bilinear %.2f dB, AI (%s) %.2f dB" % (psnr(lin8, luma), source, psnr(ai, luma)))
    print("luma PSNR vs native, held-out x >= %d: bilinear %.2f dB, AI %.2f dB" % (x0, psnr(lin8[:, x0:], luma[:, x0:]), psnr(ai[:, x0:], luma[:, x0:])))

    # Colour the way the GPU does it: bilinear colour plus the AI luma detail on all three channels.
    lr_bgr = bgr.reshape(h, 2, w, 2, 3).astype(np.float32).mean(axis=(1, 3))
    up = cv2.resize(lr_bgr, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR)
    ai_bgr = np.clip(up + (ai - lin)[..., None], 0, 255).astype(np.uint8)
    lin_bgr = np.clip(up, 0, 255).astype(np.uint8)
    panels = []
    for spec in args.crops.split(";"):
        cx, cy, cw, ch = (int(v) for v in spec.split(","))
        row = []
        for img, label in ((lin_bgr, "bilinear x2 (GPU)"), (ai_bgr, "AI x2 (" + source + ")"), (bgr, "native reference")):
            c = cv2.resize(img[cy:cy + ch, cx:cx + cw], (2 * cw, 2 * ch), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(c, (0, 0), (2 * cw, 34), (0, 0, 0), -1)
            cv2.putText(c, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            row.append(c)
            row.append(np.full((2 * ch, 6, 3), 255, np.uint8))
        panels.append(np.concatenate(row[:-1], axis=1))
        panels.append(np.full((6, panels[-1].shape[1], 3), 255, np.uint8))
    cv2.imwrite(args.out, np.concatenate(panels[:-1], axis=0))
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
