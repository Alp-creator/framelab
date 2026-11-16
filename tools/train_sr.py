#!/usr/bin/env python3
"""Train FrameLab's x2 AI super-resolution network with numpy only (no GPU, no deep-learning framework).

The network predicts the luma detail that bilinear upscaling misses. On the device the GPU draws the
bilinear upscale of the captured colour frame and adds this detail; the NPU runs the network:

    low-res luma [H,W,1] -> 3x3 conv+ReLU -> ... -> 3x3 conv -> 4 phases [H,W,4]
    phase 2*dy+dx at (y,x) = HR(2y+dy, 2x+dx) - bilinear(LR)(2y+dy, 2x+dx)

Training is quantization-aware at the end, so the int8 graph on the NPU matches the float model.
Writes native/sr_model.h (float + int8 weights) and prints PSNR against plain bilinear upscaling.

    python tools/train_sr.py --images "C:/path/to/game-screenshots/*.png" --iters 12000
"""
import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np

LUMA = np.array([0.114, 0.587, 0.299], np.float32)  # BGR order (OpenCV), BT.601 like the shader
OUT_SCALE = 1.0 / 255.0                             # residual quantization, zero point 128
OUT_ZERO = 128

# --------------------------------------------------------------------------------------------- data

def load_luma(path, halve=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("cannot read " + path)
    if halve:  # photos/JPEGs: downsample first so every pixel carries real detail, not JPEG blur
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    y = img.astype(np.float32) @ LUMA / 255.0
    return np.ascontiguousarray(y[: y.shape[0] // 2 * 2, : y.shape[1] // 2 * 2])


def synthetic_hud(seed, w=1536, h=960):
    """Game-HUD-like content: anti-aliased text, digits, rings, icons and lines over soft backgrounds."""
    rng = np.random.default_rng(seed)
    W, H = 2 * w, 2 * h  # drawn at 2x and area-downsampled, like a game rendering with AA
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        a, b, d = rng.uniform(-1, 1, 3)
        bg[..., c] = rng.uniform(0, 160) + 60 * (a * xx / W + b * yy / H) + 30 * d
    noise = cv2.resize(rng.uniform(0, 1, (H // 64, W // 64, 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = np.clip(bg + 80 * (noise - 0.5), 0, 255).astype(np.uint8)
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX_SMALL]
    words = ["FrameLab", "MISSION PASSED", "$420135152", "01:38", "200", "021", "AMMO 30/120", "SCORE",
             "Vice", "LAP 2/3", "km/h", "HP", "MAP", "SPEED 87", "x2", "READY", "NEW RECORD", "1080p"]

    def color():
        return tuple(int(v) for v in rng.integers(0, 256, 3))

    for _ in range(int(rng.integers(90, 140))):
        kind = rng.random()
        p = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        if kind < 0.5:
            text = words[int(rng.integers(len(words)))] if rng.random() < 0.6 else str(int(rng.integers(0, 10 ** int(rng.integers(1, 9)))))
            scale = float(rng.uniform(0.7, 5.0))
            thick = int(rng.integers(1, 8))
            if rng.random() < 0.4:  # outlined HUD text
                cv2.putText(img, text, p, fonts[int(rng.integers(len(fonts)))], scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
            cv2.putText(img, text, p, fonts[int(rng.integers(len(fonts)))], scale, color(), thick, cv2.LINE_AA)
        elif kind < 0.65:
            cv2.circle(img, p, int(rng.integers(8, 220)), color(), int(rng.integers(-1, 14)), cv2.LINE_AA)
        elif kind < 0.8:
            q = (p[0] + int(rng.integers(-500, 500)), p[1] + int(rng.integers(-500, 500)))
            cv2.line(img, p, q, color(), int(rng.integers(1, 10)), cv2.LINE_AA)
        elif kind < 0.9:
            pts = (np.array(p) + rng.integers(-160, 160, (int(rng.integers(3, 7)), 2))).astype(np.int32)
            cv2.fillPoly(img, [pts], color(), cv2.LINE_AA)
        else:
            q = (p[0] + int(rng.integers(20, 400)), p[1] + int(rng.integers(20, 300)))
            cv2.rectangle(img, p, q, color(), int(rng.integers(-1, 10)), cv2.LINE_AA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img.astype(np.float32) @ LUMA / 255.0)


def box2(hr):  # [..., 2H, 2W] -> [..., H, W]
    return 0.25 * (hr[..., 0::2, 0::2] + hr[..., 1::2, 0::2] + hr[..., 0::2, 1::2] + hr[..., 1::2, 1::2])


def up2(lr):
    """Bilinear x2 with half-pixel centres and clamp-to-edge: exactly what GL_LINEAR sampling draws."""
    p = np.concatenate([lr[..., :1, :], lr, lr[..., -1:, :]], axis=-2)
    a, b, c = p[..., :-2, :], p[..., 1:-1, :], p[..., 2:, :]
    rows = np.empty(lr.shape[:-2] + (2 * lr.shape[-2], lr.shape[-1]), np.float32)
    rows[..., 0::2, :] = 0.25 * a + 0.75 * b
    rows[..., 1::2, :] = 0.75 * b + 0.25 * c
    p = np.concatenate([rows[..., :1], rows, rows[..., -1:]], axis=-1)
    a, b, c = p[..., :-2], p[..., 1:-1], p[..., 2:]
    out = np.empty(rows.shape[:-1] + (2 * rows.shape[-1],), np.float32)
    out[..., 0::2] = 0.25 * a + 0.75 * b
    out[..., 1::2] = 0.75 * b + 0.25 * c
    return out


def to_phases(res):  # [B, 2P, 2Q] -> [B, P, Q, 4], channel 2*dy+dx
    B, H2, W2 = res.shape
    return np.ascontiguousarray(res.reshape(B, H2 // 2, 2, W2 // 2, 2).transpose(0, 1, 3, 2, 4).reshape(B, H2 // 2, W2 // 2, 4))


def from_phases(ph):  # inverse of to_phases
    B, H, W, _ = ph.shape
    return ph.reshape(B, H, W, 2, 2).transpose(0, 1, 3, 2, 4).reshape(B, 2 * H, 2 * W)


def degrade(hr, rng):
    """Low-res capture of a high-res frame. Mostly an exact 2x2 average (what a half-size capture of a
    native-resolution game gives); sometimes a game that rendered below native and was rescaled."""
    if rng.random() < 0.75:
        lr = box2(hr)
    else:
        H, W = hr.shape
        r = rng.uniform(0.5, 0.95)
        mid = cv2.resize(hr, (max(W // 2, int(W * r)), max(H // 2, int(H * r))), interpolation=cv2.INTER_AREA)
        lr = cv2.resize(mid, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
    return lr


class Data:
    def __init__(self, images, rng):
        self.images = images  # list of (luma, weight, x_limit_fraction)
        w = np.array([i[1] for i in images], np.float64)
        self.p = w / w.sum()
        self.rng = rng

    def crop(self, hr_size):
        rng = self.rng
        img, _, xlim = self.images[rng.choice(len(self.images), p=self.p)]
        H, W = img.shape
        wmax = int(W * xlim)
        y = int(rng.integers(0, H - hr_size + 1)) // 2 * 2
        x = int(rng.integers(0, wmax - hr_size + 1)) // 2 * 2
        c = img[y:y + hr_size, x:x + hr_size]
        if rng.random() < 0.5:
            c = c[:, ::-1]
        if rng.random() < 0.5:
            c = c[::-1, :]
        if rng.random() < 0.5:
            c = c.T
        if rng.random() < 0.3:  # brightness/contrast variety (dark night scenes, bright menus)
            c = np.clip((c - 0.5) * rng.uniform(0.6, 1.3) + 0.5 + rng.uniform(-0.25, 0.25), 0, 1)
        return np.ascontiguousarray(c, np.float32)

    def batch(self, n, lr_size):
        hr = np.stack([self.crop(2 * lr_size) for _ in range(n)])
        lr = np.stack([degrade(h, self.rng) for h in hr])
        return make_sample(hr, lr)


def make_sample(hr, lr):
    lr8 = np.round(np.clip(lr, 0, 1) * 255) / 255.0       # the device reads 8-bit luma
    base = up2(lr8.astype(np.float32))                    # the GPU's bilinear upscale
    target = to_phases(hr - base)
    return lr8.astype(np.float32)[..., None], target.astype(np.float32), hr, base

# --------------------------------------------------------------------------------------------- model


def im2col3(x):
    """[B,H,W,C] -> [B*H*W, 9*C], 3x3 neighbourhood with zero padding, order (dy, dx, c)."""
    B, H, W, C = x.shape
    xp = np.zeros((B, H + 2, W + 2, C), np.float32)
    xp[:, 1:-1, 1:-1] = x
    cols = np.empty((B, H, W, 3, 3, C), np.float32)
    for dy in range(3):
        for dx in range(3):
            cols[:, :, :, dy, dx, :] = xp[:, dy:dy + H, dx:dx + W, :]
    return cols.reshape(B * H * W, 9 * C)


def col2im3(d, B, H, W, C):
    d = d.reshape(B, H, W, 3, 3, C)
    xp = np.zeros((B, H + 2, W + 2, C), np.float32)
    for dy in range(3):
        for dx in range(3):
            xp[:, dy:dy + H, dx:dx + W, :] += d[:, :, :, dy, dx, :]
    return xp[:, 1:-1, 1:-1, :]


def quant_w(w):
    """Symmetric per-output-channel int8: what the NPU graph uses. w is [9*Cin, Cout]."""
    s = np.maximum(np.abs(w).max(axis=0), 1e-8) / 127.0
    q = np.clip(np.round(w / s), -127, 127)
    return q, s.astype(np.float32)


class Net:
    def __init__(self, chans, seed):
        rng = np.random.default_rng(seed)
        self.chans = chans
        self.W, self.b = [], []
        for i in range(len(chans) - 1):
            cin, cout = chans[i], chans[i + 1]
            std = math.sqrt(2.0 / (9 * cin)) if i < len(chans) - 2 else 1e-3
            self.W.append((rng.standard_normal((9 * cin, cout)) * std).astype(np.float32))
            self.b.append(np.zeros(cout, np.float32))
        self.act = None  # per-layer activation scales once quantization-aware training starts
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    @property
    def layers(self):
        return len(self.W)

    def forward(self, x, keep=True):
        B, H, Wd, _ = x.shape
        cache, a = [], x
        for i in range(self.layers):
            w = self.W[i]
            if self.act is not None:
                q, s = quant_w(w)
                w = (q * s).astype(np.float32)
            cols = im2col3(a)
            z = cols @ w + self.b[i]
            if i < self.layers - 1:
                if self.act is not None:
                    s = self.act[i]
                    mask = (z > 0) & (z < 255 * s)
                    out = np.clip(np.round(z / s), 0, 255) * s
                else:
                    mask = z > 0
                    out = z * mask
                a = out.astype(np.float32).reshape(B, H, Wd, -1)
            else:
                if self.act is not None:
                    mask = np.abs(z) < 0.5
                    z = (np.clip(np.round(z / OUT_SCALE) + OUT_ZERO, 0, 255) - OUT_ZERO) * OUT_SCALE
                else:
                    mask = None
                a = z.astype(np.float32).reshape(B, H, Wd, -1)
            if keep:
                cache.append((cols, mask, w, a.shape[-1] if i < self.layers - 1 else None))
        return a, cache

    def backward(self, cache, dout, shape):
        B, H, Wd = shape
        grads_w, grads_b = [None] * self.layers, [None] * self.layers
        d = dout.reshape(B * H * Wd, -1)
        for i in reversed(range(self.layers)):
            cols, mask, w, _ = cache[i]
            if mask is not None:
                d = d * mask
            grads_w[i] = cols.T @ d
            grads_b[i] = d.sum(0)
            if i > 0:
                d = col2im3(d @ w.T, B, H, Wd, self.chans[i]).reshape(B * H * Wd, -1)
        return grads_w, grads_b

    def step(self, gw, gb, lr, wd=0.0):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params, grads = self.W + self.b, gw + gb
        corr = math.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for k, (p, g) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            p -= lr * corr * self.m[k] / (np.sqrt(self.v[k]) + eps)

    def calibrate(self, xs, pct=99.99):
        """Activation scales for the int8 graph from real data (percentile, not max, to ignore outliers)."""
        acts = [[] for _ in range(self.layers - 1)]
        for x in xs:
            a = x
            B, H, Wd, _ = x.shape
            for i in range(self.layers - 1):
                a = np.maximum(im2col3(a) @ self.W[i] + self.b[i], 0).reshape(B, H, Wd, -1)
                acts[i].append(a.ravel()[::7])
        self.act = [max(float(np.percentile(np.concatenate(v), pct)), 1e-3) / 255.0 for v in acts]

    def forward_int8(self, x):
        """Bit-level simulation of the NPU int8 graph (per-channel weights, zero padding, fused ReLU)."""
        B, H, Wd, _ = x.shape
        a, s_in = np.round(x * 255), 1.0 / 255.0
        for i in range(self.layers):
            q, sw = quant_w(self.W[i])
            bias = np.round(self.b[i] / (s_in * sw))
            acc = im2col3(a.astype(np.float32)).astype(np.float64) @ q.astype(np.float64) + bias  # exact integers
            last = i == self.layers - 1
            s_out = OUT_SCALE if last else self.act[i]
            zero = OUT_ZERO if last else 0
            out = np.floor(acc * (s_in * sw / s_out) + 0.5) + zero
            a = np.clip(out, 0, 255).reshape(B, H, Wd, -1)
            s_in = s_out
        return ((a - OUT_ZERO) * OUT_SCALE).astype(np.float32)

    def macs(self):
        return sum(9 * self.chans[i] * self.chans[i + 1] for i in range(self.layers))

    def params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

# --------------------------------------------------------------------------------------------- eval / export


def psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 1) - b) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def evaluate(net, val, int8=False):
    lr8, _, hr, base = val
    out = []
    for k in range(0, len(lr8), 8):
        ph = net.forward_int8(lr8[k:k + 8]) if int8 else net.forward(lr8[k:k + 8], keep=False)[0]
        out.append(base[k:k + 8] + from_phases(ph))
    sr = np.concatenate(out)
    return psnr(sr, hr), psnr(base, hr)


def bicubic_psnr(val):
    lr8, _, hr, _ = val
    up = np.stack([cv2.resize(l[..., 0], (l.shape[1] * 2, l.shape[0] * 2), interpolation=cv2.INTER_CUBIC) for l in lr8])
    return psnr(up, hr)


def ohwi(w, cin, cout):  # [9*Cin, Cout] (dy,dx,c) -> NNAPI CONV_2D filter [Cout, 3, 3, Cin]
    return np.ascontiguousarray(w.reshape(3, 3, cin, cout).transpose(3, 0, 1, 2))


def c_array(kind, name, values, fmt):
    values = np.asarray(values).ravel()
    lines = [", ".join(fmt % v for v in values[i:i + 12]) for i in range(0, len(values), 12)]
    return "static const %s %s[%d] = {\n    %s};\n" % (kind, name, len(values), ",\n    ".join(lines))


def export(models, path, info):
    out = ["// Generated by tools/train_sr.py. Do not edit by hand.",
           "// x2 luma super resolution: phase 2*dy+dx = HR(2y+dy,2x+dx) - bilinear(LR), quantized 128 + 255*r.",
           "#pragma once", "#include <cstdint>", "", "namespace srmodel {", "",
           "struct Layer { int cin, cout; bool relu; const float* w; const float* b; const int8_t* qw; const float* qs; const int32_t* qb; float outScale; int outZero; };",
           "struct Model { const char* name; int layers; const Layer* layer; int macsPerPixel; const char* info; };", ""]
    names = []
    for name, net in models:
        names.append(name)
        s_in = 1.0 / 255.0
        layer_rows = []
        for i in range(net.layers):
            cin, cout = net.chans[i], net.chans[i + 1]
            q, sw = quant_w(net.W[i])
            qb = np.round(net.b[i] / (s_in * sw)).astype(np.int64)
            last = i == net.layers - 1
            s_out = OUT_SCALE if last else net.act[i]
            p = "%s_%d" % (name, i)
            out.append(c_array("float", p + "_w", ohwi(net.W[i], cin, cout), "%.9g"))
            out.append(c_array("float", p + "_b", net.b[i], "%.9g"))
            out.append(c_array("int8_t", p + "_qw", ohwi(q, cin, cout).astype(np.int64), "%d"))
            out.append(c_array("float", p + "_qs", sw, "%.9g"))
            out.append(c_array("int32_t", p + "_qb", qb, "%d"))
            layer_rows.append("    {%d, %d, %s, %s_w, %s_b, %s_qw, %s_qs, %s_qb, %.9gf, %d}," % (
                cin, cout, "false" if last else "true", p, p, p, p, p, s_out, OUT_ZERO if last else 0))
            s_in = s_out
        out.append("static const Layer %s_layers[%d] = {\n%s\n};" % (name, net.layers, "\n".join(layer_rows)))
        out.append('static const Model %s = {"%s", %d, %s_layers, %d, "%s"};\n' % (name, name, net.layers, name, net.macs(), info[name]))
    out.append("static const Model* const MODELS[] = {%s};" % ", ".join("&" + n for n in names))
    out.append("static constexpr int MODEL_COUNT = %d;" % len(names))
    out.append("")
    out.append("}  // namespace srmodel")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def save_weights(net, path):
    np.savez(path, chans=np.array(net.chans), act=np.array(net.act if net.act else []),
             **{"W%d" % i: w for i, w in enumerate(net.W)}, **{"b%d" % i: b for i, b in enumerate(net.b)})


def load_weights(path):
    z = np.load(path)
    net = Net([int(c) for c in z["chans"]], 0)
    net.W = [z["W%d" % i].astype(np.float32) for i in range(net.layers)]
    net.b = [z["b%d" % i].astype(np.float32) for i in range(net.layers)]
    net.act = [float(a) for a in z["act"]] or None
    return net

# --------------------------------------------------------------------------------------------- main


def build_data(args):
    imgs, val_imgs = [], []
    for spec in args.images:
        pattern, _, weight = spec.partition("@")  # "frames/*.png@8": sampling weight per image
        for f in sorted(glob.glob(pattern)):
            y = load_luma(f)
            if y.std() < 0.02:  # black/secure captures carry nothing
                continue
            imgs.append((y, float(weight or 6), 0.75))  # game frames: the right quarter is held out
            val_imgs.append(y[:, int(y.shape[1] * 0.75) // 2 * 2:])
            print("game/app frame %s %s" % (os.path.basename(f), y.shape))
    photos = sorted(glob.glob(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Web", "**", "*.jpg"), recursive=True))
    photos = [f for f in photos if os.path.getsize(f) > 200_000]
    for k, f in enumerate(photos):
        y = load_luma(f, halve=True)
        if k % 6 == 5:
            val_imgs.append(y)
        else:
            imgs.append((y, 1.0, 1.0))
    print("photos: %d" % len(photos))
    for s in range(6):
        imgs.append((synthetic_hud(100 + s), 2.0, 1.0))
    val_imgs.append(synthetic_hud(999))
    return imgs, val_imgs


def validation_set(val_imgs, size=256, per_image=6, seed=7):
    rng = np.random.default_rng(seed)
    hr = []
    for img in val_imgs:
        H, W = img.shape
        for _ in range(per_image):
            y = int(rng.integers(0, H - size + 1)) // 2 * 2
            x = int(rng.integers(0, W - size + 1)) // 2 * 2
            hr.append(img[y:y + size, x:x + size])
    hr = np.stack(hr).astype(np.float32)
    return make_sample(hr, box2(hr))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[], help="extra high-resolution frames (game screenshots), glob patterns")
    ap.add_argument("--chans", default="1,16,16,16,4", help="layer widths, first 1 and last 4")
    ap.add_argument("--name", default="sr16")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--qat", type=float, default=0.25, help="fraction of iterations with int8 fake quantization")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--patch", type=int, default=48, help="low-res patch size")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="build/sr", help="checkpoint directory")
    ap.add_argument("--export", nargs="*", help="only export: name=checkpoint.npz pairs into --header")
    ap.add_argument("--header", default="native/sr_model.h")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.export:
        imgs, val_imgs = build_data(args)
        val = validation_set(val_imgs)
        models, info = [], {}
        print("validation: bilinear %.2f dB, bicubic %.2f dB" % (evaluate(load_weights(args.export[0].split("=")[1]), val)[1], bicubic_psnr(val)))
        for pair in args.export:
            name, path = pair.split("=")
            net = load_weights(path)
            f, b = evaluate(net, val)
            q, _ = evaluate(net, val, int8=True)
            info[name] = "%d params, %d MAC/px; validation PSNR: bilinear %.2f dB, float %.2f dB, int8 %.2f dB" % (net.params(), net.macs(), b, f, q)
            print(name, info[name])
            models.append((name, net))
        export(models, args.header, info)
        print("wrote " + args.header)
        return

    chans = [int(c) for c in args.chans.split(",")]
    assert chans[0] == 1 and chans[-1] == 4
    imgs, val_imgs = build_data(args)
    rng = np.random.default_rng(args.seed)
    data = Data(imgs, rng)
    val = validation_set(val_imgs)
    net = Net(chans, args.seed)
    f0, b0 = evaluate(net, val)
    print("%s: %d params, %d MAC per low-res pixel. validation: bilinear %.2f dB, bicubic %.2f dB" % (
        args.name, net.params(), net.macs(), b0, bicubic_psnr(val)), flush=True)
    qat_start = int(args.iters * (1 - args.qat))
    t0, eps, run = time.time(), 1e-3, 0.0
    best = -1.0
    for it in range(1, args.iters + 1):
        if it == qat_start + 1:
            net.calibrate([data.batch(args.batch, args.patch)[0] for _ in range(24)])
            print("int8 activation scales:", ["%.4g" % s for s in net.act], flush=True)
            net.m = [np.zeros_like(p) for p in net.W + net.b]
            net.v = [np.zeros_like(p) for p in net.W + net.b]
            net.t = 0
        x, t, _, _ = data.batch(args.batch, args.patch)
        y, cache = net.forward(x)
        diff = y - t
        loss = np.sqrt(diff * diff + eps * eps)
        run = 0.98 * run + 0.02 * float(loss.mean()) if it > 1 else float(loss.mean())
        grad = (diff / loss) / diff.size
        gw, gb = net.backward(cache, grad, x.shape[:3])
        if it <= qat_start:
            warm = min(1.0, it / 300)
            lr = args.lr * warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * it / qat_start)))
        else:
            lr = args.lr * 0.03 * (1 - (it - qat_start) / max(1, args.iters - qat_start)) + 1e-6
        net.step(gw, gb, lr)
        if it % 500 == 0 or it == args.iters:
            f, b = evaluate(net, val)
            msg = "it %6d loss %.5f lr %.2e | val PSNR %.3f dB (bilinear %.3f, +%.3f) | %.0f s" % (
                it, run, lr, f, b, f - b, time.time() - t0)
            if net.act is not None:
                q, _ = evaluate(net, val, int8=True)
                msg += " | int8 %.3f dB" % q
            print(msg, flush=True)
            save_weights(net, os.path.join(args.out, args.name + ".npz"))
    print("done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
