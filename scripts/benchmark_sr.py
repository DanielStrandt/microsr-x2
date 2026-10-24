#!/usr/bin/env python3
"""Evaluate MicroSR-X2 on classic x2 SR datasets stored as HR images.

Expected layout: DATA_ROOT/{Set5,Set14,BSD100,Urban100,Manga109}/HR/*.{png,jpg,...}
Images are not downloaded or redistributed by this script. LR inputs are
generated with MATLAB-style antialiased bicubic downsampling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microsr_model import (FP32_PARAMETER_BYTES, LAYERS,
                           PACKED_WEIGHT_BYTES, WEIGHT_COUNT, MicroSRX2)  # noqa: E402

DATASET_COUNTS = {"Set5": 5, "Set14": 14, "BSD100": 100, "Urban100": 100, "Manga109": 109}
DATASETS = tuple(DATASET_COUNTS)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCALE = 2


def cubic(x: np.ndarray) -> np.ndarray:
    """Keys cubic kernel used by MATLAB imresize (a=-0.5)."""
    ax = np.abs(x)
    out = np.zeros_like(ax, dtype=np.float64)
    m1 = ax <= 1
    m2 = (ax > 1) & (ax < 2)
    out[m1] = 1.5 * ax[m1] ** 3 - 2.5 * ax[m1] ** 2 + 1
    out[m2] = -0.5 * ax[m2] ** 3 + 2.5 * ax[m2] ** 2 - 4 * ax[m2] + 2
    return out


def resize_axis(image: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    """Separable antialiased cubic resize with symmetric edge extension."""
    in_size = image.shape[axis]
    scale = out_size / in_size
    kernel_width = 4.0 / scale if scale < 1 else 4.0
    out_pos = np.arange(1, out_size + 1, dtype=np.float64)
    in_pos = out_pos / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(in_pos - kernel_width / 2).astype(np.int64)
    taps = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(taps, dtype=np.int64)[None, :]
    weights = scale * cubic((in_pos[:, None] - indices) * min(1.0, scale))
    weights /= weights.sum(axis=1, keepdims=True)
    # MATLAB's symmetric extension repeats edge pixels.
    period = 2 * in_size
    folded = np.mod(indices - 1, period)
    indices = np.where(folded < in_size, folded, period - 1 - folded)
    moved = np.moveaxis(image.astype(np.float64, copy=False), axis, 0)
    gathered = moved[indices]
    result = np.sum(gathered * weights.reshape((out_size, taps) + (1,) * (image.ndim - 1)), axis=1)
    return np.moveaxis(result, 0, axis)


def matlab_imresize(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    out = resize_axis(image, out_w, 1)
    out = resize_axis(out, out_h, 0)
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(err * err))
    return math.inf if mse == 0 else 10.0 * math.log10((255.0**2) / mse)


def y_channel(rgb: np.ndarray) -> np.ndarray:
    # MATLAB rgb2ycbcr coefficients for uint8 RGB -> uint8 YCbCr.
    rgb = rgb.astype(np.float64)
    y = (65.481 * rgb[..., 0] + 128.553 * rgb[..., 1] + 24.966 * rgb[..., 2]) / 255.0 + 16.0
    return np.clip(np.floor(y + 0.5), 0, 255).astype(np.uint8)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM, valid window, 11x11 / sigma 1.5."""
    size, sigma = 11, 1.5
    # Float32 keeps the full 300+ image suite practical on CPU; the resulting
    # SSIM rounding is well below the displayed precision.
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    gaussian = torch.exp(-(axis ** 2) / (2 * sigma ** 2))
    gaussian /= gaussian.sum()
    channels = 1 if a.ndim == 2 else a.shape[2]
    window = (gaussian[:, None] * gaussian[None, :]).reshape(1, 1, size, size).repeat(channels, 1, 1, 1)
    if channels == 1:
        x = torch.as_tensor(np.array(a, copy=True), dtype=torch.float32)[None, None]
        y = torch.as_tensor(np.array(b, copy=True), dtype=torch.float32)[None, None]
    else:
        x = torch.as_tensor(np.array(a, copy=True), dtype=torch.float32).permute(2, 0, 1)[None]
        y = torch.as_tensor(np.array(b, copy=True), dtype=torch.float32).permute(2, 0, 1)[None]
    mu_x, mu_y = F.conv2d(x, window, groups=channels), F.conv2d(y, window, groups=channels)
    correction = size * size / (size * size - 1)
    var_x = correction * (F.conv2d(x * x, window, groups=channels) - mu_x * mu_x)
    var_y = correction * (F.conv2d(y * y, window, groups=channels) - mu_y * mu_y)
    cov = correction * (F.conv2d(x * y, window, groups=channels) - mu_x * mu_y)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
    return float(score.mean().item())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Root containing dataset-name/HR image folders")
    parser.add_argument("--model", type=Path, default=ROOT / "model" / "microsr_v6.msr")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        help="Optional explicit dataset selection; incomplete sets are skipped")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads (default: 1)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.device == "mps" and not (getattr(torch.backends, "mps", None)
                                       and torch.backends.mps.is_available()):
        parser.error("MPS requested but unavailable")
    if args.device == "cuda":
        torch.backends.cudnn.allow_tf32 = False
    model = MicroSRX2.load(args.model, device=args.device).eval()
    model_hash = sha256(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    per_image: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for dataset in (args.datasets or list(DATASETS)):
        hr_dir = args.data_root / dataset / "HR"
        images = find_images(hr_dir) if hr_dir.is_dir() else []
        expected = DATASET_COUNTS[dataset]
        if len(images) != expected:
            print(f"SKIP {dataset}: expected {expected} complete HR images, found {len(images)}")
            continue
        manifest = hashlib.sha256()
        for image_path in images:
            manifest.update(image_path.name.encode("utf-8"))
            manifest.update(bytes.fromhex(sha256(image_path)))
        stored_parameter_values = WEIGHT_COUNT + sum(shape[0] * 2 for _, shape, _, _ in LAYERS)
        scores: dict[str, list[float]] = {k: [] for k in
                                           ("bicubic_rgb_psnr_db", "microsr_rgb_psnr_db",
                                            "bicubic_y_psnr_db", "microsr_y_psnr_db",
                                            "bicubic_rgb_ssim", "microsr_rgb_ssim",
                                            "bicubic_y_ssim", "microsr_y_ssim")}
        timings: list[float] = []
        peak_allocated: list[int] = []
        with torch.inference_mode():
            warmup = torch.zeros((1, 3, 32, 32), dtype=torch.float32, device=args.device)
            model(warmup)
            if args.device == "cuda":
                torch.cuda.synchronize()
        for path in images:
            with Image.open(path) as src:
                hr = np.asarray(src.convert("RGB"), dtype=np.uint8)
            h = (hr.shape[0] // SCALE) * SCALE
            w = (hr.shape[1] // SCALE) * SCALE
            hr = hr[:h, :w]
            lr = matlab_imresize(hr, h // SCALE, w // SCALE)
            tensor = torch.from_numpy(lr.copy()).permute(2, 0, 1).unsqueeze(0).to(
                device=args.device, dtype=torch.float32).div_(255.0)
            with torch.inference_mode():
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start = time.perf_counter()
                sr = model(tensor).squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).round()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
                if args.device == "cuda":
                    peak_allocated.append(torch.cuda.max_memory_allocated())
                sr_rgb = sr.cpu().numpy().astype(np.uint8)
            # Report a true bicubic reference using the same MATLAB-style
            # antialiased cubic resizer used to synthesize LR inputs.
            base_rgb = matlab_imresize(lr, h, w)
            # Standard scale-factor border shave, applied equally to both methods.
            shave = SCALE
            target = hr[shave:-shave, shave:-shave]
            sr_eval = sr_rgb[shave:-shave, shave:-shave]
            base_eval = base_rgb[shave:-shave, shave:-shave]
            base_y, sr_y, target_y = y_channel(base_eval), y_channel(sr_eval), y_channel(target)
            row = {"dataset": dataset, "image": path.name,
                   "bicubic_rgb_psnr_db": psnr(base_eval, target),
                   "microsr_rgb_psnr_db": psnr(sr_eval, target),
                   "bicubic_y_psnr_db": psnr(base_y, target_y),
                   "microsr_y_psnr_db": psnr(sr_y, target_y),
                   "bicubic_rgb_ssim": ssim(base_eval, target),
                   "microsr_rgb_ssim": ssim(sr_eval, target),
                   "bicubic_y_ssim": ssim(base_y, target_y),
                   "microsr_y_ssim": ssim(sr_y, target_y),
                   "inference_seconds": timings[-1],
                   "inference_megapixels_per_second": (lr.shape[0] * lr.shape[1] / 1e6) / timings[-1],
                   "peak_cuda_allocated_bytes": peak_allocated[-1] if args.device == "cuda" else None}
            per_image.append(row)
            for key in scores:
                scores[key].append(float(row[key]))
        means = {key: statistics.fmean(values) for key, values in scores.items()}
        summaries.append({"dataset": dataset, "images": len(images),
                          "dataset_manifest_sha256": manifest.hexdigest(),
                          "bicubic_rgb_mean_psnr_db": means["bicubic_rgb_psnr_db"],
                          "microsr_rgb_mean_psnr_db": means["microsr_rgb_psnr_db"],
                          "bicubic_y_mean_psnr_db": means["bicubic_y_psnr_db"],
                          "microsr_y_mean_psnr_db": means["microsr_y_psnr_db"],
                          "bicubic_rgb_mean_ssim": means["bicubic_rgb_ssim"],
                          "microsr_rgb_mean_ssim": means["microsr_rgb_ssim"],
                          "bicubic_y_mean_ssim": means["bicubic_y_ssim"],
                          "microsr_y_mean_ssim": means["microsr_y_ssim"],
                          "microsr_minus_bicubic_y_db": means["microsr_y_psnr_db"] - means["bicubic_y_psnr_db"],
                          "microsr_minus_bicubic_rgb_db": means["microsr_rgb_psnr_db"] - means["bicubic_rgb_psnr_db"],
                          "mean_inference_seconds_per_image": statistics.fmean(timings),
                          "median_inference_seconds_per_image": statistics.median(timings),
                          "p95_inference_seconds_per_image": float(np.percentile(timings, 95)),
                          "mean_lr_megapixels_per_second": statistics.fmean(
                              row["inference_megapixels_per_second"] for row in per_image[-len(images):]),
                          "peak_cuda_allocated_bytes": max(peak_allocated) if peak_allocated else None})
        print(f"{dataset}: {len(images)} images | bicubic Y {means['bicubic_y_psnr_db']:.4f} dB / "
              f"{means['bicubic_y_ssim']:.6f} SSIM | MicroSR Y {means['microsr_y_psnr_db']:.4f} dB / "
              f"{means['microsr_y_ssim']:.6f} SSIM")

    if not summaries:
        print("No complete benchmark datasets were available; no result files written.")
        return 2
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    metadata = {"protocol": "x2 synthetic BI; MATLAB-style antialiased bicubic downsampling and bicubic reference; "
                            "RGB model input; uint8 round/clamp; shave 2 HR pixels; "
                            "per-image metrics then arithmetic mean; 11x11 Gaussian SSIM, sigma 1.5",
                "model_path": str(args.model), "model_sha256": model_hash,
                "model_file_bytes": args.model.stat().st_size,
                "packed_int2_weight_bytes": PACKED_WEIGHT_BYTES,
                "weight_values": WEIGHT_COUNT,
                "stored_parameter_values": stored_parameter_values,
                "fp32_scale_bias_bytes": FP32_PARAMETER_BYTES,
                "peak_gpu_memory_metric": "CUDA max allocated bytes per image; null on non-CUDA runs",
                "device": args.device, "torch_version": torch.__version__,
                "torch_threads": args.threads,
                "numpy_version": np.__version__, "pillow_version": Image.__version__,
                "python_version": platform.python_version(), "summaries": summaries}
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output / 'per_image.csv'} and {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

