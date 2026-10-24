#!/usr/bin/env python3
r"""MicroSR-X2 V6 visual / exposure / color / clipping / checkerboard audit.

Use alongside microsr_int2_local.py (V6) and a TRUSTED local checkpoint.

Recommended (same held-out split and bicubic operations as the trainer):
  python microsr_color_audit.py --checkpoint "./runs/microsr_x2_v6_full/best.pt" \
    --image-dir "./data" --count 8 --crops-per-image 2 \
    --output "./runs/microsr_x2_v6_full/color_audit"

Optional side-by-side V5 control:
  python microsr_color_audit.py --checkpoint "./runs/microsr_x2_v6_full/best.pt" \
    --compare-checkpoint "./runs/microsr_x2_v5_fresh/best.pt" \
    --image-dir "./data" --count 8 \
    --output "./runs/microsr_x2_v6_full/color_audit_vs_v5"

To inspect actual native-LR photos (no HR reference, so no PSNR):
  python microsr_color_audit.py --checkpoint "./runs/microsr_x2_v6_full/best.pt" \
    --mode upscale --selection folder --input-dir "./photos" \
    --count 8 --output "./runs/microsr_x2_v6_full/native_color_audit"

Every test uses the saved selected INT2 weights and the trainer's PyTorch
bicubic operators. Does NOT train, modify, or save any model weights.

What the audit CAN tell you: whether its tested images have measurable
brightness/RGB cast shifts, severe clipping, low-frequency color changes,
and whether synthetic flat patches show phase/checkerboard artifacts.
What it CANNOT do: certify all possible images are free of color issues;
this is a targeted diagnostic, not a perceptual study. The existing 205-image
held-out set has already been used for development, not an untouched test set.
"""

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

POSITIONS = [
    ("center", 0.5, 0.5),
    ("top_left", 0.0, 0.0),
    ("bottom_right", 1.0, 1.0),
    ("top_right", 1.0, 0.0),
    ("bottom_left", 0.0, 1.0),
]
Y_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Trusted V6 best.pt (or other Spatial model)")
    p.add_argument("--compare-checkpoint", default=None,
                   help="Optional V5 best.pt control; visual/metric comparisons share EXACT samples")
    p.add_argument("--model-module", default=None,
                   help="V6 microsr_int2_local.py; defaults to next to this audit script")
    p.add_argument("--image-dir", default="data",
                   help="The top-level training image directory, used to reproduce the held-out split")
    p.add_argument("--selection", choices=("heldout", "train", "folder"), default="heldout",
                   help="heldout is recommended; always searches immediate files only")
    p.add_argument("--input-dir", default=None,
                   help="For --selection folder: a separate folder containing photos to inspect")
    p.add_argument("--inputs", nargs="+", default=None,
                   help="Optional explicit image paths, bypasses automatic sampling")
    p.add_argument("--mode", choices=("reconstruct", "upscale"), default="reconstruct",
                   help="reconstruct: HR source->bicubic down->model->HR; upscale: native LR->model")
    p.add_argument("--count", type=int, default=8, help="Distinct source images to inspect")
    p.add_argument("--crops-per-image", type=int, default=2, choices=[1, 2, 3, 4, 5])
    p.add_argument("--patch-size", type=int, default=512,
                   help="Per-source crop: HR pixels in reconstruct mode, LR pixels in upscale mode")
    p.add_argument("--shave", type=int, default=2,
                   help="Discard this many output border pixels for metrics (trainer validation uses 2); visuals remain full")
    p.add_argument("--sample-seed", type=int, default=1729, help="Deterministic image sampling seed")
    p.add_argument("--max-source-mp", type=float, default=50.0,
                   help="Safety: skip sources above megapixel limit; 0 disables this check")
    p.add_argument("--output", default="./runs/microsr_x2_v6_full/color_audit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--difference-gain", type=float, default=8.0,
                   help="Visual ONLY: amplify signed SR-bicubic differences by this factor")
    p.add_argument("--overview-thumb", type=int, default=240,
                   help="Image tile size in overview contact sheet")
    p.add_argument("--skip-synthetic", action="store_true",
                   help="Omit flat-gray, solid-color, grayscale-ramp tests (not recommended)")
    p.add_argument("--luma-warning", type=float, default=1.0,
                   help="Warn if mean SR-vs-bicubic luma shift exceeds this many 8-bit levels")
    p.add_argument("--cast-warning", type=float, default=1.5,
                   help="Warn if R-G or B-G mean shift exceeds this many 8-bit levels")
    p.add_argument("--clip-warning-pct", type=float, default=0.1,
                   help="Warn if this %% of RGB samples newly exceed [0,1] before clamp")
    p.add_argument("--phase-warning", type=float, default=1.0,
                   help="Warn if flat-color 2x2 phase luma differs by more than these levels")
    args = p.parse_args()
    if args.count < 1 or args.patch_size < 32 or args.patch_size % 2:
        p.error("--count >= 1; --patch-size must be even and >= 32")
    if args.shave < 0 or args.shave * 2 >= args.patch_size:
        p.error("--shave must be nonnegative and smaller than half the patch size")
    if args.difference_gain <= 0 or args.overview_thumb < 64:
        p.error("--difference-gain >0 and --overview-thumb >=64 required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA was requested but is not available")
    if args.selection == "folder" and args.inputs is None and not args.input_dir:
        p.error("--selection folder requires --input-dir (or --inputs)")
    if args.max_source_mp < 0:
        p.error("--max-source-mp must be >=0")
    return args


def import_trainer(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V6 trainer module not found: {path}")
    spec = importlib.util.spec_from_file_location("microsr_int2_local_for_audit", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = ("load_model", "local_image_split", "validation_fingerprint", "image_tensor",
                "bicubic_down", "bicubic_up")
    for attr in required:
        if not hasattr(mod, attr):
            raise RuntimeError(f"The trainer module is missing {attr}; use the V6 version")
    return mod


def list_top_level(folder, extensions):
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory missing: {root}")
    return sorted(x for x in root.iterdir() if x.is_file() and x.suffix.lower() in extensions)


def source_candidates(args, core, ck):
    if args.inputs is not None:
        paths = [Path(item) for item in args.inputs]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError("One or more --inputs are not readable image files")
    elif args.selection == "folder":
        paths = list_top_level(args.input_dir, core.IMAGE_EXTENSIONS)
    else:
        cfg = ck.get("config", {})
        seed = int(cfg.get("seed", 42))
        val_fraction = float(cfg.get("val_fraction", 0.05))
        minimum = int(cfg.get("patch", 64)) + 2 * int(cfg.get("context", 8))
        train_paths, val_paths = core.local_image_split(args.image_dir, seed, val_fraction, minimum)
        saved_fingerprint = cfg.get("val_fingerprint")
        computed = core.validation_fingerprint(val_paths)
        if saved_fingerprint and saved_fingerprint != computed:
            raise ValueError(f"Dataset/split fingerprint mismatch: checkpoint={saved_fingerprint}, "
                             f"directory={computed}. Cannot claim matched held-out tests.")
        print(f"Split fingerprint {computed}; selection={args.selection}; no subfolders used.")
        paths = val_paths if args.selection == "heldout" else train_paths
    if not paths:
        raise RuntimeError("No supported images found (subdirectories are intentionally ignored)")
    paths = list(dict.fromkeys(paths))
    random.Random(args.sample_seed).shuffle(paths)
    return paths


def luma(image):
    return np.tensordot(image.astype(np.float64, copy=False), Y_WEIGHTS, axes=([-1], [0]))


def to_rgb_numpy(t):
    return t.detach().squeeze(0).permute(1, 2, 0).float().cpu().numpy().astype(np.float32)


def png_image(x):
    x = np.clip(x, 0, 1)
    a = np.rint(x * 255.0).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def signed_difference(a, b, multiplier):
    # Mid-gray means unchanged; brighter / darker signs of each RGB correction.
    return Image.fromarray(np.clip(np.rint(127.5 + (a - b) * 255.0 * multiplier), 0, 255)
                           .astype(np.uint8), mode="RGB")


def load_font(size=19):
    for name in ("DejaVuSans.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, ValueError):
            pass
    return ImageFont.load_default()


def add_label(image, caption, label_height=32):
    result = Image.new("RGB", (image.width, image.height + label_height), (246, 246, 246))
    result.paste(image, (0, label_height))
    ImageDraw.Draw(result).text((7, 6), caption, fill=(20, 20, 20), font=load_font(16))
    return result


def create_row(parts, thumb=None):
    # Parts: list of (caption, PIL.Image) pairs.
    if thumb:
        resized = []
        for title, im in parts:
            if im.width != thumb:
                height = max(1, round(im.height * (thumb / im.width)))
                im = im.resize((thumb, height), Image.Resampling.LANCZOS)
            resized.append((title, im))
        parts = resized
    tiles = [add_label(img, title) for title, img in parts]
    height = max(t.height for t in tiles)
    canvas = Image.new("RGB", (sum(x.width for x in tiles), height), "white")
    col = 0
    for tile in tiles:
        canvas.paste(tile, (col, 0))
        col += tile.width
    return canvas


def save_overview(rows, path):
    if not rows:
        return
    width = max(im.width for _, im in rows)
    total_h = sum(im.height + 29 for _, im in rows)
    if width * total_h > 80_000_000:
        print("Large overview; reducing contact sheet resolution to fit memory")
        factor = math.sqrt(80_000_000 / (width * total_h))
        rows = [(name, im.resize((max(1, round(im.width * factor)),
                                  max(1, round(im.height * factor))), Image.Resampling.LANCZOS))
                for name, im in rows]
        width = max(im.width for _, im in rows)
        total_h = sum(im.height + 29 for _, im in rows)
    canvas = Image.new("RGB", (width, total_h), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    y = 0
    font = load_font(16)
    for name, im in rows:
        draw.text((8, y + 3), name[:110], fill=(20, 20, 20), font=font)
        canvas.paste(im, (0, y + 29))
        y += im.height + 29
    canvas.save(path, optimize=True)


def psnr(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def mse(x, y):
    a = x.astype(np.float64) - y.astype(np.float64)
    return float(np.mean(a * a))


def block_averages(image, block=16):
    h, w, _ = image.shape
    hh = h // block
    ww = w // block
    if not hh or not ww:
        return np.mean(image, axis=(0, 1), keepdims=True)
    data = image[:hh*block, :ww*block, :]
    return data.reshape(hh, block, ww, block, 3).mean(axis=(1, 3))


def color_metrics(sr, bc, gt=None, raw_sr=None, raw_bc=None, args=None):
    # All shifts measured BEFORE 8-bit saving, in display-encoded sRGB units.
    # Luma is Rec.709 Y' on these encoded values, NOT physical linear light.
    diff = (sr.astype(np.float64) - bc.astype(np.float64)) * 255.0
    mean_rgb = np.mean(diff, axis=(0, 1))
    mean_luma = float(np.mean(luma(sr) - luma(bc)) * 255.0)
    rg_cast = float(mean_rgb[0] - mean_rgb[1])
    bg_cast = float(mean_rgb[2] - mean_rgb[1])
    block_delta = (block_averages(sr) - block_averages(bc)).astype(np.float64) * 255.0
    block_luma = np.sum(block_delta * Y_WEIGHTS, axis=-1)
    block_chroma = block_delta - block_luma[..., None]
    chroma_blocks = np.linalg.norm(block_chroma, axis=-1)
    p95_block_chroma = float(np.percentile(chroma_blocks, 95))
    out = {
        "mean_luma_shift_lsb": mean_luma,
        "mean_red_shift_lsb": float(mean_rgb[0]),
        "mean_green_shift_lsb": float(mean_rgb[1]),
        "mean_blue_shift_lsb": float(mean_rgb[2]),
        "red_minus_green_cast_shift_lsb": rg_cast,
        "blue_minus_green_cast_shift_lsb": bg_cast,
        "p95_low_frequency_chroma_change_lsb": p95_block_chroma,
        "mean_abs_luma_change_lsb": float(np.mean(np.abs(luma(sr) - luma(bc))) * 255.0),
        "p99_abs_rgb_change_lsb": float(np.percentile(np.abs(diff), 99)),
        "out_of_gamut_sr_pct": 0.0,
        "new_out_of_gamut_sr_pct": 0.0,
        "out_of_gamut_bc_pct": 0.0,
    }
    if raw_sr is not None and raw_bc is not None:
        oc_sr = (raw_sr < 0) | (raw_sr > 1)
        oc_bc = (raw_bc < 0) | (raw_bc > 1)
        out["out_of_gamut_sr_pct"] = float(np.mean(oc_sr) * 100)
        out["new_out_of_gamut_sr_pct"] = float(np.mean(oc_sr & ~oc_bc) * 100)
        out["out_of_gamut_bc_pct"] = float(np.mean(oc_bc) * 100)
    if gt is not None:
        out["sr_mse"] = mse(sr, gt)
        out["bc_mse"] = mse(bc, gt)
        out["sr_psnr_db"] = psnr(out["sr_mse"])
        out["bicubic_psnr_db"] = psnr(out["bc_mse"])
        out["psnr_gain_db"] = out["sr_psnr_db"] - out["bicubic_psnr_db"]
        gt_rgb_sr = np.mean((sr.astype(np.float64)-gt.astype(np.float64))*255, axis=(0, 1))
        gt_rgb_bc = np.mean((bc.astype(np.float64)-gt.astype(np.float64))*255, axis=(0, 1))
        for name, arr in (("sr_vs_truth", gt_rgb_sr), ("bicubic_vs_truth", gt_rgb_bc)):
            out[f"{name}_mean_R_bias_lsb"] = float(arr[0])
            out[f"{name}_mean_G_bias_lsb"] = float(arr[1])
            out[f"{name}_mean_B_bias_lsb"] = float(arr[2])
            out[f"{name}_luma_bias_lsb"] = float(np.dot(arr, Y_WEIGHTS))
        sr_block, bc_block, gt_block = map(block_averages, (sr, bc, gt))
        out["sr_low_freq_rgb_mae_vs_truth_lsb"] = float(np.mean(np.abs(sr_block - gt_block)) * 255)
        out["bc_low_freq_rgb_mae_vs_truth_lsb"] = float(np.mean(np.abs(bc_block - gt_block)) * 255)
    flags = []
    if abs(mean_luma) > args.luma_warning:
        flags.append("mean_luma_shift")
    if max(abs(rg_cast), abs(bg_cast)) > args.cast_warning:
        flags.append("mean_color_cast")
    if out["new_out_of_gamut_sr_pct"] > args.clip_warning_pct:
        flags.append("new_clipping")
    # Heuristic: low frequency localized chroma change; not proof of wrong colors.
    if p95_block_chroma > 3.0:
        flags.append("local_color_shift_check")
    if gt is not None and out["sr_low_freq_rgb_mae_vs_truth_lsb"] > out["bc_low_freq_rgb_mae_vs_truth_lsb"] + 0.5:
        flags.append("low_frequency_fidelity_regression")
    out["warnings"] = flags
    return out


@torch.inference_mode()
def run_model(core, model, lr):
    # Inference always FP32; no autocast; no replacement for INT2 forward.
    # Reflect padding, learned scales, and saved selected EMA/raw weights are
    # exactly those of core.load_model + core.SpatialV*.forward.
    raw = model(lr)
    return to_rgb_numpy(raw.clamp(0, 1)), to_rgb_numpy(raw)


@torch.inference_mode()
def run_crop(core, model, control, im, position, args):
    w, h = im.size
    side = min((w // 2) * 2, (h // 2) * 2, args.patch_size)
    if side < 32:
        raise ValueError(f"Image is too small after even cropping: {w}x{h}")
    loc_name, fx, fy = position
    x = min(w-side, 2*round((w-side)*fx / 2))
    y = min(h-side, 2*round((h-side)*fy / 2))
    with im.crop((x, y, x+side, y+side)) as crop:
        src = core.image_tensor(crop).unsqueeze(0).to(args.device)
    if args.mode == "reconstruct":
        hr = src
        lr = core.bicubic_down(hr)
        gt = to_rgb_numpy(hr)
    else:
        lr = src
        gt = None
    bc_raw = core.bicubic_up(lr)
    bc = to_rgb_numpy(bc_raw.clamp(0, 1))
    sr, sr_raw = run_model(core, model, lr)
    old, old_raw = (run_model(core, control, lr) if control is not None else (None, None))
    return dict(name=loc_name, location=[int(x), int(y), int(side)],
                gt=gt, bicubic=bc, sr=sr, raw_sr=sr_raw,
                raw_bc=to_rgb_numpy(bc_raw), control=old, raw_control=old_raw)


def profile_of_flat(sr, rgb, raw_sr, args):
    trim = 16
    inside = sr[trim:-trim, trim:-trim]
    raw_inside = raw_sr[trim:-trim, trim:-trim]
    base = np.asarray(rgb, np.float64) / 255.0
    means = inside.mean(axis=(0, 1)).astype(np.float64) * 255
    yshift = float(np.dot(means - np.asarray(rgb), Y_WEIGHTS))
    cast = [float((means[0]-rgb[0]) - (means[1]-rgb[1])),
            float((means[2]-rgb[2]) - (means[1]-rgb[1]))]
    yy = luma(inside) * 255
    phase_y = [float(yy[i::2, j::2].mean()) for i in (0, 1) for j in (0, 1)]
    phase_y_range = max(phase_y) - min(phase_y)
    phase_rgb = [inside[i::2, j::2].mean(axis=(0, 1))*255 for i in (0, 1) for j in (0, 1)]
    phase_rgb_range = float(np.max(np.ptp(np.asarray(phase_rgb), axis=0)))
    clipped = float(np.mean((raw_inside < 0) | (raw_inside > 1)) * 100)
    flags = []
    if abs(yshift) > args.luma_warning:
        flags.append("flat_exposure_shift")
    if max(abs(v) for v in cast) > args.cast_warning:
        flags.append("flat_rgb_cast")
    if phase_y_range > args.phase_warning:
        flags.append("2x2_luma_checkerboard")
    if phase_rgb_range > args.phase_warning:
        flags.append("2x2_rgb_checkerboard")
    if clipped > args.clip_warning_pct:
        flags.append("flat_clipping")
    return {"input_R": int(rgb[0]), "input_G": int(rgb[1]), "input_B": int(rgb[2]),
            "output_mean_R": float(means[0]), "output_mean_G": float(means[1]),
            "output_mean_B": float(means[2]), "mean_luma_shift_lsb": yshift,
            "red_minus_green_cast_shift_lsb": cast[0],
            "blue_minus_green_cast_shift_lsb": cast[1],
            "phase_2x2_luma_range_lsb": phase_y_range,
            "phase_2x2_rgb_max_range_lsb": phase_rgb_range,
            "out_of_gamut_pct": clipped,
            "max_abs_pixel_change_lsb": float(np.max(np.abs((inside - base) * 255))),
            "warnings": flags}


def write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(key for rec in records for key in rec))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for rec in records:
            row = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                   for k, v in rec.items()}
            writer.writerow(row)


@torch.inference_mode()
def synthetic_tests(core, model, control, args, output):
    print("\nTesting flat grays, solid colors and a grayscale ramp...")
    patterns = [
        ("gray_016", (16, 16, 16)),
        ("gray_064", (64, 64, 64)),
        ("gray_128", (128, 128, 128)),
        ("gray_192", (192, 192, 192)),
        ("gray_240", (240, 240, 240)),
        ("red", (220, 30, 30)),
        ("green", (30, 220, 30)),
        ("blue", (30, 30, 220)),
        ("yellow", (220, 220, 30)),
        ("cyan", (30, 220, 220)),
        ("magenta", (220, 30, 220)),
        ("skin_like", (190, 135, 110)),
    ]
    rows, grid = [], []
    for name, rgb in patterns:
        source = np.broadcast_to(np.asarray(rgb, np.float32).reshape(1, 1, 3)/255.0,
                                 (128, 128, 3)).copy()
        lr = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).to(args.device)
        sr, raw = run_model(core, model, lr)
        bc = to_rgb_numpy(core.bicubic_up(lr).clamp(0, 1))
        measurements = profile_of_flat(sr, rgb, raw, args)
        measurements["pattern"] = name
        if control is not None:
            old, oldraw = run_model(core, control, lr)
            old_metrics = profile_of_flat(old, rgb, oldraw, args)
            measurements["control_luma_shift_lsb"] = old_metrics["mean_luma_shift_lsb"]
            measurements["control_phase_luma_range_lsb"] = old_metrics["phase_2x2_luma_range_lsb"]
        rows.append(measurements)
        small = [("Input / bicubic", png_image(bc))]
        if control is not None:
            small.append(("V5 control", png_image(old)))
        small.append(("V6 INT2", png_image(sr)))
        small.append(("Delta x8", signed_difference(sr, bc, args.difference_gain)))
        grid.append((name, create_row(small, thumb=160)))
    # Neutral ramp in all three channels; catches tone curve / tint changes.
    ramp = np.linspace(0, 1, 128, dtype=np.float32)
    source = np.broadcast_to(ramp[None, :, None], (128, 128, 3)).copy()
    lr = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0).to(args.device)
    sr, raw = run_model(core, model, lr)
    bc = to_rgb_numpy(core.bicubic_up(lr).clamp(0, 1))
    if control is not None:
        old, _ = run_model(core, control, lr)
    ramp_tiles = [("Neutral gray ramp", png_image(bc))]
    if control is not None:
        ramp_tiles.append(("V5 control", png_image(old)))
    ramp_tiles.extend([("V6 INT2", png_image(sr)),
                       ("Delta x8", signed_difference(sr, bc, args.difference_gain))])
    grid.append(("neutral_ramp", create_row(ramp_tiles, thumb=160)))
    ramp_stats = color_metrics(sr, bc, raw_sr=raw, raw_bc=bc, args=args)
    ramp_stats["pattern"] = "neutral_ramp"
    # Ramp itself has no super-resolution textures; mean color cast is not desirable.
    write_csv(output / "synthetic_flat_color_metrics.csv", rows)
    save_overview(grid, output / "synthetic_color_tests.png")
    (output / "synthetic_ramp_metrics.json").write_text(
        json.dumps(ramp_stats, indent=2), encoding="utf-8")
    for row in rows:
        print(f"  {row['pattern']:>10}: luma={row['mean_luma_shift_lsb']:+.3f} "
              f"cast(R-G)={row['red_minus_green_cast_shift_lsb']:+.3f} "
              f"phase={row['phase_2x2_luma_range_lsb']:.3f} "
              f"warnings={','.join(row['warnings']) or 'none'}")
    return rows, ramp_stats


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    core = import_trainer(args.model_module or (Path(__file__).resolve().parent / "microsr_int2_local.py"))
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
    print(f"Loading EXACT V6 selected INT2 inference weights from {ckpt}")
    model, ck = core.load_model(str(ckpt), args.device, use_inference=True)
    model.eval()
    control = None
    if args.compare_checkpoint:
        control, _ = core.load_model(args.compare_checkpoint, args.device, use_inference=True)
        control.eval()
        if model.architecture not in ("spatial_lr_dw2_shuffle_v1", "spatial_lr_dw2_shuffle_v3") \
                or control.architecture not in ("spatial_lr_dw2_shuffle_v1", "spatial_lr_dw2_shuffle_v3"):
            raise ValueError("The optional V5/V6 comparison is intended for Spatial V1/V3 models")
    print(f"Architecture={model.architecture}; codebook={model.quant_grid}; "
          f"selected={ck.get('inference_weights', 'raw')}; device={args.device}")
    print("No model training or checkpoint modification occurs.")
    candidates = source_candidates(args, core, ck)
    processed, rejected = [], []
    count, sr_sum, bc_sum, control_sum, n_patches = 0, 0.0, 0.0, 0.0, 0
    rows = []
    overview_rows = []
    crops_dir = output / "crops"
    crops_dir.mkdir(exist_ok=True)
    print(f"Sampling {args.count} images, {args.crops_per_image} location(s) each; "
          f"mode={args.mode}; patch={args.patch_size}; seed={args.sample_seed}")
    for path in candidates:
        if count >= args.count:
            break
        try:
            with Image.open(path) as im:
                w, h = im.size
                if args.max_source_mp and w * h > args.max_source_mp * 1e6:
                    raise ValueError(f"source {w}x{h} exceeds {args.max_source_mp} MP safety limit")
                if min(w, h) < 32:
                    raise ValueError("source too small to audit")
                warning = []
                if im.info.get("icc_profile"):
                    warning.append("Embedded ICC: PIL RGB conversion follows trainer; no ICC color management")
                if im.mode not in ("RGB", "L"):
                    warning.append(f"Source mode {im.mode} converted to RGB as in trainer")
                sample_overview = []
                sample_rows = []
                for position in POSITIONS[:args.crops_per_image]:
                    data = run_crop(core, model, control, im, position, args)
                    def roi(image):
                        return image[args.shave:-args.shave, args.shave:-args.shave] \
                            if args.shave and image is not None else image
                    values = color_metrics(roi(data["sr"]), roi(data["bicubic"]),
                                           roi(data["gt"]), roi(data["raw_sr"]),
                                           roi(data["raw_bc"]), args)
                    rec = {"image": path.name, "location": data["name"],
                           "input_width": w, "input_height": h,
                           "crop_left": data["location"][0],
                           "crop_top": data["location"][1],
                           "crop_size": data["location"][2],
                           **values}
                    if data["control"] is not None:
                        rec["control_mse"] = mse(roi(data["control"]), roi(data["gt"])) if data["gt"] is not None else None
                        rec["control_psnr_db"] = psnr(rec["control_mse"]) if rec["control_mse"] is not None else None
                        old_stats = color_metrics(roi(data["control"]), roi(data["bicubic"]),
                                                  roi(data["gt"]), roi(data["raw_control"]),
                                                  roi(data["raw_bc"]), args)
                        rec["control_mean_luma_shift_lsb"] = old_stats["mean_luma_shift_lsb"]
                        rec["control_rg_cast_shift_lsb"] = old_stats["red_minus_green_cast_shift_lsb"]
                        rec["control_bg_cast_shift_lsb"] = old_stats["blue_minus_green_cast_shift_lsb"]
                        rec["control_new_clipping_pct"] = old_stats["new_out_of_gamut_sr_pct"]
                        rec["v6_minus_control_mean_luma_lsb"] = (
                            rec["mean_luma_shift_lsb"] - rec["control_mean_luma_shift_lsb"])
                    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path.stem)[:65]
                    prefix = f"{count+1:02d}_{safe}_{data['name']}"
                    folder = crops_dir / prefix
                    folder.mkdir(exist_ok=True)
                    png_image(data["bicubic"]).save(folder / "bicubic.png")
                    png_image(data["sr"]).save(folder / "v6_int2.png")
                    signed_difference(data["sr"], data["bicubic"], args.difference_gain).save(
                        folder / f"signed_difference_x{args.difference_gain:g}.png")
                    tiles = []
                    if data["gt"] is not None:
                        png_image(data["gt"]).save(folder / "original_hr_reference.png")
                        tiles.append(("Original HR", png_image(data["gt"])))
                    tiles.append(("Bicubic 2x", png_image(data["bicubic"])))
                    if data["control"] is not None:
                        png_image(data["control"]).save(folder / "v5_control.png")
                        tiles.append(("V5 control", png_image(data["control"])))
                    tiles.append(("V6 INT2", png_image(data["sr"])))
                    tiles.append((f"V6 - bicubic x{args.difference_gain:g}",
                                  signed_difference(data["sr"], data["bicubic"], args.difference_gain)))
                    big_row = create_row(tiles)
                    big_row.save(folder / "side_by_side.png", optimize=True)
                    small_row = create_row(tiles, thumb=args.overview_thumb)
                    sample_overview.append((path.name + " / " + data["name"], small_row))
                    sample_rows.append(rec)
                    print(f"  {prefix}: luma={values['mean_luma_shift_lsb']:+.3f} LSB "
                          f"R-G={values['red_minus_green_cast_shift_lsb']:+.3f} LSB "
                          f"new_clip={values['new_out_of_gamut_sr_pct']:.3f}% "
                          f"flags={','.join(values['warnings']) or 'none'}", flush=True)
                rows.extend(sample_rows)
                processed.append({"path": str(path.resolve()), "source_rgb": im.mode,
                                  "icc_or_mode_warnings": warning})
                count += 1
        except (OSError, MemoryError, ValueError) as e:
            print(f"  SKIP {path.name}: {e}")
            rejected.append({"path": str(path), "error": str(e)})
            continue
        overview_rows.extend(sample_overview)
        for r in sample_rows:
            if args.mode == "reconstruct":
                sr_sum += r["sr_mse"]
                bc_sum += r["bc_mse"]
                if control is not None:
                    control_sum += r["control_mse"]
            n_patches += 1
    if not processed:
        raise RuntimeError("No suitable images were audited; review source selection and size limits")
    if count < args.count:
        print(f"WARNING: only {count}/{args.count} requested source images were suitable")
    write_csv(output / "per_crop_color_metrics.csv", rows)
    save_overview(overview_rows, output / "overview_comparisons.png")
    synth_rows, ramp = ([], None)
    if not args.skip_synthetic:
        synth_rows, ramp = synthetic_tests(core, model, control, args, output)
    warns = Counter(w for row in rows for w in row["warnings"])
    synth_warns = Counter(w for row in synth_rows for w in row["warnings"])
    summary = {
        "model_checkpoint": str(ckpt.resolve()),
        "model_architecture": model.architecture,
        "inference_weights": ck.get("inference_weights", "raw"),
        "int2_grid": model.quant_grid,
        "control_checkpoint": str(Path(args.compare_checkpoint).resolve()) if args.compare_checkpoint else None,
        "training_module": str(Path(args.model_module or (Path(__file__).resolve().parent / "microsr_int2_local.py")).resolve()),
        "mode": args.mode,
        "source_selection": "explicit" if args.inputs else args.selection,
        "source_images_audited": len(processed),
        "source_images_skipped": rejected,
        "processed_sources": processed,
        "patches": n_patches,
        "patch_size_input": args.patch_size,
        "metric_border_shave": args.shave,
        "sample_seed": args.sample_seed,
        "rgb_color_space": "PIL RGB conversion as in trainer; metrics on gamma-encoded sRGB-like values",
        "per_crop_warning_counts": dict(warns),
        "synthetic_warning_counts": dict(synth_warns),
        "warned_real_crops": sum(bool(x["warnings"]) for x in rows),
        "warned_flat_colors": sum(bool(x["warnings"]) for x in synth_rows),
        "mean_absolute_luma_shift_lsb_across_crops": float(np.mean([abs(r["mean_luma_shift_lsb"]) for r in rows])),
        "largest_absolute_luma_shift_lsb": float(max(abs(r["mean_luma_shift_lsb"]) for r in rows)),
        "largest_mean_cast_shift_lsb": float(max(max(abs(r["red_minus_green_cast_shift_lsb"]),
                                                 abs(r["blue_minus_green_cast_shift_lsb"])) for r in rows)),
        "max_new_clipping_pct": float(max(r["new_out_of_gamut_sr_pct"] for r in rows)),
        "synthetic_ramp": ramp,
        "warning_thresholds": {"luma_lsb": args.luma_warning,
                               "cast_lsb": args.cast_warning,
                               "clipping_percent": args.clip_warning_pct,
                               "phase_lsb": args.phase_warning},
        "interpretation": ("This is a SAMPLE-BASED DIAGNOSTIC, not a color-safety guarantee. "
                           "All flags are heuristics. Inspect the saved reference/bicubic/SR and "
                           "signed difference images directly. Using development held-out images "
                           "does not replace an independent final test set."),
    }
    if args.mode == "reconstruct":
        summary["bicubic_psnr_db"] = psnr(bc_sum / n_patches)
        summary["v6_int2_psnr_db"] = psnr(sr_sum / n_patches)
        summary["gain_over_bicubic_db"] = summary["v6_int2_psnr_db"] - summary["bicubic_psnr_db"]
        if control is not None:
            summary["v5_control_psnr_db"] = psnr(control_sum / n_patches)
            summary["v6_gain_vs_v5_db"] = summary["v6_int2_psnr_db"] - summary["v5_control_psnr_db"]
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "selected_images.json").write_text(json.dumps(processed, indent=2), encoding="utf-8")
    print("\n=== V6 COLOR / EXPOSURE AUDIT ===")
    print(f"Real images: {len(processed)}, inspected crops: {n_patches}")
    if args.mode == "reconstruct":
        print(f"Matched-crop PSNR: bicubic={summary['bicubic_psnr_db']:.4f} dB; "
              f"V6={summary['v6_int2_psnr_db']:.4f} dB; "
              f"gain={summary['gain_over_bicubic_db']:+.4f} dB")
        if control is not None:
            print(f"V5 control={summary['v5_control_psnr_db']:.4f} dB; "
                  f"V6 vs V5={summary['v6_gain_vs_v5_db']:+.4f} dB")
    print(f"Largest mean brightness shift vs bicubic: "
          f"{summary['largest_absolute_luma_shift_lsb']:.3f} /255")
    print(f"Largest mean RGB color-cast shift vs bicubic: "
          f"{summary['largest_mean_cast_shift_lsb']:.3f} /255")
    print(f"Largest newly out-of-range fraction (pre-clamp): "
          f"{summary['max_new_clipping_pct']:.4f}% of RGB samples")
    print(f"Flagged real crops: {summary['warned_real_crops']}/{n_patches}; "
          f"flagged synthetic flat colors: {summary['warned_flat_colors']}/{len(synth_rows)}")
    print(f"Saved overview: {output / 'overview_comparisons.png'}")
    if synth_rows:
        print(f"Saved flat-color/ramp diagnostics: {output / 'synthetic_color_tests.png'}")
    print(f"Saved CSV metrics: {output / 'per_crop_color_metrics.csv'}")
    print(f"Saved detailed report: {output / 'audit_summary.json'}")
    print("Caution: flag-free sampled results do not guarantee color fidelity on all images.")


if __name__ == "__main__":
    main()

