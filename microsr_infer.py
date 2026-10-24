#!/usr/bin/env python3
"""MicroSR-X2 2x native-image upscaling CLI using microsr_model.py.

Examples (PowerShell):
  python microsr_infer.py --input "F:\\photos\\photo.jpg" --output "F:\\photos\\photo_x2.png"
  python microsr_infer.py --input "F:\\photos" --output "F:\\photos_upscaled"
  python microsr_infer.py --self-test

Default model: ./model/microsr_v6.msr, relative to THIS script.
By default, only top-level images in a directory are processed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from microsr_model import MicroSRX2, SUPPORTED_IMAGES


DEFAULT_MODEL = Path(__file__).resolve().parent / "model" / "microsr_v6.msr"


def choose_device(selection: str) -> str:
    if selection == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if selection == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or auto")
    if selection == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but unavailable. Use --device cpu or auto")
    return selection


def image_paths(source: Path, recursive: bool, skip_upscaled_suffix: str = "") -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_IMAGES:
            raise ValueError(f"Unsupported input file suffix: {source}")
        return [source]
    if source.is_dir():
        it = source.rglob("*") if recursive else source.iterdir()
        return sorted(p for p in it if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGES
                      and (not skip_upscaled_suffix or not p.stem.lower().endswith(skip_upscaled_suffix.lower())))
    raise FileNotFoundError(f"Cannot find input file or directory: {source}")


def destination_for(src: Path, root: Path, output: Path | None, recursive: bool, suffix: str) -> Path:
    if root.is_file():
        return output if output is not None else src.with_name(src.stem + suffix + ".png")
    output_root = output if output is not None else root.with_name(root.name + "_upscaled")
    relative = src.relative_to(root)
    return output_root / relative.parent / (src.stem + suffix + ".png")


def save_image(image: Image.Image, output: Path, *, quality: int = 95) -> None:
    ext = output.suffix.lower()
    if ext not in SUPPORTED_IMAGES:
        raise ValueError(f"Unsupported output extension {ext!r}. Use PNG for lossless output")
    if image.mode == "RGBA" and ext in (".jpg", ".jpeg", ".bmp"):
        raise ValueError("Output has transparency: choose PNG/WebP/TIFF, not JPEG/BMP")
    params = {}
    if image.info.get("icc_profile"):
        if ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
            params["icc_profile"] = image.info["icc_profile"]
    if ext in (".jpg", ".jpeg"):
        params.update(quality=quality, subsampling=0)
    if ext == ".webp":
        params.update(lossless=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never copy EXIF orientation: ImageOps.exif_transpose was already applied.
    image.save(output, **params)


def run_self_test(model: MicroSRX2, device: str, tile: int) -> None:
    # Full vs tile comparison is a deterministic regression test of the actual
    # packaged model, including bicubic skip and reflect-padding boundaries.
    torch.manual_seed(123)
    print("Self-test: checking native input shapes, exact 2x output, tile parity, and RGB finiteness")
    model.eval()
    with torch.inference_mode():
        for height, width in ((17, 23), (45, 39), (64, 65), (5, 7)):
            lr = torch.rand(1, 3, height, width, dtype=torch.float32, device=device)
            full = model(lr)
            tiled = model.upscale_tensor(lr, tile_size=11, halo=4)
            error = (full - tiled).abs().max().item()
            if full.shape[-2:] != (2*height, 2*width) or not torch.isfinite(tiled).all():
                raise AssertionError(f"Invalid output for {height}x{width} input")
            if error > 2e-5:
                raise AssertionError(f"Tiled inference differs from full inference by {error:.6g}")
            print(f"  {width}x{height} -> {width*2}x{height*2}: max tiled difference {error:.9g}")
    print("PASS: packed INT2 model inference and border-safe tiling are consistent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="Compact .msr, trainer export prefix, .bin/.json path or export folder")
    parser.add_argument("--input", help="Input image file OR folder; native LR images, not bicubic-preupsampled")
    parser.add_argument("--output", help="Output PNG file for one input, or output folder for input directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--tile", type=int, default=384,
                        help="LR tile edge length (default 384, 0=untiled; choose tiles for big images)")
    parser.add_argument("--recursive", action="store_true", help="For folder inputs, include nested folders")
    parser.add_argument("--include-upscaled", action="store_true",
                        help="For folder input, also process files whose names already end in --suffix")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting already existing output files")
    parser.add_argument("--suffix", default="_x2", help="Suffix added to output names (default _x2)")
    parser.add_argument("--discard-alpha", action="store_true", help="Do not preserve transparency")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Quality if output is explicitly JPEG (PNG recommended)")
    parser.add_argument("--self-test", action="store_true", help="Verify this model's tiled results match whole-image inference")
    args = parser.parse_args()
    if args.tile < 0 or not (1 <= args.jpeg_quality <= 100) or not args.suffix:
        parser.error("--tile>=0, --jpeg-quality in [1,100], and non-empty --suffix required")
    if not args.self_test and not args.input:
        parser.error("--input is required (or use --self-test)")
    device = choose_device(args.device)
    if device == "cuda":
        # Avoid reduced-precision FP32 cuDNN math where it might affect PSNR.
        torch.backends.cudnn.allow_tf32 = False
    print(f"Loading exact four-level INT2 model: {args.model}")
    model = MicroSRX2.load(args.model, device=device)
    model.eval()
    print(f"Model ready: architecture=Spatial LR DW2 shuffle, grid={model.grid}, 396 INT2 weights; device={device}")
    if args.self_test:
        run_self_test(model, device, args.tile)
        if not args.input:
            return 0
    source = Path(args.input)
    output = Path(args.output) if args.output else None
    sources = image_paths(source, args.recursive, "" if args.include_upscaled else args.suffix)
    if not sources:
        raise RuntimeError(f"No readable image file extensions found directly in {source}")
    if source.is_dir() and output is not None and output.exists() and not output.is_dir():
        raise ValueError("--output must be a directory when --input is a directory")
    if source.is_file() and output is not None and output.exists() and output.is_dir():
        raise ValueError("--output must be a file when --input is a file")
    failures = 0
    saved = 0
    for i, src in enumerate(sources, 1):
        dest = destination_for(src, source, output, args.recursive, args.suffix)
        if dest.resolve() == src.resolve():
            print(f"ERROR: refusing to overwrite source image: {src}", file=sys.stderr)
            failures += 1
            continue
        if dest.exists() and not args.overwrite:
            print(f"[{i}/{len(sources)}] SKIP existing: {dest}")
            continue
        try:
            start = time.perf_counter()
            with Image.open(src) as image:
                size = image.size
                result = model.upscale_pil(image, tile_size=args.tile, preserve_alpha=not args.discard_alpha)
            save_image(result, dest, quality=args.jpeg_quality)
            elapsed = time.perf_counter() - start
            print(f"[{i}/{len(sources)}] {src.name}: {size[0]}x{size[1]} -> "
                  f"{result.width}x{result.height} in {elapsed:.2f}s | {dest}")
            saved += 1
        except (OSError, ValueError, RuntimeError) as err:
            print(f"[{i}/{len(sources)}] ERROR {src}: {err}", file=sys.stderr)
            failures += 1
    print(f"Finished: {saved} written, {failures} failed, {len(sources)-saved-failures} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
