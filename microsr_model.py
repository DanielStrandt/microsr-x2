#!/usr/bin/env python3
"""Standalone, inference-only MicroSR-X2 2-bit weight model.

Requires PyTorch + NumPy + Pillow. No training code, checkpoint unpickling,
learned extra operations, internet, or GPU-specific kernel is needed.

Input: RGB image in [0,1], NCHW float32, interpreted as *native low resolution*.
Output: 2x RGB float32, bicubic skip plus real four-level INT2 convolution.

Accepted model files:
  - Original trainer export: microsr_int2.bin + microsr_int2.json
  - Compact self-contained .msr (495 bytes: 12-byte header + 483 bytes of params)

CLI to pack an existing trainer export:
  python microsr_model.py pack --export microsr_int2 --output model/microsr_v6.msr

A .msr contains the packed 396 two-bit codes, all 48 FP32 scales and all 48
FP32 biases, and a 12-byte header/checksum. Neither .msr nor the trainer's
export stores FP32 master weights. Packed weights are decoded to a small float
scratch tensor for convolution because ordinary PyTorch convolutions are FP32;
this is weight-quantized inference, not an integer-only convolution engine.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

ARCHITECTURES = {"spatial_lr_dw2_shuffle_v1", "spatial_lr_dw2_shuffle_v3"}
LAYERS = (
    ("expand", (12, 3, 1, 1), 1, 1),
    ("dw1", (12, 1, 3, 3), 12, 3),
    ("dw2", (12, 1, 3, 3), 12, 3),
    ("project", (12, 12, 1, 1), 1, 1),
)
WEIGHT_COUNT = sum(math.prod(shape) for _, shape, _, _ in LAYERS)
PACKED_WEIGHT_BYTES = (WEIGHT_COUNT + 3) // 4
FP32_PARAMETER_BYTES = 4 * sum(shape[0] * 2 for _, shape, _, _ in LAYERS)
MSR_MAGIC = b"MSR2"
MSR_VERSION = 1
MSR_HEADER = struct.Struct("<4sBBBBI")  # magic, version, grid, channels, flags, crc32(payload)
MSR_BYTES = MSR_HEADER.size + PACKED_WEIGHT_BYTES + FP32_PARAMETER_BYTES
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _ensure_finite(values: Any, count: int, name: str, positive: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (count,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain {count} finite numbers")
    if positive and not (arr > 0.0).all():
        raise ValueError(f"{name} must contain strictly positive numbers")
    if (abs(arr) > np.finfo(np.float32).max).any():
        raise ValueError(f"{name} is outside float32 range")
    return arr.astype("<f4")


def _unpack_2bit(packed: bytes) -> np.ndarray:
    if len(packed) != PACKED_WEIGHT_BYTES:
        raise ValueError(f"Expected {PACKED_WEIGHT_BYTES} packed INT2 bytes; got {len(packed)}")
    raw = np.frombuffer(packed, dtype=np.uint8)
    indexes = np.arange(WEIGHT_COUNT, dtype=np.int32)
    return ((raw[indexes // 4] >> ((indexes % 4) * 2)) & 3).astype(np.int8)


def _validate_grid(grid: str) -> None:
    if grid not in ("asym", "balanced"):
        raise ValueError(f"Unsupported INT2 grid {grid!r}")


def _check_meta(meta: dict[str, Any]) -> tuple[str, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    if not isinstance(meta, dict) or meta.get("format") != "MicroSR-X2-INT2-packet-v3":
        raise ValueError("Unsupported export format (expected MicroSR-X2-INT2-packet-v3)")
    if meta.get("architecture") not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {meta.get('architecture')!r}")
    if meta.get("channels") != 12:
        raise ValueError("Only the exact 12-channel / 396-weight Spatial architecture is supported")
    grid = meta.get("int2_grid")
    _validate_grid(grid)
    entries = meta.get("layers")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("Export metadata must contain the four exact Spatial convolution layers")
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for spec, entry in zip(LAYERS, entries):
        name, shape, groups, kernel = spec
        if (not isinstance(entry, dict) or entry.get("name") != name
                or entry.get("shape") != list(shape)
                or entry.get("groups") != groups or entry.get("kernel") != kernel):
            raise ValueError(f"Unexpected layer layout at {name}: this export is not compatible")
        scales = _ensure_finite(entry.get("scales"), shape[0], f"{name}.scales", positive=True)
        bias = _ensure_finite(entry.get("bias"), shape[0], f"{name}.bias")
        arrays.append((scales, bias))
    return grid, tuple(arrays)


def load_trainer_export(path: str | Path) -> tuple[str, bytes, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """Read trainer-generated .bin + .json; never uses torch.load or pickle."""
    prefix = Path(path)
    if prefix.is_dir():
        prefix = prefix / "microsr_int2"
    if prefix.suffix in (".bin", ".json"):
        prefix = prefix.with_suffix("")
    json_path = prefix.with_suffix(".json")
    bin_path = prefix.with_suffix(".bin")
    if not json_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"Need BOTH original export files: {bin_path} AND {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    grid, arrays = _check_meta(meta)
    packed = bin_path.read_bytes()
    _unpack_2bit(packed)
    return grid, packed, arrays


def pack_msr(export_prefix: str | Path, output_path: str | Path) -> Path:
    """Losslessly repack trainer export into a compact, self-contained .msr."""
    grid, packed, layers = load_trainer_export(export_prefix)
    data = [packed]
    for scale, bias in layers:
        data.append(scale.astype("<f4", copy=False).tobytes())
        data.append(bias.astype("<f4", copy=False).tobytes())
    payload = b"".join(data)
    assert len(payload) == PACKED_WEIGHT_BYTES + FP32_PARAMETER_BYTES
    grid_id = 0 if grid == "asym" else 1
    header = MSR_HEADER.pack(MSR_MAGIC, MSR_VERSION, grid_id, 12, 0,
                             zlib.crc32(payload) & 0xFFFFFFFF)
    output = Path(output_path)
    if output.suffix.lower() != ".msr":
        raise ValueError("Compact output file must have a .msr extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + payload)
    if output.stat().st_size != MSR_BYTES:
        raise AssertionError("Compact model has incorrect size")
    return output


def load_msr(path: str | Path) -> tuple[str, bytes, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    contents = Path(path).read_bytes()
    if len(contents) != MSR_BYTES:
        raise ValueError(f"Invalid .msr size: got {len(contents)} bytes, expected {MSR_BYTES}")
    magic, ver, grid_id, chans, flags, stored_crc = MSR_HEADER.unpack_from(contents)
    if magic != MSR_MAGIC or ver != MSR_VERSION or chans != 12 or flags != 0 or grid_id not in (0, 1):
        raise ValueError("This is not a compatible MicroSR-X2 Spatial .msr file")
    payload = contents[MSR_HEADER.size:]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != stored_crc:
        raise ValueError("Model checksum failed: file is corrupted or incomplete")
    packed = payload[:PACKED_WEIGHT_BYTES]
    offset = PACKED_WEIGHT_BYTES
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for name, shape, _, _ in LAYERS:
        count = shape[0]
        length = count * 4
        scales = np.frombuffer(payload[offset:offset + length], dtype="<f4").copy()
        offset += length
        bias = np.frombuffer(payload[offset:offset + length], dtype="<f4").copy()
        offset += length
        scales = _ensure_finite(scales, count, f"{name}.scales", positive=True)
        bias = _ensure_finite(bias, count, f"{name}.bias")
        arrays.append((scales, bias))
    assert offset == len(payload)
    return ("asym" if grid_id == 0 else "balanced"), packed, tuple(arrays)


def _resolve_model(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        compact = path / "microsr_v6.msr"
        if compact.is_file():
            return compact
        compact = path / "microsr_int2.msr"
        if compact.is_file():
            return compact
        return path / "microsr_int2"
    return path


class Int2Conv(nn.Module):
    """Inference convolution with persistent INT2 codes + FP32 scales/biases."""

    def __init__(self, codes: np.ndarray, scales: np.ndarray, bias: np.ndarray,
                 shape: tuple[int, ...], groups: int, kernel: int, grid: str):
        super().__init__()
        self.groups = groups
        self.kernel = kernel
        self.grid = grid
        self.register_buffer("codes", torch.from_numpy(np.array(codes, dtype=np.int8, copy=True).reshape(shape)))
        self.register_buffer("scales", torch.from_numpy(scales.copy()))
        self.register_buffer("bias", torch.from_numpy(bias.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stored 2-bit lanes are decoded only at inference time. This performs
        # *exactly* the trainer's four-level conv, not a float-trained fallback.
        if self.grid == "asym":
            levels = torch.where(self.codes < 2, self.codes, self.codes - 4).to(torch.float32)
        else:
            levels = self.codes.to(torch.float32) - 1.5
        weights = levels * self.scales.reshape(-1, 1, 1, 1)
        if self.kernel == 3:
            x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, weights, self.bias, groups=self.groups)


class MicroSRX2(nn.Module):
    """Standalone packaged V1/V3/V6 Spatial model; 2x native RGB super-resolution."""

    def __init__(self, grid: str, packed: bytes,
                 layer_params: tuple[tuple[np.ndarray, np.ndarray], ...]):
        super().__init__()
        _validate_grid(grid)
        if len(layer_params) != 4:
            raise ValueError("Expected four convolution layers")
        self.grid = grid
        self.channels = 12
        wire = _unpack_2bit(packed)
        index = 0
        for (name, shape, groups, kernel), (scales, bias) in zip(LAYERS, layer_params):
            n = math.prod(shape)
            self.add_module(name, Int2Conv(wire[index:index+n], scales, bias,
                                           shape, groups, kernel, grid))
            index += n
        assert index == WEIGHT_COUNT
        self.eval()

    @classmethod
    def load(cls, model_path: str | Path, device: str | torch.device = "cpu") -> "MicroSRX2":
        path = _resolve_model(model_path)
        if path.suffix.lower() == ".msr":
            grid, packed, params = load_msr(path)
        else:
            grid, packed, params = load_trainer_export(path)
        model = cls(grid, packed, params)
        model.to(device)
        model.eval()
        return model

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.ndim != 4 or lr.shape[1] != 3 or lr.shape[2] < 2 or lr.shape[3] < 2:
            raise ValueError("Input must be an NCHW RGB tensor, with H and W >= 2")
        if lr.dtype != torch.float32:
            raise ValueError("Use float32 RGB tensors in [0,1] for exact matching")
        base = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        x = F.relu(self.expand(lr))
        x = F.relu(self.dw1(x))
        x = F.relu(self.dw2(x))
        return base + F.pixel_shuffle(self.project(x), 2)

    @torch.inference_mode()
    def upscale_tensor(self, lr: torch.Tensor, tile_size: int = 0, halo: int = 4) -> torch.Tensor:
        """Native NCHW float32 LR -> NCHW float32 HR; returns UNCLAMPED floats.

        tile_size is measured in LR pixels; tile_size=0 processes as one piece.
        Halo>=4 makes tiled output agree with un-tiled within FP32 precision.
        The output tensor is allocated on the SAME device as lr.
        """
        if tile_size < 0 or halo < 4:
            raise ValueError("tile_size must be >=0 and halo must be >=4 LR pixels")
        if tile_size == 0:
            return self(lr)
        if lr.ndim != 4 or lr.shape[1] != 3 or min(lr.shape[-2:]) < 2:
            raise ValueError("Expected NCHW RGB image with H,W>=2")
        n, c, h, w = lr.shape
        out = torch.empty((n, c, h*2, w*2), dtype=torch.float32, device=lr.device)
        for y0 in range(0, h, tile_size):
            y1 = min(y0 + tile_size, h)
            ys, ye = max(0, y0 - halo), min(h, y1 + halo)
            for x0 in range(0, w, tile_size):
                x1 = min(x0 + tile_size, w)
                xs, xe = max(0, x0 - halo), min(w, x1 + halo)
                patch = self(lr[:, :, ys:ye, xs:xe])
                out[:, :, 2*y0:2*y1, 2*x0:2*x1] = patch[:, :, 2*(y0-ys):2*(y1-ys), 2*(x0-xs):2*(x1-xs)]
        return out

    @torch.inference_mode()
    def upscale_pil(self, image: Image.Image, tile_size: int = 384,
                    *, preserve_alpha: bool = True, halo: int = 4) -> Image.Image:
        """8-bit Pillow input -> lossless-in-memory RGB/RGBA image, 2x dimensions.

        Processes LR patches on model's GPU/CPU while assembling the final uint8
        output in system RAM, so large photos do not allocate full FP32 HR on GPU.
        The model is NOT responsible for source JPEG decoding or output encoding.
        """
        if tile_size < 0 or halo < 4:
            raise ValueError("tile_size must be >=0; halo must be >=4")
        src = ImageOps.exif_transpose(image)
        if src.mode in ("I", "F") or src.mode.startswith("I;16"):
            raise ValueError("16-bit or floating-point source unsupported: convert intentionally to 8-bit RGB first")
        keep_alpha = preserve_alpha and ("A" in src.getbands() or (src.mode == "P" and "transparency" in src.info))
        pixels = np.asarray(src.convert("RGBA" if keep_alpha else "RGB"), dtype=np.uint8)
        h, w, depth = pixels.shape
        if h < 2 or w < 2:
            raise ValueError("Image must be at least 2x2 low-resolution pixels")
        channels = 4 if keep_alpha else 3
        output = np.empty((2*h, 2*w, channels), dtype=np.uint8)
        device = next(self.buffers()).device
        tile = max(h, w) if tile_size == 0 else tile_size
        for y0 in range(0, h, tile):
            y1 = min(y0+tile, h)
            ys, ye = max(0, y0-halo), min(h, y1+halo)
            for x0 in range(0, w, tile):
                x1 = min(x0+tile, w)
                xs, xe = max(0, x0-halo), min(w, x1+halo)
                crop = np.array(pixels[ys:ye, xs:xe], dtype=np.uint8, copy=True, order="C")
                lr = torch.from_numpy(crop[:, :, :3]).permute(2, 0, 1).unsqueeze(0).to(
                    device=device, dtype=torch.float32).div_(255.0)
                sr = self(lr)
                a, b = 2*(y0-ys), 2*(x0-xs)
                rgb = sr[:, :, a:a+2*(y1-y0), b:b+2*(x1-x0)]
                rgb8 = rgb.clamp(0, 1).mul(255).round().to(torch.uint8)
                output[2*y0:2*y1, 2*x0:2*x1, :3] = rgb8.squeeze(0).permute(1, 2, 0).cpu().numpy()
                if keep_alpha:
                    alpha = torch.from_numpy(crop[:, :, 3].copy()).unsqueeze(0).unsqueeze(0).to(
                        device=device, dtype=torch.float32).div_(255.0)
                    alpha_sr = F.interpolate(alpha, scale_factor=2, mode="bicubic", align_corners=False)
                    alpha_sr = alpha_sr[:, :, a:a+2*(y1-y0), b:b+2*(x1-x0)]
                    output[2*y0:2*y1, 2*x0:2*x1, 3] = alpha_sr.clamp(0, 1).mul(255).round().to(torch.uint8).squeeze().cpu().numpy().reshape((2*(y1-y0), 2*(x1-x0)))
        result = Image.fromarray(output, "RGBA" if keep_alpha else "RGB")
        # Keep the ICC *tag* when pixels stay in their original 8-bit RGB space.
        # EXIF orientation has been applied to pixels, so never copy old orientation.
        if src.mode in ("RGB", "RGBA") and src.info.get("icc_profile"):
            result.info["icc_profile"] = src.info["icc_profile"]
        return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Pack or inspect standalone MicroSR-X2 INT2 model")
    sub = parser.add_subparsers(dest="command", required=True)
    pk = sub.add_parser("pack", help="Convert the trainer's .bin + .json export to a single 495-byte .msr")
    pk.add_argument("--export", required=True, help="Prefix or path to trainer-exported .bin/.json files")
    pk.add_argument("--output", default="model/microsr_v6.msr")
    info = sub.add_parser("info", help="Validate a .msr or exported .bin + .json model")
    info.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.command == "pack":
        target = pack_msr(args.export, args.output)
        print(f"Created {target} ({target.stat().st_size} bytes; 396 INT2 codes + 48 scales + 48 biases)")
        print("V6 trainer and PyTorch checkpoint are NOT required for inference.")
    else:
        mdl = MicroSRX2.load(args.model)
        print(f"Valid MicroSR-X2 Spatial export: grid={mdl.grid}, 2x RGB, 396 INT2 weights")
        print(f"Parameter payload: {PACKED_WEIGHT_BYTES} packed bytes + {FP32_PARAMETER_BYTES} FP32 bytes = 483 bytes")
        print(f"Compact .msr size: {MSR_BYTES} bytes including header/checksum")
        for name, *_ in LAYERS:
            conv = getattr(mdl, name)
            wire = conv.codes.reshape(-1).cpu().numpy().astype(np.uint8)
            hist = np.bincount(wire, minlength=4)
            print(f"  {name}: {wire.size} codes, wire-lane histogram {hist.tolist()}")


if __name__ == "__main__":
    _cli()

