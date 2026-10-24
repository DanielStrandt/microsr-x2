#!/usr/bin/env python3
r"""MicroSR-X2 V6: joint INT2 refinement, coordinated weight pairs and multi-guard selection.

Spatial V1 and Spatial V3 have the same 3->12->DW12->DW12->12 pixel-shuffle
backbone (396 INT2 codes = 99 packed bytes; 48 biases + 48 scales = 492
trainable scalar parameters). V3 changes quantization-aware TRAINING, not the
parameter budget. Optional symmetric midrise four-level grid is an explicit,
separately reported inference variant, not the default.

Recommended V6: continue directly from our best *refined* V5 checkpoint.
All proposals use TRAINING images only. Two disjoint per-round guard sets and
one fixed anchor must each avoid regression. 2-code edits are tested jointly.
No new inference weights, scales, biases, learned operations, or codebooks.
 python microsr_int2_local.py refine-v6 --checkpoint \
   ./runs/microsr_x2_v5_fresh/best.pt --fork-from-refinement \
   --image-dir "./data" --out-dir ./runs/microsr_x2_v6_joint \
   --rounds 4 --cycles-per-round 2 --val-images 0
 Resume via: python microsr_int2_local.py refine-v6 \
   --resume ./runs/microsr_x2_v6_joint/last.pt --image-dir "./data"

Recommended (begin from best Spatial V1 EMA, fresh Adam and EMA):
 python microsr_int2_local.py train --architecture spatial_v3 \
   --finetune-from ./runs/microsr_x2_spatial_ema/best.pt \
   --image-dir "./data" --steps 30000 --lr 1e-5 \
   --lr-min 2e-7 --schedule plateau --plateau-patience 2 \
   --plateau-factor 0.5 --plateau-min-delta 0.005

To compare against legacy STE, repeat with --quant-grad ste and --out-dir
./runs/spatial_v3_ste_control. Checkpoint resume autodetects quantization config.

Controlled ablations, all branches from SAME checkpoint and SAME train crops:
 python microsr_int2_local.py ablate --checkpoint \
   ./runs/microsr_x2_spatial_v3_lsq/best.pt --image-dir "./data" \
   --out-dir ./runs/microsr_x2_v5_ablation --val-images 0

Fresh-crop follow-on refinement from a V4-refined checkpoint (separate run):
 python microsr_int2_local.py refine --checkpoint \
   ./runs/microsr_x2_spatial_discrete/best.pt --fork-from-refinement \
   --crop-seed 92761 --out-dir ./runs/microsr_x2_v5_fresh \
   --image-dir "./data" --val-images 0

"""
import argparse
import csv
import hashlib
import copy
import gc
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_DIR = "data"


def local_image_split(root, seed=42, val_fraction=0.05, min_size=80):
    """Only immediate files in root; never descend into subdirectories."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    paths = []
    skipped = 0
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
            if min(w, h) < min_size:
                skipped += 1
                continue
            paths.append(path)
        except (OSError, ValueError):
            skipped += 1
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 readable images >= {min_size}px directly inside {root}; found {len(paths)}")
    random.Random(seed).shuffle(paths)
    n_val = max(1, min(len(paths) - 1, round(len(paths) * val_fraction)))
    print(f"Using {root}: {len(paths) - n_val} training images, {n_val} held-out validation images; {skipped} skipped")
    print("Only top-level image files are used; subfolders are ignored.")
    return paths[n_val:], paths[:n_val]


def validation_fingerprint(paths):
    # Fingerprint validation membership AND deterministic ordering.
    content = "\n".join(p.name for p in paths).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def image_tensor(image):
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)


def bicubic_down(hr):
    # Torch antialias=True is critical for realistic downsampling.
    return F.interpolate(hr, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True)


def bicubic_up(lr):
    return F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)


def quant_codes(weight, scale, grid="asym"):
    """Returns *integer* 2-bit codes 0..3. Both grids have four values.

    asym code 0,1,2,3 -> -2,-1,0,+1 (legacy inference-compatible)
    balanced code 0,1,2,3 -> -1.5,-0.5,+0.5,+1.5 (zero-free midrise)
    """
    z = weight / scale
    if grid == "asym":
        return torch.round(z).clamp(-2, 1).to(torch.int8) + 2
    if grid == "balanced":
        return torch.round(z + 1.5).clamp(0, 3).to(torch.int8)
    raise ValueError(f"Unknown INT2 grid: {grid}")


def decode_codes(codes, grid="asym"):
    if grid == "asym":
        return codes.to(torch.float32) - 2
    if grid == "balanced":
        return codes.to(torch.float32) - 1.5
    raise ValueError(f"Unknown INT2 grid: {grid}")


def fake_int2(w, scale, grad_mode="ste", grid="asym"):
    """Hard four-level QAT forward, exact round/clamp inference.

    STE master gradient = identity. Trainable quantization scales receive the
    approximate (q - w/s) gradient; LSQ optionally normalizes that scale
    gradient by 1/sqrt(weights-per-output-channel * max positive code level).
    No new inference parameters, codebooks, tensors, or learned offsets.
    """
    if grad_mode not in ("ste", "lsq"):
        raise ValueError(f"Unknown quantization gradient mode: {grad_mode}")
    if grad_mode == "lsq":
        n = w[0].numel()
        qmax = 1.5 if grid == "balanced" else 2.0
        g = 1.0 / math.sqrt(max(1.0, n * qmax))
        scale = scale.detach() + (scale - scale.detach()) * g
    z = w / scale
    if grid == "asym":
        q = torch.round(z).clamp(-2, 1)
    elif grid == "balanced":
        q = (torch.round(z + 1.5) - 1.5).clamp(-1.5, 1.5)
    else:
        raise ValueError(f"Unknown INT2 grid: {grid}")
    q_ste = z + (q - z).detach()
    return q_ste * scale


class QuantConv(nn.Module):
    """Fake-INT2 conv; per-output-channel learnable scales, FP32 biases."""
    def __init__(self, cin, cout, kernel=3, groups=1):
        super().__init__()
        self.kernel = kernel
        self.groups = groups
        # Non-parameter settings; restored from checkpoint metadata.
        self.grad_mode = "ste"
        self.grid = "asym"
        self.weight = nn.Parameter(torch.empty(cout, cin // groups, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.log_scale = nn.Parameter(torch.zeros(cout))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        with torch.no_grad():
            initial = self.weight.abs().flatten(1).mean(dim=1).clamp_min(1e-5)
            self.log_scale.copy_(initial.log())

    def quantized(self):
        scale = self.log_scale.clamp(-14, 5).exp().view(-1, 1, 1, 1)
        return fake_int2(self.weight, scale, self.grad_mode, self.grid)

    def forward(self, x):
        if self.kernel > 1:
            pad = self.kernel // 2
            x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, self.quantized(), self.bias, groups=self.groups)


class SpatialV1(nn.Module):
    """Existing baseline: 396 INT2 conv weights; 48 biases + 48 scales."""
    architecture = "spatial_lr_dw2_shuffle_v1"
    def __init__(self, channels=12):
        super().__init__()
        if channels != 12:
            raise ValueError("For a strict 396-weight comparison, --channels must be 12")
        self.channels = channels
        self.expand = QuantConv(3, 12, kernel=1)
        self.dw1 = QuantConv(12, 12, kernel=3, groups=12)
        self.dw2 = QuantConv(12, 12, kernel=3, groups=12)
        self.project = QuantConv(12, 12, kernel=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        # Make zero-initialized residual head trainable early with INT2 QAT:
        # the previous Kaiming-derived project scale was often ~0.1-0.3,
        # delaying the FIRST nonzero INT2 code until thousands of updates.
        # This affects training initialization only; adds no parameters.
        with torch.no_grad():
            self.project.log_scale.fill_(math.log(0.02))

    def quant_layers(self):
        return (("expand", self.expand), ("dw1", self.dw1),
                ("dw2", self.dw2), ("project", self.project))

    def forward(self, lr):
        base = bicubic_up(lr)
        x = F.relu(self.expand(lr))
        x = F.relu(self.dw1(x))
        x = F.relu(self.dw2(x))
        return base + F.pixel_shuffle(self.project(x), 2)


class SpatialV3(SpatialV1):
    """V1-identical inference backbone; optional improved 4-level QAT.

    The state_dict is bit-for-bit shape-compatible with Spatial V1. With the
    asymmetric codebook, selected V1 EMA weights give identical predictions
    before fine-tuning. No extra operations or parameters at inference.
    """
    architecture = "spatial_lr_dw2_shuffle_v3"


class BottleneckMixV2(nn.Module):
    """Exactly the SAME parameter and INT2 weight budget as SpatialV1.

    3->6 pointwise: 18 weights; depthwise 3x3 (6): 54;
    learned 6->12 pointwise mix: 72; depthwise 3x3 (12): 108;
    12->12 phase project: 144. Total = 396 INT2 weights.
    Layer outputs = 6+6+12+12+12 = 48 scales + 48 biases.
    Total trainable scalar parameters = 492; nominal LR RF = 5x5.

    NO fixed/Hadamard mixing, no additional trainable parameters.
    The new computation is genuine learned channel mixing between spatial layers.
    """
    architecture = "spatial_bottleneck_mix_shuffle_v2"
    def __init__(self, channels=12):
        super().__init__()
        if channels != 12:
            raise ValueError("Mix V2 has a strict 396-weight budget and requires --channels 12")
        self.channels = channels
        self.expand = QuantConv(3, 6, kernel=1)
        self.dw1 = QuantConv(6, 6, kernel=3, groups=6)
        self.mix = QuantConv(6, 12, kernel=1)
        self.dw2 = QuantConv(12, 12, kernel=3, groups=12)
        self.project = QuantConv(12, 12, kernel=1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        # Make zero-initialized residual head trainable early with INT2 QAT:
        # the previous Kaiming-derived project scale was often ~0.1-0.3,
        # delaying the FIRST nonzero INT2 code until thousands of updates.
        # This affects training initialization only; adds no parameters.
        with torch.no_grad():
            self.project.log_scale.fill_(math.log(0.02))

    def quant_layers(self):
        return (("expand", self.expand), ("dw1", self.dw1),
                ("mix", self.mix), ("dw2", self.dw2),
                ("project", self.project))

    def forward(self, lr):
        base = bicubic_up(lr)
        x = F.relu(self.expand(lr))
        x = F.relu(self.dw1(x))
        x = F.relu(self.mix(x))
        x = F.relu(self.dw2(x))
        return base + F.pixel_shuffle(self.project(x), 2)


ARCHITECTURES = {"spatial_v1": SpatialV1, "spatial_v3": SpatialV3,
                 "mix_v2": BottleneckMixV2}
ARCH_FROM_ID = {cls.architecture: cls for cls in ARCHITECTURES.values()}
MicroSR = SpatialV1  # Backward-compatible import for older scripts.


def build_model(architecture, channels=12):
    if architecture in ARCHITECTURES:
        cls = ARCHITECTURES[architecture]
    elif architecture in ARCH_FROM_ID:
        cls = ARCH_FROM_ID[architecture]
    else:
        raise ValueError(f"Unknown architecture {architecture!r}. Options: {list(ARCHITECTURES)}")
    model = cls(channels)
    weights = sum(layer.weight.numel() for _, layer in model.quant_layers())
    output_parameters = sum(layer.bias.numel() + layer.log_scale.numel()
                            for _, layer in model.quant_layers())
    total = sum(p.numel() for p in model.parameters())
    if (weights, output_parameters, total) != (396, 96, 492):
        raise AssertionError(f"Unexpected model budget: {weights} INT2 weights, "
                             f"{output_parameters} biases/scales, {total} total")
    return model


def configure_quantization(model, grad_mode="ste", grid="asym"):
    if grad_mode not in ("ste", "lsq") or grid not in ("asym", "balanced"):
        raise ValueError("quant-grad must be ste/lsq and quant-grid asym/balanced")
    for _, layer in model.quant_layers():
        layer.grad_mode = grad_mode
        layer.grid = grid
    model.quant_grad = grad_mode
    model.quant_grid = grid
    return model


def effective_code_summary(model):
    sections = []
    for name, layer in model.quant_layers():
        scale = layer.log_scale.detach().clamp(-14, 5).exp().view(-1, 1, 1, 1)
        codes = quant_codes(layer.weight.detach(), scale, layer.grid)
        levels = torch.bincount(codes.reshape(-1).to(torch.int64), minlength=4).tolist()
        sections.append(f"{name}=[{','.join(str(int(x)) for x in levels)}]")
    return "codes_per_level_0to3: " + " ".join(sections)


class RandomCropStream(IterableDataset):
    """Worker-local crop reservoir; decode one image to produce many crops.

    The reservoir is periodically refreshed so training samples remain diverse.
    Each worker receives a disjoint memory budget. No subdirectories are scanned.
    """
    def __init__(self, paths, patch, context, seed, cache_bytes,
                 crops_per_image=48, refresh_crops=256, pool_mb_per_worker=256, max_source_mp=50.0):
        self.max_source_mp = max_source_mp
        self.paths = list(paths)
        self.full = patch + 2 * context
        self.seed = seed
        self.cache_bytes = cache_bytes
        self.crops_per_image = crops_per_image
        self.refresh_crops = refresh_crops
        self.pool_mb_per_worker = pool_mb_per_worker

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + wid * 1000003 + int(torch.initial_seed()) % (2**31))
        bytes_per_crop = self.full * self.full * 3
        # A hard per-worker ceiling avoids committing the whole 12 GB budget
        # as one gigantic array on Windows (and starving image decoders).
        worker_budget = min(self.cache_bytes // workers, int(self.pool_mb_per_worker * 1024**2))
        capacity = max(1, min(250000, worker_budget // bytes_per_crop))
        # Store uint8, not float32. Crop buffers are independent of source images.
        pool = np.empty((capacity, self.full, self.full, 3), dtype=np.uint8)
        filled = 0
        cursor = 0
        yielded = 0
        ordered = list(self.paths)
        rng.shuffle(ordered)
        path_cursor = 0

        def next_path():
            nonlocal path_cursor
            if path_cursor >= len(ordered):
                rng.shuffle(ordered)
                path_cursor = 0
            item = ordered[path_cursor]
            path_cursor += 1
            return item

        def replenish(count):
            nonlocal filled, cursor
            remaining = count
            failures = 0
            while remaining > 0:
                path = next_path()
                n = min(self.crops_per_image, remaining)
                try:
                    # IMPORTANT: NEVER convert the entire source image to RGB/NumPy.
                    # Large panoramas and TIFFs otherwise create multi-GB temporary
                    # buffers and kill Windows DataLoader workers.
                    with Image.open(path) as im:
                        w, h = im.size
                        # Some TIFF/panorama decoders allocate the entire source
                        # even for a small PIL.crop(). Skip giant sources to
                        # keep Windows worker RAM bounded (training only).
                        if w * h > self.max_source_mp * 1_000_000:
                            raise ValueError(f"Oversized training image: {w}x{h}")
                        for _ in range(n):
                            if w >= self.full and h >= self.full:
                                x = 2 * rng.randrange((w - self.full) // 2 + 1)
                                y = 2 * rng.randrange((h - self.full) // 2 + 1)
                                patch = im.crop((x, y, x + self.full, y + self.full))
                                crop = np.asarray(patch.convert("RGB"), dtype=np.uint8)
                            else:
                                # Decode only the original small image, then pad
                                # reflectively to get a valid training context.
                                small = np.asarray(im.convert("RGB"), dtype=np.uint8)
                                ph = max(0, self.full - h)
                                pw = max(0, self.full - w)
                                small = np.pad(small, ((ph//2, ph-ph//2),
                                                       (pw//2, pw-pw//2), (0,0)),
                                               mode="reflect")
                                crop = small[:self.full, :self.full]
                            pool[cursor] = crop
                            cursor = (cursor + 1) % capacity
                            filled = min(capacity, filled + 1)
                            del crop
                except (OSError, ValueError, MemoryError) as exc:
                    failures += 1
                    gc.collect()
                    if failures >= 30:
                        raise RuntimeError(
                            "Unable to refill crop pool after 30 problematic images; "
                            "check available RAM and input image formats"
                        ) from exc
                    continue
                remaining -= n

        # Avoid multi-GB startup stalls. Fill only a modest pool initially.
        replenish(min(capacity, max(256, self.refresh_crops)))
        while True:
            if yielded and yielded % self.refresh_crops == 0:
                replenish(min(capacity, self.refresh_crops))
            idx = rng.randrange(filled)
            crop = pool[idx]
            if rng.getrandbits(1):
                crop = crop[:, ::-1]
            if rng.getrandbits(1):
                crop = crop[::-1]
            rot = rng.randrange(4)
            if rot:
                crop = np.rot90(crop, rot)
            yielded += 1
            yield torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)


def make_loader(paths, args):
    stream = RandomCropStream(paths, args.patch, args.context, args.seed + 123,
                              int(args.cache_gb * 1024**3),
                              args.crops_per_image, args.refresh_crops,
                              args.pool_mb_per_worker, args.max_source_mp)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=torch.cuda.is_available() and str(args.device).startswith("cuda"))
    if args.workers:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch)
    return DataLoader(stream, **kwargs)


def central(t, context):
    return t[..., context:-context, context:-context] if context else t


def mse_psnr(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


@torch.no_grad()
def evaluate(model, paths, device, max_images=20, shave=2, val_patch=512, positions=5):
    """Fixed, bounded HR patches from several locations per held-out image.

    Scores are average per-patch MSE converted to PSNR. Never creates a full
    resolution RGB tensor. Position offsets are stable between evaluations.
    """
    locations = [(0.5, 0.5)] if positions == 1 else [
        (0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    locations = locations[:positions]
    model.eval()
    sr_error = bc_error = 0.0
    count = 0
    try:
        for path in tqdm(paths[:max_images] if max_images else paths,
                         desc="Validation", leave=False):
            with Image.open(path) as im:
                w, h = im.size
                side = min(w - w % 2, h - h % 2, val_patch)
                side -= side % 2
                if side <= 2 * shave + 4:
                    continue
                for fx, fy in locations:
                    left = 2 * round(((w - side) * fx) / 2)
                    top = 2 * round(((h - side) * fy) / 2)
                    cropped = im.crop((left, top, left + side, top + side))
                    hr = image_tensor(cropped).unsqueeze(0).to(device)
                    del cropped
                    lr = bicubic_down(hr)
                    sr = model(lr).clamp(0, 1)
                    bc = bicubic_up(lr).clamp(0, 1)
                    if shave:
                        hr = hr[..., shave:-shave, shave:-shave]
                        sr = sr[..., shave:-shave, shave:-shave]
                        bc = bc[..., shave:-shave, shave:-shave]
                    sr_error += F.mse_loss(sr, hr).item()
                    bc_error += F.mse_loss(bc, hr).item()
                    count += 1
                    del hr, lr, sr, bc
    finally:
        model.train()
    if not count:
        raise RuntimeError("No suitable validation images found")
    return {"sr_psnr": mse_psnr(sr_error / count),
            "bicubic_psnr": mse_psnr(bc_error / count), "patches": count,
            "validation": f"{len(locations)} deterministic positions/image, max {val_patch}px HR"}


def learning_rate(step, total, peak, floor, warmup):
    if step <= warmup and warmup:
        return peak * (0.2 + 0.8 * step / warmup)
    t = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return floor + (peak - floor) * (0.5 + 0.5 * math.cos(math.pi * t))


class ModelEMA:
    """EMA of FP32 master weights and quantization scales; inference still uses INT2."""

    def __init__(self, model, decay=0.995):
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # Faster adaptation at the beginning; gradually approach requested decay.
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1 - d)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


@torch.no_grad()
def quantization_comparison(raw_model, ema_model):
    """Differences in deployable INT2 codes and FP32 master/scales."""
    mismatches = total = 0
    max_master_gap = max_scale_gap = 0.0
    details = []
    raw_layers = list(raw_model.quant_layers())
    ema_layers = list(ema_model.quant_layers())
    if [n for n, _ in raw_layers] != [n for n, _ in ema_layers]:
        raise ValueError("Cannot compare incompatible architectures")
    for (name, a), (_, b) in zip(raw_layers, ema_layers):
        sa = a.log_scale.clamp(-14, 5).exp().view(-1, 1, 1, 1)
        sb = b.log_scale.clamp(-14, 5).exp().view(-1, 1, 1, 1)
        if a.grid != b.grid:
            raise ValueError("Cannot compare different four-level grids")
        qa = quant_codes(a.weight, sa, a.grid)
        qb = quant_codes(b.weight, sb, b.grid)
        n = int((qa != qb).sum().item())
        mismatches += n
        total += qa.numel()
        details.append(f"{name}:{n}")
        max_master_gap = max(max_master_gap, float((a.weight - b.weight).abs().max()))
        max_scale_gap = max(max_scale_gap, float((sa - sb).abs().max()))
    return (f"int2_code_differences={mismatches}/{total} "
            f"by_layer=({','.join(details)}) "
            f"max_master_gap={max_master_gap:.3g} "
            f"max_scale_gap={max_scale_gap:.3g}")


def save_checkpoint(path, model, opt, step, best, config, crop_rng_state,
                    ema=None, inference_weights="raw", scheduler_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
             "best_gain": best, "config": config, "inference_weights": inference_weights,
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "crop_rng": crop_rng_state, "scheduler_state": scheduler_state}
    if ema is not None:
        state["ema_model"] = ema.model.state_dict()
        state["ema_updates"] = ema.updates
    torch.save(state, tmp)
    tmp.replace(path)


def load_model(checkpoint, device, use_inference=True):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    arch = cfg.get("architecture")
    if arch not in ARCH_FROM_ID:
        raise ValueError(f"Unknown/incompatible checkpoint architecture {arch!r}")
    model = build_model(arch, cfg.get("channels", 12)).to(device)
    configure_quantization(model, cfg.get("quant_grad", "ste"),
                           cfg.get("quant_grid", "asym"))
    chosen = ck.get("inference_weights", "raw")
    weights = ck.get("ema_model") if use_inference and chosen == "ema" else ck["model"]
    if weights is None:
        raise ValueError("Checkpoint selects EMA inference but contains no EMA weights")
    model.load_state_dict(weights, strict=True)
    return model, ck


def append_metrics(run, record):
    path = Path(run) / "validation.csv"
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(record)


def train(args):
    if args.crops_per_image < 1 or args.refresh_crops < 1:
        raise ValueError("Crop pool refill parameters must be positive")
    if args.pool_mb_per_worker < 16 or args.max_source_mp < 1:
        raise ValueError("--pool-mb-per-worker >=16 and --max-source-mp >=1 are required")
    if args.workers < 0 or args.cache_gb <= 0 or args.prefetch < 1:
        raise ValueError("--workers must be >=0, --cache-gb >0, --prefetch >=1")
    if args.patch <= 0 or args.patch % 2 or args.context % 2:
        raise ValueError("--patch and --context must be even")
    if args.context < 8:
        raise ValueError("--context must be >=8 HR pixels")
    if not args.patches or any(v < 16 or v % 2 for v in args.patches):
        raise ValueError("--patches must all be even and >=16")
    if not 0 < args.ema_decay < 1:
        raise ValueError("--ema-decay must be in (0,1)")
    if args.resume and args.finetune_from:
        raise ValueError("Use either --resume OR --finetune-from")
    if args.steps < 1 or args.val_every < 1 or args.log_every < 1:
        raise ValueError("--steps, --val-every, --log-every must be positive")
    if args.snapshot_every < 1:
        raise ValueError("--snapshot-every must be positive")
    if args.plateau_patience < 1 or not 0 < args.plateau_factor < 1:
        raise ValueError("--plateau-patience >=1, --plateau-factor in (0,1) required")
    if args.plateau_min_delta < 0 or args.min_gain_delta < 0:
        raise ValueError("Plateau and save improvement thresholds must be nonnegative")
    if args.lr_min <= 0 or args.lr <= 0 or args.lr_min > args.lr:
        raise ValueError("Learning rates require 0 < --lr-min <= --lr")
    if args.val_positions not in (1, 5) or args.val_patch < 16 or args.val_patch % 2:
        raise ValueError("Invalid validation positions/patch size")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be in (0,1)")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    source_ck = None
    if args.resume or args.finetune_from:
        # Inspect metadata before instantiating any model or choosing output folder.
        source_ck = torch.load(args.resume or args.finetune_from,
                               map_location="cpu", weights_only=False)
        if args.resume and source_ck.get("config", {}).get("discrete_refinement"):
            raise ValueError("Discrete search checkpoints must resume with the 'refine' "
                             "command. To begin QAT from refined weights, use "
                             "'train --finetune-from <discrete_best.pt>' instead.")
        checkpoint_arch = source_ck.get("config", {}).get("architecture")
        if args.resume:
            # Exact continuation uses the original settings. To change the
            # learning rate or other training recipe, start a --finetune-from
            # branch instead of silently altering a resumed run.
            saved_cfg = source_ck.get("config", {})
            for key in ("lr", "lr_min", "ema_decay", "patches", "context",
                        "val_images", "val_patch", "val_positions",
                        "plateau_patience", "plateau_factor", "min_gain_delta",
                        "warmup", "restore_best_on_plateau", "plateau_min_delta",
                        "quant_grad", "quant_grid"):
                if key in saved_cfg:
                    setattr(args, key, saved_cfg[key])
        if checkpoint_arch not in ARCH_FROM_ID:
            raise ValueError(f"Unsupported checkpoint architecture {checkpoint_arch!r}")
        if args.architecture is None:
            architecture = checkpoint_arch
        else:
            architecture = build_model(args.architecture, args.channels).architecture
            if architecture != checkpoint_arch:
                v1_to_v3 = (bool(args.finetune_from) and
                            checkpoint_arch == SpatialV1.architecture and
                            architecture == SpatialV3.architecture)
                if not v1_to_v3:
                    raise ValueError("Checkpoint architecture incompatible with selected run: "
                                     f"{checkpoint_arch} -> {architecture}. "
                                     "Only selected V1 -> V3 fine-tuning is supported.")
        if source_ck["config"].get("channels", 12) != args.channels:
            raise ValueError("Checkpoint --channels does not match requested --channels")
        # Keep the original dataset split, including seed and fraction, on resume.
        cfg = source_ck["config"]
        if args.seed != cfg.get("seed", args.seed) or args.val_fraction != cfg.get("val_fraction", args.val_fraction):
            print("Using checkpoint seed/validation fraction to retain identical held-out images.")
            args.seed = cfg.get("seed", args.seed)
            args.val_fraction = cfg.get("val_fraction", args.val_fraction)
    else:
        architecture = build_model(args.architecture or "spatial_v3", args.channels).architecture

    is_v2 = architecture == BottleneckMixV2.architecture
    is_v3 = architecture == SpatialV3.architecture
    source_quant_grid = (source_ck or {}).get("config", {}).get("quant_grid", "asym")
    if args.quant_grad is None:
        args.quant_grad = "lsq" if is_v3 and not args.resume else "ste"
    if args.quant_grid is None:
        args.quant_grid = source_quant_grid if source_ck else "asym"
    if args.finetune_from and args.quant_grid != source_quant_grid:
        raise ValueError("A different quantization grid changes inference semantics. "
                         "Start from scratch for --quant-grid balanced; do not silently "
                         "reinterpret the weights of an existing checkpoint.")
    if args.schedule is None:
        if args.resume:
            fallback = "plateau" if is_v2 or is_v3 or source_ck.get("config", {}).get("finetune_mode", False) else "cosine"
            args.schedule = source_ck.get("config", {}).get("schedule", fallback)
        else:
            args.schedule = "plateau" if is_v2 or is_v3 or args.finetune_from else "cosine"
    if args.finetune_from:
        args.schedule = "plateau"
        args.warmup = 0
    elif args.resume:
        args.warmup = source_ck.get("config", {}).get("warmup", args.warmup)
    if args.warmup < 0 or args.warmup >= args.steps:
        raise ValueError("--warmup must be >=0 and < --steps")

    if args.out_dir is None:
        if args.resume:
            args.out_dir = str(Path(args.resume).parent)
        elif args.finetune_from:
            if is_v3:
                args.out_dir = f"./runs/microsr_x2_spatial_v3_{args.quant_grad}"
            else:
                args.out_dir = "./runs/microsr_x2_mix_v2_finetune" if is_v2 else "./runs/microsr_x2_spatial_finetune"
        else:
            if is_v3:
                args.out_dir = f"./runs/microsr_x2_spatial_v3_{args.quant_grid}_{args.quant_grad}"
            else:
                args.out_dir = "./runs/microsr_x2_mix_v2" if is_v2 else "./runs/microsr_x2_spatial_ema"
    run = Path(args.out_dir)
    if args.finetune_from and run.resolve() == Path(args.finetune_from).resolve().parent:
        raise ValueError("Fine-tune output directory must differ from source checkpoint directory")
    if args.resume and run.resolve() != Path(args.resume).resolve().parent:
        raise ValueError("--resume must use the checkpoint's existing output directory; "
                         "use --finetune-from for a new run")
    if not args.resume and (run / "best.pt").exists():
        raise FileExistsError(f"{run}/best.pt already exists. Use --resume or another "
                              "--out-dir; refusing to overwrite a previous best model")

    train_paths, valid_paths = local_image_split(args.image_dir, args.seed, args.val_fraction,
                                                args.patch + 2 * args.context)
    fingerprint = validation_fingerprint(valid_paths)
    if args.resume and source_ck["config"].get("val_fingerprint") and source_ck["config"]["val_fingerprint"] != fingerprint:
        raise ValueError("Held-out validation image membership/order has changed since checkpoint")
    # Filter only giant TRAINING sources; never drop validation images silently.
    training_sources = []
    omitted_giant = 0
    for path in train_paths:
        try:
            with Image.open(path) as im:
                if im.width * im.height > args.max_source_mp * 1_000_000:
                    omitted_giant += 1
                    continue
        except (OSError, ValueError):
            continue
        training_sources.append(path)
    if omitted_giant:
        print(f"Training source safety: excluded {omitted_giant} images larger than "
              f"{args.max_source_mp:g} megapixels (held-out validation unchanged).")
    if not training_sources:
        raise ValueError("No memory-safe training images remained")

    model = build_model(architecture, args.channels).to(device)
    configure_quantization(model, args.quant_grad, args.quant_grid)
    ema = ModelEMA(model, args.ema_decay)
    start = 0
    best = float("-inf")
    scheduler_state = {"lr": args.lr, "bad_validations": 0, "reductions": 0}
    fine_mode = bool(args.finetune_from)
    if args.finetune_from:
        loaded, source_ck = load_model(args.finetune_from, device, use_inference=True)
        model.load_state_dict(loaded.state_dict())
        ema = ModelEMA(model, args.ema_decay)
        print(f"Fine-tuning {architecture} from selected="
              f"{source_ck.get('inference_weights','raw')} checkpoint {args.finetune_from}; "
              f"fresh Adam and EMA, {args.steps} additional steps")
    if args.resume:
        loaded, source_ck = load_model(args.resume, device, use_inference=False)
        model.load_state_dict(loaded.state_dict())
        ema = ModelEMA(model, args.ema_decay)
        if source_ck.get("ema_model") is not None:
            ema.model.load_state_dict(source_ck["ema_model"])
            ema.updates = int(source_ck.get("ema_updates", 0))
        start = int(source_ck["step"])
        if source_ck.get("scheduler_state"):
            scheduler_state.update(source_ck["scheduler_state"])
        fine_mode = bool(source_ck.get("config", {}).get("finetune_mode", False))
        scheme = source_ck.get("config", {}).get("validation_scheme")
        expected_scheme = ("spatial_five_position_ema_v3_lsq" if is_v3 else
                           "five_position_ema_v3" if is_v2 else
                           "spatial_five_position_ema_v2")
        if scheme == expected_scheme and (source_ck.get("config", {}).get("val_images", args.val_images) == args.val_images):
            best = float(source_ck.get("best_gain", float("-inf")))
        else:
            print("Validation protocol changed; resetting best metric to avoid incomparable scores")
        if "torch_rng" in source_ck:
            torch.set_rng_state(source_ck["torch_rng"])
        if "python_rng" in source_ck:
            random.setstate(source_ck["python_rng"])
        print(f"Resuming architecture={architecture}, step={start}/{args.steps}, schedule={args.schedule}")

    # Construct optimizer AFTER the active model is chosen; preserves fixed resume bug.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.resume:
        if source_ck.get("config", {}).get("optimizer_binding_fixed") and source_ck.get("optimizer"):
            opt.load_state_dict(source_ck["optimizer"])
            print("Restored Adam optimizer moments from verified checkpoint")
        else:
            print("Reset Adam optimizer moments (legacy unsafe checkpoint)")
    def check_optimizer_binding():
        if {id(p) for p in model.parameters()} != {id(p) for g in opt.param_groups for p in g["params"]}:
            raise RuntimeError("Optimizer is detached from the live model")
    check_optimizer_binding()

    if args.resume:
        scheduler_state.setdefault("plateau_best", best)
    elif args.finetune_from:
        scheduler_state["plateau_best"] = float("-inf")
    else:
        scheduler_state["plateau_best"] = float("-inf")
    config = {
        "architecture": architecture, "channels": args.channels,
        "quant_grad": args.quant_grad, "quant_grid": args.quant_grid,
        "patch": args.patch, "patches": list(args.patches), "context": args.context,
        "seed": args.seed, "image_dir": str(args.image_dir),
        "val_fraction": args.val_fraction, "val_fingerprint": fingerprint,
        "validation_scheme": ("spatial_five_position_ema_v3_lsq" if is_v3 else
                              "five_position_ema_v3" if is_v2 else
                              "spatial_five_position_ema_v2"),
        "val_positions": args.val_positions, "val_patch": args.val_patch,
        "val_images": args.val_images, "lr": args.lr, "lr_min": args.lr_min,
        "steps": args.steps, "ema_decay": args.ema_decay,
        "optimizer_binding_fixed": True, "schedule": args.schedule,
        "warmup": args.warmup, "finetune_mode": fine_mode,
        "finetune_source": str(args.finetune_from) if args.finetune_from else
                           source_ck.get("config", {}).get("finetune_source") if source_ck else None,
        "plateau_patience": args.plateau_patience,
        "plateau_factor": args.plateau_factor,
        "restore_best_on_plateau": args.restore_best_on_plateau,
        "min_gain_delta": args.min_gain_delta,
        "plateau_min_delta": args.plateau_min_delta,
        "max_source_mp": args.max_source_mp,
    }
    weighted = sum(layer.weight.numel() for _, layer in model.quant_layers())
    print(f"Architecture={architecture}; INT2 weights={weighted}, packed INT2 bytes={(weighted+3)//4}; "
          f"FP32 biases/scales=96, total trainable={sum(p.numel() for p in model.parameters())}")
    print(f"Device={device}; training_images={len(training_sources)}; "
          f"held_out_images={len(valid_paths)}; fingerprint={fingerprint}")
    print(f"Quantization=grid:{args.quant_grid}, training_gradient:{args.quant_grad}; "
          "all inference convolutions use FOUR levels, never full precision")
    print(f"Schedule={args.schedule}; start_lr={args.lr:.4g}; min_lr={args.lr_min:.4g}; "
          f"plateau_patience={args.plateau_patience}; "
          f"plateau_min_delta={args.plateau_min_delta:.4g} dB; "
          f"restore_best={args.restore_best_on_plateau}")
    run.mkdir(parents=True, exist_ok=True)
    args.patch = max(args.patches)
    loader = make_loader(training_sources, args)
    batches = iter(loader)
    print(f"DataLoader workers={args.workers}; max crop reservoir={args.pool_mb_per_worker} MiB/worker; "
          f"max training image={args.max_source_mp:g} MP; pinned_memory={loader.pin_memory}", flush=True)

    if args.finetune_from and not args.resume:
        eval0 = evaluate(model, valid_paths, device, args.val_images,
                         val_patch=args.val_patch, positions=args.val_positions)
        best = eval0["sr_psnr"] - eval0["bicubic_psnr"]
        scheduler_state["plateau_best"] = best
        print(f"FINETUNE BASELINE: SR={eval0['sr_psnr']:.4f} "
              f"bicubic={eval0['bicubic_psnr']:.4f} gain={best:+.4f} dB")
        save_checkpoint(run / "best.pt", model, opt, 0, best, config, None,
                        ema, inference_weights="raw", scheduler_state=scheduler_state)
        save_checkpoint(run / "last.pt", model, opt, 0, best, config, None,
                        ema, inference_weights="raw", scheduler_state=scheduler_state)

    log_at = time.perf_counter()
    data_total = train_total = 0.0
    measurements = 0
    for step in range(start+1, args.steps+1):
        model.train()
        data_begin = time.perf_counter()
        hr = next(batches).to(device, non_blocking=True).float().div_(255.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        data_end = time.perf_counter()
        patch = random.choice(args.patches)
        full = patch + 2 * args.context
        if hr.shape[-1] > full:
            y = 2 * random.randrange((hr.shape[-2]-full)//2+1)
            x = 2 * random.randrange((hr.shape[-1]-full)//2+1)
            hr = hr[..., y:y+full, x:x+full]
        if args.schedule == "plateau":
            lr_rate = scheduler_state["lr"]
            if args.warmup and step <= args.warmup and start == 0:
                lr_rate *= (0.2+0.8*step/args.warmup)
        else:
            lr_rate = learning_rate(step, args.steps, args.lr, args.lr_min, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr_rate
        lr_image = bicubic_down(hr)
        prediction = model(lr_image)
        loss = F.mse_loss(central(prediction, args.context), central(hr, args.context))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_end = time.perf_counter()
        data_total += data_end - data_begin
        train_total += train_end - data_end
        measurements += 1
        if step == 1 or step % args.log_every == 0:
            duration = time.perf_counter() - log_at
            print(f"step={step}/{args.steps} mse={loss.item():.8f} "
                  f"crop_psnr={mse_psnr(loss.item()):.3f} dB patch={patch} lr={lr_rate:.7g} "
                  f"data={1000*data_total/measurements:.1f}ms "
                  f"gpu_train={1000*train_total/measurements:.1f}ms "
                  f"steps_per_sec={measurements/max(duration,1e-9):.2f}", flush=True)
            log_at = time.perf_counter()
            data_total = train_total = 0.0
            measurements = 0
        if step % args.snapshot_every == 0:
            save_checkpoint(run / f"step_{step:07d}.pt", model, opt, step, best,
                            config, None, ema, scheduler_state=scheduler_state)
        if step % args.val_every == 0 or step == args.steps:
            # Save before validation: recoverable even if image decoding fails.
            save_checkpoint(run / "last.pt", model, opt, step, best,
                            config, None, ema, scheduler_state=scheduler_state)
            raw = evaluate(model, valid_paths, device, args.val_images,
                           val_patch=args.val_patch, positions=args.val_positions)
            avg = evaluate(ema.model, valid_paths, device, args.val_images,
                           val_patch=args.val_patch, positions=args.val_positions)
            raw_gain = raw["sr_psnr"] - raw["bicubic_psnr"]
            ema_gain = avg["sr_psnr"] - avg["bicubic_psnr"]
            winner = "ema" if ema_gain > raw_gain else "raw"
            gain = max(raw_gain, ema_gain)
            print(f"VALID step={step}: bicubic={raw['bicubic_psnr']:.4f} dB "
                  f"raw={raw['sr_psnr']:.4f} dB ({raw_gain:+.4f}) "
                  f"EMA={avg['sr_psnr']:.4f} dB ({ema_gain:+.4f}) "
                  f"selected={winner} {quantization_comparison(model, ema.model)}", flush=True)
            improved = gain > best + args.min_gain_delta
            plateau_reference = scheduler_state.get("plateau_best", best)
            meaningful = gain > plateau_reference + args.plateau_min_delta
            if meaningful:
                scheduler_state["plateau_best"] = gain
                scheduler_state["bad_validations"] = 0
            elif args.schedule == "plateau":
                scheduler_state["bad_validations"] += 1
            if improved:
                best = gain
                save_checkpoint(run / "best.pt", model, opt, step, best, config,
                                None, ema, inference_weights=winner,
                                scheduler_state=scheduler_state)
                print(f"NEW BEST: gain={best:+.4f} dB ({winner})", flush=True)
            append_metrics(run, {
                "step": step, "architecture": architecture,
                "bicubic_psnr": f"{raw['bicubic_psnr']:.7f}",
                "raw_psnr": f"{raw['sr_psnr']:.7f}", "ema_psnr": f"{avg['sr_psnr']:.7f}",
                "raw_gain": f"{raw_gain:.7f}", "ema_gain": f"{ema_gain:.7f}",
                "selected": winner, "best_gain": f"{best:.7f}",
                "learning_rate": f"{lr_rate:.10g}",
                "count_patches": raw["patches"], "improved": int(improved),
                "meaningful_for_plateau": int(meaningful),
                "bad_validations": scheduler_state["bad_validations"],
            })
            if (args.schedule == "plateau" and not meaningful and
                    scheduler_state["bad_validations"] >= args.plateau_patience):
                old_lr = scheduler_state["lr"]
                next_lr = max(args.lr_min, old_lr*args.plateau_factor)
                scheduler_state["bad_validations"] = 0
                if next_lr < old_lr:
                    scheduler_state["lr"] = next_lr
                    scheduler_state["reductions"] += 1
                    print(f"PLATEAU: reducing LR {old_lr:.5g} -> {next_lr:.5g}")
                    if args.restore_best_on_plateau and (run / "best.pt").is_file():
                        # Restore the checkpoint-selected best inference WEIGHTS,
                        # but do not swap the optimizer's attached module.
                        best_model, _ = load_model(run / "best.pt", device,
                                                   use_inference=True)
                        model.load_state_dict(best_model.state_dict())
                        del best_model
                        ema = ModelEMA(model, args.ema_decay)
                        opt = torch.optim.Adam(model.parameters(), lr=next_lr)
                        check_optimizer_binding()
                        print("PLATEAU: restored best inference weights; reset Adam and EMA")
            save_checkpoint(run / "last.pt", model, opt, step, best, config,
                            None, ema, scheduler_state=scheduler_state)
    print(f"Final INT2 distribution: {effective_code_summary(model)}")
    print(f"Done. Best {architecture} validation gain: {best:+.4f} dB")


# -----------------------------------------------------------------------------
# V4 DIRECT DISCRETE INT2 REFINEMENT (NO NEW INFERENCE PARAMETERS)
# -----------------------------------------------------------------------------

def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def _atomic_json_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temp.replace(path)


def _reference_codes(model):
    result = {}
    for name, layer in model.quant_layers():
        with torch.no_grad():
            s = layer.log_scale.detach().clamp(-14, 5).exp().view(-1, 1, 1, 1)
            result[name] = quant_codes(layer.weight.detach(), s, layer.grid).detach().clone()
    return result


@torch.no_grad()
def snap_to_int2_centers(model):
    """Encode the deployed INT2 codes explicitly in the FP32 master tensors.

    This changes NO effective code/scale/bias and adds NO inference state. It
    prevents a long-unused master weight value from influencing code edits.
    """
    before = _reference_codes(model)
    for name, layer in model.quant_layers():
        scale = layer.log_scale.detach().clamp(-14, 5).exp().view(-1, 1, 1, 1)
        layer.weight.copy_(decode_codes(before[name], layer.grid).to(layer.weight.dtype) * scale)
    after = _reference_codes(model)
    for name in before:
        if not torch.equal(before[name], after[name]):
            raise RuntimeError(f'INT2 snapping changed deployed codes in layer {name}')


@torch.no_grad()
def set_int2_code(model, layer_name, index, desired):
    """Set one deployed code, preserving other codes and existing scale/bias."""
    layers = dict(model.quant_layers())
    layer = layers[layer_name]
    index = int(index)
    desired = int(desired)
    if desired not in (0, 1, 2, 3) or not 0 <= index < layer.weight.numel():
        raise ValueError('Invalid INT2 code or weight index')
    old_codes = _reference_codes(model)[layer_name]
    old = int(old_codes.reshape(-1)[index].item())
    channel = index // layer.weight[0].numel()
    scale = layer.log_scale.detach()[channel].clamp(-14, 5).exp()
    new_value = (torch.tensor(float(desired), device=scale.device) -
                 (2.0 if layer.grid == 'asym' else 1.5)) * scale
    layer.weight.view(-1)[index].copy_(new_value)
    verification = quant_codes(
        layer.weight.detach(),
        layer.log_scale.detach().clamp(-14, 5).exp().view(-1, 1, 1, 1),
        layer.grid)
    if int(verification.reshape(-1)[index]) != desired:
        raise RuntimeError('INT2 code edit did not survive quantization')
    return old


@torch.no_grad()
def set_int2_scale(model, layer_name, channel, new_log_scale, original_codes=None):
    """Change one channel's learned scale while preserving all its INT2 codes."""
    layer = dict(model.quant_layers())[layer_name]
    channel = int(channel)
    if not -14 < new_log_scale < 5:
        raise ValueError('Scale proposal exceeds the allowed exponent range')
    codes = (_reference_codes(model)[layer_name][channel] if original_codes is None
             else original_codes)
    layer.log_scale[channel].fill_(float(new_log_scale))
    s = layer.log_scale[channel].exp()
    layer.weight[channel].copy_(decode_codes(codes, layer.grid).to(layer.weight.dtype) * s)
    now = quant_codes(layer.weight.detach()[channel:channel+1],
                      s.reshape(1, 1, 1, 1), layer.grid)[0]
    if not torch.equal(codes, now):
        raise RuntimeError('Scale proposal inadvertently modified INT2 codes')


def make_discrete_manifest(train_paths, search_patches, guard_patches, patch_size,
                           context, seed, max_source_mp):
    """Disjoint image identities; sample crops from training membership ONLY."""
    full = patch_size + 2 * context
    eligible = []
    for p in train_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
            if min(w, h) >= full and w * h <= max_source_mp * 1_000_000:
                eligible.append((p.name, w, h))
        except (OSError, ValueError):
            continue
    total = search_patches + guard_patches
    if len(eligible) < total:
        raise ValueError(f'Refinement needs {total} distinct training images with sides >= {full}px; '
                         f'only {len(eligible)} eligible. Lower --search-patches and --guard-patches.')
    rng = random.Random(seed + 93517)
    rng.shuffle(eligible)
    manifest = {'version': 1, 'patch_size': patch_size, 'context': context,
                'seed': seed, 'search': [], 'guard': []}
    for i, (name, w, h) in enumerate(eligible[:total]):
        spec = {'name': name,
                'left': 2 * rng.randrange((w - full) // 2 + 1),
                'top': 2 * rng.randrange((h - full) // 2 + 1),
                'flip': bool(rng.getrandbits(1)), 'rotate': rng.randrange(4)}
        (manifest['search'] if i < search_patches else manifest['guard']).append(spec)
    return manifest


def assert_discrete_manifest(manifest, train_paths, valid_paths, args):
    train_names = {p.name for p in train_paths}
    held_names = {p.name for p in valid_paths}
    specs = manifest.get('search', []) + manifest.get('guard', [])
    names = [s['name'] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError('The search and guard image identities are not disjoint')
    if any(name not in train_names or name in held_names for name in names):
        raise ValueError('Refinement manifest uses an image outside the training split')
    if manifest.get('patch_size') != args.patch_size or manifest.get('context') != args.context:
        raise ValueError('Refinement patch/context settings changed since initial run')
    if len(manifest['search']) != args.search_patches or len(manifest['guard']) != args.guard_patches:
        raise ValueError('Refinement crop counts changed since initial run')


def build_discrete_cache(specs, image_dir, patch_size, context, device):
    """Fixed bicubic LR inputs and HR targets; all data fits in modest GPU RAM."""
    full = patch_size + 2 * context
    lrs, targets = [], []
    for item in tqdm(specs, desc='Build fixed TRAIN crop cache', leave=False):
        path = Path(image_dir) / item['name']
        with Image.open(path) as im:
            if item['left'] < 0 or item['top'] < 0 or (item['left']+full > im.width or
                                                       item['top']+full > im.height):
                raise ValueError(f'Dataset image size changed since patch manifest: {path}')
            crop = im.crop((item['left'], item['top'], item['left']+full, item['top']+full))
            if item['flip']:
                crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for _ in range(item['rotate']):
                crop = crop.transpose(Image.Transpose.ROTATE_90)
            rgb = crop.convert('RGB')
            arr = np.asarray(rgb, dtype=np.uint8).copy()
        hr = (torch.from_numpy(arr).permute(2, 0, 1)
              .unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0))
        with torch.no_grad():
            lrs.append(bicubic_down(hr))
            targets.append(central(hr, context))
    return {'lr': torch.cat(lrs, dim=0), 'target': torch.cat(targets, dim=0),
            'count': len(specs)}


@torch.no_grad()
def score_discrete(model, cache, batch_size, context):
    model.eval()
    error_sum = torch.zeros((), device=cache['lr'].device, dtype=torch.float64)
    count = cache['count']
    for i in range(0, count, batch_size):
        lr = cache['lr'][i:i+batch_size]
        gt = cache['target'][i:i+batch_size]
        pred = central(model(lr).clamp_(0.0, 1.0), context)
        per_patch = (pred - gt).square().mean(dim=(1, 2, 3)).to(torch.float64)
        error_sum += per_patch.sum()
    return float((error_sum / count).item())


def discrete_gradient_ranking(model, cache, args):
    """Rank candidate code flips cheaply; actual acceptance uses *true* MSE.

    Straight-through gradients ONLY choose which single-code edits to test.
    Nothing is accepted on gradient predictions or held-out PSNR.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    limit = min(cache['count'], args.rank_patches)
    for i in range(0, limit, args.probe_batch):
        lr = cache['lr'][i:min(i+args.probe_batch, limit)]
        gt = cache['target'][i:min(i+args.probe_batch, limit)]
        pred = central(model(lr).clamp(0, 1), args.context)
        loss = F.mse_loss(pred, gt) * (len(lr) / limit)
        loss.backward()
    ranked = []
    for name, layer in model.quant_layers():
        codes = _reference_codes(model)[name].flatten().tolist()
        gradients = (layer.weight.grad.detach().flatten() if layer.weight.grad is not None
                     else torch.zeros_like(layer.weight).flatten())
        scales = layer.log_scale.detach().clamp(-14, 5).exp().repeat_interleave(layer.weight[0].numel())
        for idx, current in enumerate(codes):
            g = float((gradients[idx] * scales[idx]).item())
            for new_code in range(4):
                if current != new_code:
                    expected = g * (new_code - current)
                    ranked.append((expected, name, idx, int(current), new_code))
    model.zero_grad(set_to_none=True)
    ranked.sort(key=lambda item: item[0])
    rng = random.Random(args.seed + 707213 * (args._round + 1))
    # Mix promising gradient-ranked changes with diverse exploratory changes.
    take = min(args.probes_per_round, len(ranked))
    n_priority = min(take, round(take * (1.0 - args.explore_fraction)))
    priority = ranked[:n_priority]
    rest = ranked[n_priority:]
    exploratory = rng.sample(rest, min(take-n_priority, len(rest)))
    ordered = priority + exploratory
    return [{'layer': name, 'index': idx, 'old': before, 'new': after,
             'estimated_delta': est} for est, name, idx, before, after in ordered]


def scale_candidate_plan(model, args):
    items = []
    for name, layer in model.quant_layers():
        for channel in range(layer.weight.shape[0]):
            logscale = float(layer.log_scale.detach()[channel].item())
            for sign in (-1, 1):
                shift = sign * math.log1p(args.scale_fraction)
                if -14 < logscale + shift < 5:
                    items.append({'layer': name, 'channel': channel,
                                  'expected_logscale': logscale, 'shift': shift})
    rng = random.Random(args.seed + 171341 * (args._round + 1))
    rng.shuffle(items)
    return items[:args.scale_probes]


def calibrate_discrete_bias(model, search, guard, args, initial_search, initial_guard):
    """Optional Adam on EXISTING 48 FP32 biases; reject regressions on train guard.

    Only 48 existing biases are modified. Convolution codes and scales stay fixed.
    Entire routine is transactional: if no guard-approved improvement, restore.
    """
    if args.bias_steps <= 0:
        return initial_search, initial_guard, False
    biases = [layer.bias for _, layer in model.quant_layers()]
    originals = [p.detach().clone() for p in biases]
    optimizer = torch.optim.Adam(biases, lr=args.bias_lr)
    best_biases = None
    best_search, best_guard = initial_search, initial_guard
    for iteration in range(args.bias_steps):
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, search['count'], args.probe_batch):
            lr = search['lr'][i:i+args.probe_batch]
            gt = search['target'][i:i+args.probe_batch]
            prediction = central(model(lr).clamp(0, 1), args.context)
            loss = F.mse_loss(prediction, gt) * (len(lr) / search['count'])
            # No need to propagate gradients to the quantized master weights.
            gradients = torch.autograd.grad(loss, biases, allow_unused=True)
            for p, g in zip(biases, gradients):
                if g is not None:
                    if p.grad is None:
                        p.grad = g.detach().clone()
                    else:
                        p.grad.add_(g.detach())
        optimizer.step()
        if ((iteration+1) % args.bias_check_every == 0 or
                iteration+1 == args.bias_steps):
            candidate_search = score_discrete(model, search, args.probe_batch, args.context)
            if candidate_search < best_search * (1 - args.min_relative_gain):
                candidate_guard = score_discrete(model, guard, args.probe_batch, args.context)
                if candidate_guard <= best_guard * (1 + args.guard_tolerance):
                    best_search, best_guard = candidate_search, candidate_guard
                    best_biases = [p.detach().clone() for p in biases]
    with torch.no_grad():
        for p, old in zip(biases, best_biases if best_biases is not None else originals):
            p.copy_(old)
    model.zero_grad(set_to_none=True)
    return best_search, best_guard, best_biases is not None


def _refine_save(run, model, state, config, reported_gain=None):
    payload = {'model': model.state_dict(), 'optimizer': None,
               'ema_model': None, 'ema_updates': 0,
               'step': state['accepted_codes'] + state['accepted_scales'] + state['accepted_bias_rounds'],
               'best_gain': reported_gain,
               'config': config, 'inference_weights': 'raw',
               'refinement_state': copy.deepcopy(state)}
    _atomic_torch_save(payload, run / 'last.pt')
    # Every accepted change must improve the TRAIN guard objective, thus the
    # current committed model is our best according to this train-only metric.
    _atomic_torch_save(payload, run / 'best.pt')


def _refine_csv(run, row):
    csv_path = run / 'refinement.csv'
    columns = ['round', 'phase', 'probes', 'accepted_codes', 'accepted_scales',
               'accepted_bias_rounds', 'search_psnr', 'guard_psnr',
               'search_mse', 'guard_mse']
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)


def refine(args):
    """Direct discrete-coordinate INT2 optimization, evaluated on train images."""
    if bool(args.checkpoint) == bool(args.resume):
        raise ValueError('Use exactly one of --checkpoint (new) or --resume (continue)')
    if args.rounds < 1 or args.search_patches < 1 or args.guard_patches < 1:
        raise ValueError('Positive --rounds, --search-patches, --guard-patches required')
    if args.patch_size < 32 or args.patch_size % 2 or args.context < 8 or args.context % 2:
        raise ValueError('--patch-size even and >=32; --context even and >=8 required')
    if args.mode not in ('all', 'codes', 'scales', 'bias'):
        raise ValueError('Refinement --mode must be all/codes/scales/bias')
    if args.mode in ('all', 'codes') and args.probes_per_round < 1:
        raise ValueError('Code-search modes need --probes-per-round >= 1')
    if args.probe_batch < 1 or args.rank_patches < 1:
        raise ValueError('Probe batch and rank patches must be positive')
    if args.scale_probes < 0 or args.bias_steps < 0 or args.scale_fraction <= 0:
        raise ValueError('Invalid scale/bias refinement settings')
    if args.bias_steps and (args.bias_lr <= 0 or args.bias_check_every < 1):
        raise ValueError('Bias calibration needs positive learning rate and check interval')
    if not 0 <= args.explore_fraction <= 1 or args.guard_tolerance < 0 or args.min_relative_gain < 0:
        raise ValueError('explore fraction must be [0,1]; MSE thresholds nonnegative')
    if args.checkpoint_every < 1 or args.max_source_mp < 1:
        raise ValueError('Checkpoint frequency/max source size must be positive')
    device = torch.device(args.device)
    if args.resume:
        run = Path(args.resume).resolve().parent
        model, source_ck = load_model(args.resume, device, use_inference=True)
        cfg = source_ck.get('config', {})
        if not cfg.get('discrete_refinement'):
            raise ValueError('Not a discrete-refinement checkpoint; use --checkpoint instead')
        settings = cfg['refine_settings']
        for key, value in settings.items():
            setattr(args, key, value)
        args.out_dir = str(run)
        args._round = 0
        state = copy.deepcopy(source_ck['refinement_state'])
        print(f'Resuming discrete refinement: round={state["round"]+1}/{args.rounds}, '
              f'phase={state["phase"]}, total_probes={state["total_probes"]}')
    else:
        model, source_ck = load_model(args.checkpoint, device, use_inference=True)
        cfg_src = source_ck.get('config', {})
        if cfg_src.get('discrete_refinement') and not args.fork_from_refinement:
            raise ValueError('Refined source checkpoint: use --fork-from-refinement to start '
                             'a NEW search on fresh training crops, or --resume to continue it')
        args.seed = cfg_src.get('seed', 42)
        args.val_fraction = cfg_src.get('val_fraction', 0.05)
        args.out_dir = args.out_dir or './runs/microsr_x2_spatial_discrete'
        run = Path(args.out_dir).resolve()
        if run.resolve() == Path(args.checkpoint).resolve().parent:
            raise ValueError('Refinement directory cannot overwrite source checkpoint directory')
        if (run / 'best.pt').exists() or (run / 'last.pt').exists():
            raise FileExistsError(f'{run} already contains checkpoints; choose a new --out-dir')
        args._round = 0
        state = {'round': 0, 'phase': 'plan_codes', 'cursor': 0,
                 'proposals': [], 'total_probes': 0, 'accepted_codes': 0,
                 'accepted_scales': 0, 'accepted_bias_rounds': 0,
                 'search_mse': None, 'guard_mse': None,
                 'source_checkpoint': str(args.checkpoint),
                 'source_selected': source_ck.get('inference_weights', 'raw')}
        print(f'Discrete refinement of {model.architecture} selected='
              f'{state["source_selected"]} from {args.checkpoint}')

    if model.architecture not in (SpatialV1.architecture, SpatialV3.architecture):
        raise ValueError('Discrete INT2 search targets the successful Spatial 12-channel backbone; '
                         'do not run it on Mix V2')
    if args.seed is None or args.val_fraction is None:
        raise ValueError('Source checkpoint has incomplete dataset split metadata')
    if args.crop_seed is None:
        args.crop_seed = args.seed
    if args.mode not in ('all', 'codes', 'scales', 'bias'):
        raise ValueError(f'Unknown refinement mode {args.mode!r}')
    if args.image_dir is None:
        args.image_dir = cfg.get('image_dir', DEFAULT_IMAGE_DIR) if args.resume else cfg_src.get('image_dir', DEFAULT_IMAGE_DIR)
    train_paths, valid_paths = local_image_split(args.image_dir, args.seed, args.val_fraction, 80)
    reference_cfg = source_ck.get('config', {})
    expected_fp = reference_cfg.get('val_fingerprint')
    if expected_fp is not None and validation_fingerprint(valid_paths) != expected_fp:
        raise ValueError('Dataset validation split changed; cannot use this checkpoint safely')
    print(f'Refinement split fingerprint={validation_fingerprint(valid_paths)}; '
          f'validation images={len(valid_paths)} (NEVER used for weight selection)')
    if args.resume:
        manifest = json.loads((run / 'refinement_manifest.json').read_text(encoding='utf-8'))
        config = cfg
        if (config.get('val_fingerprint') and
                config['val_fingerprint'] != validation_fingerprint(valid_paths)):
            raise ValueError('Validation membership/order changed since refinement started')
    else:
        run.mkdir(parents=True, exist_ok=True)
        manifest = make_discrete_manifest(train_paths, args.search_patches, args.guard_patches,
                                          args.patch_size, args.context, args.crop_seed,
                                          args.max_source_mp)
        manifest['split_fingerprint'] = validation_fingerprint(valid_paths)
        _atomic_json_save(manifest, run / 'refinement_manifest.json')
        configure_quantization(model, model.quant_grad, model.quant_grid)
        snap_to_int2_centers(model)
        settings = {key: getattr(args, key) for key in (
            'rounds', 'search_patches', 'guard_patches', 'patch_size', 'context',
            'probe_batch', 'rank_patches', 'probes_per_round', 'explore_fraction',
            'scale_probes', 'scale_fraction', 'bias_steps', 'bias_lr',
            'bias_check_every', 'min_relative_gain', 'guard_tolerance',
            'checkpoint_every', 'max_source_mp', 'seed', 'val_fraction',
            'mode', 'crop_seed', 'fork_from_refinement')}
        config = dict(reference_cfg)
        config.update({'discrete_refinement': True,
                       'refinement_scheme': 'train_only_code_coordinate_search_v2',
                       'refine_mode': args.mode,
                       'refine_settings': settings, 'source_checkpoint': str(args.checkpoint),
                       'val_fingerprint': validation_fingerprint(valid_paths)})

    assert_discrete_manifest(manifest, train_paths, valid_paths, args)
    model.eval()
    torch.manual_seed(args.seed + 67413)
    random.seed(args.seed + 67413)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    search_cache = build_discrete_cache(manifest['search'], args.image_dir,
                                        args.patch_size, args.context, device)
    guard_cache = build_discrete_cache(manifest['guard'], args.image_dir,
                                       args.patch_size, args.context, device)
    if state['search_mse'] is None:
        state['search_mse'] = score_discrete(model, search_cache, args.probe_batch, args.context)
        state['guard_mse'] = score_discrete(model, guard_cache, args.probe_batch, args.context)
        _refine_save(run, model, state, config)
        _refine_csv(run, {'round': 0, 'phase': 'baseline',
                          'probes': 0, 'accepted_codes': 0, 'accepted_scales': 0,
                          'accepted_bias_rounds': 0,
                          'search_psnr': mse_psnr(state['search_mse']),
                          'guard_psnr': mse_psnr(state['guard_mse']),
                          'search_mse': state['search_mse'], 'guard_mse': state['guard_mse']})
    else:
        # Detect corrupt checkpoints, changed images, or inconsistent settings.
        check_a = score_discrete(model, search_cache, args.probe_batch, args.context)
        check_b = score_discrete(model, guard_cache, args.probe_batch, args.context)
        if abs(check_a - state['search_mse']) > max(1e-10, 1e-5*state['search_mse']) or \
           abs(check_b - state['guard_mse']) > max(1e-10, 1e-5*state['guard_mse']):
            raise RuntimeError('Saved refinement objective changed on resume; '
                               'check dataset and precision/determinism settings')
    print(f'FIXED TRAIN-only MSE: search={state["search_mse"]:.9g} '
          f'({mse_psnr(state["search_mse"]):.4f} dB), '
          f'guard={state["guard_mse"]:.9g} '
          f'({mse_psnr(state["guard_mse"]):.4f} dB). '
          f'Search samples={args.search_patches}, separate guard samples={args.guard_patches}.')
    print(f'Model budget: 396 deployed INT2 codes (99 packed bytes), '
          f'48 biases + 48 scales (492 total scalar parameters). '
          f'Grid={model.quant_grid}; quant-grad={model.quant_grad}; '
          f'edit_mode={args.mode}; crop_seed={args.crop_seed}.')

    this_invocation_probes = 0
    while state['round'] < args.rounds:
        args._round = state['round']
        if state['phase'] == 'plan_codes':
            if args.mode in ('all', 'codes'):
                state['proposals'] = discrete_gradient_ranking(model, search_cache, args)
                state['cursor'] = 0
                state['phase'] = 'codes'
                _refine_save(run, model, state, config)
                print(f'ROUND {state["round"]+1}/{args.rounds}: '
                      f'prioritized {len(state["proposals"])} single-code candidates', flush=True)
            else:
                state['proposals'] = []
                state['cursor'] = 0
                state['phase'] = 'plan_scales'
                _refine_save(run, model, state, config)

        if state['phase'] == 'codes':
            while state['cursor'] < len(state['proposals']):
                cand = state['proposals'][state['cursor']]
                name, idx = cand['layer'], cand['index']
                current_codes = _reference_codes(model)[name].reshape(-1)
                current = int(current_codes[idx].item())
                if current == cand['old']:
                    previous = set_int2_code(model, name, idx, cand['new'])
                    trial_search = score_discrete(model, search_cache, args.probe_batch, args.context)
                    good = False
                    if trial_search < state['search_mse'] * (1 - args.min_relative_gain):
                        trial_guard = score_discrete(model, guard_cache, args.probe_batch, args.context)
                        if trial_guard <= state['guard_mse'] * (1 + args.guard_tolerance):
                            state['search_mse'], state['guard_mse'] = trial_search, trial_guard
                            state['accepted_codes'] += 1
                            good = True
                            print(f'  ACCEPT code {name}[{idx}] {previous}->{cand["new"]}: '
                                  f'search={mse_psnr(trial_search):.5f} dB '
                                  f'guard={mse_psnr(trial_guard):.5f} dB', flush=True)
                    if not good:
                        set_int2_code(model, name, idx, previous)
                    state['total_probes'] += 1
                    this_invocation_probes += 1
                state['cursor'] += 1
                if (state['total_probes'] % args.checkpoint_every == 0 or
                        state['cursor'] == len(state['proposals'])):
                    _refine_save(run, model, state, config)
                if args.stop_after_probes and this_invocation_probes >= args.stop_after_probes:
                    _refine_save(run, model, state, config)
                    print('Stopped at requested probe budget; resume with refine --resume '
                          f'"{run / "last.pt"}"')
                    return
            state['phase'] = 'plan_scales'
            state['proposals'] = []
            state['cursor'] = 0
            _refine_save(run, model, state, config)

        if state['phase'] == 'plan_scales':
            if args.mode in ('all', 'scales'):
                state['proposals'] = scale_candidate_plan(model, args)
                state['cursor'] = 0
                state['phase'] = 'scales'
            else:
                state['proposals'] = []
                state['cursor'] = 0
                state['phase'] = 'bias'
            _refine_save(run, model, state, config)
        if state['phase'] == 'scales':
            while state['cursor'] < len(state['proposals']):
                cand = state['proposals'][state['cursor']]
                layer = dict(model.quant_layers())[cand['layer']]
                channel = cand['channel']
                previous_log = float(layer.log_scale.detach()[channel])
                if abs(previous_log - cand['expected_logscale']) < 1e-7:
                    before = _reference_codes(model)[cand['layer']][channel].clone()
                    set_int2_scale(model, cand['layer'], channel, previous_log + cand['shift'], before)
                    trial_search = score_discrete(model, search_cache, args.probe_batch, args.context)
                    good = False
                    if trial_search < state['search_mse'] * (1 - args.min_relative_gain):
                        trial_guard = score_discrete(model, guard_cache, args.probe_batch, args.context)
                        if trial_guard <= state['guard_mse'] * (1 + args.guard_tolerance):
                            state['search_mse'], state['guard_mse'] = trial_search, trial_guard
                            state['accepted_scales'] += 1
                            good = True
                            print(f'  ACCEPT scale {cand["layer"]}[{channel}] '
                                  f'{math.expm1(cand["shift"]):+.2%}: '
                                  f'guard={mse_psnr(trial_guard):.5f} dB', flush=True)
                    if not good:
                        set_int2_scale(model, cand['layer'], channel, previous_log, before)
                    state['total_probes'] += 1
                    this_invocation_probes += 1
                state['cursor'] += 1
                if (state['total_probes'] % args.checkpoint_every == 0 or
                        state['cursor'] == len(state['proposals'])):
                    _refine_save(run, model, state, config)
                if args.stop_after_probes and this_invocation_probes >= args.stop_after_probes:
                    _refine_save(run, model, state, config)
                    print('Stopped at requested probe budget; resume with refine --resume '
                          f'"{run / "last.pt"}"')
                    return
            state['phase'] = 'bias'
            state['proposals'] = []
            state['cursor'] = 0
            _refine_save(run, model, state, config)

        if state['phase'] == 'bias':
            if args.mode in ('all', 'bias'):
                next_a, next_b, approved = calibrate_discrete_bias(
                    model, search_cache, guard_cache, args,
                    state['search_mse'], state['guard_mse'])
            else:
                next_a, next_b, approved = state['search_mse'], state['guard_mse'], False
            state['search_mse'], state['guard_mse'] = next_a, next_b
            if approved:
                state['accepted_bias_rounds'] += 1
                print(f'  ACCEPT bias calibration: guard={mse_psnr(next_b):.5f} dB', flush=True)
            state['round'] += 1
            state['phase'] = 'plan_codes'
            state['proposals'] = []
            state['cursor'] = 0
            _refine_save(run, model, state, config)
            print(f'END ROUND {state["round"]}/{args.rounds}: '
                  f'codes_accepted={state["accepted_codes"]}, '
                  f'scale_accepted={state["accepted_scales"]}, '
                  f'bias_rounds={state["accepted_bias_rounds"]}, '
                  f'probes={state["total_probes"]}, '
                  f'search={mse_psnr(next_a):.5f} dB, '
                  f'guard={mse_psnr(next_b):.5f} dB', flush=True)
            _refine_csv(run, {'round': state['round'], 'phase': 'round_complete',
                              'probes': state['total_probes'],
                              'accepted_codes': state['accepted_codes'],
                              'accepted_scales': state['accepted_scales'],
                              'accepted_bias_rounds': state['accepted_bias_rounds'],
                              'search_psnr': mse_psnr(next_a), 'guard_psnr': mse_psnr(next_b),
                              'search_mse': next_a, 'guard_mse': next_b})

    # Held-out validation is REPORT ONLY, never involved in selection/acceptance.
    print('TRAIN-ONLY SEARCH COMPLETE. Evaluating held-out images for reporting only.', flush=True)
    gain = None
    if args.val_images >= 0:
        result = evaluate(model, valid_paths, device, args.val_images,
                          val_patch=args.val_patch, positions=args.val_positions)
        gain = result['sr_psnr'] - result['bicubic_psnr']
        print(f'FINAL HELD-OUT REPORT (not used for model selection): '
              f'bicubic={result["bicubic_psnr"]:.4f} dB '
              f'INT2={result["sr_psnr"]:.4f} dB '
              f'gain={gain:+.4f} dB, patches={result["patches"]}', flush=True)
        _atomic_json_save({'checkpoint': str(run / 'best.pt'),
                           'selected_on': 'disjoint TRAIN search/guard MSE only',
                           'validation_is_report_only': True, 'validation': result,
                           'gain_db': gain, 'training_search_mse': state['search_mse'],
                           'training_guard_mse': state['guard_mse'],
                           'accepted_codes': state['accepted_codes'],
                           'accepted_scales': state['accepted_scales'],
                           'accepted_bias_rounds': state['accepted_bias_rounds']},
                          run / 'final_report.json')
    _refine_save(run, model, state, config, reported_gain=gain)
    print(f'DONE: {effective_code_summary(model)}')
    print(f'Export with: python microsr_int2_local.py export --checkpoint '
          f'"{run / "best.pt"}" --output "{run / "microsr_int2"}"')


# -----------------------------------------------------------------------------
# V6: TRAIN-ONLY JOINT SINGLE/PAIR INT2 SEARCH, MULTI-GUARD AND CROP ROTATION
# -----------------------------------------------------------------------------

def v6_make_manifest(train_paths, valid_paths, args):
    """Create fixed, reproducible train-only crops before ANY weight search.

    The anchor uses distinct training images; each round has disjoint search
    and independent guard image identities (also disjoint from the anchor).
    A round rotates to new crop/image samples but never touches validation.
    """
    anchor_seed = args.crop_seed + 8163
    anchor_manifest = make_discrete_manifest(
        train_paths, args.anchor_patches, 1, args.patch_size, args.context,
        anchor_seed, args.max_source_mp)
    anchor = anchor_manifest['search']
    used_anchor = {spec['name'] for spec in anchor}
    pool = [p for p in train_paths if p.name not in used_anchor]
    manifest = {
        'version': 6,
        'validation_fingerprint': validation_fingerprint(valid_paths),
        'patch_size': args.patch_size, 'context': args.context,
        'seed': args.crop_seed,
        'anchor': anchor, 'rounds': [],
    }
    for rnd in range(args.rounds):
        sub = make_discrete_manifest(
            pool, args.search_patches, args.guard_sets * args.guard_patches,
            args.patch_size, args.context, args.crop_seed + 100003 * (rnd+1),
            args.max_source_mp)
        guards = [sub['guard'][i*args.guard_patches:(i+1)*args.guard_patches]
                  for i in range(args.guard_sets)]
        manifest['rounds'].append({'round': rnd, 'search': sub['search'], 'guards': guards})
    return manifest


def v6_check_manifest(manifest, train_paths, valid_paths, args):
    if manifest.get('version') != 6:
        raise ValueError('Invalid V6 manifest version')
    if manifest['validation_fingerprint'] != validation_fingerprint(valid_paths):
        raise ValueError('Validation split changed since V6 manifest was created')
    if (manifest['patch_size'], manifest['context']) != (args.patch_size, args.context):
        raise ValueError('V6 patch/context settings do not match saved manifest')
    if len(manifest['rounds']) != args.rounds or len(manifest['anchor']) != args.anchor_patches:
        raise ValueError('V6 round or anchor count differs from saved manifest')
    train_names = {p.name for p in train_paths}
    val_names = {p.name for p in valid_paths}
    anchor_names = {spec['name'] for spec in manifest['anchor']}
    if len(anchor_names) != args.anchor_patches:
        raise ValueError('V6 anchor repeats an image identity')
    if anchor_names - train_names or anchor_names & val_names:
        raise ValueError('V6 anchor uses non-training images')
    for rnd, block in enumerate(manifest['rounds']):
        if block['round'] != rnd or len(block['search']) != args.search_patches:
            raise ValueError('Invalid V6 round search manifest')
        if len(block['guards']) != args.guard_sets or any(
                len(g) != args.guard_patches for g in block['guards']):
            raise ValueError('Invalid V6 multi-guard manifest')
        specs = block['search'] + [s for guard in block['guards'] for s in guard]
        names = [s['name'] for s in specs]
        if len(set(names)) != len(names) or set(names) & anchor_names:
            raise ValueError(f'V6 round {rnd+1} train crop identities overlap guards/anchor')
        if set(names) - train_names or set(names) & val_names:
            raise ValueError(f'V6 round {rnd+1} uses non-training images')


def v6_cache_round(manifest, rnd, args, device):
    block = manifest['rounds'][rnd]
    print(f'V6 ROUND {rnd+1}/{args.rounds}: loading fresh search and '
          f'{args.guard_sets} disjoint guard sets (training images only)', flush=True)
    search = build_discrete_cache(block['search'], args.image_dir,
                                  args.patch_size, args.context, device)
    guards = [build_discrete_cache(g, args.image_dir, args.patch_size,
                                   args.context, device) for g in block['guards']]
    return search, guards


def v6_objectives(model, search, guards, anchor, args):
    """Baseline/resume check; each number is average per-patch MSE."""
    return (score_discrete(model, search, args.probe_batch, args.context),
            [score_discrete(model, guard, args.probe_batch, args.context)
             for guard in guards],
            score_discrete(model, anchor, args.probe_batch, args.context))


def v6_accept(model, search, guards, anchor, state, args):
    """Pure TRAIN-only acceptance. Search must improve; EVERY guard must not regress."""
    trial_search = score_discrete(model, search, args.probe_batch, args.context)
    if not trial_search < state['search_mse'] * (1 - args.min_relative_gain):
        return None
    trial_guards = []
    for guard, old_mse in zip(guards, state['guard_mse']):
        mse = score_discrete(model, guard, args.probe_batch, args.context)
        if mse > old_mse * (1 + args.guard_tolerance):
            return None
        trial_guards.append(mse)
    anchor_mse = score_discrete(model, anchor, args.probe_batch, args.context)
    if anchor_mse > state['anchor_mse'] * (1 + args.guard_tolerance):
        return None
    return trial_search, trial_guards, anchor_mse


def v6_commit_objectives(state, accepted):
    state['search_mse'], state['guard_mse'], state['anchor_mse'] = accepted


def v6_plan_pairs(model, failed, args):
    """Build coordinated proposals from previously rejected single edits.

    Individual edits were rejected at the time tested, but because other
    updates can occur in between, these are not claimed to be mathematically
    synergistic pairs. Every pair is actually tested jointly before acceptance.
    """
    codes = _reference_codes(model)
    unique = {}
    for candidate in failed:
        name, idx = candidate['layer'], int(candidate['index'])
        if int(codes[name].view(-1)[idx]) != int(candidate['old']):
            continue
        key = (name, idx, int(candidate['new']))
        if key not in unique or candidate['estimated_delta'] < unique[key]['estimated_delta']:
            unique[key] = candidate
    rng = random.Random(args.crop_seed + 191413 * (args._round+1))
    ordered = sorted(unique.values(), key=lambda c: c['estimated_delta'])
    if len(ordered) < 2 or args.pair_probes <= 0:
        return []
    pool_count = min(args.pair_pool, len(ordered))
    n_top = max(2, round(pool_count * 0.65))
    top = ordered[:n_top]
    rest = ordered[n_top:]
    pool = top + rng.sample(rest, min(pool_count-len(top), len(rest)))
    seen = set()
    proposals = []

    def add_pair(a, b, reason):
        if a['layer'] == b['layer'] and a['index'] == b['index']:
            return
        one = (a['layer'], int(a['index']), int(a['new']))
        two = (b['layer'], int(b['index']), int(b['new']))
        canonical = tuple(sorted((one, two)))
        if canonical in seen:
            return
        seen.add(canonical)
        proposals.append({'first': dict(a), 'second': dict(b),
                          'reason': reason,
                          'estimated_delta': (a['estimated_delta'] + b['estimated_delta'])})

    # First preferentially search code pairs within the same learned output
    # channel, where two kernels/coefficients can compensate for each other.
    layers = dict(model.quant_layers())
    for a in top[:min(len(top), max(24, args.pair_probes // 2))]:
        width = layers[a['layer']].weight[0].numel()
        channel = int(a['index']) // width
        same = [b for b in pool if b['layer'] == a['layer'] and
                int(b['index']) // width == channel and b['index'] != a['index']]
        if same:
            for b in rng.sample(same, min(3, len(same))):
                add_pair(a, b, 'same_output_channel')

    # Also explore cross-layer and more distant interactions. Include some
    # proposals with less favorable individual gradient estimates, since
    # these are the cases where coordination can matter most.
    trials = 0
    while len(proposals) < max(args.pair_probes * 3, args.pair_probes+64) and trials < args.pair_probes * 40:
        a, b = rng.sample(pool, 2)
        add_pair(a, b, 'cross_layer' if a['layer'] != b['layer'] else 'same_layer')
        trials += 1
    near = sorted(proposals, key=lambda c: c['estimated_delta'])
    n_priority = min(args.pair_probes, round(args.pair_probes * (1-args.pair_explore_fraction)))
    chosen = near[:n_priority]
    remaining = near[n_priority:]
    chosen += rng.sample(remaining, min(args.pair_probes-len(chosen), len(remaining)))
    return chosen


def v6_bias_calibrate(model, search, guards, anchor, state, args):
    """Calibrate ONLY existing biases; commit the best multi-guard-safe state."""
    if args.bias_steps <= 0:
        return False
    biases = [layer.bias for _, layer in model.quant_layers()]
    originals = [p.detach().clone() for p in biases]
    optimizer = torch.optim.Adam(biases, lr=args.bias_lr)
    best_biases, best_result = None, None
    # New improvements must beat the best accepted interim bias state,
    # rather than merely beating the original state repeatedly.
    comparison = {'search_mse': state['search_mse'],
                  'guard_mse': list(state['guard_mse']),
                  'anchor_mse': state['anchor_mse']}
    for step in range(args.bias_steps):
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, search['count'], args.probe_batch):
            lr = search['lr'][i:i+args.probe_batch]
            gt = search['target'][i:i+args.probe_batch]
            pred = central(model(lr).clamp(0, 1), args.context)
            loss = F.mse_loss(pred, gt) * (len(lr) / search['count'])
            gradients = torch.autograd.grad(loss, biases, allow_unused=True)
            for p, g in zip(biases, gradients):
                if g is not None:
                    if p.grad is None:
                        p.grad = g.detach().clone()
                    else:
                        p.grad.add_(g.detach())
        optimizer.step()
        if (step+1) % args.bias_check_every == 0 or step+1 == args.bias_steps:
            candidate = v6_accept(model, search, guards, anchor, comparison, args)
            if candidate is not None:
                best_result = candidate
                best_biases = [p.detach().clone() for p in biases]
                v6_commit_objectives(comparison, candidate)
    with torch.no_grad():
        for p, old in zip(biases, best_biases if best_biases is not None else originals):
            p.copy_(old)
    model.zero_grad(set_to_none=True)
    if best_result is not None:
        v6_commit_objectives(state, best_result)
    return best_result is not None


def v6_save(run, model, state, config, reported_gain=None):
    payload = {
        'model': model.state_dict(), 'optimizer': None, 'ema_model': None,
        'ema_updates': 0,
        'step': state['accepted_singles'] + 2*state['accepted_pairs'] +
                state['accepted_scales'] + state['accepted_bias_rounds'],
        'best_gain': reported_gain,
        'config': config, 'inference_weights': 'raw',
        'v6_state': copy.deepcopy(state),
    }
    _atomic_torch_save(payload, run / 'last.pt')
    # All committed configurations strictly improve the search and never
    # regress on any training guard. Held-out validation NEVER selects best.pt.
    _atomic_torch_save(payload, run / 'best.pt')


def v6_write_metrics(run, state, phase):
    columns = ['round', 'cycle', 'phase', 'probes', 'single_edits',
               'pair_edits', 'scale_edits', 'bias_rounds', 'search_psnr',
               'guard_min_psnr', 'anchor_psnr', 'search_mse',
               'guards_mse_json', 'anchor_mse']
    values = {'round': state['round']+1, 'cycle': state['cycle']+1,
              'phase': phase, 'probes': state['total_probes'],
              'single_edits': state['accepted_singles'],
              'pair_edits': state['accepted_pairs'],
              'scale_edits': state['accepted_scales'],
              'bias_rounds': state['accepted_bias_rounds'],
              'search_psnr': mse_psnr(state['search_mse']),
              'guard_min_psnr': min(mse_psnr(v) for v in state['guard_mse']),
              'anchor_psnr': mse_psnr(state['anchor_mse']),
              'search_mse': state['search_mse'],
              'guards_mse_json': json.dumps(state['guard_mse']),
              'anchor_mse': state['anchor_mse']}
    with (run / 'v6_refinement.csv').open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(values)


def refine_v6(args):
    """V6 coordinated discrete refinement; independent training-only guards.

    Safe to start from a V5 refined best.pt with --fork-from-refinement.
    New output directory required. --resume restores the exact pending
    candidate plan, cursor, model, objectives and crop manifests.
    """
    if bool(args.checkpoint) == bool(args.resume):
        raise ValueError('V6 requires exactly one of --checkpoint or --resume')
    if (args.rounds < 1 or args.cycles_per_round < 1 or args.search_patches < 2 or
            args.guard_patches < 1 or args.guard_sets < 2 or args.anchor_patches < 1 or
            args.patch_size < 32 or args.patch_size % 2 or args.context < 8 or args.context % 2):
        raise ValueError('Invalid round/cycle/crop settings (guards>=2, even patch/context)')
    if (args.single_probes < 1 or args.pair_probes < 1 or args.pair_pool < 2 or
            args.scale_probes < 0 or args.scale_fraction <= 0 or
            args.probe_batch < 1 or args.rank_patches < 1 or args.checkpoint_every < 1 or
            args.bias_steps < 0 or args.bias_lr <= 0 or args.bias_check_every < 1 or
            args.max_source_mp <= 0):
        raise ValueError('Invalid V6 probe, scale, bias or checkpoint settings')
    if (not 0 <= args.explore_fraction <= 1 or
            not 0 <= args.pair_explore_fraction <= 1 or
            args.min_relative_gain < 0 or args.guard_tolerance < 0):
        raise ValueError('V6 exploration fractions/acceptance thresholds are invalid')
    device = torch.device(args.device)

    if args.resume:
        run = Path(args.resume).resolve().parent
        model, ck = load_model(args.resume, device, use_inference=True)
        cfg = ck.get('config', {})
        if not cfg.get('v6_refinement') or 'v6_state' not in ck:
            raise ValueError('This is not a V6 checkpoint. Use refine-v6 --checkpoint '
                             'to start a new V6 run, or refine --resume for V4/V5.')
        settings = cfg['v6_settings']
        for key, val in settings.items():
            setattr(args, key, val)
        state = copy.deepcopy(ck['v6_state'])
        args.image_dir = args.image_dir or cfg.get('image_dir', DEFAULT_IMAGE_DIR)
        print(f'V6 RESUME: round {state["round"]+1}/{args.rounds}, '
              f'cycle {state["cycle"]+1}/{args.cycles_per_round}, '
              f'phase={state["phase"]} cursor={state["cursor"]} '
              f'probes={state["total_probes"]}', flush=True)
    else:
        model, source_ck = load_model(args.checkpoint, device, use_inference=True)
        source_cfg = source_ck.get('config', {})
        if source_cfg.get('discrete_refinement') and not args.fork_from_refinement:
            raise ValueError('A refined source needs --fork-from-refinement. '
                             'V6 starts a new independent search; it cannot overwrite V5.')
        if model.architecture not in (SpatialV1.architecture, SpatialV3.architecture):
            raise ValueError('V6 supports Spatial V1/V3 12-channel models, not Mix V2')
        args.seed = source_cfg.get('seed', 42)
        args.val_fraction = source_cfg.get('val_fraction', 0.05)
        args.image_dir = args.image_dir or source_cfg.get('image_dir', DEFAULT_IMAGE_DIR)
        args.out_dir = args.out_dir or './runs/microsr_x2_v6_joint'
        run = Path(args.out_dir).resolve()
        if run == Path(args.checkpoint).resolve().parent:
            raise ValueError('V6 output folder cannot equal the source checkpoint folder')
        if (run / 'best.pt').exists() or (run / 'last.pt').exists():
            raise FileExistsError(f'{run} already contains a checkpoint; choose a new --out-dir')
        state = {
            'round': 0, 'cycle': 0, 'phase': 'enter_round', 'cursor': 0,
            'proposals': [], 'failed_singles': [], 'total_probes': 0,
            'accepted_singles': 0, 'accepted_pairs': 0, 'accepted_scales': 0,
            'accepted_bias_rounds': 0, 'search_mse': None, 'guard_mse': [],
            'anchor_mse': None, 'source_checkpoint': str(Path(args.checkpoint).resolve()),
            'source_selected': source_ck.get('inference_weights', 'raw'),
        }
        print(f'V6 source: {args.checkpoint} selected={state["source_selected"]}. '
              f'Original checkpoint is not modified.', flush=True)

    if model.architecture not in (SpatialV1.architecture, SpatialV3.architecture):
        raise ValueError('V6 only supports 396-weight Spatial V1/V3 architectures')
    if args.seed is None or args.val_fraction is None:
        raise ValueError('V6 checkpoint lacks training/validation split settings')
    train_paths, valid_paths = local_image_split(args.image_dir, args.seed, args.val_fraction, 80)
    fp = validation_fingerprint(valid_paths)
    if args.resume:
        if cfg.get('val_fingerprint') and cfg['val_fingerprint'] != fp:
            raise ValueError('V6 validation split fingerprint changed; cannot resume safely')
        manifest = json.loads((run / 'v6_manifest.json').read_text(encoding='utf-8'))
        config = cfg
    else:
        prev_fp = source_cfg.get('val_fingerprint')
        if prev_fp and prev_fp != fp:
            raise ValueError('V6 source validation split differs from current dataset')
        run.mkdir(parents=True, exist_ok=True)
        manifest = v6_make_manifest(train_paths, valid_paths, args)
        _atomic_json_save(manifest, run / 'v6_manifest.json')
        configure_quantization(model, model.quant_grad, model.quant_grid)
        snap_to_int2_centers(model)
        setting_names = (
            'rounds', 'cycles_per_round', 'search_patches', 'guard_patches',
            'guard_sets', 'anchor_patches', 'patch_size', 'context',
            'single_probes', 'pair_probes', 'pair_pool', 'probe_batch',
            'rank_patches', 'explore_fraction', 'pair_explore_fraction',
            'scale_probes', 'scale_fraction', 'bias_steps', 'bias_lr',
            'bias_check_every', 'min_relative_gain', 'guard_tolerance',
            'checkpoint_every', 'max_source_mp', 'seed', 'val_fraction',
            'crop_seed', 'fork_from_refinement', 'val_images', 'val_patch',
            'val_positions')
        settings = {key: getattr(args, key) for key in setting_names}
        config = dict(source_cfg)
        config.update({'discrete_refinement': True, 'v6_refinement': True,
                       'refinement_scheme': 'v6_train_only_joint_single_pair_multiguard',
                       'v6_settings': settings,
                       'source_checkpoint': str(Path(args.checkpoint).resolve()),
                       'val_fingerprint': fp, 'image_dir': str(args.image_dir)})
    v6_check_manifest(manifest, train_paths, valid_paths, args)
    random.seed(args.seed + 775621)
    torch.manual_seed(args.seed + 775621)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model.eval()
    anchor = build_discrete_cache(manifest['anchor'], args.image_dir,
                                  args.patch_size, args.context, device)
    print(f'V6: 396 INT2 weights = 99 bytes, 48 scales + 48 biases. '
          f'{args.rounds} crop-rotation rounds x {args.cycles_per_round} alternating cycles; '
          f'anchor={args.anchor_patches}, search={args.search_patches}, '
          f'guards={args.guard_sets}x{args.guard_patches}. '
          'HELD-OUT IMAGES ARE REPORT-ONLY.', flush=True)
    this_invocation_probes = 0
    loaded_round = None
    search = guards = None

    while state['round'] < args.rounds:
        if loaded_round != state['round']:
            # Free stale GPU crops before moving to the next round.
            del search, guards
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            search, guards = v6_cache_round(manifest, state['round'], args, device)
            loaded_round = state['round']
            if state['phase'] != 'enter_round':
                observed = v6_objectives(model, search, guards, anchor, args)
                stored = [state['search_mse'], *state['guard_mse'], state['anchor_mse']]
                check = [observed[0], *observed[1], observed[2]]
                if len(stored) != len(check) or any(
                        abs(a-b) > max(1e-10, 1e-5*abs(a)) for a, b in zip(stored, check)):
                    raise RuntimeError('V6 resume objective mismatch; dataset, checkpoint, '
                                       'or scoring configuration may have changed')
        args._round = state['round'] * args.cycles_per_round + state['cycle']

        if state['phase'] == 'enter_round':
            observed = v6_objectives(model, search, guards, anchor, args)
            if state['anchor_mse'] is not None and abs(observed[2]-state['anchor_mse']) > max(
                    1e-10, 1e-5*abs(state['anchor_mse'])):
                raise RuntimeError('V6 anchor objective changed between rounds')
            v6_commit_objectives(state, observed)
            state['phase'] = 'plan_singles'
            state['cursor'] = 0
            state['proposals'] = []
            state['failed_singles'] = []
            v6_save(run, model, state, config)
            v6_write_metrics(run, state, 'round_start')
            print(f'ROUND {state["round"]+1}: search={mse_psnr(state["search_mse"]):.5f} dB '
                  f'guards={[round(mse_psnr(g),5) for g in state["guard_mse"]]} '
                  f'anchor={mse_psnr(state["anchor_mse"]):.5f} dB', flush=True)

        if state['phase'] == 'plan_singles':
            # Recompute cheap gradients after the latest code/scale/bias changes.
            # Existing V5 ranking function is used only to prioritize proposals.
            rank_args = argparse.Namespace(**vars(args))
            rank_args.probes_per_round = args.single_probes
            state['proposals'] = discrete_gradient_ranking(model, search, rank_args)
            state['cursor'] = 0
            state['failed_singles'] = []
            state['phase'] = 'singles'
            v6_save(run, model, state, config)
            print(f'ROUND {state["round"]+1}/{args.rounds} '
                  f'CYCLE {state["cycle"]+1}/{args.cycles_per_round}: '
                  f'{len(state["proposals"])} single-code candidates', flush=True)

        if state['phase'] == 'singles':
            while state['cursor'] < len(state['proposals']):
                cand = state['proposals'][state['cursor']]
                name, idx = cand['layer'], int(cand['index'])
                current = int(_reference_codes(model)[name].view(-1)[idx])
                if current == int(cand['old']):
                    previous = set_int2_code(model, name, idx, int(cand['new']))
                    accepted = v6_accept(model, search, guards, anchor, state, args)
                    if accepted is None:
                        set_int2_code(model, name, idx, previous)
                        state['failed_singles'].append(cand)
                    else:
                        v6_commit_objectives(state, accepted)
                        state['accepted_singles'] += 1
                        print(f'  ACCEPT single {name}[{idx}] '
                              f'{previous}->{cand["new"]} search='
                              f'{mse_psnr(accepted[0]):.5f} dB '
                              f'guards_min={min(mse_psnr(x) for x in accepted[1]):.5f} dB',
                              flush=True)
                    state['total_probes'] += 1
                    this_invocation_probes += 1
                state['cursor'] += 1
                if state['cursor'] == len(state['proposals']) or (
                        state['total_probes'] % args.checkpoint_every == 0):
                    v6_save(run, model, state, config)
                if args.stop_after_probes and this_invocation_probes >= args.stop_after_probes:
                    v6_save(run, model, state, config)
                    print(f'V6 PAUSED. Resume: python microsr_int2_local.py refine-v6 '
                          f'--resume "{run / "last.pt"}" --image-dir "{args.image_dir}"')
                    return
            state['phase'] = 'plan_pairs'
            state['cursor'] = 0
            state['proposals'] = []
            v6_save(run, model, state, config)

        if state['phase'] == 'plan_pairs':
            state['proposals'] = v6_plan_pairs(model, state['failed_singles'], args)
            state['cursor'] = 0
            state['phase'] = 'pairs'
            v6_save(run, model, state, config)
            print(f'  Coordinated pairs: {len(state["proposals"])} '
                  f'(drawn from {len(state["failed_singles"])} rejected singles)', flush=True)

        if state['phase'] == 'pairs':
            while state['cursor'] < len(state['proposals']):
                cand = state['proposals'][state['cursor']]
                a, b = cand['first'], cand['second']
                codes = _reference_codes(model)
                if (int(codes[a['layer']].view(-1)[int(a['index'])]) == int(a['old']) and
                    int(codes[b['layer']].view(-1)[int(b['index'])]) == int(b['old'])):
                    old_a = set_int2_code(model, a['layer'], a['index'], a['new'])
                    old_b = set_int2_code(model, b['layer'], b['index'], b['new'])
                    accepted = v6_accept(model, search, guards, anchor, state, args)
                    if accepted is None:
                        set_int2_code(model, b['layer'], b['index'], old_b)
                        set_int2_code(model, a['layer'], a['index'], old_a)
                    else:
                        v6_commit_objectives(state, accepted)
                        state['accepted_pairs'] += 1
                        print(f'  ACCEPT PAIR {a["layer"]}[{a["index"]}] '
                              f'{old_a}->{a["new"]} + {b["layer"]}[{b["index"]}] '
                              f'{old_b}->{b["new"]}: '
                              f'search={mse_psnr(accepted[0]):.5f} dB '
                              f'anchor={mse_psnr(accepted[2]):.5f} dB', flush=True)
                    state['total_probes'] += 1
                    this_invocation_probes += 1
                state['cursor'] += 1
                if state['cursor'] == len(state['proposals']) or (
                        state['total_probes'] % args.checkpoint_every == 0):
                    v6_save(run, model, state, config)
                if args.stop_after_probes and this_invocation_probes >= args.stop_after_probes:
                    v6_save(run, model, state, config)
                    print(f'V6 PAUSED. Resume: python microsr_int2_local.py refine-v6 '
                          f'--resume "{run / "last.pt"}" --image-dir "{args.image_dir}"')
                    return
            state['phase'] = 'plan_scales'
            state['cursor'] = 0
            state['proposals'] = []
            state['failed_singles'] = []
            v6_save(run, model, state, config)

        if state['phase'] == 'plan_scales':
            state['proposals'] = scale_candidate_plan(model, args)
            state['cursor'] = 0
            state['phase'] = 'scales'
            v6_save(run, model, state, config)

        if state['phase'] == 'scales':
            while state['cursor'] < len(state['proposals']):
                cand = state['proposals'][state['cursor']]
                layer = dict(model.quant_layers())[cand['layer']]
                ch = int(cand['channel'])
                oldlog = float(layer.log_scale.detach()[ch])
                if abs(oldlog - cand['expected_logscale']) < 1e-7:
                    original_codes = _reference_codes(model)[cand['layer']][ch].clone()
                    set_int2_scale(model, cand['layer'], ch, oldlog+cand['shift'], original_codes)
                    accepted = v6_accept(model, search, guards, anchor, state, args)
                    if accepted is None:
                        set_int2_scale(model, cand['layer'], ch, oldlog, original_codes)
                    else:
                        v6_commit_objectives(state, accepted)
                        state['accepted_scales'] += 1
                        print(f'  ACCEPT scale {cand["layer"]}[{ch}] '
                              f'{math.expm1(cand["shift"]):+.2%}: '
                              f'search={mse_psnr(accepted[0]):.5f} dB', flush=True)
                    state['total_probes'] += 1
                    this_invocation_probes += 1
                state['cursor'] += 1
                if state['cursor'] == len(state['proposals']) or (
                        state['total_probes'] % args.checkpoint_every == 0):
                    v6_save(run, model, state, config)
                if args.stop_after_probes and this_invocation_probes >= args.stop_after_probes:
                    v6_save(run, model, state, config)
                    print(f'V6 PAUSED. Resume: python microsr_int2_local.py refine-v6 '
                          f'--resume "{run / "last.pt"}" --image-dir "{args.image_dir}"')
                    return
            state['phase'] = 'bias'
            state['cursor'] = 0
            state['proposals'] = []
            v6_save(run, model, state, config)

        if state['phase'] == 'bias':
            if v6_bias_calibrate(model, search, guards, anchor, state, args):
                state['accepted_bias_rounds'] += 1
                print(f'  ACCEPT multi-guard bias calibration: '
                      f'anchor={mse_psnr(state["anchor_mse"]):.5f} dB', flush=True)
            v6_write_metrics(run, state, 'cycle_complete')
            print(f'END ROUND {state["round"]+1} CYCLE {state["cycle"]+1}: '
                  f'accepted_singles={state["accepted_singles"]}, '
                  f'accepted_pairs={state["accepted_pairs"]}, '
                  f'accepted_scales={state["accepted_scales"]}, '
                  f'bias_rounds={state["accepted_bias_rounds"]}, '
                  f'probes={state["total_probes"]}, '
                  f'search={mse_psnr(state["search_mse"]):.5f} dB, '
                  f'anchor={mse_psnr(state["anchor_mse"]):.5f} dB', flush=True)
            state['cycle'] += 1
            state['cursor'] = 0
            state['proposals'] = []
            state['failed_singles'] = []
            if state['cycle'] >= args.cycles_per_round:
                state['round'] += 1
                state['cycle'] = 0
                state['phase'] = 'enter_round'
                state['search_mse'] = None
                state['guard_mse'] = []
            else:
                state['phase'] = 'plan_singles'
            v6_save(run, model, state, config)

    print('V6 TRAIN-ONLY SEARCH COMPLETE. Held-out validation is REPORT ONLY.', flush=True)
    gain = None
    if args.val_images >= 0:
        result = evaluate(model, valid_paths, device, args.val_images,
                          val_patch=args.val_patch, positions=args.val_positions)
        gain = result['sr_psnr'] - result['bicubic_psnr']
        print(f'FINAL HELD-OUT REPORT (never used for candidate acceptance): '
              f'bicubic={result["bicubic_psnr"]:.4f} dB '
              f'INT2={result["sr_psnr"]:.4f} dB '
              f'gain={gain:+.4f} dB, patches={result["patches"]}', flush=True)
        _atomic_json_save({
            'checkpoint': str(run / 'best.pt'),
            'selection': 'train search MSE improvements, no regression on 2+ train guards and anchor',
            'validation_is_report_only': True,
            'validation': result, 'gain_db': gain,
            'source_checkpoint': state['source_checkpoint'],
            'accepted_singles': state['accepted_singles'],
            'accepted_pairs': state['accepted_pairs'],
            'accepted_scales': state['accepted_scales'],
            'accepted_bias_rounds': state['accepted_bias_rounds'],
            'total_probes': state['total_probes'],
            'anchor_mse': state['anchor_mse'],
        }, run / 'v6_final_report.json')
    v6_save(run, model, state, config, reported_gain=gain)
    print(f'DONE: {effective_code_summary(model)}')
    print(f'Export: python microsr_int2_local.py export --checkpoint '
          f'"{run / "best.pt"}" --output "{run / "microsr_int2"}"')



# -----------------------------------------------------------------------------
# V5 ABLATION SUITE: EXACT SAME SOURCE CHECKPOINT AND TRAIN CROPS PER ARM
# -----------------------------------------------------------------------------

def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _ablation_summary(root, plan, summary):
    """Atomic, restart-safe suite report. Never selects weights on validation."""
    columns = ['mode', 'status', 'baseline_gain_db', 'refined_gain_db',
               'delta_gain_db', 'search_psnr', 'guard_psnr', 'accepted_codes',
               'accepted_scales', 'accepted_bias_rounds', 'total_probes', 'checkpoint']
    csv_path = root / 'ablation_summary.csv'
    temp = csv_path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for mode in plan['modes']:
            if mode in summary:
                writer.writerow({k: summary[mode].get(k, '') for k in columns})
    temp.replace(csv_path)
    _atomic_json_save({'protocol': plan, 'results': summary}, root / 'ablation_summary.json')


def verify_ablation_purity(source_ckpt, trial_ckpt, mode):
    """Prove component-only ablations did not modify forbidden deployed tensors."""
    source, _ = load_model(source_ckpt, 'cpu', use_inference=True)
    trial, _ = load_model(trial_ckpt, 'cpu', use_inference=True)
    if source.architecture != trial.architecture or source.quant_grid != trial.quant_grid:
        raise RuntimeError('Ablation model architecture/codebook changed unexpectedly')
    for (name, old), (_, new) in zip(source.quant_layers(), trial.quant_layers()):
        old_scale = old.log_scale.detach()
        new_scale = new.log_scale.detach()
        old_bias = old.bias.detach()
        new_bias = new.bias.detach()
        old_codes = quant_codes(old.weight.detach(), old_scale.clamp(-14, 5).exp().view(-1, 1, 1, 1), old.grid)
        new_codes = quant_codes(new.weight.detach(), new_scale.clamp(-14, 5).exp().view(-1, 1, 1, 1), new.grid)
        if mode in ('scales', 'bias') and not torch.equal(old_codes, new_codes):
            raise AssertionError(f'{mode} ablation unexpectedly changed deployed INT2 codes in {name}')
        if mode in ('codes', 'bias') and not torch.equal(old_scale, new_scale):
            raise AssertionError(f'{mode} ablation unexpectedly changed scales in {name}')
        if mode in ('codes', 'scales') and not torch.equal(old_bias, new_bias):
            raise AssertionError(f'{mode} ablation unexpectedly changed biases in {name}')
    return True


def ablate(args, refine_defaults):
    """Compare independent code-only/scale-only/bias-only/all optimizers fairly.

    Every arm is initialized from the same checkpoint, uses the same deterministic
    search/guard manifest and runtime settings, and independently writes its own
    best.pt. The all-three arm is not initialized from any single-component arm.
    Held-out validation is used for *reporting*, never candidate acceptance.
    """
    if args.val_images < 0:
        raise ValueError('Ablation requires a held-out final report; --val-images must be >=0')
    if args.rounds < 1 or args.search_patches < 1 or args.guard_patches < 1:
        raise ValueError('Ablation rounds/search/guard counts must be positive')
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'No source checkpoint: {checkpoint}')
    root = Path(args.out_dir).resolve()
    if root == checkpoint.parent or checkpoint.parent in root.parents:
        # Source and output should be in separate run trees. A source inside
        # the suite directory is also checked below, before any modification.
        if checkpoint.parent == root:
            raise ValueError('Ablation suite cannot write into source checkpoint folder')
    if checkpoint == root / 'best.pt' or checkpoint == root / 'last.pt':
        raise ValueError('Source checkpoint lies in ablation output directory')
    modes = args.modes
    if len(modes) != len(set(modes)):
        raise ValueError('Each ablation mode may appear only once')
    device = torch.device(args.device)
    model, ck = load_model(checkpoint, device, use_inference=True)
    if model.architecture not in (SpatialV1.architecture, SpatialV3.architecture):
        raise ValueError('Ablation is for the successful 12-channel Spatial model only')
    if ck.get('config', {}).get('discrete_refinement'):
        raise ValueError('For clean component ablations start from pre-refinement '
                         'Spatial V3/V1 best.pt, not V4 refined best.pt')
    cfg = ck.get('config', {})
    image_dir = args.image_dir or cfg.get('image_dir', DEFAULT_IMAGE_DIR)
    seed = cfg.get('seed', 42)
    fraction = cfg.get('val_fraction', 0.05)
    train_paths, val_paths = local_image_split(image_dir, seed, fraction, 80)
    val_fp = validation_fingerprint(val_paths)
    if cfg.get('val_fingerprint') and cfg['val_fingerprint'] != val_fp:
        raise ValueError('Ablation source checkpoint does not match current held-out split')
    # Never infer best selection from validation; this is an experiment protocol.
    trial_options = {k: getattr(args, k) for k in (
        'rounds', 'search_patches', 'guard_patches', 'patch_size', 'context',
        'probe_batch', 'rank_patches', 'probes_per_round', 'explore_fraction',
        'scale_probes', 'scale_fraction', 'bias_steps', 'bias_lr',
        'bias_check_every', 'min_relative_gain', 'guard_tolerance',
        'checkpoint_every', 'max_source_mp', 'crop_seed', 'val_images',
        'val_patch', 'val_positions')}
    plan = {'suite_version': 1, 'checkpoint': str(checkpoint),
            'checkpoint_sha256': _file_sha256(checkpoint), 'image_dir': str(Path(image_dir).resolve()),
            'split_seed': seed, 'val_fraction': fraction, 'val_fingerprint': val_fp,
            'modes': list(modes), 'trial_options': trial_options,
            'code_weights': 396, 'packed_int2_bytes': 99,
            'selection_rule': 'training-only MSE; held-out used for reporting only'}
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / 'ablation_plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding='utf-8'))
        if previous != plan:
            raise ValueError(f'Ablation settings/source differ from existing plan: {plan_path}. '
                             'Use the exact same command to resume, or a NEW --out-dir')
    else:
        # Prevent accidentally adopting an unrelated directory.
        if any((root / x).exists() for x in ('ablation_summary.json', 'codes', 'scales', 'bias', 'all')):
            raise FileExistsError('Existing suite data without matching plan; choose new --out-dir')
        _atomic_json_save(plan, plan_path)

    baseline_path = root / 'baseline_eval.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        print('ABLATION BASELINE: evaluating source checkpoint on held-out images (report only)', flush=True)
        baseline = evaluate(model, val_paths, device, args.val_images,
                            val_patch=args.val_patch, positions=args.val_positions)
        baseline['gain_db'] = baseline['sr_psnr'] - baseline['bicubic_psnr']
        _atomic_json_save(baseline, baseline_path)
    print(f'ABLATION SOURCE: {checkpoint.name}; baseline full held-out gain '
          f'{baseline["gain_db"]:+.6f} dB. Every arm restarts from SAME weights.', flush=True)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    summary = {}
    for index, mode in enumerate(modes, 1):
        trial_dir = root / mode
        print(f'\n=== ABLATION ARM {index}/{len(modes)}: {mode} ===', flush=True)
        final_path = trial_dir / 'final_report.json'
        if final_path.exists():
            print(f'Existing completed arm: {trial_dir}, verifying and reusing results.', flush=True)
        else:
            options = dict(refine_defaults)
            options.update(trial_options)
            options.update({'checkpoint': str(checkpoint), 'resume': None,
                            'image_dir': image_dir, 'out_dir': str(trial_dir),
                            'mode': mode, 'fork_from_refinement': False,
                            'seed': None, 'val_fraction': None, 'stop_after_probes': 0,
                            'device': args.device})
            if (trial_dir / 'last.pt').is_file():
                options['resume'] = str(trial_dir / 'last.pt')
                options['checkpoint'] = None
                print('Resuming interrupted arm from saved search state.', flush=True)
            elif trial_dir.exists() and any(trial_dir.iterdir()):
                raise FileExistsError(f'Incomplete arm directory without last.pt: {trial_dir}')
            refine(argparse.Namespace(**options))
        if not final_path.is_file():
            raise RuntimeError(f'Ablation arm {mode} did not complete its final report')
        # Byte-identical crop plans are a strict fairness requirement.
        manifest_path = trial_dir / 'refinement_manifest.json'
        own_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        first_manifest_path = root / modes[0] / 'refinement_manifest.json'
        first_manifest = json.loads(first_manifest_path.read_text(encoding='utf-8'))
        if own_manifest != first_manifest:
            raise RuntimeError(f'ERROR: {mode} trained on different crops than the first arm!')
        result = json.loads(final_path.read_text(encoding='utf-8'))
        model_ck = torch.load(trial_dir / 'best.pt', map_location='cpu', weights_only=False)
        if model_ck['config'].get('refine_mode', 'all') != mode:
            raise RuntimeError(f'Ablation checkpoint for {mode} has incorrect refinement mode')
        if Path(model_ck['refinement_state']['source_checkpoint']).resolve() != checkpoint:
            raise RuntimeError(f'Ablation {mode} checkpoint did not start from suite source')
        verify_ablation_purity(checkpoint, trial_dir / 'best.pt', mode)
        state = model_ck['refinement_state']
        refined_gain = float(result['gain_db'])
        row = {'mode': mode, 'status': 'complete',
               'baseline_gain_db': float(baseline['gain_db']),
               'refined_gain_db': refined_gain,
               'delta_gain_db': refined_gain - float(baseline['gain_db']),
               'search_psnr': mse_psnr(state['search_mse']),
               'guard_psnr': mse_psnr(state['guard_mse']),
               'accepted_codes': state['accepted_codes'],
               'accepted_scales': state['accepted_scales'],
               'accepted_bias_rounds': state['accepted_bias_rounds'],
               'total_probes': state['total_probes'],
               'checkpoint': str(trial_dir / 'best.pt')}
        summary[mode] = row
        _ablation_summary(root, plan, summary)
        print(f'ARM {mode}: held-out gain={refined_gain:+.6f} dB, '
              f'delta from same baseline={row["delta_gain_db"]:+.6f} dB, '
              f'codes={state["accepted_codes"]}, scales={state["accepted_scales"]}, '
              f'bias_rounds={state["accepted_bias_rounds"]}; '
              f'parameter isolation=VERIFIED', flush=True)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('\n=== FINAL MATCHED ABLATION COMPARISON ===', flush=True)
    print(f'Source gain: {baseline["gain_db"]:+.6f} dB')
    for mode in modes:
        row = summary[mode]
        print(f'{mode:>6s}: held-out={row["refined_gain_db"]:+.6f} dB '
              f'delta={row["delta_gain_db"]:+.6f} dB '
              f'(codes={row["accepted_codes"]} scales={row["accepted_scales"]} '
              f'bias_rounds={row["accepted_bias_rounds"]})')
    print(f'CSV: {root / "ablation_summary.csv"}')
    print('All four checkpoints remain separate. Validation is REPORT ONLY; '
          'publish selected results on an untouched independent test set.')


def export(args):
    model, ck = load_model(args.checkpoint, "cpu", use_inference=True)
    model.eval()
    codes = []
    meta = {
        "format": "MicroSR-X2-INT2-packet-v3",
        "weight_source": ck.get("inference_weights", "raw"),
        "architecture": model.architecture,
        "channels": model.channels,
        "interpolation": "PyTorch bicubic skip; Torch antialiased bicubic downsampling used for training and evaluation",
        "int2_grid": model.quant_grid,
        "int2_levels": [-2, -1, 0, 1] if model.quant_grid == "asym" else [-1.5, -0.5, 0.5, 1.5],
        "packing": "4 two-bit codes per byte, least-significant lane first; "
                   "asym uses signed 2-bit two's-complement code; "
                   "balanced uses unsigned indices 0/1/2/3",
        "layers": [],
    }
    for name, layer in model.quant_layers():
        scales = layer.log_scale.detach().clamp(-14, 5).exp().numpy().astype("<f4")
        weights = layer.weight.detach().numpy()
        q = quant_codes(layer.weight.detach(),
                        layer.log_scale.detach().clamp(-14, 5).exp().view(-1, 1, 1, 1),
                        layer.grid).reshape(-1).cpu().numpy().astype(np.uint8)
        codes.extend(q.tolist())
        meta["layers"].append({
            "name": name, "shape": list(weights.shape), "groups": layer.groups,
            "kernel": layer.kernel, "scales": scales.tolist(),
            "bias": layer.bias.detach().numpy().tolist(),
        })
    packed = bytearray((len(codes)+3)//4)
    wire_codes = [((v - 2) & 3) if model.quant_grid == "asym" else v
                  for v in codes]
    for i, code in enumerate(wire_codes):
        packed[i//4] |= int(code) << (2*(i%4))
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".bin").write_bytes(packed)
    prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    decoded_wire = [(packed[i//4] >> (2*(i%4))) & 3 for i in range(len(codes))]
    decoded = [((v + 2) & 3) if model.quant_grid == "asym" else v
               for v in decoded_wire]
    if decoded != codes:
        raise AssertionError("INT2 packing failed roundtrip verification")
    bias_scale_bytes_fp32 = 4*sum(layer.bias.numel()+layer.log_scale.numel()
                                  for _, layer in model.quant_layers())
    print(f"INT2 weights: {len(codes)} codes packed in {len(packed)} bytes")
    print(f"Learned FP32 scales/biases: {bias_scale_bytes_fp32} bytes. "
          f"Minimum raw inference array payload: {len(packed)+bias_scale_bytes_fp32} bytes "
          f"(excluding format headers, runtime, bicubic implementation, padding etc.)")
    print(f"JSON metadata disk size: {prefix.with_suffix('.json').stat().st_size} bytes")
    print("Export stores parameters only; not a standalone implementation of inference.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train V1, V3 quantization experiment, or legacy Mix V2")
    t.add_argument("--architecture", choices=list(ARCHITECTURES), default=None,
                   help="Default spatial_v3 for new runs. Resume/finetune autodetects checkpoint type")
    t.add_argument("--quant-grad", choices=["ste", "lsq"], default=None,
                   help="V3 defaults to LSQ-normalized scale gradients; STE is legacy control")
    t.add_argument("--quant-grid", choices=["asym", "balanced"], default=None,
                   help="asym is V1-compatible [-2,-1,0,1]; balanced has no zero level")
    t.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    t.add_argument("--val-fraction", type=float, default=0.05)
    t.add_argument("--out-dir", default=None, help="Separate automatic directory per architecture")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--cache-gb", type=float, default=12.0)
    t.add_argument("--pool-mb-per-worker", type=int, default=256)
    t.add_argument("--max-source-mp", type=float, default=50.0,
                   help="Exclude training images above this megapixel limit to protect Windows RAM")
    t.add_argument("--crops-per-image", type=int, default=48)
    t.add_argument("--refresh-crops", type=int, default=256)
    t.add_argument("--prefetch", type=int, default=2)
    t.add_argument("--patch", type=int, default=64)
    t.add_argument("--patches", type=int, nargs="+", default=[32, 64, 128])
    t.add_argument("--context", type=int, default=8)
    t.add_argument("--channels", type=int, default=12)
    t.add_argument("--lr", type=float, default=0.00001,
                   help="Start from a strong checkpoint at 1e-5 for V3; no guarantee of +1dB")
    t.add_argument("--lr-min", type=float, default=0.0000002)
    t.add_argument("--schedule", choices=["plateau", "cosine"], default=None)
    t.add_argument("--ema-decay", type=float, default=0.995)
    t.add_argument("--warmup", type=int, default=300)
    t.add_argument("--snapshot-every", type=int, default=10000)
    t.add_argument("--val-every", type=int, default=2000)
    t.add_argument("--val-images", type=int, default=20,
                   help="20 reproduces old checkpoint selection; 0 evaluates all 205")
    t.add_argument("--val-patch", type=int, default=512)
    t.add_argument("--val-positions", type=int, default=5, choices=[1,5])
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume", default=None)
    t.add_argument("--finetune-from", default=None)
    t.add_argument("--plateau-patience", type=int, default=2)
    t.add_argument("--plateau-factor", type=float, default=0.5)
    t.add_argument("--restore-best-on-plateau", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--min-gain-delta", type=float, default=0.0002,
                   help="Min improvement for saving best checkpoint in dB")
    t.add_argument("--plateau-min-delta", type=float, default=0.005,
                   help="Min gain improvement that resets LR plateau patience in dB")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e = sub.add_parser("eval", help="Evaluate the best-selected raw or EMA inference weights")
    e.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    e.add_argument("--val-fraction", type=float, default=None)
    e.add_argument("--seed", type=int, default=None)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--val-images", type=int, default=0)
    e.add_argument("--val-patch", type=int, default=512)
    e.add_argument("--val-positions", type=int, default=5, choices=[1,5])
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r = sub.add_parser("refine", help="Train-only discrete INT2 code/scale/bias coordinate search")
    r.add_argument("--checkpoint", default=None, help="New search: best Spatial V1/V3 checkpoint")
    r.add_argument("--resume", default=None, help="Continue a discrete search from its last.pt")
    r.add_argument("--mode", choices=["all", "codes", "scales", "bias"], default="all",
                   help="Ablation component to optimize: codes, scales, biases, or all three")
    r.add_argument("--crop-seed", type=int, default=None,
                   help="Change fixed TRAIN crop identities/locations; default source split seed")
    r.add_argument("--fork-from-refinement", action="store_true",
                   help="Start a NEW search from a previously refined checkpoint (fresh crops recommended)")
    r.add_argument("--image-dir", default=None)
    r.add_argument("--out-dir", default=None)
    r.add_argument("--rounds", type=int, default=4)
    r.add_argument("--search-patches", type=int, default=128,
                   help="Distinct TRAIN images to rank/score candidate code changes")
    r.add_argument("--guard-patches", type=int, default=128,
                   help="Disjoint TRAIN images which must not regress on accepted changes")
    r.add_argument("--patch-size", type=int, default=128)
    r.add_argument("--context", type=int, default=8)
    r.add_argument("--probe-batch", type=int, default=16)
    r.add_argument("--rank-patches", type=int, default=48)
    r.add_argument("--probes-per-round", type=int, default=400)
    r.add_argument("--explore-fraction", type=float, default=0.25)
    r.add_argument("--scale-probes", type=int, default=80,
                   help="Additional 48-scale scalar proposals, 0 to disable")
    r.add_argument("--scale-fraction", type=float, default=0.02)
    r.add_argument("--bias-steps", type=int, default=20,
                   help="Adam fine-tune existing 48 biases, guarded by TRAIN MSE; 0 disables")
    r.add_argument("--bias-lr", type=float, default=0.00001)
    r.add_argument("--bias-check-every", type=int, default=5)
    r.add_argument("--min-relative-gain", type=float, default=0.000001)
    r.add_argument("--guard-tolerance", type=float, default=0.0,
                   help="Allowed fractional TRAIN guard MSE regression; default ZERO")
    r.add_argument("--checkpoint-every", type=int, default=25)
    r.add_argument("--max-source-mp", type=float, default=50.0)
    r.add_argument("--seed", type=int, default=None, help="Inherited from checkpoint")
    r.add_argument("--val-fraction", type=float, default=None,
                   help="Inherited from checkpoint")
    r.add_argument("--val-images", type=int, default=20,
                   help="REPORT ONLY after search: 20 old protocol, 0 all; -1 skips")
    r.add_argument("--val-patch", type=int, default=512)
    r.add_argument("--val-positions", type=int, default=5, choices=[1, 5])
    r.add_argument("--stop-after-probes", type=int, default=0,
                   help="Optional finite run for testing; resume with refine --resume")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = sub.add_parser("ablate", help="Four-way fixed-crop, fixed-source discrete refinement ablation")
    a.add_argument("--checkpoint", required=True,
                   help="UNREFINED Spatial V1/V3 best.pt, identical starting point for every arm")
    a.add_argument("--image-dir", default=None)
    a.add_argument("--out-dir", default="./runs/microsr_x2_v5_ablation")
    a.add_argument("--modes", nargs="+", choices=["codes", "scales", "bias", "all"],
                   default=["codes", "scales", "bias", "all"])
    a.add_argument("--rounds", type=int, default=4)
    a.add_argument("--search-patches", type=int, default=128)
    a.add_argument("--guard-patches", type=int, default=128)
    a.add_argument("--patch-size", type=int, default=128)
    a.add_argument("--context", type=int, default=8)
    a.add_argument("--probe-batch", type=int, default=16)
    a.add_argument("--rank-patches", type=int, default=48)
    a.add_argument("--probes-per-round", type=int, default=400)
    a.add_argument("--explore-fraction", type=float, default=0.25)
    a.add_argument("--scale-probes", type=int, default=80)
    a.add_argument("--scale-fraction", type=float, default=0.02)
    a.add_argument("--bias-steps", type=int, default=20)
    a.add_argument("--bias-lr", type=float, default=0.00001)
    a.add_argument("--bias-check-every", type=int, default=5)
    a.add_argument("--min-relative-gain", type=float, default=0.000001)
    a.add_argument("--guard-tolerance", type=float, default=0.0)
    a.add_argument("--checkpoint-every", type=int, default=25)
    a.add_argument("--max-source-mp", type=float, default=50.0)
    a.add_argument("--crop-seed", type=int, default=42,
                   help="Use the SAME crop seed for every arm, independent of checkpoint split")
    a.add_argument("--val-images", type=int, default=0,
                   help="0 evaluates all held-out validation images, same protocol for every arm")
    a.add_argument("--val-patch", type=int, default=512)
    a.add_argument("--val-positions", type=int, default=5, choices=[1, 5])
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    v = sub.add_parser("refine-v6", help="V6: multi-guard, rotating-crop, joint single/pair INT2 optimization")
    v.add_argument("--checkpoint", default=None,
                   help="Start from V5 refined best.pt; use --fork-from-refinement")
    v.add_argument("--resume", default=None, help="Resume exact V6 state from last.pt")
    v.add_argument("--fork-from-refinement", action="store_true",
                   help="Explicitly allow starting from a V4/V5/V6 refined checkpoint")
    v.add_argument("--image-dir", default=None)
    v.add_argument("--out-dir", default="./runs/microsr_x2_v6_joint")
    v.add_argument("--rounds", type=int, default=4,
                   help="Rotate to independent TRAIN crops each round")
    v.add_argument("--cycles-per-round", type=int, default=2,
                   help="Alternate single codes, pairs, scales, biases within each round")
    v.add_argument("--search-patches", type=int, default=192)
    v.add_argument("--guard-patches", type=int, default=96,
                   help="Distinct TRAIN images PER guard set")
    v.add_argument("--guard-sets", type=int, default=2)
    v.add_argument("--anchor-patches", type=int, default=64,
                   help="Fixed training-only anchor set across all rounds")
    v.add_argument("--patch-size", type=int, default=128)
    v.add_argument("--context", type=int, default=8)
    v.add_argument("--probe-batch", type=int, default=16)
    v.add_argument("--rank-patches", type=int, default=64)
    v.add_argument("--single-probes", type=int, default=240)
    v.add_argument("--pair-probes", type=int, default=96)
    v.add_argument("--pair-pool", type=int, default=200)
    v.add_argument("--explore-fraction", type=float, default=0.25)
    v.add_argument("--pair-explore-fraction", type=float, default=0.30)
    v.add_argument("--scale-probes", type=int, default=64)
    v.add_argument("--scale-fraction", type=float, default=0.015)
    v.add_argument("--bias-steps", type=int, default=12)
    v.add_argument("--bias-lr", type=float, default=0.000005)
    v.add_argument("--bias-check-every", type=int, default=4)
    v.add_argument("--min-relative-gain", type=float, default=0.000001)
    v.add_argument("--guard-tolerance", type=float, default=0.0,
                   help="Allowed regression on EACH guard and anchor; 0 is strict")
    v.add_argument("--checkpoint-every", type=int, default=20)
    v.add_argument("--max-source-mp", type=float, default=50.0)
    v.add_argument("--seed", type=int, default=None,
                   help="Inherited from source checkpoint, used only for split")
    v.add_argument("--val-fraction", type=float, default=None,
                   help="Inherited from source checkpoint")
    v.add_argument("--crop-seed", type=int, default=173129)
    v.add_argument("--val-images", type=int, default=0,
                   help="REPORT ONLY; 0=all 205, -1=skip")
    v.add_argument("--val-patch", type=int, default=512)
    v.add_argument("--val-positions", type=int, default=5, choices=[1,5])
    v.add_argument("--stop-after-probes", type=int, default=0,
                   help="Testing/interruption: pause after N candidate probes")
    v.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    x = sub.add_parser("export", help="Pack selected INT2 model codes and FP32 scale/bias metadata")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--output", default="./runs/microsr_x2_mix_v2/microsr_int2")
    args = p.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "refine":
        refine(args)
    elif args.command == "refine-v6":
        refine_v6(args)
    elif args.command == "ablate":
        refine_defaults = {action.dest: action.default for action in r._actions
                           if action.dest != 'help'}
        ablate(args, refine_defaults)
    elif args.command == "eval":
        model, ck = load_model(args.checkpoint, args.device)
        cfg = ck["config"]
        seed = args.seed if args.seed is not None else cfg.get("seed", 42)
        fraction = args.val_fraction if args.val_fraction is not None else cfg.get("val_fraction", 0.05)
        _, paths = local_image_split(args.image_dir, seed, fraction,
                                     cfg.get("patch",64)+2*cfg.get("context",8))
        saved_hash = cfg.get("val_fingerprint")
        if saved_hash and saved_hash != validation_fingerprint(paths):
            raise ValueError("Validation image split changed since training; cannot compare")
        print(json.dumps(evaluate(model, paths, args.device, args.val_images,
                                  val_patch=args.val_patch, positions=args.val_positions), indent=2))
    else:
        export(args)


if __name__ == "__main__":
    main()
