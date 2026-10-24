#!/usr/bin/env python3
"""Compare a trusted trainer .pt against the compact INT2 .msr on fixed LR tensors.

Loading .pt files executes PyTorch deserialization. ONLY use checkpoint files
that you created or that come from a trusted source. Public inference does not
load .pt files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'research'))

import torch
from microsr_model import MicroSRX2
from research import microsr_int2_local as training


def check_parity(checkpoint: str, model_path: str, device: str = 'cpu',
                 tolerance: float = 2e-5) -> float:
    if tolerance <= 0:
        raise ValueError('Tolerance must be positive')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    if device == 'cuda':
        torch.backends.cudnn.allow_tf32 = False
    trained, info = training.load_model(checkpoint, device, use_inference=True)
    trained.eval()
    published = MicroSRX2.load(model_path, device=device)
    print(f"Checkpoint architecture: {trained.architecture}; selected={info.get('inference_weights', 'raw')}")
    print(f"Compact model grid: {published.grid}")
    expected_grid = info.get('config', {}).get('quant_grid', 'asym')
    if expected_grid != published.grid:
        raise AssertionError(f'Grid mismatch: {expected_grid} != {published.grid}')
    torch.manual_seed(20261023)
    maximum = 0.0
    with torch.inference_mode():
        for height, width in ((7, 9), (23, 37), (67, 35), (128, 127)):
            image = torch.rand((1, 3, height, width), dtype=torch.float32, device=device)
            true = trained(image)
            packed = published(image)
            gap = float((true - packed).abs().max())
            maximum = max(maximum, gap)
            print(f'  {width}x{height} -> {width*2}x{height*2}: max abs difference = {gap:.9g}')
    if maximum > tolerance:
        raise AssertionError(f'CHECKPOINT PARITY FAILED: maximum {maximum} exceeds {tolerance}')
    print(f'PASS: inference parity maximum absolute error {maximum:.9g} <= {tolerance}')
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='TRUSTED training .pt checkpoint')
    parser.add_argument('--model', default=str(ROOT / 'model' / 'microsr_v6.msr'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--tolerance', type=float, default=2e-5)
    args = parser.parse_args()
    check_parity(args.checkpoint, args.model, args.device, args.tolerance)


if __name__ == '__main__':
    main()
