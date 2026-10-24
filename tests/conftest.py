from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from microsr_model import LAYERS, PACKED_WEIGHT_BYTES


@pytest.fixture
def synthetic_export(tmp_path):
    """Create deterministic, explicitly UNTRAINED 396-code model export for tests."""
    rng = np.random.default_rng(31)
    wire = rng.integers(0, 4, size=396, dtype=np.uint8)
    packed = bytearray(PACKED_WEIGHT_BYTES)
    for i, value in enumerate(wire):
        packed[i // 4] |= int(value) << (2 * (i % 4))
    layers = []
    for name, shape, groups, kernel in LAYERS:
        scales = np.linspace(0.001, 0.008, shape[0], dtype=np.float32)
        biases = rng.uniform(-0.001, 0.001, size=shape[0]).astype(np.float32)
        layers.append({
            'name': name, 'shape': list(shape), 'groups': groups, 'kernel': kernel,
            'scales': scales.tolist(), 'bias': biases.tolist(),
        })
    meta = {
        'format': 'MicroSR-X2-INT2-packet-v3',
        'architecture': 'spatial_lr_dw2_shuffle_v3',
        'channels': 12,
        'int2_grid': 'asym',
        'layers': layers,
    }
    path = tmp_path / 'microsr_int2'
    path.with_suffix('.bin').write_bytes(bytes(packed))
    path.with_suffix('.json').write_text(json.dumps(meta), encoding='utf-8')
    return path


@pytest.fixture
def synthetic_model_file(synthetic_export):
    from microsr_model import pack_msr
    return pack_msr(synthetic_export, synthetic_export.parent / 'microsr_v6.msr')

