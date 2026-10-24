from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from microsr_model import (MicroSRX2, MSR_BYTES, PACKED_WEIGHT_BYTES, WEIGHT_COUNT,
                           _unpack_2bit, load_msr, load_trainer_export, pack_msr)


def test_exact_size_and_roundtrip(synthetic_export, synthetic_model_file):
    assert WEIGHT_COUNT == 396
    assert PACKED_WEIGHT_BYTES == 99
    assert MSR_BYTES == 495
    assert synthetic_model_file.stat().st_size == 495
    ga, wa, pa = load_trainer_export(synthetic_export)
    gb, wb, pb = load_msr(synthetic_model_file)
    assert ga == gb == 'asym'
    assert wa == wb
    assert _unpack_2bit(wa).shape == (396,)
    for (s0, b0), (s1, b1) in zip(pa, pb):
        np.testing.assert_array_equal(s0, s1)
        np.testing.assert_array_equal(b0, b1)


def test_original_export_and_compact_model_predict_same(synthetic_export, synthetic_model_file):
    original = MicroSRX2.load(synthetic_export)
    compact = MicroSRX2.load(synthetic_model_file)
    torch.manual_seed(51)
    x = torch.rand((1, 3, 27, 33))
    with torch.inference_mode():
        difference = (original(x) - compact(x)).abs().max().item()
    assert difference == 0.0


def test_crc_corruption_rejected(synthetic_model_file):
    corrupted = synthetic_model_file.with_name('bad.msr')
    data = bytearray(synthetic_model_file.read_bytes())
    data[-1] ^= 32
    corrupted.write_bytes(data)
    with pytest.raises(ValueError, match='checksum'):
        MicroSRX2.load(corrupted)


def test_invalid_file_length_rejected(synthetic_model_file):
    wrong = synthetic_model_file.with_name('short.msr')
    wrong.write_bytes(synthetic_model_file.read_bytes()[:-1])
    with pytest.raises(ValueError, match='size'):
        load_msr(wrong)


def test_invalid_export_scale_rejected(synthetic_export):
    path = synthetic_export.with_suffix('.json')
    meta = json.loads(path.read_text(encoding='utf-8'))
    meta['layers'][0]['scales'][0] = 0.0
    path.write_text(json.dumps(meta), encoding='utf-8')
    with pytest.raises(ValueError, match='strictly positive'):
        load_trainer_export(synthetic_export)


def test_model_is_exactly_the_expected_backbone(synthetic_model_file):
    net = MicroSRX2.load(synthetic_model_file)
    params = sum(p.numel() for p in net.parameters())
    assert params == 0
    # Stored small INT2 buffers, per-channel scales and biases; no master weights.
    assert sum(getattr(net, name).codes.numel() for name in ('expand', 'dw1', 'dw2', 'project')) == 396

