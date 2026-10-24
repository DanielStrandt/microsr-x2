from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from microsr_model import MicroSRX2


@pytest.mark.parametrize('height,width,tile', [(5, 7, 3), (13, 19, 8), (32, 35, 11)])
def test_tiling_matches_full_frame(synthetic_model_file, height, width, tile):
    model = MicroSRX2.load(synthetic_model_file)
    torch.manual_seed(99)
    low = torch.rand(1, 3, height, width)
    with torch.inference_mode():
        whole = model(low)
        tiled = model.upscale_tensor(low, tile_size=tile, halo=4)
    assert tuple(whole.shape) == (1, 3, 2 * height, 2 * width)
    torch.testing.assert_close(whole, tiled, atol=2e-5, rtol=0.0)


def test_opaque_image_is_lossless_png_ready(synthetic_model_file):
    model = MicroSRX2.load(synthetic_model_file)
    pixels = np.random.default_rng(0).integers(0, 256, (9, 11, 3), dtype=np.uint8)
    with Image.fromarray(pixels, mode='RGB') as src:
        output = model.upscale_pil(src, tile_size=5)
    assert output.size == (22, 18)
    assert output.mode == 'RGB'
    assert np.asarray(output).dtype == np.uint8


def test_alpha_is_preserved(synthetic_model_file):
    model = MicroSRX2.load(synthetic_model_file)
    rgba = np.zeros((11, 7, 4), dtype=np.uint8)
    rgba[:, :, 0] = 100
    rgba[:, :, 3] = 127
    with Image.fromarray(rgba, mode='RGBA') as src:
        output = model.upscale_pil(src, tile_size=6)
    assert output.size == (14, 22)
    assert output.mode == 'RGBA'
    assert np.max(np.abs(np.asarray(output)[:, :, 3].astype('int32') - 127)) <= 1


def test_float_images_are_explicitly_rejected(synthetic_model_file):
    model = MicroSRX2.load(synthetic_model_file)
    with Image.new('F', (10, 9), color=0.5) as src:
        with pytest.raises(ValueError, match='16-bit or floating'):
            model.upscale_pil(src)
