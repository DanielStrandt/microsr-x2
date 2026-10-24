from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from microsr_gui import run_inference
from microsr_infer import image_paths

ROOT = Path(__file__).resolve().parents[1]


def test_cli_upscales_a_file(synthetic_model_file, tmp_path):
    path = tmp_path / 'sample.png'
    Image.fromarray(np.full((8, 13, 3), 80, dtype=np.uint8)).save(path)
    dest = tmp_path / 'upscaled.png'
    command = [sys.executable, str(ROOT / 'microsr_infer.py'),
               '--model', str(synthetic_model_file), '--input', str(path),
               '--output', str(dest), '--device', 'cpu', '--tile', '8']
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
    assert dest.is_file()
    with Image.open(dest) as output:
        assert output.size == (26, 16)
    assert 'Finished: 1 written' in completed.stdout


def test_gui_worker_without_display(synthetic_model_file, tmp_path):
    path = tmp_path / 'sample.png'
    Image.fromarray(np.full((8, 13, 3), 135, dtype=np.uint8)).save(path)
    dest = tmp_path / 'gui_result.png'
    outcome = run_inference(path, dest, synthetic_model_file, 'cpu', 8)
    assert outcome['output_size'] == (26, 16)
    assert outcome['device'] == 'cpu'
    assert dest.is_file()


def test_folder_search_does_not_recurse_by_default(tmp_path):
    (tmp_path / 'top.png').write_bytes(b'not decoded in directory listing')
    deep = tmp_path / 'nested'
    deep.mkdir()
    (deep / 'inside.png').write_bytes(b'not decoded in directory listing')
    assert [p.name for p in image_paths(tmp_path, recursive=False)] == ['top.png']
    assert len(image_paths(tmp_path, recursive=True)) == 2

