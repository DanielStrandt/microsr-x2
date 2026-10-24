#!/usr/bin/env python3
"""Build a distributable ZIP with the exact 495-byte V6 .msr model (if provided)."""
from __future__ import annotations

import argparse
import hashlib
import struct
import zlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
RELEASE_NAME = 'microsr-x2-v6-release'
BASE_FILES = (
    'microsr_model.py',
    'microsr_infer.py',
    'microsr_gui.py',
    'Launch_MicroSR_GUI.bat',
    'requirements.txt',
    'README.md',
    'LICENSE',
    'CONTRIBUTING.md',
    'SECURITY.md',
    'CHANGELOG.md',
    'model/README.md',
    'model/LICENSE_NOTICE.md',
    'docs/ARCHITECTURE.md',
    'docs/MODEL_FORMAT.md',
    'docs/BENCHMARKS.md',
    'docs/TROUBLESHOOTING.md',
    'docs/RELEASE.md',
    'docs/TRAINING.md',
)
RESEARCH_FILES = (
    'requirements-research.txt',
    'research/microsr_int2_local.py',
    'research/microsr_color_audit.py',
)


def validate_model(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Cannot build a model-inclusive release: {path} missing')
    data = path.read_bytes()
    if len(data) != 495:
        raise ValueError(f'Invalid model size: {len(data)}, expected 495 bytes')
    magic, version, grid, channels, flags, checksum = struct.unpack_from('<4sBBBBI', data)
    if (magic, version, channels, flags) != (b'MSR2', 1, 12, 0) or grid not in (0, 1):
        raise ValueError('Incompatible model header')
    if (zlib.crc32(data[12:]) & 0xFFFFFFFF) != checksum:
        raise ValueError('Model CRC32 failed')


def make_release(*, allow_no_model: bool = False, include_research: bool = False,
                 dest: Path | None = None) -> Path:
    model_path = ROOT / 'model' / 'microsr_v6.msr'
    files = list(BASE_FILES)
    if model_path.is_file():
        validate_model(model_path)
        files.append('model/microsr_v6.msr')
    elif not allow_no_model:
        raise FileNotFoundError(
            f'No packaged model at {model_path}. Pack your actual V6 export first, '
            'or explicitly pass --allow-no-model for source-only ZIPs.'
        )
    if include_research:
        files.extend(RESEARCH_FILES)
    target = dest if dest is not None else ROOT / 'dist' / (RELEASE_NAME + '.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=9) as out:
        for rel in sorted(files):
            path = ROOT / rel
            if not path.is_file():
                raise FileNotFoundError(f'Missing release file {rel}')
            out.write(path, f'{RELEASE_NAME}/{rel}')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    checksum = target.with_name(target.name + '.sha256')
    checksum.write_text(f'{digest}  {target.name}\n', encoding='ascii')
    print(f'Built: {target} ({target.stat().st_size:,} bytes)')
    print(f'SHA256: {digest}')
    if model_path.is_file():
        print('Includes a complete 495-byte compact model.')
    else:
        print('SOURCE ONLY: model file absent, release cannot upscale until you add weights.')
    print(f'Checksum: {checksum}')
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-no-model', action='store_true', help='Build source-only release ZIP without .msr')
    parser.add_argument('--include-research', action='store_true', help='Include optional trainer and color audit')
    parser.add_argument('--output', type=Path, help='Custom release ZIP path')
    args = parser.parse_args()
    make_release(allow_no_model=args.allow_no_model, include_research=args.include_research, dest=args.output)


if __name__ == '__main__':
    main()
