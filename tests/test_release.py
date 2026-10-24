from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_builder', ROOT / 'scripts' / 'build_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_release_builder_checks_model_or_warns(tmp_path):
    # Never mutate the checked-in repository: a source-only release should work,
    # but the default must reject missing weights.
    present = (ROOT / 'model' / 'microsr_v6.msr').is_file()
    if present:
        pytest.skip('Real model is present; check packaging in the next test')
    with pytest.raises(FileNotFoundError):
        release.make_release(dest=tmp_path / 'should_not_exist.zip')
    path = release.make_release(allow_no_model=True, dest=tmp_path / 'source.zip')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert any(n.endswith('/microsr_gui.py') for n in names)
        assert not any(n.endswith('microsr_v6.msr') for n in names)
    assert path.with_name(path.name + '.sha256').is_file()

