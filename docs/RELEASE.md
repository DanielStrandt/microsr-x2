# GitHub publication and release checklist

This project folder contains the prepared source and model. It is not a Git checkout until initialized and committed.

## Before publishing

- [ ] Confirm you have the rights to distribute the actual model weights and that the terms in `model/LICENSE_NOTICE.md` are correct.
- [ ] Review the proposed `LICENSE`, attribution and contributor credits.
- [x] Include the 495-byte `model/microsr_v6.msr` file.
- [ ] Run `python microsr_model.py info --model model/microsr_v6.msr` and `python microsr_infer.py --self-test`.
- [ ] When you have a trusted `.pt`, run `python scripts/verify_checkpoint_parity.py --checkpoint <trusted V6 best.pt> --model model/microsr_v6.msr`.
- [ ] Run `python -m pytest -q`, including on Windows if available.
- [ ] Verify a real GUI upscale and output PNG, and review exposure/clipping artifacts.
- [ ] Review published quality statements in `README.md`/`docs/BENCHMARKS.md`; independently evaluate an untouched test dataset if making benchmark claims.
- [ ] Check `git status` before pushing; no dataset or `.pt` checkpoints should be staged.

## Create a public GitHub repo

From this project folder:

```powershell
git init
git add .
git commit -m "Initial MicroSR-X2 release"
# Requires the GitHub CLI and an authenticated account:
gh repo create microsr-x2 --public --source . --remote origin --push
```

If you do not use GitHub CLI, create an empty repository at https://github.com/new and follow its existing-repository push directions. This package does not create or publish a remote automatically.

## Create a release ZIP

```powershell
python scripts/build_release.py
```

By default, this **requires a valid 495-byte `model/microsr_v6.msr`**, and creates `dist/microsr-x2-v6-release.zip` containing the runnable inference modules, GUI, launcher, dependencies, model and license. It writes a SHA-256 checksum file beside the ZIP. To build a source-only demonstration ZIP, use `--allow-no-model`.

GitHub Releases: create tag `v0.1.0`, create a Release on GitHub and attach the ZIP and its checksum. This does not auto-publish releases.

## Suggested release notes

MicroSR-X2 V6: 2× RGB super-resolution with a four-level INT2 convolution backbone (396 codes / 99 packed code bytes). Model payload includes 48 FP32 scales and 48 FP32 biases, yielding a 495-byte `.msr` model file with header/checksum. Desktop GUI and batch CLI available. Runtime uses floating-point PyTorch and bicubic upsampling. Development validation results are not independent test benchmarks.
