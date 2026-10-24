# MicroSR-X2: 2× Super-Resolution with 99 Bytes of INT2 Weight Codes

> **396 packed 2-bit weight codes · 492 stored parameter values · 495-byte `.msr` model file · +0.5323 dB over bicubic in development validation**

MicroSR-X2 is an experiment in how far a small neural network can improve 2× image upscaling. Its four-level quantized convolution weights occupy 99 packed bytes. The model also stores 48 FP32 channel scales and 48 FP32 biases; a 12-byte header and CRC32 bring the complete `.msr` file to 495 bytes. Inference uses PyTorch floating-point operations and a bicubic residual connection.

The spatial model expands RGB input to 12 feature channels, applies two 3×3 depthwise convolutions, then projects to 12 channels for PixelShuffle(2) residual output. The learned residual is added to a bicubic-upscaled input. This compact topology is an engineering experiment; the app still depends on PyTorch, NumPy and Pillow.

The reported **+0.5323 dB PSNR** gain over bicubic comes from a 205-image development validation split with five fixed crops per image. That split was consulted during development, so this is not an independent benchmark. The original validation images and split manifest are not included. See [benchmark methodology](docs/BENCHMARKS.md) before making quality comparisons.

- **396 convolution weights, quantized to four levels** (99 packed bytes).
- **48 FP32 channel scales + 48 FP32 biases** (384 additional bytes).
- **495-byte `.msr` model file** = 483 bytes of parameter data + 12 bytes of format header and CRC32.
- PyTorch-based inference, bicubic residual connection, CPU/CUDA support, tiled processing, EXIF orientation, and RGBA images.
- Optional research trainer, color/exposure audit, and tests.

> **Model terms:** the repository includes `model/microsr_v6.msr`. The MIT license covers the software source only; see [model weight terms](model/LICENSE_NOTICE.md) for the separate terms.

## Start here (Windows)

Requires Python 3.10+ with `tkinter` (usually included with the standard Python Windows installer).

```powershell
cd path\to\microsr-x2
python -m pip install -r requirements.txt
python microsr_gui.py
```

Or double-click **`Launch_MicroSR_GUI.bat`**. The GUI lets you browse for an input image, choose a 2× PNG output path, select CPU/GPU and tile size, and view side-by-side previews. The included model is selected by default; you can also browse for another `.msr` model or a trainer-exported `.bin`/`.json` pair.

If your CUDA-enabled PyTorch is already working, you generally do not need to reinstall it. For a CUDA-enabled installation on a new machine, use the current [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for your platform and driver.

### CLI: upscale one image

```powershell
python microsr_infer.py --model "./model/microsr_v6.msr" --input "F:\photos\example.jpg" --output "F:\photos\example_x2.png" --device auto --tile 384
```

### CLI: upscale a folder

```powershell
python microsr_infer.py --input "F:\photos" --output "F:\photos_upscaled" --device cuda --tile 384
```

By default, it does not recurse into subfolders, does not overwrite outputs, skips filenames already ending in `_x2`, and saves PNGs. Use `--recursive` or `--overwrite` explicitly if desired. The model accepts **native low-resolution images**; it does the 2× bicubic skip inside the model. Do **not** pre-upscale your input image.

## Package *your* V6 weights

With a trusted V6 training checkpoint at `<checkpoint-path>`:

```powershell
# Run the optional research trainer; .pt files must be from a trusted source.
python research/microsr_int2_local.py export --checkpoint "<checkpoint-path>" --output "<export-directory>"

# Convert the export pair to the compact 495-byte model file.
python microsr_model.py pack --export "<export-directory>" --output "./model/microsr_v6.msr"

# Inspect and verify it.
python microsr_model.py info --model "./model/microsr_v6.msr"
python microsr_infer.py --model "./model/microsr_v6.msr" --self-test --device cpu
```

If you already have an exported `.bin`/`.json` pair, use that directory in the pack command. The model packer stores four-level weight codes, scales and biases; it does not embed the Python code, PyTorch runtime, training data or FP32 master weights. No PyTorch training checkpoint is required to run inference.

**Before publishing a release**, verify your final `.msr` against your V6 checkpoint:

```powershell
python scripts/verify_checkpoint_parity.py --checkpoint "<trusted-checkpoint-path>" --model "./model/microsr_v6.msr" --device cpu
```

Only run `--checkpoint` with `.pt` files you trust: the training format is based on PyTorch serialization. Public inference never loads pickle checkpoints.

## Python use

```python
from PIL import Image
from microsr_model import MicroSRX2

model = MicroSRX2.load("model/microsr_v6.msr", device="cpu")
with Image.open("input.jpg") as src:
    output = model.upscale_pil(src, tile_size=384)
output.save("input_x2.png")
```

## Quantization and measured quality

The **weights are genuinely four-level quantized**, not unrestricted FP32 convolution weights. The default asymmetric levels are `-2, -1, 0, +1` times one FP32 scale per output channel. The 396 INT2 codes fit in 99 bytes. Activations, scales, biases, convolutions, and the bicubic baseline use floating-point calculations. This is **INT2 weight-quantized inference**, *not* a fully integer or 495-byte standalone executable.

A previous local V6 experiment reported the following **development validation** figures, using a fixed 205-image split and five deterministic crops per image (1,025 patches, up to 512 HR pixels per crop):

| Method | PSNR (dB) | Gain over bicubic |
|---|---:|---:|
| Bicubic | 33.1472 | — |
| Spatial V3 before discrete refinement | 33.5423 | +0.3952 |
| Spatial V5 refined | 33.5942 | +0.4471 |
| Spatial V6 refined | **33.6794** | **+0.5323** |

These are **user-reported training/evaluation results**, not CI benchmarks; the original images and split manifest are not bundled. Validation had been consulted during development, so it is **not an untouched independent test set**. They should not be presented as a new independent benchmark or a claim of being the world's smallest/most accurate model. See [Benchmark methodology](docs/BENCHMARKS.md) for reproducible and publication-ready evaluation guidance.

## Project structure

```text
microsr-x2-int2/
├── microsr_model.py             # Exact INT2 model and .msr loader/packer
├── microsr_infer.py             # File/folder CLI
├── microsr_gui.py               # Tkinter desktop interface
├── Launch_MicroSR_GUI.bat       # Windows double-click launcher
├── model/                       # Included 495-byte V6 .msr model
├── tests/                       # Synthetic, model-independent regression tests
├── research/                    # Optional trainer and color/exposure audit
├── scripts/                     # Release builder and checkpoint parity verifier
├── docs/                        # Format, architecture, benchmark, publishing notes
└── .github/workflows/ci.yml     # Windows + Linux automated tests
```

## Run the tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The included tests use deterministic synthetic weights; they do **not** validate your private V6 weights unless you run the parity command yourself. The test suite does not require a GPU, image dataset, GUI display or remote network connection.

## Publish on GitHub

1. Confirm you have the rights to distribute `model/microsr_v6.msr` and review its separate terms in `model/LICENSE_NOTICE.md`.
2. Review the **MIT** software license in `LICENSE` and confirm the attribution is appropriate.
3. Run the tests and, when you have the original trusted checkpoint, the model parity verifier. Ensure no dataset, `.pt` checkpoint, or private files are included.
4. From the repository root, publish the project:

   ```powershell
   git add .
   git commit -m "Initial MicroSR-X2 release"
   gh repo create microsr-x2 --public --source . --remote origin --push
   ```

   The final command uses the optional [GitHub CLI](https://cli.github.com/) after you authenticate with `gh auth login`; alternatively create an empty repository on GitHub, then push from Git. **This downloaded ZIP does not publish anything automatically.**

5. For a model-inclusive release ZIP, run `python scripts/build_release.py`, then upload `dist/microsr-x2-v6-release.zip` to a GitHub Release. See [Release instructions](docs/RELEASE.md).

## Limitations

- 2× enlargement only; 8-bit RGB/RGBA inputs; not a general-purpose photograph restoration model.
- `.msr` is the **model data file**, not an executable; PyTorch, NumPy and Pillow are required.
- File output is clamped to valid 8-bit RGB; internal floating-point outputs may extend beyond `[0,1]` around edges/highlights.
- Color profiling is limited; ICC tags may be retained, but this is not an end-to-end color-managed workflow.
- The public inference API does not execute or load untrusted `.pt` checkpoint files.

See [Architecture](docs/ARCHITECTURE.md), [Model format](docs/MODEL_FORMAT.md), [Training and audit](docs/TRAINING.md), [Contributing](CONTRIBUTING.md), and [Security](SECURITY.md). See also [Troubleshooting](docs/TROUBLESHOOTING.md).

## License

The MIT license in `LICENSE` applies to the software source. The model weights are governed separately by [model/LICENSE_NOTICE.md](model/LICENSE_NOTICE.md).
