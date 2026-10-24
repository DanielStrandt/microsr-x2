# Research and model export (optional)

Most users need **only** the GUI/CLI and a `.msr` model. The `research/` directory contains the full trainer and the original color/exposure audit utility for reproducible experimentation.

Install the additional research dependency:

```powershell
python -m pip install -r requirements-research.txt
```

Training/checkpoint loading uses PyTorch serialization. **Never load untrusted `.pt` files.** No checkpoint file is needed for public inference.

## Export a trusted V6 checkpoint

From the repository root:

```powershell
python research/microsr_int2_local.py export --checkpoint "../runs/microsr_x2_v6_full/best.pt" --output "../microsr_int2"
python microsr_model.py pack --export "../microsr_int2" --output "model/microsr_v6.msr"
python scripts/verify_checkpoint_parity.py --checkpoint "../runs/microsr_x2_v6_full/best.pt" --model "model/microsr_v6.msr"
```

Replace `<checkpoint-path>` and `<export-directory>` with locations on your machine. The checkpoint must be one you created or otherwise trust.

## Full V6 joint discrete refinement

The `research/microsr_int2_local.py` module also supports training, validation, code/scale/bias refinement, V5 ablation, and V6 coordinated-pair refinement. The original V6 experiment used 24 rounds and 3 cycles/round, with 320 single-code probes and 128 pair probes per cycle, starting from a V5 refined checkpoint. It is **not** gradient training from scratch; it explores changes to the already quantized weights and calibrates existing scales and biases.

See `python research/microsr_int2_local.py refine-v6 --help` for the authoritative current flags. Specify your dataset path with `--image-dir`; only top-level images are used in training/evaluation. The developer validation results are not an independent final test.

## Color and exposure audit

```powershell
python research/microsr_color_audit.py --checkpoint "<trusted-checkpoint-path>" --image-dir "./data" --count 8 --crops-per-image 3 --output "./runs/color_audit"
```

The audit checks real-image mean luminance/RGB shifts and out-of-range samples, plus flat colors, grayscale ramps and 2×2 pixel-phase artifacts. In the previous local test, the maximum average luma shift was 0.113/255 and maximum average red–green cast shift was 0.355/255 across 24 sampled crops; some crops had as many as 3.6522% pre-clamp out-of-range channels. These observations are **sample-specific** and not proof that clipping/artifacts never occur.
