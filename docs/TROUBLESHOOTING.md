# Troubleshooting

## `FileNotFoundError: model/microsr_v6.msr`

The repository includes `model/microsr_v6.msr`. If it is missing, export and pack a model.
From the repository folder:

```powershell
python microsr_model.py pack --export "../microsr_int2" --output "./model/microsr_v6.msr"
```

If you have a model in another location, copy it into the repository's `model/` directory.

## `can't open file '...microsr_model.py'`

Your PowerShell working directory doesn't contain that file. Run `dir` and
`cd` into the folder holding `microsr_model.py`.

## CUDA unavailable

Run `--device cpu` to confirm basic operation. `--device cuda` requires a
CUDA-enabled PyTorch installation and compatible NVIDIA drivers. `--device auto`
falls back to CPU.

## GUI doesn't open

Install Python 3.10+ with tkinter and the dependencies from `requirements.txt`.
Start from PowerShell with `python microsr_gui.py` to read error messages.
On non-GUI/headless Linux CI, the inference tests can run without a display.

## Large image or out-of-memory error

Use a smaller native LR tile size (`--tile 256` or `--tile 128`) or CPU inference.
The output still needs system RAM for its full 2× RGB/RGBA pixels.

## Image unexpectedly appears unchanged

MicroSR uses a bicubic skip plus a learned residual, so the visual difference may
be subtle. Check the output dimensions and compare it against a separately
upscaled bicubic reference at 100% zoom. The input must be the **original LR**
image rather than a pre-upscaled image.

## `.pt` checkpoint fails to load in the GUI

This is intentional. The inference app loads `.msr` or the trainer's `.bin/.json`
export pair—not `.pt`. Run the trusted research trainer's `export` command, then
pack it using `microsr_model.py`.

