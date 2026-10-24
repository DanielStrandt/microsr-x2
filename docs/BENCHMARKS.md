# Classic 2× super-resolution benchmarks

## Dataset suite

| Dataset | Images | Common evaluation focus |
| --- | ---: | --- |
| Set5 | 5 | Small introductory natural-image benchmark |
| Set14 | 14 | Broader image content |
| BSD100 | 100 | Natural-image reconstruction |
| Urban100 | 100 | Repeated structures and architectural detail |
| Manga109 | 109 | Manga line art, text-like edges, and illustrations |

The first four are established image-SR benchmarks. EDSR's reference implementation describes evaluation on Y-channel PSNR and links the benchmark providers. Manga109 is separately provided by the University of Tokyo and requires users to accept its dataset-specific terms. Obtain the data from the dataset providers or other sources whose distribution terms you are entitled to follow. Do not check dataset images, generated crops, LR images, or SR outputs into this repository.

## Measured results (2026-10-23)

| Metric | Why it is logged |
| --- | --- |
| Bicubic PSNR / SSIM | Establishes the same-protocol interpolation baseline. |
| MicroSR PSNR / SSIM | Measures reconstruction quality for the model. |
| Gain over bicubic (dB) | States the primary quality comparison directly. |
| RGB and Y-channel metrics | Makes full-color results available alongside the Y-channel values common in published SR comparisons. |
| Inference time and throughput | Documents practical model-call speed for the recorded hardware and image sizes. |
| Parameter count and model-file size | Shows the compact stored representation. |
| Peak CUDA allocated memory | Records GPU memory required by an individual forward call when CUDA is used. |

The included V6 model was evaluated on the four complete datasets obtainable from the EDSR-linked benchmark archive. Scores are means of per-image metrics under the protocol below.

| Dataset | Images | Bicubic Y PSNR | MicroSR Y PSNR | Gain | Bicubic Y SSIM | MicroSR Y SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Set5 | 5 | 33.6486 | **34.5494** | **+0.9008 dB** | 0.92923 | **0.93861** |
| Set14 | 14 | 30.3177 | **31.0899** | **+0.7722 dB** | 0.86893 | **0.88992** |
| BSD100 | 100 | 29.5445 | **30.2639** | **+0.7195 dB** | 0.84232 | **0.87052** |
| Urban100 | 100 | 26.8676 | **27.6051** | **+0.7375 dB** | 0.83953 | **0.86257** |

RGB-channel scores and SSIM are also recorded in the machine-readable result file. On this CPU run, mean per-image model-call times were **10.3 ms (Set5), 19.1 ms (Set14), 11.5 ms (BSD100), and 82.0 ms (Urban100)**. These vary with image dimensions and are not a speed guarantee for other hardware. Peak GPU memory was **not measured** because this run used CPU. The included model has **492 stored parameter values**, including 396 2-bit weight codes and 96 FP32 scale/bias values; its complete `.msr` file is **495 bytes** (99 packed weight bytes, 384 scale/bias bytes, 12 header/checksum bytes).

Manga109 was skipped because it requires access through its official gated flow; no alternate mirror was used. These results describe this exact evaluator and model hash, recorded in [`benchmark_results_x2.json`](benchmark_results_x2.json), with all 219 per-image measurements in [`benchmark_per_image_x2.csv`](benchmark_per_image_x2.csv). They are same-protocol comparisons against the included evaluator's bicubic output, not a claim of independent training or direct equivalence to every paper's evaluation implementation. The earlier `+0.5323 dB` development result remains separate.

## Reproduce

The downloader fetches the four datasets offered through the EDSR-linked benchmark archive and checks the expected image counts. Dataset files stay local under the ignored `data/` directory:

```powershell
python scripts/download_benchmarks.py
```

Manga109 is skipped because access is gated. If you have authorized access, place the 109 HR pages in `data/benchmarks/Manga109/HR/` and include it explicitly in the benchmark command.

The local directory layout is:

```text
data/benchmarks/
├── Set5/HR/*
├── Set14/HR/*
├── BSD100/HR/*
├── Urban100/HR/*
└── Manga109/HR/*
```

Install the normal project requirements into an environment with PyTorch, then run the four complete sets from the repository root:

```powershell
python scripts/benchmark_sr.py --data-root "./data/benchmarks" --model "./model/microsr_v6.msr" --output "./benchmark-results" --device cpu
```

The default run automatically skips any set that is missing images or has the wrong image count. To run only a chosen subset, name the datasets after `--datasets`, for example:

```powershell
python scripts/benchmark_sr.py --data-root "./data/benchmarks" --datasets Set5 Set14 BSD100 Urban100 --device cpu
```

The evaluator uses scale ×2. It crops bottom/right HR pixels so both dimensions divide by two, synthesizes LR with antialiased MATLAB-style bicubic downsampling, and feeds the resulting RGB image to the model. The bicubic reference is the same cubic resize applied directly from LR to HR. Both reference and model outputs are clipped to `[0,255]`, rounded to 8-bit, and evaluated after shaving two HR pixels from each edge. It reports RGB PSNR and BT.601 Y-channel PSNR per image, then averages those per-image values within each dataset. It also reports Y-channel SSIM using an 11×11 Gaussian window (σ=1.5), valid filtering, and sample covariance correction. The model's internal bicubic skip remains the PyTorch `interpolate(..., mode="bicubic", align_corners=False)` implementation, so that implementation detail is part of the measured system.

`per_image.csv` records per-image RGB/Y PSNR and SSIM, model-call time, throughput, and CUDA peak allocated memory when measured. `summary.json` records protocol, model SHA-256, model size and parameter counts, dataset image-manifest hashes, runtime versions, device, and per-dataset means plus latency statistics. Timing is hardware-specific and excludes image decoding, LR synthesis, and file I/O; one untimed warm-up call is run before each dataset. GPU memory is the maximum CUDA allocated memory observed during an individual forward call; it is null on CPU/MPS runs. For a paper or headline claim, record the exact dataset release and source, evaluation code commit, model hash, hardware, and software versions alongside the reported values. Do not compare figures produced with different color channels, degradation kernels, border shaving, rounding, or aggregation rules as if they were directly equivalent.

## Dataset terms and references

- The [EDSR-PyTorch benchmark instructions](https://github.com/sanghyun-son/EDSR-PyTorch) link Set5, Set14, BSD100, and Urban100 sources; the downloader uses its linked benchmark archive.
- [Manga109 official site and terms](https://manga109.github.io/manga109-project-website/en/) distinguish academic/non-commercial Manga109 from Manga109-s. The official terms prohibit redistribution of the images; Manga109-s additionally requires clearly identifying the dataset when publishing results.
- Keep all downloaded datasets local and follow the exact license or access agreement attached to the copy you use.

## Previously observed development data

The V6 development run reported bicubic `33.1472 dB` and Spatial V6 `33.6794 dB` (**+0.5323 dB**) on a **205-image development validation split** with 5 fixed crops per image (1,025 crops). The V3 result on the same split was `33.5423279 dB`; V5 was `33.5942 dB`.

These values were reported during the local experiments. This repository does **not** contain the original images, final user-trained model file, validation split manifest or output tensors, so its automated tests cannot independently reproduce these exact figures. Validation had been consulted during development, so this is not an untouched independent test set.

