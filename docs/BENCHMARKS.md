# Reproducibility and benchmark claims

## Previously observed development data

The V6 development run reported bicubic `33.1472 dB` and Spatial V6 `33.6794 dB` (**+0.5323 dB**) on a **205-image development validation split** with 5 fixed crops per image (1,025 crops). The V3 result on the same split was `33.5423279 dB`; V5 was `33.5942 dB`.

These values were reported during the local experiments. This repository does **not** contain the original images, final user-trained model file, validation split manifest or output tensors, so its automated tests cannot independently reproduce these exact figures.

## Recommended publication evaluation

1. Lock the exact commit, model `.msr` SHA-256, architecture and inference code before final evaluation.
2. Use separate **untouched** datasets (e.g. Set5, Set14, BSD100, Urban100 when permitted), rather than reusing a development validation set consulted while optimizing.
3. Fully specify the degradation procedure, HR/LR alignment, crop border shaving, color space (RGB or luminance/Y), metric aggregation (mean PSNR vs PSNR from global MSE) and whether any clamping occurs. Bicubic downsampling with antialiasing and bicubic upsampling use library-specific implementations.
4. Report both the **weight payload (99 B)** and the **complete model file (495 B)**, along with compute/runtime and device. Don't describe the whole inference application as 495 bytes.
5. Avoid **“world's smallest”** or other records without a scoped, dated literature review covering complete model storage, runtime assumptions and image fidelity.
6. Show representative output images with permission, as well as error analyses for clipping, colors and saturation. The separate color audit can help find artifacts; a flag-free report alone doesn't prove perfect color fidelity.

The tiny architecture has an explicit bicubic skip, so comparisons must describe it as **bicubic-assisted residual SR**. The deployment CLI is intended for natural low-resolution photos; the training protocol uses synthetic bicubic degradation and may not transfer to arbitrary real-world image artifacts.
