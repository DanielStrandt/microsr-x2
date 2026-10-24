# Architecture and model accounting

Input: 8-bit RGB converted to NCHW FP32 `[0,1]` *at its native resolution*.

| Stage | Operation | INT2 codes | Bias scalars | Scale scalars |
|---|---|---:|---:|---:|
| expand | 1×1 Conv2d, RGB 3 → 12 | 36 | 12 | 12 |
| dw1 | depthwise 3×3, reflect pad, 12 channels | 108 | 12 | 12 |
| dw2 | depthwise 3×3, reflect pad, 12 channels | 108 | 12 | 12 |
| project | 1×1 Conv2d, 12 → 12 | 144 | 12 | 12 |
| **Total** | | **396** | **48** | **48** |

`ReLU(expand) → ReLU(dw1) → ReLU(dw2) → project → PixelShuffle(2)` predicts a three-channel HR residual. Its output is added to a separately bicubic-upscaled native LR input (`align_corners=False`, PyTorch bicubic interpolation). The two depthwise layers give a nominal 5×5 receptive field in low-resolution pixels. Float convolution weights are reconstructed as `quantized_code × per-output-channel FP32 scale` in runtime scratch buffers.

## What “2-bit” means here

The model uses **396 four-level discrete convolution codes**. With the default asymmetric grid those levels are `-2,-1,0,+1`, scaled per channel. Packed storage: 396 × 2 / 8 = **99 bytes**. The learned 48 FP32 scales and 48 FP32 biases use 384 further bytes. The minimal raw parameter payload is **483 bytes**; the published `.msr` format adds 12 bytes of versioning/checksum for a total **495 bytes**.

The full package and the PyTorch runtime are much larger. The model still performs bicubic interpolation and floating-point operations. There is no true packed-bit arithmetic engine or INT2 activation quantization, and neither inference compute cost nor memory consumption is 495 bytes.

## Inference scope

- Inference is 2× per spatial dimension (4× pixel count).
- Full RGB is predicted. The trainer's HR reference crops are downsampled with bicubic antialiasing to define synthetic LR inputs; live photos can have different degradation distributions.
- The saved `.msr` contains no optimizer, floating master weights or training state.
- Tiled inference uses low-resolution halos around each tile so outputs agree with full-frame prediction within floating-point tolerance on tested shapes.
- For RGBA images, alpha uses bicubic scaling, not learned color convolution.
