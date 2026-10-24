# Portable `.msr` model format — version 1

Model: `microsr_v6.msr`; fixed total **495 bytes**.

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 4 | ASCII magic `MSR2` |
| 4 | 1 | File format version `1` |
| 5 | 1 | Quantization grid: `0` asymmetric or `1` balanced |
| 6 | 1 | Number of feature channels: `12` |
| 7 | 1 | Reserved flags `0` |
| 8 | 4 | Little-endian CRC32 of following 483-byte payload |
| 12 | 99 | 396 packed 2-bit wire codes, four codes per byte, least-significant first |
| 111 | 384 | Per layer: 12 FP32 scales followed by 12 FP32 biases |

Layer order: `expand`, `dw1`, `dw2`, `project`. The 2-bit wire codes are **not** directly signed integers:

- Asymmetric wire `0,1,2,3` maps to quantized levels `0,+1,-2,-1`.
- Balanced wire `0,1,2,3` maps to quantized levels `-1.5,-0.5,+0.5,+1.5`.

Layers have exact weight shapes `(12,3,1,1)`, `(12,1,3,3)` depthwise, `(12,1,3,3)` depthwise, `(12,12,1,1)`.

Scales must be finite strictly-positive FP32 values; biases finite FP32. The loader rejects incompatible version/channel flags, malformed export metadata, nonfinite parameters and corrupted CRC. **CRC32 detects incidental corruption, not malicious tampering or cryptographic authenticity.** Obtain models from a trusted publisher and verify release hashes if available.

## Pack a model

```powershell
python microsr_model.py pack --export "../microsr_int2" --output "model/microsr_v6.msr"
python microsr_model.py info --model "model/microsr_v6.msr"
```

The original trainer `.bin` and `.json` files can also be loaded as an export pair. Export metadata is much larger than the compact binary model because the scales/biases are human-readable in JSON.
