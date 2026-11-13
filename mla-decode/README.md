# MLA decode

**16th / 283 · 32.8 µs ranked geomean**

Multi-head latent attention decode for DeepSeek-R1: 16 query heads, a 576-dim compressed KV cache, batch 4–256 and KV length 1K–8K.

## Approach

- Cast Q to FP8 with a fixed scale (a single `copy_`), then use aiter's persistent fp8 × fp8 paged ASM decode kernel.
- Per-shape page size, KV splits and granularity: page size 8 for 8K contexts, 2 for most 1K ones. For batch 32 / KV 1024, a non-persistent single-split mode skips the reduce kernel entirely.
- Every shape gets a pre-built closure holding its metadata and buffers, so the per-call Python overhead is close to zero.

Going from 65.6 µs to 32.8 µs was mostly page size and split tuning.

## What didn't work

- **Page size 16:** fails correctness.
- **Page size 4 at 1K context:** fails correctness.
- **Non-persistent mode for other shapes:** fails correctness or runs slower.
- **Pure Triton flash decode:** 3–70× slower than the ASM kernel.
- **Triton kernel reading MXFP4 KV directly:** about 16× slower; dequantization dominates.
- **BF16 Q (no FP8 cast):** about 1.5× slower.

The top entries were around 12–24 µs using custom ASM, which is where the remaining gap is.
