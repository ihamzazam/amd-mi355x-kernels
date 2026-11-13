# MXFP4 GEMM

**11th / 434 · 8.00 µs ranked geomean**

BF16 activations × pre-shuffled MXFP4 weights → BF16 output, across six shapes from M=4 to M=256.

## Approach

The final version is a standalone Triton kernel with no aiter dependency.

- A single kernel quantizes each A tile to FP4 on the fly and runs the scaled matmul. There's no separate quantization pass.
- Quantization uses the gfx950 hardware instruction `v_cvt_scalef32_pk_fp4_bf16` through inline asm. That's about 16 instructions per K-step instead of roughly 430 in software, so quantization is effectively free.
- B is read directly in its pre-shuffled layout.
- Tile sizes, split-K, `waves_per_eu`, pipeline stages and cache modifiers are tuned per shape. The small-M, large-K shape splits K 14 ways and finishes with a small reduce kernel.

Earlier versions built on aiter's Triton GEMM got to 8.54 µs. Rewriting it as a standalone kernel and retuning every shape brought it to 8.00 µs.

## What didn't work

- **Separate quant + ASM GEMM:** correct, but about 2.5× slower because of the extra pass.
- **Non-preshuffled kernel:** failed correctness on the ranked shapes.
- **Larger tiles (BLOCK_M=32 with BLOCK_K=512):** register spilling.
- **Grouped tile ordering:** 2–10% worse; plain sequential order used L2 better.
- **Triton → ASM hybrid for large M:** about 2× faster on those shapes, but the extra module builds pushed it past the runner's compile timeout.
