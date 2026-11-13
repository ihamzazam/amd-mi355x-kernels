# MoE, MXFP4

**2nd / 305 · 70.6 µs ranked geomean**

DeepSeek-R1 mixture-of-experts layer with MXFP4 weights: 7168 hidden size, top-8 routed experts plus one shared expert. It runs across seven batch/expert shapes.

## Approach

The baseline is aiter's two-stage fused MoE: stage 1 is gate/up + SiLU, stage 2 is the down projection. Most of the gain came from swapping in faster kernels underneath it:

- The runner's aiter didn't support the newer FlyDSL MoE kernels, so the submission pulls a newer aiter, copies in the FlyDSL kernels, and patches `fused_moe.py` at import time.
- FlyDSL handles stage 2 for every shape (tile 16×256×128, atomic accumulate) and stage 1 for the 33-expert shapes. The 257-expert shape stays on CK for stage 1, where FlyDSL was slower or hit memory faults.
- All FlyDSL kernels are compiled at import, so none of the compile cost lands inside timed runs.

That took it from about 128 µs to about 70 µs.

## What didn't work

- **Split-K (KSPLIT=2) for large shapes:** much slower.
- **Other FlyDSL tile sizes** (t8, t128, larger K tiles): correctness failures or no gain.
- **CUDA/HIP graphs:** blocked by the competition runner.
- **Skipping or caching activation quantization:** failed correctness on the secret seeds.
