# AMD MI355X kernels

My submissions to phase 1 of the GPU MODE × AMD kernel competition: three inference kernels from DeepSeek-R1, optimised for the AMD Instinct MI355X.

**Top 10 overall (trophy and prize), including 2nd place out of 305 on the MoE kernel.**

| Kernel | Rank | Time (ranked geomean) |
|---|---|---|
| [MoE, MXFP4](moe-mxfp4/) | **2nd / 305** | 70.6 µs |
| [MXFP4 GEMM](mxfp4-gemm/) | 11th / 434 | 8.00 µs |
| [MLA decode](mla-decode/) | 16th / 283 | 32.8 µs |

Leaderboards: [GEMM](https://www.gpumode.com/leaderboard/763) · [MoE](https://www.gpumode.com/leaderboard/764) · [MLA](https://www.gpumode.com/leaderboard/765)

Each folder has the submitted kernel and a short write-up: the problem, what worked, and what didn't.

## Running

Submissions go through [popcorn-cli](https://github.com/gpu-mode/popcorn-cli). The `#!POPCORN` header at the top of each file picks the leaderboard and GPU.

```
popcorn submit --mode leaderboard mxfp4-gemm/submission.py
```

They expect the competition's runner environment (ROCm 7.1, PyTorch 2.10, aiter) and won't run on a stock machine.

## License

[MIT](LICENSE)
