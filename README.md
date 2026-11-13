# AMD MI355X kernels

My Phase 1 kernels from the AMD × GPU MODE E2E Model Speedrun. I finished top 10 and made the finals, with a trophy and prize money. My MoE kernel placed 2nd out of 305.

This was my first GPU kernel competition, and I learned a lot along the way, especially about low-precision formats like MXFP4 and AMD's ROCm stack.

| Kernel | Rank | Time (geomean) |
|---|---|---|
| [Fused MoE (MXFP4)](moe-mxfp4/) | **2nd / 305** | 70.6 µs |
| [FP4 GEMM](mxfp4-gemm/) | 11th / 434 | 8.00 µs |
| [MLA decode](mla-decode/) | 16th / 283 | 32.8 µs |

Leaderboards: [GEMM](https://www.gpumode.com/leaderboard/763) · [MoE](https://www.gpumode.com/leaderboard/764) · [MLA](https://www.gpumode.com/leaderboard/765)

## The competition

GPU MODE and AMD ran an open competition to speed up LLM inference on AMD's MI355X. Phase 1, which closed on 7 April 2026, covered three kernels from DeepSeek-R1, each timed against AMD's own aiter kernels. The top 10 overall went through to the finals.

Each folder has the kernel I submitted and a short write-up of what worked and what didn't.

## License

[MIT](LICENSE)
