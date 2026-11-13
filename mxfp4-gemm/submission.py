#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
Fused BF16-to-FP4 quantization + scaled GEMM with pre-shuffled weights.
Hardware-accelerated quantization via v_cvt_scalef32_pk_fp4_bf16.
Per-shape tuned tile/split-K parameters for MI355X (256 CUs).
"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
import sys as _sys
import gc as _gc

torch.set_grad_enabled(False)
_gc.disable()
_sys.setswitchinterval(1.0)


@triton.jit
def _quant_block_fp4(
    inp_bf16,
    TILE_M: tl.constexpr,
    TILE_K: tl.constexpr,
):
    """Convert BF16 tile to packed MXFP4 using hardware pk_fp4 instruction."""
    GRP_SZ: tl.constexpr = 32
    N_GROUPS: tl.constexpr = TILE_K // GRP_SZ

    vals_f32 = inp_bf16.to(tl.float32).reshape(TILE_M, N_GROUPS, GRP_SZ)

    # Block-wise absolute max with rounding to nearest power-of-2
    peak = tl.max(tl.abs(vals_f32), axis=-1, keep_dims=True)
    peak = peak.to(tl.int32, bitcast=True)
    peak = (peak + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    log2_peak = ((peak >> 23) & 0xFF).to(tl.int32) - 127
    exp_unbiased = log2_peak - 2
    exp_unbiased = tl.minimum(tl.maximum(exp_unbiased, -127), 127)
    scale_e8m0 = exp_unbiased.to(tl.uint8) + 127

    # Build IEEE754 float divisor: 2^unbiased for hw instruction
    div_bits = (exp_unbiased.to(tl.int32) + 127).to(tl.uint32) << 23
    divisor = div_bits.to(tl.float32, bitcast=True)  # [M, N_GROUPS, 1]

    # Expand divisor to per-pair level
    div_full = tl.broadcast_to(divisor, (TILE_M, N_GROUPS, GRP_SZ))
    div_full = div_full.reshape(TILE_M, TILE_K)
    div_pairs = div_full.reshape(TILE_M, TILE_K // 2, 2)
    div_even, _ = tl.split(div_pairs)
    div_per_pair = div_even.reshape(TILE_M, TILE_K // 2)

    # Interleave adjacent BF16 elements into uint32 for hw conversion
    inp_u16 = inp_bf16.to(tl.uint16, bitcast=True).reshape(TILE_M, TILE_K // 2, 2)
    lo_half, hi_half = tl.split(inp_u16)
    packed_u32 = lo_half.to(tl.uint32) | (hi_half.to(tl.uint32) << 16)
    packed_u32 = packed_u32.reshape(TILE_M, TILE_K // 2)

    # Hardware FP4 pack-convert
    fp4_raw = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_bf16 $0, $1, $2",
        "=v, v, v",
        [packed_u32, div_per_pair],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    fp4_bytes = (fp4_raw & 0xFF).to(tl.uint8)
    fp4_bytes = fp4_bytes.reshape(TILE_M, TILE_K // 2)

    return fp4_bytes, scale_e8m0.reshape(TILE_M, N_GROUPS)


@triton.heuristics(
    {
        "ALIGNED_K": lambda args: (args["K"] % (args["TILE_K"] // 2) == 0)
        and (args["SK_TILE"] % args["TILE_K"] == 0)
        and (args["K"] % (args["SK_TILE"] // 2) == 0),
    }
)
@triton.jit
def _matmul_fused_kernel(
    inp_ptr, wt_ptr, out_ptr, wsc_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_ok, stride_om, stride_on,
    stride_sn, stride_sk,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    N_SPLITS: tl.constexpr,
    SK_TILE: tl.constexpr,
    ALIGNED_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    load_modifier: tl.constexpr,
):
    tl.assume(stride_im > 0)
    tl.assume(stride_ik > 0)
    tl.assume(stride_wn > 0)
    tl.assume(stride_wk > 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_on > 0)
    tl.assume(stride_sn > 0)
    tl.assume(stride_sk > 0)

    SCALE_GRP: tl.constexpr = 32
    n_tiles_m = tl.cdiv(M, TILE_M)
    n_tiles_n = tl.cdiv(N, TILE_N)

    flat_pid = tl.program_id(axis=0)
    split_id = flat_pid % N_SPLITS
    tile_pid = flat_pid // N_SPLITS

    # Tile assignment: grouped swizzle for single-split, linear for multi-split
    if N_SPLITS == 1:
        tiles_per_grp = GROUP_M * n_tiles_n
        grp = tile_pid // tiles_per_grp
        first_m = grp * GROUP_M
        grp_sz = min(n_tiles_m - first_m, GROUP_M)
        tile_m = first_m + ((tile_pid % tiles_per_grp) % grp_sz)
        tile_n = (tile_pid % tiles_per_grp) // grp_sz
    else:
        tile_m = tile_pid // n_tiles_n
        tile_n = tile_pid % n_tiles_n

    tl.assume(tile_m >= 0)
    tl.assume(tile_n >= 0)
    tl.assume(split_id >= 0)

    if (split_id * SK_TILE // 2) < K:
        k_iters = tl.cdiv(SK_TILE // 2, TILE_K // 2)

        # A: BF16 input [M, 2*K]
        row_a = (tile_m * TILE_M + tl.arange(0, TILE_M)) % M
        col_a = split_id * SK_TILE + tl.arange(0, TILE_K)
        ptrs_a = inp_ptr + (row_a[:, None] * stride_im + col_a[None, :] * stride_ik)

        # B: pre-shuffled FP4 weights [N//16, K_packed*16]
        shuf_range = tl.arange(0, (TILE_K // 2) * 16)
        shuf_base = split_id * (SK_TILE // 2) * 16 + shuf_range
        row_b = (tile_n * (TILE_N // 16) + tl.arange(0, TILE_N // 16)) % (N // 16)
        ptrs_b = wt_ptr + (row_b[:, None] * stride_wn + shuf_base[None, :] * stride_wk)

        # B scales: shuffled E8M0 layout
        row_s = (tile_n * TILE_N + tl.arange(0, TILE_N // 32) * 32)
        col_s = (split_id * (SK_TILE // SCALE_GRP) * 32) + tl.arange(
            0, TILE_K // SCALE_GRP * 32
        )
        ptrs_s = wsc_ptr + row_s[:, None] * stride_sn + col_s[None, :] * stride_sk

        acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        for ki in range(split_id * k_iters, (split_id + 1) * k_iters):
            # Issue all loads before compute for memory-level parallelism
            if ALIGNED_K:
                a_tile = tl.load(ptrs_a, eviction_policy="evict_last")
                s_raw = tl.load(ptrs_s, cache_modifier=load_modifier)
                b_raw = tl.load(ptrs_b, cache_modifier=load_modifier)
            else:
                k_off = (ki - split_id * k_iters) * TILE_K
                a_tile = tl.load(
                    ptrs_a,
                    mask=tl.arange(0, TILE_K)[None, :] < (2 * K - split_id * SK_TILE - k_off),
                    other=0.0,
                    eviction_policy="evict_last",
                )
                s_raw = tl.load(ptrs_s, cache_modifier=load_modifier)
                b_raw = tl.load(
                    ptrs_b,
                    mask=shuf_range[None, :] < ((K - (split_id * (SK_TILE // 2) + (ki - split_id * k_iters) * (TILE_K // 2))) * 16),
                    other=0,
                    cache_modifier=load_modifier,
                )

            # On-the-fly A quantization
            inp_q, inp_sc = _quant_block_fp4(a_tile, TILE_M, TILE_K)

            # Reconstruct B scale layout from shuffled storage
            b_sc = (
                s_raw
                .reshape(
                    TILE_N // 32,
                    TILE_K // SCALE_GRP // 8,
                    4, 16, 2, 2, 1,
                )
                .permute(0, 5, 3, 1, 4, 2, 6)
                .reshape(TILE_N, TILE_K // SCALE_GRP)
            )

            # Reconstruct B tile from shuffled storage
            b_tile = (
                b_raw.reshape(1, TILE_N // 16, TILE_K // 64, 2, 16, 16)
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(TILE_N, TILE_K // 2)
                .trans(1, 0)
            )

            acc = tl.dot_scaled(
                inp_q, inp_sc, "e2m1", b_tile, b_sc, "e2m1", acc,
                fast_math=True,
            )

            ptrs_a += TILE_K * stride_ik
            ptrs_b += (TILE_K // 2) * 16 * stride_wk
            ptrs_s += TILE_K * stride_sk

        out_vals = acc.to(out_ptr.type.element_ty)

        row_o = tile_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        col_o = tile_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        ptrs_o = (
            out_ptr
            + stride_om * row_o[:, None]
            + stride_on * col_o[None, :]
            + split_id * stride_ok
        )
        mask_o = (row_o[:, None] < M) & (col_o[None, :] < N)
        tl.store(ptrs_o, out_vals, mask=mask_o, cache_modifier=".wt")


@triton.jit
def _partial_sum_kernel(
    partials_ptr, final_ptr, M, N,
    stride_pk, stride_pm, stride_pn,
    stride_fm, stride_fn,
    RED_M: tl.constexpr, RED_N: tl.constexpr,
    TRUE_SPLITS: tl.constexpr, MAX_SPLITS: tl.constexpr,
):
    """Reduce split-K partial results into final output."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    rows = (pid_m * RED_M + tl.arange(0, RED_M)) % M
    cols = (pid_n * RED_N + tl.arange(0, RED_N)) % N

    base = (
        partials_ptr
        + (rows[:, None] * stride_pm)
        + (cols[None, :] * stride_pn)
    )
    total = tl.load(base).to(tl.float32)
    for s in tl.static_range(1, MAX_SPLITS):
        if s < TRUE_SPLITS:
            total += tl.load(base + s * stride_pk).to(tl.float32)

    out_vals = total.to(final_ptr.type.element_ty)
    out_ptrs = (
        final_ptr
        + (rows[:, None] * stride_fm)
        + (cols[None, :] * stride_fn)
    )
    tl.store(out_ptrs, out_vals)


# ---------------------------------------------------------------------------
# Split-K adjustment for divisibility constraints
# ---------------------------------------------------------------------------
def _adjust_splitk(K, BK, n_splits):
    sk_tile = (
        triton.cdiv((2 * triton.cdiv(K, n_splits)), BK) * BK
    )
    while n_splits > 1 and BK > 16:
        if (
            K % (sk_tile // 2) == 0
            and sk_tile % BK == 0
            and K % (BK // 2) == 0
        ):
            break
        elif K % (sk_tile // 2) != 0 and n_splits > 1:
            n_splits = n_splits // 2
        elif sk_tile % BK != 0:
            if n_splits > 1:
                n_splits = n_splits // 2
            elif BK > 16:
                BK = BK // 2
        elif K % (BK // 2) != 0 and BK > 16:
            BK = BK // 2
        else:
            break
        sk_tile = (
            triton.cdiv((2 * triton.cdiv(K, n_splits)), BK) * BK
        )
    n_splits = triton.cdiv(K, (sk_tile // 2))
    return sk_tile, BK, n_splits


# ---------------------------------------------------------------------------
# Per-shape tuned configurations (MI355X, 256 CUs)
# ---------------------------------------------------------------------------
_TUNED_PARAMS = {
    (4, 2880, 512): {"TILE_M": 4, "TILE_N": 128, "TILE_K": 256, "GROUP_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "load_modifier": None, "N_SPLITS": 1},
    (16, 2112, 7168): {"TILE_M": 16, "TILE_N": 128, "TILE_K": 512, "GROUP_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "load_modifier": ".cg", "N_SPLITS": 14},
    (32, 4096, 512): {"TILE_M": 16, "TILE_N": 32, "TILE_K": 256, "GROUP_M": 1, "num_warps": 4, "num_stages": 3, "waves_per_eu": 3, "matrix_instr_nonkdim": 16, "load_modifier": ".cg", "N_SPLITS": 1},
    (32, 2880, 512): {"TILE_M": 8, "TILE_N": 128, "TILE_K": 256, "GROUP_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "load_modifier": None, "N_SPLITS": 1},
    (64, 7168, 2048): {"TILE_M": 16, "TILE_N": 128, "TILE_K": 256, "GROUP_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "load_modifier": ".cg", "N_SPLITS": 1},
    (256, 3072, 1536): {"TILE_M": 16, "TILE_N": 256, "TILE_K": 512, "GROUP_M": 1, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "load_modifier": None, "N_SPLITS": 1},
}

_FALLBACK_PARAMS = {
    "TILE_M": 16, "TILE_N": 32, "TILE_K": 256, "GROUP_M": 1,
    "num_warps": 2, "num_stages": 2, "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16, "load_modifier": ".cg", "N_SPLITS": 1,
}


# ---------------------------------------------------------------------------
# Caches for buffers, configs, and launch grids
# ---------------------------------------------------------------------------
_out_pool = {}
_param_pool = {}
_grid_pool = {}
_operand_pool = {}


def _get_output_bufs(m, n, n_splits, dev):
    key = (m, n, n_splits)
    if key not in _out_pool:
        final = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        partials = (
            torch.empty((n_splits, m, n), dtype=torch.float32, device=dev)
            if n_splits > 1 else None
        )
        _out_pool[key] = (final, partials)
    return _out_pool[key]


def _resolve_params(m, n, k):
    key = (m, n, k)
    if key not in _param_pool:
        params = _TUNED_PARAMS.get(key, _FALLBACK_PARAMS).copy()
        k_half = k // 2
        if params["N_SPLITS"] > 1:
            sk_tile, tile_k, n_splits = _adjust_splitk(
                k_half, params["TILE_K"], params["N_SPLITS"]
            )
            params["SK_TILE"] = sk_tile
            params["TILE_K"] = tile_k
            params["N_SPLITS"] = n_splits
        else:
            params["SK_TILE"] = 2 * k_half
            params["N_SPLITS"] = 1
        if params["TILE_K"] >= 2 * k_half:
            params["TILE_K"] = triton.next_power_of_2(2 * k_half)
            params["SK_TILE"] = 2 * k_half
            params["N_SPLITS"] = 1
        params["TILE_N"] = max(params["TILE_N"], 32)
        _param_pool[key] = params
    return _param_pool[key]


def _reshape_operands(B_data, B_scales, n, k_half):
    """Reshape pre-shuffled weight and scale tensors for kernel indexing."""
    key = B_data.data_ptr()
    if key not in _operand_pool:
        b_flat = B_data.view(torch.uint8).reshape(n // 16, k_half * 16)
        s_flat = B_scales.view(torch.uint8)
        _operand_pool[key] = (b_flat, s_flat)
    return _operand_pool[key]


def _build_grid(m, n, k, dev):
    """Precompute all launch parameters for a given problem shape."""
    key = (m, n, k)
    if key not in _grid_pool:
        params = _resolve_params(m, n, k)
        k_half = k // 2
        ns = params["N_SPLITS"]

        final, partials = _get_output_bufs(m, n, ns, dev)

        grid = (
            ns
            * triton.cdiv(m, params["TILE_M"])
            * triton.cdiv(n, params["TILE_N"]),
        )

        if ns == 1:
            sk_o, sm_o, sn_o = 0, final.stride(0), final.stride(1)
        else:
            sk_o = partials.stride(0)
            sm_o = partials.stride(1)
            sn_o = partials.stride(2)

        info = {
            'params': params,
            'k_half': k_half,
            'grid': grid,
            'ns': ns,
            'sk_o': sk_o,
            'sm_o': sm_o,
            'sn_o': sn_o,
        }

        if ns > 1:
            info['red_grid'] = (triton.cdiv(m, 16), triton.cdiv(n, 64))
            info['true_splits'] = triton.cdiv(k_half, (params["SK_TILE"] // 2))
            info['max_splits'] = triton.next_power_of_2(ns)

        _grid_pool[key] = info
    return _grid_pool[key]


def _execute_matmul(inp_mat, wt_data, wt_scales, m, n, k):
    """Run fused quantize + GEMM with optional split-K reduction."""
    info = _build_grid(m, n, k, inp_mat.device)
    final, partials = _get_output_bufs(m, n, info['ns'], inp_mat.device)

    b_flat, s_flat = _reshape_operands(wt_data, wt_scales, n, info['k_half'])

    _matmul_fused_kernel[info['grid']](
        inp_mat, b_flat,
        final if info['ns'] == 1 else partials,
        s_flat,
        m, n, info['k_half'],
        inp_mat.stride(0), inp_mat.stride(1),
        b_flat.stride(0), b_flat.stride(1),
        info['sk_o'], info['sm_o'], info['sn_o'],
        s_flat.stride(0), s_flat.stride(1),
        **info['params'],
    )

    if info['ns'] > 1:
        _partial_sum_kernel[info['red_grid']](
            partials, final, m, n,
            partials.stride(0), partials.stride(1), partials.stride(2),
            final.stride(0), final.stride(1),
            16, 64,
            info['true_splits'], info['max_splits'],
        )

    return final


def custom_kernel(data: input_t) -> output_t:
    A = data[0]
    return _execute_matmul(
        A, data[3], data[4], A.shape[0], data[1].shape[0], A.shape[1]
    )
