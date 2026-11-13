#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

import gc
import os
import sys
import subprocess
import shutil
import re
import functools

sys.setswitchinterval(1.0)

os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["AITER_USE_NT"] = "1"
os.environ["HSA_ENABLE_INTERRUPT"] = "0"

_AITER_DIR = '/home/runner/aiter'
_NEW_DIR = '/tmp/aiter_fresh'
_FLYDSL_DIR = '/tmp/flydsl_new'

# ========== Install flydsl 0.1.1 ==========
os.makedirs(_FLYDSL_DIR, exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'flydsl==0.1.1',
                '--target', _FLYDSL_DIR, '--upgrade', '--break-system-packages'],
               capture_output=True, text=True, timeout=120)
sys.path.insert(0, _FLYDSL_DIR)
for key in list(sys.modules.keys()):
    if 'flydsl' in key:
        del sys.modules[key]

# ========== Clone AITER + pin to old API ==========
subprocess.run(['git', 'checkout', '--', '.'], capture_output=True, cwd=_AITER_DIR, timeout=10)

if not os.path.exists(os.path.join(_NEW_DIR, 'aiter')):
    subprocess.run(['git', 'clone', '--depth=50', 'https://github.com/ROCm/aiter.git', _NEW_DIR],
                   capture_output=True, text=True, timeout=120)

_mk_path = os.path.join(_NEW_DIR, 'aiter', 'ops', 'flydsl', 'moe_kernels.py')
if os.path.exists(_mk_path):
    with open(_mk_path) as f:
        if '_get_compiled_stage1' not in f.read():
            r = subprocess.run(['git', 'log', '--format=%H', '-50'],
                               capture_output=True, text=True, cwd=_NEW_DIR, timeout=10)
            for commit in r.stdout.strip().split('\n')[1:]:
                if not commit.strip(): continue
                subprocess.run(['git', 'checkout', commit.strip()], capture_output=True, cwd=_NEW_DIR, timeout=10)
                with open(_mk_path) as f2:
                    if '_get_compiled_stage1' in f2.read():
                        print(f"[moe] Pinned to {commit.strip()[:12]}", file=sys.stderr)
                        break

# Copy NEWER fused_moe.py (fixes stage1 calling convention)
shutil.copy2(os.path.join(_NEW_DIR, 'aiter', 'fused_moe.py'),
             os.path.join(_AITER_DIR, 'aiter', 'fused_moe.py'))

# Copy FlyDSL kernel files
flydsl_src = os.path.join(_NEW_DIR, 'aiter', 'ops', 'flydsl')
flydsl_dst = os.path.join(_AITER_DIR, 'aiter', 'ops', 'flydsl')
for fname in ['moe_kernels.py', 'utils.py']:
    shutil.copy2(os.path.join(flydsl_src, fname), os.path.join(flydsl_dst, fname))
init_path = os.path.join(flydsl_dst, '__init__.py')
with open(init_path) as f:
    init_src = f.read()
init_src = re.sub(r'raise ImportError\([^)]*\)', 'pass  # version check bypassed', init_src)
with open(init_path, 'w') as f:
    f.write(init_src)
kernels_src = os.path.join(flydsl_src, 'kernels')
kernels_dst = os.path.join(flydsl_dst, 'kernels')
if os.path.exists(kernels_src):
    os.makedirs(kernels_dst, exist_ok=True)
    for fname in os.listdir(kernels_src):
        if fname.endswith('.py'):
            shutil.copy2(os.path.join(kernels_src, fname), os.path.join(kernels_dst, fname))

# ========== Source-patch fused_moe.py ==========
_FMOE_PATH = os.path.join(_AITER_DIR, 'aiter', 'fused_moe.py')
with open(_FMOE_PATH, 'r') as f:
    src = f.read()

# Append: FlyDSL stage1 for E≤100 ONLY, stage2 FlyDSL for all
tail = r'''

# FlyDSL stage1 E<=100 only, CK stage1 for E=257
def _fly_s1(hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out, topk, w1_scale=None, a1_scale=None, sorted_weights=None, **_kw):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
    flydsl_moe_stage1(a=hidden_states, w1=w1, sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids, out=out, topk=topk,
        tile_m=32, tile_n=256, tile_k=128, a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        act="silu", w1_scale=w1_scale, a1_scale=a1_scale, sorted_weights=sorted_weights)
    return out

def _fly_s2_atomic(inter_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out, topk, w2_scale=None, a2_scale=None, sorted_weights=None, **_kw):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    flydsl_moe_stage2(inter_states=inter_states, w2=w2, sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids, out=out, topk=topk,
        tile_m=16, tile_n=256, tile_k=128, a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        mode="atomic", w2_scale=w2_scale, a2_scale=a2_scale, sorted_weights=sorted_weights)

_fly_orig_g2sc = get_2stage_cfgs
try: _fly_orig_g2sc.cache_clear()
except: pass
_fly_ready = [False]
_fly_s1_ready = [False]

import functools as _ft

@_ft.lru_cache(maxsize=2048)
def _fly_g2sc(*a, **kw):
    md = _fly_orig_g2sc(*a, **kw)
    E_val = a[4] if len(a) >= 5 else 0
    # Stage1: FlyDSL ONLY for E<=100, CK for E=257
    if _fly_s1_ready[0] and E_val <= 100 and not md.run_1stage and md.stage1 is not None:
        md.stage1 = _ft.partial(_fly_s1)
    # Stage2: FlyDSL for ALL shapes
    if _fly_ready[0] and not md.run_1stage and md.stage2 is not None:
        md.stage2 = _ft.partial(_fly_s2_atomic)
    return md
get_2stage_cfgs = _fly_g2sc

import sys as _s
print("[moe] FlyDSL stage1 E<=100, stage2 all — source-patched", file=_s.stderr)
'''
src += tail
with open(_FMOE_PATH, 'w') as f:
    f.write(src)

print("[moe] Source patches applied", file=sys.stderr)

# ========== Fix Triton constexpr ==========
try:
    import triton.language.core as _tlc
    _ce = _tlc.constexpr
    try:
        _ce(1).__lt__(_ce(2), _semantic=None)
    except TypeError:
        for _mn in ['__lt__', '__le__', '__gt__', '__ge__', '__eq__', '__ne__',
                     '__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
                     '__truediv__', '__floordiv__', '__mod__', '__pow__',
                     '__lshift__', '__rshift__', '__and__', '__or__', '__xor__']:
            _ofn = getattr(_ce, _mn, None)
            if _ofn is not None and not getattr(_ofn, '_patched', False):
                def _mkw(fn):
                    def _w(self, *a, _semantic=None, **kw):
                        return fn(self, *a, **kw)
                    _w._patched = True
                    return _w
                setattr(_ce, _mn, _mkw(_ofn))
    del _ce, _tlc
except Exception:
    pass

# ========== Import AITER (with newer fused_moe.py, source-patched) ==========
import torch
torch.set_grad_enabled(False)
from task import input_t, output_t

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as _fmoe
from aiter.ops.quant import per_1x32_f4_quant_hip as _quant_hip
from aiter.utility import fp4_utils as _fp4_utils

# ========== Pre-compile FlyDSL ==========
try:
    from aiter.ops.flydsl.moe_kernels import _get_compiled_stage1, _get_compiled_stage2

    # Stage1 ONLY for E=33
    for inter, exp in [(512, 33), (2048, 33)]:
        try:
            _get_compiled_stage1(model_dim=7168, inter_dim=inter, experts=exp, topk=9,
                                tile_m=32, tile_n=256, tile_k=128, doweight=True,
                                a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", act="silu")
            print(f"[moe] Stage1 compiled: d={inter} E={exp}", file=sys.stderr)
        except Exception as e:
            print(f"[moe] Stage1 FAILED: d={inter} E={exp}: {str(e)[:80]}", file=sys.stderr)

    _fmoe._fly_s1_ready[0] = True

    # Stage2 for all
    for inter, exp in [(256, 257), (512, 33), (2048, 33)]:
        try:
            _get_compiled_stage2(model_dim=7168, inter_dim=inter, experts=exp, topk=9,
                                tile_m=16, tile_n=256, tile_k=128, doweight=True,
                                a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", accumulate=True)
        except: pass

    _fmoe._fly_ready[0] = True
    print("[moe] FlyDSL pre-compiled", file=sys.stderr)
except Exception as e:
    print(f"[moe] Pre-compile failed: {e}", file=sys.stderr)

# Warmup
_warmed = set()
def _warmup(data):
    (hs, guw, dw, guws, dws, guw_sh, dw_sh, guws_sh, dws_sh, tw, ti, cfg) = data
    key = (cfg["bs"], cfg["n_routed_experts"], cfg["d_expert"])
    if key in _warmed: return
    _warmed.add(key)
    try:
        _ = fused_moe(hs[:2], guw_sh, dw_sh, tw[:2], ti[:2],
                      activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                      w1_scale=guws_sh, w2_scale=dws_sh,
                      hidden_pad=cfg["d_hidden_pad"]-cfg["d_hidden"],
                      intermediate_pad=cfg["d_expert_pad"]-cfg["d_expert"])
        torch.cuda.synchronize()
    except: pass

gc.disable()
print("[moe] Ready", file=sys.stderr)

def custom_kernel(data: input_t) -> output_t:
    _warmup(data)
    (hs, guw, dw, guws, dws, guw_sh, dw_sh, guws_sh, dws_sh, tw, ti, cfg) = data
    return fused_moe(hs, guw_sh, dw_sh, tw, ti,
                     activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                     w1_scale=guws_sh, w2_scale=dws_sh,
                     hidden_pad=cfg["d_hidden_pad"]-cfg["d_hidden"],
                     intermediate_pad=cfg["d_expert_pad"]-cfg["d_expert"])

