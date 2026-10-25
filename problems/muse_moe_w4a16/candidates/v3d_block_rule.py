# okc: problem=muse_moe_w4a16
"""Routed NVFP4 experts: SGLang's Marlin W4A16 GEMMs for the two expert matmuls, with the
post-expert norm + routing-weighted sum fused into one Triton kernel (SGLang runs that step as
about 15 small PyTorch kernels below 32 tokens), and expert alignment over the 32 local experts
only. A stepping stone while the custom dequant GEMM is tuned."""

import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


@triton.jit
def _finalize(y_ptr, ids_ptr, g_ptr, w_ptr, out_ptr, eps, H: tl.constexpr, TOP_K: tl.constexpr,
              BLOCK_H: tl.constexpr):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        pair = tok * TOP_K + k
        if tl.load(ids_ptr + pair) >= 0:
            y = tl.load(y_ptr + pair.to(tl.int64) * H + offs, mask=mask, other=0.0).to(tl.float32)
            y = (y * w).to(tl.bfloat16).to(tl.float32)
            var = tl.sum(y * y, axis=0) / H
            acc += y * tl.rsqrt(var + eps) * tl.load(g_ptr + pair)
    tl.store(out_ptr + tok.to(tl.int64) * H + offs, acc.to(tl.bfloat16), mask=mask)



def _marlin_layer(weights, start, count):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
        expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {"banks": banks, "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
            "post_norm_w": weights["post_norm_w"].contiguous()}


def _experts_marlin(x, local_ids, topk_weights, bank, ws, top_k):
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    count = bank["count"]
    # Measured on H200: Marlin's 16-row M tiles beat its 32/48/64 choices at every batch above 8 (bs 128: 1.25x,
    # bs 256: 1.40x over SGLang's heuristic); 8 stays best at bs <= 8.
    block_size_m = 8 if num_tokens <= 8 else (16 if num_tokens <= 512 else 64)  # 16-row tiles lose above ~512 tokens (prefill)
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, count)
    cache2 = torch.empty((num_tokens * top_k, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * top_k * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * top_k * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * top_k * size_k].view(-1, size_k)
    tiny = num_tokens <= 8  # measured: atomic adds beat the fp32 reduce pass at bs 1 to 8
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=tiny,
                  use_fp32_reduce=not tiny, is_zp_float=False)
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=top_k,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * top_k, size_n=size_k, size_k=size_n, **common)
    return y.view(num_tokens * top_k, size_k)


def _moe_core(x, local_ids, g, st, top_k, count, I, e_off=0):
    """Layer entry point: x [M, H] bf16, local_ids [M, top_k] int32 (-1 = elsewhere), g [M, top_k] fp32."""
    bank = st["banks"][(e_off, count)]
    M, H = x.shape
    y = _experts_marlin(x, local_ids.contiguous(), g, bank, st["workspace"], top_k)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    _finalize[(M,)](y, local_ids, g, st["post_norm_w"], out, POST_NORM_EPS, H=H, TOP_K=top_k,
                    BLOCK_H=triton.next_power_of_2(H), num_warps=8)
    return out


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    M, H = x.shape
    local_ids = bank["expert_map"][data["topk_ids"].to(torch.int32)].contiguous()
    g = data["topk_weights"].to(torch.float32).contiguous()
    y = _experts_marlin(x, local_ids, g, bank, state["workspace"], cfg["top_k"])
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    _finalize[(M,)](y, local_ids, g, state["post_norm_w"], out, POST_NORM_EPS, H=H, TOP_K=cfg["top_k"],
                    BLOCK_H=triton.next_power_of_2(H), num_warps=8)
    return out
