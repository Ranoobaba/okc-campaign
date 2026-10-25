# okc: problem=muse_moe_w4a16
"""Routed NVFP4 experts (W4A16) as three Triton kernels on top of SGLang's moe_align.

  1. moe_align_block_size (SGLang, CUDA): pairs sorted by local expert, padded to BLOCK_M.
  2. _gemm1: for each (m-block, n-tile): dequantize the gate and up tiles of W13[e] from packed
     E2M1 nibbles x E4M3 block scales (lossless to BF16: a 3-bit magnitude times a 3-bit-mantissa
     scale fits BF16 exactly), tl.dot against the gathered BF16 token rows, SiLU(gate) * up in
     the epilogue -> inter [pairs, I] BF16.
  3. _gemm2: inter x dequant(W2[e]) -> y [pairs, H] BF16.
  4. _finalize: per token, sum over its local slots of g * rmsnorm((y * (1 + w)).bf16()).

The per-expert FP32 global scale is applied to the accumulator (exact: it is one scalar per
expert). Every scored shape is weight-read bound, so the kernels are organised to stream each
touched expert's weights once with many programs in flight.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


POST_NORM_EPS = 1e-8
# 2^126 folded into the scale: an E2M1 nibble c placed at FP32 bits [22:24] (e1 e0 m) with the
# sign at bit 31 reads as (1 + m/2) * 2^(e - 127) for e >= 1 and (m/2) * 2^-126 for e = 0.
TWO_POW_126 = tl.constexpr(float(2.0 ** 126))


@triton.jit
def _dequant_tile(b_ptrs, shift, sf_ptrs, sfx):
    """[BN, BK] BF16 weights from packed nibbles (byte pointers, per-element shift) and scales."""
    b = tl.load(b_ptrs)
    code = (b >> shift) & 0xF
    code = code.to(tl.int32)
    bits = ((code & 8) << 28) | ((code & 7) << 22)
    mag = bits.to(tl.float32, bitcast=True)
    sf = tl.load(sf_ptrs).to(tl.float32) * sfx
    return (mag * sf).to(tl.bfloat16)


@triton.jit
def _gemm1(a_ptr, w_ptr, sf_ptr, gs_ptr, c_ptr, sorted_ids_ptr, expert_ids_ptr, n_post_pad_ptr,
           num_valid, I: tl.constexpr, K: tl.constexpr,
           stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_cm,
           TOP_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_post = tl.load(n_post_pad_ptr)
    if pid_m * BLOCK_M >= n_post:
        return
    e = tl.load(expert_ids_ptr + pid_m)
    if e < 0:
        return
    offs_tok = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    tok_mask = offs_tok < num_valid
    rows = offs_tok // TOP_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rows[:, None] * stride_am + offs_k[None, :]
    e64 = e.to(tl.int64)
    wg = w_ptr + e64 * stride_we + offs_n[:, None] * stride_wn + (offs_k[None, :] // 2)
    wu = wg + I * stride_wn
    shift = (offs_k[None, :] % 2) * 4
    sg = sf_ptr + e64 * stride_se + offs_n[:, None] * stride_sn + (offs_k[None, :] // 16)
    su = sg + I * stride_sn
    sfx = tl.load(gs_ptr + e) * TWO_POW_126
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=tok_mask[:, None], other=0.0)
        g = _dequant_tile(wg, shift, sg, sfx)
        u = _dequant_tile(wu, shift, su, sfx)
        acc_g = tl.dot(a, tl.trans(g), acc_g)
        acc_u = tl.dot(a, tl.trans(u), acc_u)
        a_ptrs += BLOCK_K
        wg += BLOCK_K // 2
        wu += BLOCK_K // 2
        sg += BLOCK_K // 16
        su += BLOCK_K // 16
    h = acc_g * tl.sigmoid(acc_g) * acc_u
    c_ptrs = c_ptr + offs_tok[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, h.to(tl.bfloat16), mask=tok_mask[:, None])


@triton.jit
def _gemm2(a_ptr, w_ptr, sf_ptr, gs_ptr, c_ptr, sorted_ids_ptr, expert_ids_ptr, n_post_pad_ptr,
           num_valid, K: tl.constexpr,
           stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_cm,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_post = tl.load(n_post_pad_ptr)
    if pid_m * BLOCK_M >= n_post:
        return
    e = tl.load(expert_ids_ptr + pid_m)
    if e < 0:
        return
    offs_tok = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    tok_mask = offs_tok < num_valid
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_tok[:, None] * stride_am + offs_k[None, :]
    e64 = e.to(tl.int64)
    w = w_ptr + e64 * stride_we + offs_n[:, None] * stride_wn + (offs_k[None, :] // 2)
    shift = (offs_k[None, :] % 2) * 4
    s = sf_ptr + e64 * stride_se + offs_n[:, None] * stride_sn + (offs_k[None, :] // 16)
    sfx = tl.load(gs_ptr + e) * TWO_POW_126
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=tok_mask[:, None], other=0.0)
        b = _dequant_tile(w, shift, s, sfx)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        w += BLOCK_K // 2
        s += BLOCK_K // 16
    c_ptrs = c_ptr + offs_tok[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=tok_mask[:, None])


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


def prepare(weights: Weights):
    st = {k: weights[k].contiguous() for k in ("w13", "w13_sf", "w2", "w2_sf", "post_norm_w")}
    st["w13_gs"] = weights["w13_gs"].float().contiguous()
    st["w2_gs"] = weights["w2_gs"].float().contiguous()
    st["maps"] = {}
    torch.cuda.synchronize()
    return st


def _expert_map(st, start, count, n_experts, device):
    key = (start, count)
    m = st["maps"].get(key)
    if m is None:
        m = torch.full((n_experts,), -1, dtype=torch.int32, device=device)
        m[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        st["maps"][key] = m
    return m


def _block_m(pairs_per_expert):
    if pairs_per_expert <= 16:
        return 16
    if pairs_per_expert <= 48:
        return 32
    return 64


def kernel_fn(data: Inputs, st) -> torch.Tensor:
    cfg = data["config"]
    x = data["hidden_states"]
    M, H = x.shape
    I = cfg["d_expert"]
    top_k = cfg["top_k"]
    count = cfg["local_count"]
    dev = x.device
    emap = _expert_map(st, cfg["local_start"], count, cfg["n_experts"], dev)
    # Local expert id per (token, slot): -1 where the expert lives on another GPU.
    local_ids = emap[data["topk_ids"]]
    g = data["topk_weights"]
    if g.dtype != torch.float32:
        g = g.float()
    g = g.contiguous()
    pairs = M * top_k
    block_m = _block_m(pairs / count)
    sorted_ids, expert_ids, n_post = moe_align_block_size(local_ids, block_m, count)
    max_blocks = expert_ids.numel()
    inter = torch.empty((pairs, I), dtype=torch.bfloat16, device=dev)
    w13, w13_sf, w2, w2_sf = st["w13"], st["w13_sf"], st["w2"], st["w2_sf"]
    # Global expert row offsets: the weights are indexed by global id = local_start + local id.
    e_off = cfg["local_start"]
    BN1, BK1 = 64, 128
    _gemm1[(max_blocks, I // BN1)](
        x, w13[e_off:], w13_sf[e_off:], st["w13_gs"][e_off:], inter, sorted_ids, expert_ids, n_post,
        pairs, I, H, x.stride(0), w13.stride(0), w13.stride(1), w13_sf.stride(0), w13_sf.stride(1), inter.stride(0),
        TOP_K=top_k, BLOCK_M=block_m, BLOCK_N=BN1, BLOCK_K=BK1, num_warps=4, num_stages=3)
    y = torch.empty((pairs, H), dtype=torch.bfloat16, device=dev)
    BN2, BK2 = 64, 128
    _gemm2[(max_blocks, H // BN2)](
        inter, w2[e_off:], w2_sf[e_off:], st["w2_gs"][e_off:], y, sorted_ids, expert_ids, n_post,
        pairs, I, inter.stride(0), w2.stride(0), w2.stride(1), w2_sf.stride(0), w2_sf.stride(1), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=BN2, BLOCK_K=BK2, num_warps=4, num_stages=3)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    _finalize[(M,)](y, local_ids, g, st["post_norm_w"], out, POST_NORM_EPS, H=H, TOP_K=top_k,
                    BLOCK_H=triton.next_power_of_2(H), num_warps=8)
    return out
