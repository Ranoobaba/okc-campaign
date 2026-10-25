# okc: problem=muse_attention_decode
"""Gated GQA decode over a paged FP8 cache in three Triton launches (SGLang uses five).

  1. _prep: per new token: per-head RMSNorm of q (times qk_scale_factor) and k, interleaved RoPE
     on sliding layers, FP8 store of the new k and v at the token's cache row, normalized q to
     a BF16 scratch.
  2. _attn: split-KV flash decoding. One program per (request, kv head, key chunk): the group's
     8 query heads x L block tokens are the rows of one tl.dot against gathered FP8 key tiles,
     online softmax, partial (m, l, acc) out. Every key in the window / context is attended.
  3. _combine: merge the chunks, per-head RMSNorm, times sigmoid(gate), BF16 out.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
NEG_INF = float("-inf")


@triton.jit
def _rope_pair(xe, xo, pos, theta, HALF: tl.constexpr):
    """Interleaved RoPE on even/odd halves [heads, HALF] at integer position pos."""
    i = tl.arange(0, HALF).to(tl.float32)
    inv_freq = 1.0 / tl.exp2(i * (2.0 / (2 * HALF)) * tl.log2(theta))
    ang = pos.to(tl.float32) * inv_freq
    c = tl.cos(ang)[None, :]
    s = tl.sin(ang)[None, :]
    return xe * c - xo * s, xe * s + xo * c


@triton.jit
def _prep(qkv_ptr, pos_ptr, loc_ptr, kc_ptr, vc_ptr, q_out_ptr, k_scale_ptr, v_scale_ptr,
          qk_scale, theta, eps,
          HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, ROPE: tl.constexpr,
          SCALED: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    row = qkv_ptr + t * ((2 * HQ + 2 * HKV) * D)
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    de = tl.arange(0, HALF) * 2
    pos = tl.load(pos_ptr + t)
    # q: [HQ, D] as even / odd halves.
    qe = tl.load(row + hq[:, None] * D + de[None, :]).to(tl.float32)
    qo = tl.load(row + hq[:, None] * D + de[None, :] + 1).to(tl.float32)
    ms = (tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D
    r = tl.rsqrt(ms + eps)[:, None] * qk_scale
    qe = qe * r
    qo = qo * r
    if ROPE:
        qe, qo = _rope_pair(qe, qo, pos, theta, HALF)
    qbase = q_out_ptr + t * (HQ * D)
    tl.store(qbase + hq[:, None] * D + de[None, :], qe.to(tl.bfloat16))
    tl.store(qbase + hq[:, None] * D + de[None, :] + 1, qo.to(tl.bfloat16))
    # k: [HKV, D], then v.
    krow = row + HQ * D
    ke = tl.load(krow + hk[:, None] * D + de[None, :]).to(tl.float32)
    ko = tl.load(krow + hk[:, None] * D + de[None, :] + 1).to(tl.float32)
    ms = (tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / D
    r = tl.rsqrt(ms + eps)[:, None]
    ke = ke * r
    ko = ko * r
    if ROPE:
        ke, ko = _rope_pair(ke, ko, pos, theta, HALF)
    vrow = krow + HKV * D
    ve = tl.load(vrow + hk[:, None] * D + de[None, :]).to(tl.float32)
    vo = tl.load(vrow + hk[:, None] * D + de[None, :] + 1).to(tl.float32)
    if SCALED:
        ks = tl.load(k_scale_ptr)
        vs = tl.load(v_scale_ptr)
        ke = ke / ks
        ko = ko / ks
        ve = ve / vs
        vo = vo / vs
    loc = tl.load(loc_ptr + t).to(tl.int64)
    kdst = kc_ptr + loc * (HKV * D) + hk[:, None] * D + de[None, :]
    vdst = vc_ptr + loc * (HKV * D) + hk[:, None] * D + de[None, :]
    tl.store(kdst, ke.to(kc_ptr.dtype.element_ty))
    tl.store(kdst + 1, ko.to(kc_ptr.dtype.element_ty))
    tl.store(vdst, ve.to(vc_ptr.dtype.element_ty))
    tl.store(vdst + 1, vo.to(vc_ptr.dtype.element_ty))


@triton.jit
def _attn(q_ptr, kc_ptr, vc_ptr, table_ptr, seqlens_ptr, k_scale_ptr, v_scale_ptr,
          po_ptr, pm_ptr, pl_ptr, sm_scale, chunk,
          L: tl.constexpr, GROUP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
          PAGE: tl.constexpr, MAX_PAGES, WINDOW: tl.constexpr, NSPLIT: tl.constexpr,
          BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, SCALED: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    seqlen = tl.load(seqlens_ptr + b)  # cached + L new tokens
    first_q = seqlen - L
    if WINDOW > 0:
        base = tl.maximum(first_q - WINDOW + 1, 0)
    else:
        base = 0
    start = base + s * chunk
    end = tl.minimum(start + chunk, seqlen)
    r = tl.arange(0, BLOCK_Q)
    d = tl.arange(0, D)
    j = r // GROUP
    g = r % GROUP
    rmask = r < L * GROUP
    qpos = first_q + j
    pbase = ((b * HKV + h) * NSPLIT + s).to(tl.int64) * BLOCK_Q
    if start >= end:
        tl.store(pm_ptr + pbase + r, tl.full((BLOCK_Q,), NEG_INF, tl.float32))
        tl.store(pl_ptr + pbase + r, tl.zeros((BLOCK_Q,), tl.float32))
        tl.store(po_ptr + (pbase + r)[:, None] * D + d[None, :], tl.zeros((BLOCK_Q, D), tl.float32))
        return
    qrow = ((b * L + j) * HQ + h * GROUP + g).to(tl.int64)
    q = tl.load(q_ptr + qrow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
    if SCALED:
        ks = tl.load(k_scale_ptr)
        vs = tl.load(v_scale_ptr)
    m_i = tl.full((BLOCK_Q,), NEG_INF, tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, D), tl.float32)
    tbl = table_ptr + b.to(tl.int64) * MAX_PAGES
    for n0 in range(start, end, BLOCK_N):
        pos = n0 + tl.arange(0, BLOCK_N)
        kmask = pos < end
        page = tl.load(tbl + pos // PAGE, mask=kmask, other=0).to(tl.int64)
        rows = page * PAGE + (pos % PAGE)
        kptr = kc_ptr + rows[:, None] * (HKV * D) + h * D + d[None, :]
        k = tl.load(kptr, mask=kmask[:, None], other=0.0).to(tl.float32)
        if SCALED:
            k = k * ks
        sc = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
        vis = kmask[None, :] & (pos[None, :] <= qpos[:, None])
        if WINDOW > 0:
            vis = vis & (pos[None, :] > qpos[:, None] - WINDOW)
        sc = tl.where(vis, sc, NEG_INF)
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vptr = vc_ptr + rows[:, None] * (HKV * D) + h * D + d[None, :]
        v = tl.load(vptr, mask=kmask[:, None], other=0.0).to(tl.float32)
        if SCALED:
            v = v * vs
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    tl.store(pm_ptr + pbase + r, m_i)
    tl.store(pl_ptr + pbase + r, l_i)
    tl.store(po_ptr + (pbase + r)[:, None] * D + d[None, :], acc)


@triton.jit
def _combine(po_ptr, pm_ptr, pl_ptr, qkv_ptr, out_ptr, eps,
             L: tl.constexpr, GROUP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
             NSPLIT: tl.constexpr, BLOCK_Q: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.arange(0, BLOCK_Q)
    d = tl.arange(0, D)
    rmask = r < L * GROUP
    pbase = ((b * HKV + h) * NSPLIT).to(tl.int64) * BLOCK_Q
    m = tl.full((BLOCK_Q,), NEG_INF, tl.float32)
    for s in tl.static_range(NSPLIT):
        m = tl.maximum(m, tl.load(pm_ptr + pbase + s * BLOCK_Q + r))
    m_safe = tl.where(m == NEG_INF, 0.0, m)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    o = tl.zeros((BLOCK_Q, D), tl.float32)
    for s in tl.static_range(NSPLIT):
        ms = tl.load(pm_ptr + pbase + s * BLOCK_Q + r)
        w = tl.exp(ms - m_safe)
        l += w * tl.load(pl_ptr + pbase + s * BLOCK_Q + r)
        o += w[:, None] * tl.load(po_ptr + (pbase + s * BLOCK_Q + r)[:, None] * D + d[None, :])
    o = o / tl.maximum(l, 1e-30)[:, None]
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + eps)[:, None]
    j = r // GROUP
    g = r % GROUP
    tok = (b * L + j).to(tl.int64)
    head = h * GROUP + g
    gate = tl.load(qkv_ptr + tok[:, None] * ((2 * HQ + 2 * HKV) * D) + (HQ + 2 * HKV) * D + head[:, None] * D + d[None, :],
                   mask=rmask[:, None], other=0.0).to(tl.float32)
    o = o * tl.sigmoid(gate)
    tl.store(out_ptr + tok[:, None] * (HQ * D) + head[:, None] * D + d[None, :], o.to(tl.bfloat16), mask=rmask[:, None])


def _plan(batch, hkv, max_len, window, L, block_n):
    """(chunk, nsplit): key chunk per program so that about 300 programs are in flight."""
    span = min(max_len, window + L) if window else max_len
    target = max(1, -(-300 // (batch * hkv)))
    chunk = -(-span // target)
    chunk = max(block_n, -(-chunk // block_n) * block_n)
    nsplit = -(-span // chunk)
    return chunk, nsplit


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    HQ, HKV, D = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page = cfg["page_size"]
    window = cfg["window"]
    theta = cfg["rope_theta"]
    scaled = cfg["kv_scale"] != 1.0
    qkv = data["qkv"]
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    T = B * L
    dev = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    table = data["page_table"]
    q_scratch = torch.empty((T, HQ, D), dtype=torch.bfloat16, device=dev)
    _prep[(T,)](qkv, data["positions"], data["cache_loc"], k_cache, v_cache, q_scratch,
                data["k_scale"], data["v_scale"], cfg["qk_scale_factor"], theta if theta else 1.0, RMS_EPS,
                HQ=HQ, HKV=HKV, D=D, HALF=D // 2, ROPE=bool(theta), SCALED=scaled, num_warps=4)
    group = HQ // HKV
    block_q = max(16, triton.next_power_of_2(L * group))
    block_n = 128 if block_q <= 16 else 64
    max_len = table.shape[1] * page
    chunk, nsplit = _plan(B, HKV, max_len, window, L, block_n)
    po = torch.empty((B * HKV * nsplit * block_q, D), dtype=torch.float32, device=dev)
    pm = torch.empty((B * HKV * nsplit * block_q,), dtype=torch.float32, device=dev)
    pl = torch.empty_like(pm)
    _attn[(B, HKV, nsplit)](q_scratch, k_cache, v_cache, table, data["cache_seqlens"], data["k_scale"], data["v_scale"],
                            po, pm, pl, cfg["sm_scale"], chunk,
                            L=L, GROUP=group, HQ=HQ, HKV=HKV, D=D, PAGE=page, MAX_PAGES=table.stride(0), WINDOW=window,
                            NSPLIT=nsplit, BLOCK_Q=block_q, BLOCK_N=block_n, SCALED=scaled, num_warps=4, num_stages=2)
    out = torch.empty((B, L, HQ * D), dtype=torch.bfloat16, device=dev)
    _combine[(B, HKV)](po, pm, pl, qkv, out, RMS_EPS, L=L, GROUP=group, HQ=HQ, HKV=HKV, D=D, NSPLIT=nsplit,
                       BLOCK_Q=block_q, num_warps=4)
    return out
