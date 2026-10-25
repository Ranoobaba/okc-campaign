# okc: problem=muse_attention_decode
"""Gated GQA decode over a paged FP8 cache in three Triton launches (SGLang uses five), with SGLang's
FA3 path kept for 8-token verify blocks on full-attention layers, where our split-KV kernel is still slower.

  1. _prep: per new token: per-head RMSNorm of q (times qk_scale_factor) and k, interleaved RoPE
     on sliding layers, FP8 store of the new k and v at the token's cache row, normalized q to
     a BF16 scratch.
  2. _attn: split-KV flash decoding. One program per (request, kv head, key chunk): the group's
     8 query heads x L block tokens are the rows of one FP8 x FP8 tl.dot against gathered FP8 key
     tiles (queries rounded to E4M3 as FA3 does, qk_scale and k_scale folded into the softmax
     scale), online softmax, P (BF16) x V (FP8 -> BF16), partial (m, l, acc) out. Every key in the
     window / context is attended.
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
NEG_INF = tl.constexpr(float("-inf"))


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
    tl.store(qbase + hq[:, None] * D + de[None, :], qe.to(q_out_ptr.dtype.element_ty))
    tl.store(qbase + hq[:, None] * D + de[None, :] + 1, qo.to(q_out_ptr.dtype.element_ty))
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
    qk_scale = sm_scale
    if SCALED:
        qk_scale = sm_scale * tl.load(k_scale_ptr)
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
        k = tl.load(kptr, mask=kmask[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * qk_scale
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
        v = tl.load(vptr, mask=kmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    if SCALED:
        acc = acc * tl.load(v_scale_ptr)
    tl.store(pm_ptr + pbase + r, m_i)
    tl.store(pl_ptr + pbase + r, l_i)
    tl.store(po_ptr + (pbase + r)[:, None] * D + d[None, :], acc)


@triton.jit
def _attn_tok(q_ptr, kc_ptr, vc_ptr, table_ptr, seqlens_ptr, k_scale_ptr, v_scale_ptr,
          po_ptr, pm_ptr, pl_ptr, sm_scale, chunk,
          L: tl.constexpr, GROUP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
          PAGE: tl.constexpr, MAX_PAGES, WINDOW: tl.constexpr, NSPLIT: tl.constexpr,
          BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, SCALED: tl.constexpr):
    t = tl.program_id(0)  # token = b * L + j
    h = tl.program_id(1)
    s = tl.program_id(2)
    b = t // L
    j = t % L
    seqlen = tl.load(seqlens_ptr + b)  # cached + L new tokens
    first_q = seqlen - L
    qpos_t = first_q + j
    if WINDOW > 0:
        base = tl.maximum(qpos_t - WINDOW + 1, 0)
    else:
        base = 0
    start = base + s * chunk
    end = tl.minimum(start + chunk, qpos_t + 1)
    r = tl.arange(0, BLOCK_Q)
    d = tl.arange(0, D)
    g = r
    rmask = r < GROUP
    qpos = tl.zeros((BLOCK_Q,), tl.int32) + qpos_t
    pbase = ((t * HKV + h) * NSPLIT + s).to(tl.int64) * BLOCK_Q
    if start >= end:
        tl.store(pm_ptr + pbase + r, tl.full((BLOCK_Q,), NEG_INF, tl.float32))
        tl.store(pl_ptr + pbase + r, tl.zeros((BLOCK_Q,), tl.float32))
        tl.store(po_ptr + (pbase + r)[:, None] * D + d[None, :], tl.zeros((BLOCK_Q, D), tl.float32))
        return
    qrow = (t * HQ + h * GROUP + g).to(tl.int64)
    q = tl.load(q_ptr + qrow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
    qk_scale = sm_scale
    if SCALED:
        qk_scale = sm_scale * tl.load(k_scale_ptr)
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
        k = tl.load(kptr, mask=kmask[:, None], other=0.0)
        sc = tl.dot(q, tl.trans(k)) * qk_scale
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
        v = tl.load(vptr, mask=kmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    if SCALED:
        acc = acc * tl.load(v_scale_ptr)
    tl.store(pm_ptr + pbase + r, m_i)
    tl.store(pl_ptr + pbase + r, l_i)
    tl.store(po_ptr + (pbase + r)[:, None] * D + d[None, :], acc)


@triton.jit
def _combine_tok(po_ptr, pm_ptr, pl_ptr, qkv_ptr, out_ptr, eps,
             L: tl.constexpr, GROUP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
             NSPLIT: tl.constexpr, BLOCK_Q: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.arange(0, BLOCK_Q)
    d = tl.arange(0, D)
    rmask = r < GROUP
    pbase = ((t * HKV + h) * NSPLIT).to(tl.int64) * BLOCK_Q
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
    g = r
    tok = tl.zeros((BLOCK_Q,), tl.int64) + t.to(tl.int64)
    head = h * GROUP + g
    gate = tl.load(qkv_ptr + tok[:, None] * ((2 * HQ + 2 * HKV) * D) + (HQ + 2 * HKV) * D + head[:, None] * D + d[None, :],
                   mask=rmask[:, None], other=0.0).to(tl.float32)
    o = o * tl.sigmoid(gate)
    tl.store(out_ptr + tok[:, None] * (HQ * D) + head[:, None] * D + d[None, :], o.to(tl.bfloat16), mask=rmask[:, None])


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


# ---- SGLang's FA3 path, used for the shapes where it is still faster than our kernel (8-token
# verify blocks on full-attention layers) ----
from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: E402
from sglang.kernels.ops.kvcache.kvcache import store_cache  # noqa: E402
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm  # noqa: E402
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate  # noqa: E402

FP8 = torch.float8_e4m3fn
_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def _fa3_path(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8), k_cache=k_cache, v_cache=v_cache, page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)


def _window_tok_path(data):
    """Sliding layers, verify blocks: one 16-row problem per (token, kv head, chunk)."""
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
    q_scratch = torch.empty((T, HQ, D), dtype=torch.float8_e4m3fn, device=dev)
    _prep[(T,)](qkv, data["positions"], data["cache_loc"], k_cache, v_cache, q_scratch,
                data["k_scale"], data["v_scale"], 1.0, theta if theta else 1.0, RMS_EPS,
                HQ=HQ, HKV=HKV, D=D, HALF=D // 2, ROPE=bool(theta), SCALED=scaled, num_warps=4)
    group = HQ // HKV
    block_q = max(16, triton.next_power_of_2(group))
    block_n = 128
    max_len = table.shape[1] * page
    chunk, nsplit = _plan(T, HKV, max_len, window, 1, block_n)
    po = torch.empty((T * HKV * nsplit * block_q, D), dtype=torch.float32, device=dev)
    pm = torch.empty((T * HKV * nsplit * block_q,), dtype=torch.float32, device=dev)
    pl = torch.empty_like(pm)
    _attn_tok[(T, HKV, nsplit)](q_scratch, k_cache, v_cache, table, data["cache_seqlens"], data["k_scale"], data["v_scale"],
                                po, pm, pl, cfg["sm_scale"] * cfg["qk_scale_factor"], chunk,
                                L=L, GROUP=group, HQ=HQ, HKV=HKV, D=D, PAGE=page, MAX_PAGES=table.stride(0), WINDOW=window,
                                NSPLIT=nsplit, BLOCK_Q=block_q, BLOCK_N=block_n, SCALED=scaled, num_warps=4, num_stages=2)
    out = torch.empty((B, L, HQ * D), dtype=torch.bfloat16, device=dev)
    _combine_tok[(T, HKV)](po, pm, pl, qkv, out, RMS_EPS, L=L, GROUP=group, HQ=HQ, HKV=HKV, D=D, NSPLIT=nsplit,
                           BLOCK_Q=block_q, num_warps=4)
    return out


def _plan(batch, hkv, max_len, window, L, block_n):
    """(chunk, nsplit): key chunk per program so that about 300 programs are in flight."""
    span = min(max_len, window + L) if window else max_len
    # Full-context layers at batch >= 8: 4x more programs hide the per-tile gather latency (measured +10 to 17%);
    # at small batch the merge over hundreds of splits costs more than it saves.
    programs = 2400 if (not window and batch >= 8) else 300
    target = min(128, max(1, -(-programs // (batch * hkv))))
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
    if L * (HQ // HKV) > 16 and not window:
        return _fa3_path(data)  # long query blocks over the full context: FA3 still wins there
    if window and 1 < L <= 16 and B * L <= 32:
        return _window_tok_path(data)  # small verify batches: per-token split wins; larger ones re-read the window too often
    theta = cfg["rope_theta"]
    scaled = cfg["kv_scale"] != 1.0
    qkv = data["qkv"]
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    T = B * L
    dev = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    table = data["page_table"]
    q_scratch = torch.empty((T, HQ, D), dtype=torch.float8_e4m3fn, device=dev)
    _prep[(T,)](qkv, data["positions"], data["cache_loc"], k_cache, v_cache, q_scratch,
                data["k_scale"], data["v_scale"], 1.0, theta if theta else 1.0, RMS_EPS,
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
                            po, pm, pl, cfg["sm_scale"] * cfg["qk_scale_factor"], chunk,
                            L=L, GROUP=group, HQ=HQ, HKV=HKV, D=D, PAGE=page, MAX_PAGES=table.stride(0), WINDOW=window,
                            NSPLIT=nsplit, BLOCK_Q=block_q, BLOCK_N=block_n, SCALED=scaled, num_warps=4, num_stages=2)
    out = torch.empty((B, L, HQ * D), dtype=torch.bfloat16, device=dev)
    _combine[(B, HKV)](po, pm, pl, qkv, out, RMS_EPS, L=L, GROUP=group, HQ=HQ, HKV=HKV, D=D, NSPLIT=nsplit,
                       BLOCK_Q=block_q, num_warps=4)
    return out
