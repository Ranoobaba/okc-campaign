# okc: problem=muse_decode_layer
"""Whole decoder layer: our projections, attention and MoE kernels with SGLang's fused
norm/residual/routing modules in between (assembled by harness/assemble_layer.py)."""
from typing import TypedDict
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ====================== component: proj ======================

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


SCORED_M = (1, 8, 16, 32, 128, 256)
LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")


@triton.jit
def _gemv(x_ptr, w_ptr, w2_ptr, out_ptr, N, K, stride_wn,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, HAS_LO: tl.constexpr, OUT_F32: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    if HAS_LO:
        w2_ptrs = w2_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_k
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs).to(tl.float32)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        if HAS_LO:
            w2 = tl.load(w2_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(w2 * x[None, :], axis=1)
            w2_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    if OUT_F32:
        tl.store(out_ptr + offs_n, acc, mask=n_mask)
    else:
        tl.store(out_ptr + offs_n, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _skinny(x_ptr, w_ptr, w2_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
            HAS_LO: tl.constexpr, ATOMIC: tl.constexpr, OUT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k_len = K // SPLIT_K
    k0 = pid_k * k_len
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :]
    if HAS_LO:
        w2_ptrs = w2_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_len, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        if HAS_LO:
            w2 = tl.load(w2_ptrs, mask=n_mask[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(w2), acc)
            w2_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")
    elif OUT_F32:
        tl.store(out_ptrs, acc, mask=mask)
    else:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)


def _launch(x, w, w2, n, cfg):
    """One projection with config cfg: kind 'gemv', 'dot' or 'torch'. Returns the output tensor."""
    m, k = x.shape
    f32 = w2 is not None
    out_dtype = torch.float32 if f32 else torch.bfloat16
    if cfg["kind"] == "torch":
        if not f32:
            return torch.matmul(x, w.T)
        if m <= 8:
            return torch.nn.functional.linear(x.float(), cfg["router"])
        acc = torch.mm(x, cfg["hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, cfg["lo_t"], out_dtype=torch.float32)
        return acc
    if cfg["kind"] == "gemv":
        out = torch.empty((1, n), dtype=out_dtype, device=x.device)
        grid = (triton.cdiv(n, cfg["BLOCK_N"]),)
        _gemv[grid](x, w, w2 if f32 else w, out, n, k, w.stride(0),
                    BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], HAS_LO=f32, OUT_F32=f32,
                    num_warps=cfg["warps"], num_stages=cfg["stages"])
        return out
    split = cfg.get("SPLIT_K", 1)
    atomic = split > 1
    if atomic:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    grid = (triton.cdiv(n, cfg["BLOCK_N"]), triton.cdiv(m, cfg["BLOCK_M"]), split)
    _skinny[grid](x, w, w2 if f32 else w, out, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                  BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"], SPLIT_K=split,
                  HAS_LO=f32, ATOMIC=atomic, OUT_F32=f32 or atomic,
                  num_warps=cfg["warps"], num_stages=cfg["stages"])
    if atomic and not f32:
        out = out.to(torch.bfloat16)
    return out


def _candidates(layer, m, n, k):
    cands = []
    if m == 1:
        for bn in (4, 8, 16, 32):
            for bk in (256, 512, 1024):
                for warps in (4, 8):
                    if bn * bk * (2 if layer == "router" else 1) > 65536:
                        continue
                    cands.append({"kind": "gemv", "BLOCK_N": bn, "BLOCK_K": bk, "warps": warps, "stages": 3})
    bm = max(16, triton.next_power_of_2(min(m, 128)))
    bms = (bm,) if bm <= 32 else tuple(sorted({64, bm}))
    for bm in bms:
        bks = (64, 128, 256) if bm <= 32 else (64, 128)
        bns = (16, 32, 64, 128) if bm <= 32 else (64, 128)
        for bn in bns:
            if bn > n:
                continue
            for bk in bks:
                for warps in (4, 8):
                    for stages in (3, 4):
                        cands.append({"kind": "dot", "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": 1,
                                      "warps": warps, "stages": stages})
    blocks = (n // 16) * triton.cdiv(m, bm)
    if blocks < 264:  # too few blocks to pull bandwidth: try split-K
        for split in (2, 4, 8):
            if k % (split * 128):
                continue
            for bn in (16, 32):
                for bk in (64, 128):
                    cands.append({"kind": "dot", "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": split,
                                  "warps": 4, "stages": 3})
    return cands


def _time_graph(fn, flush, reps=7):
    """Median time of fn() as a CUDA graph replay with a cold L2, in microseconds."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for i in range(reps):
        flush.fill_(float(i))
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def _default_cfg(layer, m, n):
    if m == 1:
        return {"kind": "gemv", "BLOCK_N": 8, "BLOCK_K": 512, "warps": 4, "stages": 3}
    bm = max(16, triton.next_power_of_2(min(m, 128)))
    return {"kind": "dot", "BLOCK_M": bm, "BLOCK_N": 32 if bm <= 32 else 64, "BLOCK_K": 128,
            "SPLIT_K": 4 if layer == "router" and m <= 32 else 1, "warps": 4, "stages": 3}


def proj_prepare(weights: Weights):
    dev = weights["qkv_gate"].device
    state = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    router = weights["router"].contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router"] = router
    cfgs = {}
    flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)  # 256 MB, bigger than L2
    gen = torch.Generator(device=dev).manual_seed(0)
    import time

    t_start = time.time()
    for layer in LAYERS:
        w, w2 = _weights(state, layer)
        for m in SCORED_M:
            n = 1024 if layer == "unpooler" and m <= 8 else w.shape[0]
            x = torch.randn((m, w.shape[1]), generator=gen, device=dev).to(torch.bfloat16)
            torch_cfg = {"kind": "torch"}
            if layer == "router":
                torch_cfg.update(router=state["router"], hi_t=state["router_hi_t"], lo_t=state["router_lo_t"])
            t_torch = _time_graph(lambda: _launch(x, w[:n], w2[:n] if w2 is not None else None, n, torch_cfg), flush)
            best = (t_torch * 0.97, torch_cfg)  # a Triton config must beat cuBLAS by 3% to be picked
            for cfg in _candidates(layer, m, n, w.shape[1]):
                if time.time() - t_start > 240:  # stay well inside prepare's 300 s
                    break
                try:
                    t = _time_graph(lambda: _launch(x, w[:n], w2[:n] if w2 is not None else None, n, cfg), flush)
                except Exception:
                    continue
                if t < best[0]:
                    best = (t, cfg)
            cfgs[(layer, m)] = best[1]
    del flush
    state["cfgs"] = cfgs
    torch.cuda.synchronize()
    return state


def _weights(state, layer):
    if layer == "router":
        return state["router_hi"], state["router_lo"]
    return state[layer], None


def _pick(cfgs, layer, m, n):
    cfg = cfgs.get((layer, m))
    if cfg is None:
        # Nearest scored m, never below the actual m's tile needs.
        bucket = min(SCORED_M, key=lambda s: (abs(s - m), s))
        if m > 1 and bucket == 1:
            bucket = 8
        cfg = cfgs.get((layer, bucket), _default_cfg(layer, m, n))
        if cfg["kind"] == "gemv" and m != 1:
            cfg = _default_cfg(layer, m, n)
        if cfg["kind"] == "torch" and layer == "router":
            cfg = dict(cfg, router=None)  # filled below
        if cfg["kind"] == "dot" and cfg["BLOCK_M"] < min(m, 128):
            cfg = dict(cfg, BLOCK_M=max(16, triton.next_power_of_2(min(m, 128))))
    return cfg


def proj_kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg_in = data["config"]
    layer, n = cfg_in["layer"], cfg_in["n"]
    x = data["x"]
    m = x.shape[0]
    w, w2 = _weights(state, layer)
    cfg = _pick(state["cfgs"], layer, m, n)
    if cfg["kind"] == "torch" and layer == "router" and cfg.get("router") is None:
        cfg = dict(cfg, router=state["router"], hi_t=state["router_hi_t"], lo_t=state["router_lo_t"])
    if x.stride(1) != 1:
        x = x.contiguous()
    return _launch(x, w[:n], w2[:n] if w2 is not None else None, n, cfg)

# ====================== component: attn ======================

from typing import TypedDict

import torch
import triton
import triton.language as tl


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


def attn_kernel_fn(data: Inputs) -> torch.Tensor:
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

# ====================== component: moe ======================

from typing import TypedDict

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


POST_NORM_EPS = 1e-8
# 2^126 folded into the scale: an E2M1 nibble c placed at FP32 bits [22:24] (e1 e0 m) with the
# sign at bit 31 reads as (1 + m/2) * 2^(e - 127) for e >= 1 and (m/2) * 2^-126 for e = 0.
TWO_POW_126 = tl.constexpr(float(2.0 ** 126))


@triton.jit
def _dequant_pair(pk, sf, sfx, BN: tl.constexpr, BKH: tl.constexpr):
    """Packed bytes [BN, BKH] (two E2M1 codes each) and scales [BN, BKH // 8] -> the even and odd
    element tiles as BF16 [BN, BKH]. An E2M1 nibble placed at FP32 bits [22:24] with its sign at
    bit 31 reads as the E2M1 value times 2^-126; 2^126 is folded into the scale."""
    s3 = tl.reshape(sf.to(tl.float32) * sfx, (BN, BKH // 8, 1))
    s3 = tl.broadcast_to(s3, (BN, BKH // 8, 8))
    scale = tl.reshape(s3, (BN, BKH))
    lo = (pk & 0xF).to(tl.int32)
    hi = (pk >> 4).to(tl.int32)
    lo = (((lo & 8) << 28) | ((lo & 7) << 22)).to(tl.float32, bitcast=True)
    hi = (((hi & 8) << 28) | ((hi & 7) << 22)).to(tl.float32, bitcast=True)
    return (lo * scale).to(tl.bfloat16), (hi * scale).to(tl.bfloat16)


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
    BKH: tl.constexpr = BLOCK_K // 2
    offs_tok = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    tok_mask = offs_tok < num_valid
    rows = (offs_tok // TOP_K).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kh = tl.arange(0, BKH)
    ks = tl.arange(0, BKH // 8)
    a_even = a_ptr + rows[:, None] * stride_am + 2 * kh[None, :]
    a_odd = a_even + 1
    e64 = e.to(tl.int64)
    wg = w_ptr + e64 * stride_we + offs_n[:, None] * stride_wn + kh[None, :]
    wu = wg + I * stride_wn
    sg = sf_ptr + e64 * stride_se + offs_n[:, None] * stride_sn + ks[None, :]
    su = sg + I * stride_sn
    sfx = tl.load(gs_ptr + e) * TWO_POW_126
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        ae = tl.load(a_even, mask=tok_mask[:, None], other=0.0)
        ao = tl.load(a_odd, mask=tok_mask[:, None], other=0.0)
        g_lo, g_hi = _dequant_pair(tl.load(wg), tl.load(sg), sfx, BLOCK_N, BKH)
        acc_g = tl.dot(ae, tl.trans(g_lo), acc_g)
        acc_g = tl.dot(ao, tl.trans(g_hi), acc_g)
        u_lo, u_hi = _dequant_pair(tl.load(wu), tl.load(su), sfx, BLOCK_N, BKH)
        acc_u = tl.dot(ae, tl.trans(u_lo), acc_u)
        acc_u = tl.dot(ao, tl.trans(u_hi), acc_u)
        a_even += BLOCK_K
        a_odd += BLOCK_K
        wg += BKH
        wu += BKH
        sg += BKH // 8
        su += BKH // 8
    h = acc_g * tl.sigmoid(acc_g) * acc_u
    c_ptrs = c_ptr + offs_tok.to(tl.int64)[:, None] * stride_cm + offs_n[None, :]
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
    BKH: tl.constexpr = BLOCK_K // 2
    offs_tok = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    tok_mask = offs_tok < num_valid
    rows = offs_tok.to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kh = tl.arange(0, BKH)
    ks = tl.arange(0, BKH // 8)
    a_even = a_ptr + rows[:, None] * stride_am + 2 * kh[None, :]
    a_odd = a_even + 1
    e64 = e.to(tl.int64)
    w = w_ptr + e64 * stride_we + offs_n[:, None] * stride_wn + kh[None, :]
    s = sf_ptr + e64 * stride_se + offs_n[:, None] * stride_sn + ks[None, :]
    sfx = tl.load(gs_ptr + e) * TWO_POW_126
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        ae = tl.load(a_even, mask=tok_mask[:, None], other=0.0)
        ao = tl.load(a_odd, mask=tok_mask[:, None], other=0.0)
        lo, hi = _dequant_pair(tl.load(w), tl.load(s), sfx, BLOCK_N, BKH)
        acc = tl.dot(ae, tl.trans(lo), acc)
        acc = tl.dot(ao, tl.trans(hi), acc)
        a_even += BLOCK_K
        a_odd += BLOCK_K
        w += BKH
        s += BKH // 8
    c_ptrs = c_ptr + rows[:, None] * stride_cm + offs_n[None, :]
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


def moe_prepare(weights: Weights):
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


def _moe_core(x, local_ids, g, st, top_k, count, I, e_off=0):
    """x [M, H] bf16; local_ids [M, top_k] int32 (-1 = not on this GPU); g [M, top_k] fp32."""
    M, H = x.shape
    dev = x.device
    pairs = M * top_k
    block_m = _block_m(pairs / count)
    sorted_ids, expert_ids, n_post = moe_align_block_size(local_ids, block_m, count)
    max_blocks = expert_ids.numel()
    inter = torch.empty((pairs, I), dtype=torch.bfloat16, device=dev)
    w13, w13_sf, w2, w2_sf = st["w13"], st["w13_sf"], st["w2"], st["w2_sf"]
    BN1, BK1 = 64, 128
    _gemm1[(max_blocks, I // BN1)](
        x, w13[e_off:], w13_sf[e_off:], st["w13_gs"][e_off:], inter, sorted_ids, expert_ids, n_post,
        pairs, I, H, x.stride(0), w13.stride(0), w13.stride(1), w13_sf.stride(0), w13_sf.stride(1), inter.stride(0),
        TOP_K=top_k, BLOCK_M=block_m, BLOCK_N=BN1, BLOCK_K=BK1, num_warps=8, num_stages=3)
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


def moe_kernel_fn(data: Inputs, st) -> torch.Tensor:
    cfg = data["config"]
    x = data["hidden_states"]
    if not x.is_contiguous():
        x = x.contiguous()
    emap = _expert_map(st, cfg["local_start"], cfg["local_count"], cfg["n_experts"], x.device)
    local_ids = emap[data["topk_ids"]]
    g = data["topk_weights"]
    if g.dtype != torch.float32:
        g = g.float()
    return _moe_core(x, local_ids, g.contiguous(), st, cfg["top_k"], cfg["local_count"], cfg["d_expert"],
                     e_off=cfg["local_start"])

# ====================== layer glue ======================
import types

from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
UNPOOL_SLICE = HIDDEN // 8
SLICED_MAX_TOKENS = 8
ROUTER_MULTIPLIER = 0.1767766952966369
LAYER_RMS_EPS = 1e-5
LAYER_POST_NORM_EPS = 1e-8
# Shapes beyond this use SGLang's own attention and experts (prefill blocks).
DECODE_MAX_Q_LEN = 8


def _param(w):
    return torch.nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def prepare(weights):
    dev = weights["qkv_gate"].device
    proj_state = proj_prepare({k: weights[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")})
    moe_state = moe_prepare({"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
                             "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
                             "post_norm_w": weights["post_expert_norm"]})
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, LAYER_RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, LAYER_RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, LAYER_POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, LAYER_POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "proj": proj_state,
        "moe": moe_state,
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=ROUTER_MULTIPLIER, router_score_division_eps=1e-15),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, LAYER_RMS_EPS),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn),
    }
    torch.cuda.synchronize()
    return state


def _proj(state, x, layer, n):
    data = {"x": x, "config": {"m": x.shape[0], "n": n, "k": x.shape[1], "layer": layer}}
    return proj_kernel_fn(data, state["proj"])


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    x = data["attn_input"]
    # Attention block.
    qkv = _proj(state, x, "qkv_gate", 2 * hq * d + 2 * cfg["num_kv_heads"] * d)
    attn_data = {
        "qkv": qkv.view(B, L, -1), "positions": data["positions"], "cache_loc": data["cache_loc"],
        "cache_seqlens": data["cache_seqlens"], "cu_seqlens_q": data["cu_seqlens_q"],
        "scheduler_metadata": data["scheduler_metadata"], "k_cache": data["k_cache"], "v_cache": data["v_cache"],
        "k_scale": data["k_scale"], "v_scale": data["v_scale"], "page_table": data["page_table"],
        "seq_lens": data["seq_lens"],
        "config": {"batch": B, "q_len": L, "num_q_heads": hq, "num_kv_heads": cfg["num_kv_heads"], "head_dim": d,
                   "page_size": cfg["page_size"], "window": cfg["window"], "rope_theta": cfg["rope_theta"],
                   "qk_scale_factor": cfg["qk_scale_factor"], "sm_scale": cfg["sm_scale"], "kv_scale": cfg["kv_scale"]},
    }
    o = attn_kernel_fn(attn_data).view(T, hq * d)
    attn_output = _proj(state, o, "o", HIDDEN)
    # Post-attention boundary and router.
    split_router = T > SLICED_MAX_TOKENS
    ffn_input, stream, _ = state["attn_boundary"](data["stream"], attn_output, branch_dtype=torch.bfloat16,
                                                  router_dtype=None)
    logits = _proj(state, ffn_input, "router", N_EXPERTS)
    topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    # MoE block.
    expert_input = state["pre_expert_norm"](_proj(state, ffn_input, "pooler", D_EXPERT))
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    local_ids = torch.where(local_ids < LOCAL_EXPERTS, local_ids, -1).to(torch.int32)
    expert_output = _moe_core(expert_input.contiguous(), local_ids, topk_weights.float().contiguous(), state["moe"],
                              TOP_K, LOCAL_EXPERTS, D_EXPERT)
    if T <= SLICED_MAX_TOKENS:
        ffn_output = F.pad(_proj(state, expert_output, "unpooler", UNPOOL_SLICE), (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = _proj(state, expert_output, "unpooler", HIDDEN)
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
