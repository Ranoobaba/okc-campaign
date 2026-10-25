import types

from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)

@triton.jit
def _local_ids_kernel(ids_ptr, start_ptr, out_ptr, n, N_EXPERTS: tl.constexpr, LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    start = tl.load(start_ptr)
    ids = tl.load(ids_ptr + offs, mask=mask, other=0)
    loc = (ids - start + N_EXPERTS) % N_EXPERTS  # Triton's % keeps the sign of the dividend
    loc = tl.where(loc < LOCAL, loc, -1)
    tl.store(out_ptr + offs, loc.to(tl.int32), mask=mask)


@triton.jit
def _dual_proj_kernel(x_ptr, wp_ptr, whi_ptr, wlo_ptr, outp_ptr, outr_ptr, M, NP, NR, K,
                      stride_xm, stride_wp, stride_wr, stride_op, stride_or,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_R: tl.constexpr):
    """Pooler GEMM (BF16 out) and router hi/lo GEMM (FP32 atomics, K split SPLIT_R ways) in one launch:
    programs [0, NP/BN) are pooler tiles, the rest are router tiles x K slices. Both read the same x."""
    pid = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    n_pool_tiles = NP // BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if pid < n_pool_tiles:
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
        w_ptrs = wp_ptr + offs_n[:, None] * stride_wp + offs_k[None, :]
        for _ in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w = tl.load(w_ptrs)
            acc = tl.dot(x, tl.trans(w), acc)
            x_ptrs += BLOCK_K
            w_ptrs += BLOCK_K
        tl.store(outp_ptr + offs_m[:, None] * stride_op + offs_n[None, :], acc.to(tl.bfloat16), mask=m_mask[:, None])
    else:
        r = pid - n_pool_tiles
        tile = r // SPLIT_R
        ks = r % SPLIT_R
        k_len = K // SPLIT_R
        offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (ks * k_len + offs_k)[None, :]
        hi_ptrs = whi_ptr + offs_n[:, None] * stride_wr + (ks * k_len + offs_k)[None, :]
        lo_ptrs = wlo_ptr + offs_n[:, None] * stride_wr + (ks * k_len + offs_k)[None, :]
        for _ in range(0, k_len, BLOCK_K):
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(tl.load(hi_ptrs)), acc)
            acc = tl.dot(x, tl.trans(tl.load(lo_ptrs)), acc)
            x_ptrs += BLOCK_K
            hi_ptrs += BLOCK_K
            lo_ptrs += BLOCK_K
        tl.atomic_add(outr_ptr + offs_m[:, None] * stride_or + offs_n[None, :], acc, mask=m_mask[:, None], sem="relaxed")


def _dual_proj(state, x):
    """ffn_input [T, 8192] -> (pooler output bf16 [T, 4096], router logits fp32 [T, 256]) in one launch (T <= 32)."""
    T, K = x.shape
    ps = state["proj"]
    wp = ps["pooler"]
    whi, wlo = ps["router_hi"], ps["router_lo"]
    outp = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=x.device)
    outr = torch.zeros((T, N_EXPERTS), dtype=torch.float32, device=x.device)
    bm = 16 if T <= 16 else 32
    bn, bk, split_r = 32, 128, 4
    grid = (D_EXPERT // bn + (N_EXPERTS // bn) * split_r, triton.cdiv(T, bm))
    _dual_proj_kernel[grid](x, wp, whi, wlo, outp, outr, T, D_EXPERT, N_EXPERTS, K,
                            x.stride(0), wp.stride(0), whi.stride(0), outp.stride(0), outr.stride(0),
                            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_R=split_r, num_warps=4, num_stages=3)
    return outp, outr


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


def _proj(state, x, layer, n, out=None):
    data = {"x": x, "config": {"m": x.shape[0], "n": n, "k": x.shape[1], "layer": layer}}
    return proj_kernel_fn(data, state["proj"], out=out)


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
    o = (_fa3_path(attn_data) if L > 16 else attn_kernel_fn(attn_data)).view(T, hq * d)  # prefill blocks: FA3
    attn_output = _proj(state, o, "o", HIDDEN)
    # Post-attention boundary and router.
    split_router = T > SLICED_MAX_TOKENS
    ffn_input, stream, _ = state["attn_boundary"](data["stream"], attn_output, branch_dtype=torch.bfloat16,
                                                  router_dtype=None)
    # (A fused router+pooler launch was measured at 1.248 vs 1.260 for separate launches: the Triton pooler
    # path loses more than the saved launch; kept separate.)
    logits = _proj(state, ffn_input, "router", N_EXPERTS)
    pooled = _proj(state, ffn_input, "pooler", D_EXPERT)
    topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    # MoE block.
    expert_input = state["pre_expert_norm"](pooled)
    local_ids = torch.empty((T, TOP_K), dtype=torch.int32, device=x.device)
    n_ids = T * TOP_K
    _local_ids_kernel[(triton.cdiv(n_ids, 1024),)](topk_ids, data["local_start"], local_ids, n_ids,
                                                   N_EXPERTS=N_EXPERTS, LOCAL=LOCAL_EXPERTS, BLOCK=1024, num_warps=4)
    expert_output = _moe_core(expert_input.contiguous(), local_ids, topk_weights.float().contiguous(), state["moe"],
                              TOP_K, LOCAL_EXPERTS, D_EXPERT)
    if T <= SLICED_MAX_TOKENS:
        ffn_output = torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=x.device)
        _proj(state, expert_output, "unpooler", UNPOOL_SLICE, out=ffn_output[:, :UNPOOL_SLICE])
    else:
        ffn_output = _proj(state, expert_output, "unpooler", HIDDEN)
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
