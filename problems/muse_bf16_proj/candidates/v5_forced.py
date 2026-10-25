# okc: problem=muse_bf16_proj
"""Dense projections as weight-streaming skinny GEMMs in Triton.

Every scored shape (m <= 256) is HBM-bound on reading the weight once, so the kernel is built
around streaming W [N, K] (K contiguous) through the SMs with enough blocks in flight, while the
small activation tile x [m, K] stays in L2. Two kernels:

  _gemv    m == 1: pure load + FMA, no tensor cores, BLOCK_N rows of W per program.
  _skinny  m <= 256: tl.dot on a [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] tile per step,
           optional split-K with fp32 atomics (used for the 256-row router, which otherwise
           has too few blocks to pull bandwidth).

The router keeps SGLang's numerics above 8 tokens at every m: its FP32 weight is split into a
BF16 hi + lo pair (lossless to ~16 mantissa bits) and both products accumulate in FP32.

prepare() repacks the weights (contiguous, router hi/lo) and tunes a small config table per
(layer, m) for the scored m values by timing CUDA-graph replays with a cold L2, the way the
judge times. kernel_fn picks the tuned config (nearest m bucket for unseen m) and launches.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


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


# ---- CUDA skinny GEMM (m <= 32): 16 weight rows per block, 4 warps split K, cp.async 3-stage pipeline, mma.sync ----
from torch.utils.cpp_extension import load_inline  # noqa: E402

_SK_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>
#define BK 64
#define STAGES 3
#define XS 72
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* smem) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// MT = number of 16-row A tiles (1 for m <= 16, 2 for m <= 32). Block: 16 output columns (W rows), 4 warps on K.
template <int MT>
__global__ void __launch_bounds__(128) skinny_kernel(const __nv_bfloat16* __restrict__ X, long x_stride,
                                                     const __nv_bfloat16* __restrict__ W, long w_stride,
                                                     __nv_bfloat16* __restrict__ O, long o_stride, int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    // each warp owns a private pipeline over its K slice: chunks warp, warp+4, ...
    extern __shared__ __align__(16) unsigned char smem_raw[];
    typedef __nv_bfloat16 XsT[STAGES][MT * 16][XS];
    typedef __nv_bfloat16 WsT[STAGES][16][XS];
    typedef float RedT[MT * 16][16];
    XsT* xs = reinterpret_cast<XsT*>(smem_raw);                                   // [4]
    WsT* wsm = reinterpret_cast<WsT*>(smem_raw + 4 * sizeof(XsT));                 // [4]
    RedT* red = reinterpret_cast<RedT*>(smem_raw + 4 * sizeof(XsT) + 4 * sizeof(WsT));  // [4]
    const int nK = K / BK;              // chunks in total
    const int my_chunks = (nK - warp + 3) / 4;
    auto load_stage = [&](int stage, int ci) {
        int kc = ci * 4 + warp;
        // x: MT*16 rows x 8 chunks of 16B
        for (int c = lane; c < MT * 16 * 8; c += 32) {
            int r = c >> 3, cc = c & 7;
            bool ok = r < M;
            const __nv_bfloat16* src = ok ? X + (long)r * x_stride + kc * BK + cc * 8 : X;
            cp_async16(&xs[warp][stage][r][cc * 8], src, ok);
        }
        // w: 16 rows x 8 chunks
        for (int c = lane; c < 16 * 8; c += 32) {
            int r = c >> 3, cc = c & 7;
            bool ok = n0 + r < N;
            const __nv_bfloat16* src = ok ? W + (long)(n0 + r) * w_stride + kc * BK + cc * 8 : W;
            cp_async16(&wsm[warp][stage][r][cc * 8], src, ok);
        }
        cp_async_commit();
    };
    float acc[MT][2][4];
    #pragma unroll
    for (int t = 0; t < MT; t++)
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[t][f][i] = 0.f;
    for (int s = 0; s < STAGES - 1; s++) { if (s < my_chunks) load_stage(s, s); else cp_async_commit(); }
    for (int ci = 0; ci < my_chunks; ci++) {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        int nxt = ci + STAGES - 1;
        if (nxt < my_chunks) load_stage(nxt % STAGES, nxt); else cp_async_commit();
        const int st = ci % STAGES;
        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t b[4];  // two n-frags: rows 0-7 and 8-15 of the W tile, k 0-7 | 8-15
            // ldmatrix x4 over W rows: lanes 0-7 rows 0-7 k0, 8-15 rows 8-15 k0, 16-23 rows 0-7 k0+8, 24-31 rows 8-15 k0+8
            ldmatrix_x4(b, &wsm[warp][st][lane & 15][ks * 16 + (lane >> 4) * 8]);
            uint32_t bf0[2] = {b[0], b[2]};
            uint32_t bf1[2] = {b[1], b[3]};
            #pragma unroll
            for (int t = 0; t < MT; t++) {
                uint32_t a[4];
                ldmatrix_x4(a, &xs[warp][st][t * 16 + (lane & 15)][ks * 16 + (lane >> 4) * 8]);
                mma_bf16(acc[t][0], a, bf0);
                mma_bf16(acc[t][1], a, bf1);
            }
        }
        __syncwarp();
    }
    cp_async_wait<0>();
    // cross-warp reduction over the K slices
    #pragma unroll
    for (int t = 0; t < MT; t++)
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int r = t * 16 + (lane >> 2) + h * 8;
                int c = f * 8 + (lane & 3) * 2;
                red[warp][r][c] = acc[t][f][h * 2];
                red[warp][r][c + 1] = acc[t][f][h * 2 + 1];
            }
    __syncthreads();
    for (int i = tid; i < MT * 16 * 16; i += 128) {
        int r = i >> 4, c = i & 15;
        if (r < M && n0 + c < N) {
            float v = red[0][r][c] + red[1][r][c] + red[2][r][c] + red[3][r][c];
            O[(long)r * o_stride + n0 + c] = __float2bfloat16(v);
        }
    }
}

template <int MT> size_t smem_bytes() {
    return 4 * (size_t)STAGES * MT * 16 * XS * 2 + 4 * (size_t)STAGES * 16 * XS * 2 + 4 * (size_t)MT * 16 * 16 * 4;
}
void skinny(torch::Tensor X, torch::Tensor W, torch::Tensor O) {
    int M = X.size(0), K = X.size(1), N = W.size(0);
    dim3 grid((N + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(skinny_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes<1>());
        cudaFuncSetAttribute(skinny_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes<2>());
        attr_set = true;
    }
    if (M <= 16)
        skinny_kernel<1><<<grid, 128, smem_bytes<1>(), stream>>>((const __nv_bfloat16*)X.data_ptr(), X.stride(0), (const __nv_bfloat16*)W.data_ptr(), W.stride(0),
                                                   (__nv_bfloat16*)O.data_ptr(), O.stride(0), M, N, K);
    else
        skinny_kernel<2><<<grid, 128, smem_bytes<2>(), stream>>>((const __nv_bfloat16*)X.data_ptr(), X.stride(0), (const __nv_bfloat16*)W.data_ptr(), W.stride(0),
                                                   (__nv_bfloat16*)O.data_ptr(), O.stride(0), M, N, K);
}
"""
_sk = None


def _sk_mod():
    global _sk
    if _sk is None:
        _sk = load_inline(name="okc_proj_skinny", cpp_sources="void skinny(torch::Tensor X, torch::Tensor W, torch::Tensor O);",
                          cuda_sources=_SK_SRC, functions=["skinny"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
    return _sk


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
    if cfg["kind"] == "cuda":
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _sk_mod().skinny(x, w, out)
        return out
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


def prepare(weights: Weights):
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
    _sk_mod()
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
            cands = _candidates(layer, m, n, w.shape[1])
            if layer != "router" and 1 < m <= 32:
                cands.append({"kind": "cuda"})
            for cfg in cands:
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
        if cfg["kind"] == "cuda" and (m > 32 or layer == "router"):
            cfg = {"kind": "torch"}
        if cfg["kind"] == "torch" and layer == "router":
            cfg = dict(cfg, router=None)  # filled below
        if cfg["kind"] == "dot" and cfg["BLOCK_M"] < min(m, 128):
            cfg = dict(cfg, BLOCK_M=max(16, triton.next_power_of_2(min(m, 128))))
    return cfg


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg_in = data["config"]
    layer, n = cfg_in["layer"], cfg_in["n"]
    x = data["x"]
    m = x.shape[0]
    w, w2 = _weights(state, layer)
    cfg = _pick(state["cfgs"], layer, m, n)
    if layer != "router" and 1 < m <= 32:
        cfg = {"kind": "cuda"}
    if cfg["kind"] == "torch" and layer == "router" and cfg.get("router") is None:
        cfg = dict(cfg, router=state["router"], hi_t=state["router_hi_t"], lo_t=state["router_lo_t"])
    if x.stride(1) != 1:
        x = x.contiguous()
    return _launch(x, w[:n], w2[:n] if w2 is not None else None, n, cfg)
