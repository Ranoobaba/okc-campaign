# okc: problem=muse_moe_w4a16
"""Routed NVFP4 experts: a CUDA streaming W4A16 kernel (cp.async pipeline, byte-permute E2M1 dequant,
mma.sync) for batches up to 32 tokens, SGLang's Marlin W4A16 GEMMs above that, with the
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
def _silu_mul(acc_ptr, ids_ptr, gs_ptr, out_ptr, I: tl.constexpr, BLOCK: tl.constexpr):
    pair = tl.program_id(0).to(tl.int64)
    e = tl.load(ids_ptr + pair)
    if e < 0:
        return
    gs = tl.load(gs_ptr + e)
    for c0 in range(0, I, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        g = tl.load(acc_ptr + pair * (2 * I) + offs) * gs
        u = tl.load(acc_ptr + pair * (2 * I) + I + offs) * gs
        tl.store(out_ptr + pair * I + offs, (g * tl.sigmoid(g) * u).to(tl.bfloat16))


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


# ---- CUDA W4A16 streaming grouped GEMM for small batches ----
from torch.utils.cpp_extension import load_inline  # noqa: E402

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// One block = one m-block of 16 (token, slot) pairs of one expert x one tile of 64 output rows.
// Streams the expert's packed E2M1 rows and E4M3 block scales through a 3-stage cp.async pipeline,
// dequantizes in registers with byte permutes (exact to BF16), and multiplies with mma.sync m16n8k16.
// GATED: two weight tiles (gate rows n, up rows I + n) and SiLU(gate) * up in the epilogue.

#define BK 64
#define BN 32
#define STAGES 4
#define THREADS 128
#define X_STRIDE 72   // bf16 elements per smem x row (64 + 8 pad: conflict-free ldmatrix)

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

// E2M1 magnitudes as BF16: 0, .5, 1, 1.5, 2, 3, 4, 6 -> low bytes / high bytes pools for prmt.
#define LO_POOL_A 0xC0800000u  // bytes: [0]=0x00 [1]=0x00 [2]=0x80 [3]=0xC0
#define LO_POOL_B 0xC0804000u  // bytes: [4]=0x00 [5]=0x40 [6]=0x80 [7]=0xC0
#define HI_POOL_A 0x3F3F3F00u  // [0]=0x00 [1]=0x3F [2]=0x3F [3]=0x3F
#define HI_POOL_B 0x40404040u  // [4..7]=0x40

// Eight nibbles -> four bf16x2 words (v0,v1) (v2,v3) (v4,v5) (v6,v7): magnitude via two byte-permute
// lookups (low / high bytes), sign via a third permute into the high bytes.
__device__ __forceinline__ void dequant8(uint32_t n8, uint32_t& w01, uint32_t& w23, uint32_t& w45, uint32_t& w67) {
    uint32_t mag = n8 & 0x77777777u;
    uint32_t lo_a = __byte_perm(LO_POOL_A, LO_POOL_B, mag);
    uint32_t lo_b = __byte_perm(LO_POOL_A, LO_POOL_B, mag >> 16);
    uint32_t hi_a = __byte_perm(HI_POOL_A, HI_POOL_B, mag);
    uint32_t hi_b = __byte_perm(HI_POOL_A, HI_POOL_B, mag >> 16);
    uint32_t nsg = (~n8 >> 3) & 0x11111111u;          // 1 where the sign bit is clear
    hi_a |= __byte_perm(0x00000080u, 0u, nsg);          // byte i: 0x80 if signed else 0x00
    hi_b |= __byte_perm(0x00000080u, 0u, nsg >> 16);
    w01 = __byte_perm(lo_a, hi_a, 0x5140);
    w23 = __byte_perm(lo_a, hi_a, 0x7362);
    w45 = __byte_perm(lo_b, hi_b, 0x5140);
    w67 = __byte_perm(lo_b, hi_b, 0x7362);
}

// Four E4M3 scales (one uint32) -> four bf16x2 words, each scale duplicated, times 2^62 (exponent +182):
// E4M3 (s eeee mmm) -> BF16 bits ((s)<<15) | (((e<<3|m) << 4) + (182 << 7)). Exact for normal scales; the
// generator keeps scales in [8, 448], so the subnormal case cannot occur. 2^62 keeps the product in range and
// is undone (with 2^64 from the E2M1 encoding) by out_scale.
__device__ __forceinline__ void scales4(uint32_t sv, uint32_t& s0, uint32_t& s1, uint32_t& s2, uint32_t& s3) {
    uint32_t x01 = __byte_perm(sv, 0u, 0x4140);          // [b0, 0, b1, 0]
    uint32_t x23 = __byte_perm(sv, 0u, 0x4342);          // [b2, 0, b3, 0]
    uint32_t b01 = ((x01 & 0x00800080u) << 8) | (((x01 & 0x007F007Fu) << 4) + 0x5B005B00u);
    uint32_t b23 = ((x23 & 0x00800080u) << 8) | (((x23 & 0x007F007Fu) << 4) + 0x5B005B00u);
    s0 = __byte_perm(b01, 0u, 0x1010);
    s1 = __byte_perm(b01, 0u, 0x3232);
    s2 = __byte_perm(b23, 0u, 0x1010);
    s3 = __byte_perm(b23, 0u, 0x3232);
}

__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
    __nv_bfloat162 x = reinterpret_cast<__nv_bfloat162&>(a);
    __nv_bfloat162 y = reinterpret_cast<__nv_bfloat162&>(b);
    __nv_bfloat162 z = __hmul2(x, y);
    return reinterpret_cast<uint32_t&>(z);
}

template <bool GATED>
__global__ void __launch_bounds__(THREADS) w4a16_kernel(
    const __nv_bfloat16* __restrict__ A, long a_stride,            // activations [rows, K], row = token (GATED) or pair
    const uint8_t* __restrict__ W, long w_stride_e, long w_stride_n,  // [E, Nw, K/2]
    const uint8_t* __restrict__ SF, long s_stride_e, long s_stride_n, // [E, Nw, K/16]
    const float* __restrict__ GS,                                   // [E]
    const int32_t* __restrict__ sorted_ids, const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ n_post_pad,
    float* __restrict__ C, long c_stride,                           // fp32 [pairs, N] (gate) ; +N_out for up when GATED
    int num_valid, int top_k, int K, int N_out, int splitk, float out_scale) {
    const int mb = blockIdx.x;
    if (mb * 16 >= *n_post_pad) return;
    const int e = expert_ids[mb];
    if (e < 0) return;
    const int n0 = blockIdx.y * BN;
    const int z = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    __shared__ __align__(16) __nv_bfloat16 xs[STAGES][16][X_STRIDE];
    __shared__ __align__(16) uint8_t ws[STAGES][GATED ? 2 : 1][BN][BK / 2];
    __shared__ __align__(16) uint8_t ss[STAGES][GATED ? 2 : 1][BN][BK / 16];
    __shared__ int rows_s[16];

    if (tid < 16) {
        int pid = sorted_ids[mb * 16 + tid];
        rows_s[tid] = pid < num_valid ? (GATED ? pid / top_k : pid) : -1;
    }
    __syncthreads();

    const uint8_t* wbase = W + (long)e * w_stride_e;
    const uint8_t* sbase = SF + (long)e * s_stride_e;
    const int tiles_per_half = N_out / BN;
    const int nK_all = K / BK;
    const int nK_total = nK_all;
    const int kc0 = z * nK_total / splitk;
    const int nK = (z + 1) * nK_total / splitk - kc0;

    // Per-thread copy assignments, fixed across chunks (only the chunk offset changes).
    const int xr = tid >> 3, xc = tid & 7;                      // 128 threads = 16 rows x 8 chunks of 16B
    const int xrow = rows_s[xr];
    const __nv_bfloat16* xsrc = xrow >= 0 ? A + (long)xrow * a_stride + xc * 8 : A;
    const uint8_t* wsrc[GATED ? 2 : 1];
    const uint8_t* ssrc[GATED ? 2 : 1];
    #pragma unroll
    for (int mat = 0; mat < (GATED ? 2 : 1); mat++) {
        long tile = (long)(mat * tiles_per_half + blockIdx.y) * nK_all;
        wsrc[mat] = wbase + tile * (BN * (BK / 2));
        ssrc[mat] = sbase + tile * (BN * (BK / 16));
    }
    auto load_stage = [&](int stage, int kci) {
        const int kc = kc0 + kci;
        cp_async16(&xs[stage][xr][xc * 8], xsrc + kc * BK, xrow >= 0);
        #pragma unroll
        for (int mat = 0; mat < (GATED ? 2 : 1); mat++) {
            if (tid < BN * 2) cp_async16(&ws[stage][mat][tid >> 1][(tid & 1) * 16], wsrc[mat] + (long)kc * (BN * (BK / 2)) + tid * 16, true);
            if (tid < BN) {
                unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(&ss[stage][mat][tid][0]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(ssrc[mat] + (long)kc * (BN * (BK / 16)) + tid * 4));
            }
        }
        cp_async_commit();
    };

    float acc[GATED ? 2 : 1][4];
    #pragma unroll
    for (int m = 0; m < (GATED ? 2 : 1); m++)
        #pragma unroll
        for (int i = 0; i < 4; i++) acc[m][i] = 0.f;

    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nK) load_stage(s, s); else cp_async_commit();
    }
    const int nrow = warp * 8 + (lane >> 2);
    const int g = lane & 3;
    for (int kc = 0; kc < nK; kc++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int nxt = kc + STAGES - 1;
        if (nxt < nK) load_stage(nxt % STAGES, nxt); else cp_async_commit();
        const int st = kc % STAGES;
        uint32_t a[4][4];
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) ldmatrix_x4(a[ks], &xs[st][lane & 15][ks * 16 + (lane >> 4) * 8]);
        #pragma unroll
        for (int mat = 0; mat < (GATED ? 2 : 1); mat++) {
            uint2 w8 = *reinterpret_cast<const uint2*>(&ws[st][mat][nrow][g * 8]);
            uint32_t sv = *reinterpret_cast<const uint32_t*>(&ss[st][mat][nrow][0]);
            uint32_t w[8];
            dequant8(w8.x, w[0], w[1], w[2], w[3]);   // ks 0: (w0,w1) ; ks 1: (w2,w3)
            dequant8(w8.y, w[4], w[5], w[6], w[7]);   // ks 2, ks 3
            uint32_t sc[4];
            scales4(sv, sc[0], sc[1], sc[2], sc[3]);
            #pragma unroll
            for (int ks = 0; ks < 4; ks++) {
                uint32_t b[2] = {hmul2_bf16(w[2 * ks], sc[ks]), hmul2_bf16(w[2 * ks + 1], sc[ks])};
                mma_bf16(acc[mat], a[ks], b);
            }
        }
    }
    cp_async_wait<0>();

    #pragma unroll
    for (int h = 0; h < 2; h++) {
        int r = (lane >> 2) + h * 8;
        int pid = sorted_ids[mb * 16 + r];
        if (pid >= num_valid) continue;
        int n = n0 + warp * 8 + (lane & 3) * 2;
        #pragma unroll
        for (int mat = 0; mat < (GATED ? 2 : 1); mat++) {
            float* dst = C + (long)pid * c_stride + mat * N_out + n;
            atomicAdd(reinterpret_cast<float2*>(dst), make_float2(acc[mat][h * 2] * out_scale, acc[mat][h * 2 + 1] * out_scale));
        }
    }
}

void w4a16_moe(torch::Tensor A, torch::Tensor W, torch::Tensor SF, torch::Tensor GS, torch::Tensor sorted_ids,
               torch::Tensor expert_ids, torch::Tensor n_post_pad, torch::Tensor C, int64_t num_valid, int64_t top_k,
               int64_t max_blocks, bool gated, int64_t splitk) {
    const float out_scale = 2.1684043449710089e-19f;  // 2^-62
    int K = A.size(1);
    int N_out = gated ? C.size(1) / 2 : C.size(1);
    dim3 grid(max_blocks, N_out / BN, splitk);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (gated) {
        w4a16_kernel<true><<<grid, THREADS, 0, stream>>>(
            (const __nv_bfloat16*)A.data_ptr(), A.stride(0), (const uint8_t*)W.data_ptr(), W.stride(0), W.stride(1),
            (const uint8_t*)SF.data_ptr(), SF.stride(0), SF.stride(1), (const float*)GS.data_ptr(),
            sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(), n_post_pad.data_ptr<int32_t>(),
            (float*)C.data_ptr(), C.stride(0), (int)num_valid, (int)top_k, K, N_out, (int)splitk, out_scale);
    } else {
        w4a16_kernel<false><<<grid, THREADS, 0, stream>>>(
            (const __nv_bfloat16*)A.data_ptr(), A.stride(0), (const uint8_t*)W.data_ptr(), W.stride(0), W.stride(1),
            (const uint8_t*)SF.data_ptr(), SF.stride(0), SF.stride(1), (const float*)GS.data_ptr(),
            sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(), n_post_pad.data_ptr<int32_t>(),
            (float*)C.data_ptr(), C.stride(0), (int)num_valid, (int)top_k, K, N_out, (int)splitk, out_scale);
    }
}
"""
_cuda = None


def _cuda_mod():
    global _cuda
    if _cuda is None:
        _cuda = load_inline(
            name="okc_moe_w4a16",
            cpp_sources="void w4a16_moe(torch::Tensor A, torch::Tensor W, torch::Tensor SF, torch::Tensor GS, torch::Tensor sorted_ids,"
                        " torch::Tensor expert_ids, torch::Tensor n_post_pad, torch::Tensor C, int64_t num_valid, int64_t top_k,"
                        " int64_t max_blocks, bool gated, int64_t splitk);",
            cuda_sources=_CUDA_SRC, functions=["w4a16_moe"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
            verbose=False)
    return _cuda


CUDA_MAX_TOKENS = 32  # above this, Marlin's larger M tiles win


def _apply_gs(y, local_ids, gs2):
    """The finalize RMS-normalizes each expert output row, so a positive per-row scalar (the W2 global
    scale) cancels exactly: nothing to apply. Kept as a hook for clarity."""
    return y
_REPACK_IDX = [ks * 8 + half * 4 + g for g in range(4) for ks in range(4) for half in range(2)]


def _repack(w, bn=32):
    """Lossless repack into contiguous (32-row tile, 64-element chunk) blocks of 32 x 32 bytes, with the
    lane permutation inside each row's 32 bytes: new[g*8 + ks*2 + half] = old[ks*8 + half*4 + g].
    Layout: [E, N/bn, K/64, bn, 32] flattened to the original [E, N, K/2] byte count."""
    E, N, KB = w.shape
    idx = torch.tensor(_REPACK_IDX, device=w.device)
    out = torch.empty_like(w)
    for e in range(E):  # per expert to bound temporary memory
        t = w[e].view(N // bn, bn, KB // 32, 32)[:, :, :, idx]      # [tiles, bn, chunks, 32]
        out[e] = t.permute(0, 2, 1, 3).reshape(N, KB)
    return out


def _repack_scales(sf, bn=32):
    """[E, N, K/16] fp8 scales -> contiguous [E, N/bn, K/64, bn, 4] blocks, same byte count."""
    E, N, KS = sf.shape
    out = torch.empty_like(sf)
    for e in range(E):
        t = sf[e].view(N // bn, bn, KS // 4, 4)
        out[e] = t.permute(0, 2, 1, 3).reshape(N, KS)
    return out


def _experts_cuda(x, local_ids, g, st, top_k, count, I, e_off):
    """Small-batch path: split-K streaming W4A16 kernels on the raw NVFP4 weights, fp32 atomics."""
    M, H = x.shape
    dev = x.device
    pairs = M * top_k
    sorted_ids, expert_ids, n_post = moe_align_block_size(local_ids, 16, count)
    max_blocks = expert_ids.numel()
    splitk = 8 if M <= 8 else 4
    mod = _cuda_mod()
    w13, w13_sf, w2, w2_sf = st["raw_w13"], st["raw_w13_sf"], st["raw_w2"], st["raw_w2_sf"]
    acc1 = torch.zeros((pairs, 2 * I), dtype=torch.float32, device=dev)
    mod.w4a16_moe(x, w13[e_off:], w13_sf[e_off:], st["raw_w13_gs"][e_off:], sorted_ids, expert_ids, n_post, acc1,
                  pairs, top_k, max_blocks, True, splitk)
    inter = torch.empty((pairs, I), dtype=torch.bfloat16, device=dev)
    _silu_mul[(pairs,)](acc1, local_ids.view(-1), st["raw_w13_gs"][e_off:], inter, I=I, BLOCK=1024, num_warps=4)
    y = torch.zeros((pairs, H), dtype=torch.float32, device=dev)
    mod.w4a16_moe(inter, w2[e_off:], w2_sf[e_off:], st["raw_w2_gs"][e_off:], sorted_ids, expert_ids, n_post, y,
                  pairs, 1, max_blocks, False, splitk)
    # finalize applies the per-expert global scale of W2 through gs-scaled routing weights
    return y, st["raw_w2_gs"][e_off:]


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
    _cuda_mod()  # compile now (untimed)
    torch.cuda.synchronize()
    return {"banks": banks, "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
            "post_norm_w": weights["post_norm_w"].contiguous(),
            "raw_w13": _repack(weights["w13"]), "raw_w13_sf": _repack_scales(weights["w13_sf"]),
            "raw_w13_gs": weights["w13_gs"].float().contiguous(),
            "raw_w2": _repack(weights["w2"]), "raw_w2_sf": _repack_scales(weights["w2_sf"]),
            "raw_w2_gs": weights["w2_gs"].float().contiguous()}


def _experts_marlin(x, local_ids, topk_weights, bank, ws, top_k):
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    count = bank["count"]
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * top_k / count / block_size_m < 0.9:
            break
    block_size_m = max(block_size_m, 16) if num_tokens >= 16 else block_size_m  # measured: 16 beats 8 at bs 16
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
    if M <= CUDA_MAX_TOKENS:
        y, gs2 = _experts_cuda(x, local_ids.contiguous(), g, st, top_k, count, I, e_off)
        y = _apply_gs(y, local_ids, gs2)
    else:
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
    if M <= CUDA_MAX_TOKENS:
        y, gs2 = _experts_cuda(x, local_ids, g, state, cfg["top_k"], cfg["local_count"], cfg["d_expert"], cfg["local_start"])
        y = _apply_gs(y, local_ids, gs2)
    else:
        y = _experts_marlin(x, local_ids, g, bank, state["workspace"], cfg["top_k"])
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    _finalize[(M,)](y, local_ids, g, state["post_norm_w"], out, POST_NORM_EPS, H=H, TOP_K=cfg["top_k"],
                    BLOCK_H=triton.next_power_of_2(H), num_warps=8)
    return out
