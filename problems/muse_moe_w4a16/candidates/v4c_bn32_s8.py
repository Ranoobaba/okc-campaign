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
#define STAGES 8
#define THREADS 64
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

// Four nibbles (bits 0-3, 4-7, 8-11, 12-15 of n4) -> two bf16x2 words: (v0,v1), (v2,v3).
__device__ __forceinline__ void dequant4(uint32_t n4, uint32_t& w01, uint32_t& w23) {
    uint32_t mag = n4 & 0x7777u;                 // 3-bit magnitude codes in each nibble
    // selector for prmt: byte i of result = pool byte sel_i (sel nibbles at bits 0,4,8,12)
    uint32_t sel = (mag & 0x7u) | ((mag >> 4) & 0x7u) << 4 | ((mag >> 8) & 0x7u) << 8 | ((mag >> 12) & 0x7u) << 12;
    uint32_t lo = __byte_perm(LO_POOL_A, LO_POOL_B, sel);   // low bytes of v0..v3
    uint32_t hi = __byte_perm(HI_POOL_A, HI_POOL_B, sel);   // high bytes of v0..v3
    w01 = __byte_perm(lo, hi, 0x5140);   // bytes: lo0 hi0 lo1 hi1
    w23 = __byte_perm(lo, hi, 0x7362);   // bytes: lo2 hi2 lo3 hi3
    uint32_t sg = n4 & 0x8888u;          // sign bits at nibble bit 3
    w01 ^= ((sg & 0x8u) << 12) | ((sg & 0x80u) << 24);
    w23 ^= ((sg & 0x800u) << 4) | ((sg & 0x8000u) << 16);
}

__device__ __forceinline__ uint32_t bf16x2_scale(uint8_t sf) {
    // E4M3 -> BF16, duplicated into both halves.
    __half h = __half(__nv_cvt_fp8_to_halfraw(sf, __NV_E4M3));
    float f = __half2float(h);
    __nv_bfloat16 b = __float2bfloat16(f);
    uint32_t u = reinterpret_cast<uint16_t&>(b);
    return u | (u << 16);
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
    __nv_bfloat16* __restrict__ C, long c_stride,                   // [pairs, N]
    int num_valid, int top_k, int K, int N_out) {
    const int mb = blockIdx.x;
    if (mb * 16 >= *n_post_pad) return;
    const int e = expert_ids[mb];
    if (e < 0) return;
    const int n0 = blockIdx.y * BN;
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
    const int nK = K / BK;

    auto load_stage = [&](int stage, int kc) {
        // x: 16 rows x 8 chunks of 16B
        for (int c = tid; c < 16 * 8; c += THREADS) {
            int r = c >> 3, cc = c & 7;
            int row = rows_s[r];
            const __nv_bfloat16* src = row >= 0 ? A + (long)row * a_stride + kc * BK + cc * 8 : A;
            cp_async16(&xs[stage][r][cc * 8], src, row >= 0);
        }
        // w: BN rows x 2 chunks of 16B per matrix
        for (int c = tid; c < BN * 2 * (GATED ? 2 : 1); c += THREADS) {
            int mat = c / (BN * 2), rem = c % (BN * 2), r = rem >> 1, cc = rem & 1;
            const uint8_t* src = wbase + (long)((mat ? N_out : 0) + n0 + r) * w_stride_n + kc * (BK / 2) + cc * 16;
            cp_async16(&ws[stage][mat][r][cc * 16], src, true);
        }
        // scales: BN rows x 4B per matrix
        for (int c = tid; c < BN * (GATED ? 2 : 1); c += THREADS) {
            int mat = c / BN, r = c % BN;
            const uint8_t* src = sbase + (long)((mat ? N_out : 0) + n0 + r) * s_stride_n + kc * (BK / 16);
            unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(&ss[stage][mat][r][0]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(d), "l"(src));
        }
        cp_async_commit();
    };
    auto load_scales = [&](int, int) {};

    float acc[GATED ? 2 : 1][2][4];
    #pragma unroll
    for (int m = 0; m < (GATED ? 2 : 1); m++)
        #pragma unroll
        for (int f = 0; f < 2; f++)
            #pragma unroll
            for (int i = 0; i < 4; i++) acc[m][f][i] = 0.f;

    // prologue
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nK) { load_stage(s, s); load_scales(s, s); } else { cp_async_commit(); }
    }

    for (int kc = 0; kc < nK; kc++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        int nxt = kc + STAGES - 1;
        if (nxt < nK) { load_stage(nxt % STAGES, nxt); load_scales(nxt % STAGES, nxt); } else { cp_async_commit(); }
        const int st = kc % STAGES;
        // this warp's 16 output rows: n = n0 + warp*16 + {0..15}; two n-frags of 8
        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            uint32_t a[4];
            // ldmatrix x4: lanes 0-15 rows 0-15 at col ks*16, lanes 16-31 rows 0-15 at col ks*16+8
            ldmatrix_x4(a, &xs[st][lane & 15][ks * 16 + (lane >> 4) * 8]);
            #pragma unroll
            for (int mat = 0; mat < (GATED ? 2 : 1); mat++) {
                #pragma unroll
                for (int f = 0; f < 2; f++) {
                    int nrow = warp * 16 + f * 8 + (lane >> 2);
                    int kb = ks * 8 + (lane & 3);            // byte index of k pair (ks*16 + (lane&3)*2)
                    uint8_t b0 = ws[st][mat][nrow][kb];      // k = ks*16 + (lane&3)*2 + {0,1}
                    uint8_t b1 = ws[st][mat][nrow][kb + 4];  // k + 8
                    uint32_t n4 = (uint32_t)b0 | ((uint32_t)b1 << 8);
                    uint32_t w01, w23;
                    dequant4(n4, w01, w23);
                    uint32_t sc = bf16x2_scale(ss[st][mat][nrow][ks]);
                    uint32_t b[2] = {hmul2_bf16(w01, sc), hmul2_bf16(w23, sc)};
                    mma_bf16(acc[mat][f], a, b);
                }
            }
        }
    }
    cp_async_wait<0>();

    // epilogue: C frag -> rows (lane>>2) and +8, cols frag*8 + (lane&3)*2 + {0,1}
    const float gs = GS[e];
    #pragma unroll
    for (int f = 0; f < 2; f++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            int r = (lane >> 2) + h * 8;
            int pid = sorted_ids[mb * 16 + r];
            if (pid >= num_valid) continue;
            int n = n0 + warp * 16 + f * 8 + (lane & 3) * 2;
            float v0, v1;
            if (GATED) {
                float g0 = acc[0][f][h * 2] * gs, g1 = acc[0][f][h * 2 + 1] * gs;
                float u0 = acc[1][f][h * 2] * gs, u1 = acc[1][f][h * 2 + 1] * gs;
                v0 = g0 / (1.f + __expf(-g0)) * u0;
                v1 = g1 / (1.f + __expf(-g1)) * u1;
            } else {
                v0 = acc[0][f][h * 2] * gs;
                v1 = acc[0][f][h * 2 + 1] * gs;
            }
            __nv_bfloat162 o = __floats2bfloat162_rn(v0, v1);
            *reinterpret_cast<__nv_bfloat162*>(C + (long)pid * c_stride + n) = o;
        }
    }
}

void w4a16_moe(torch::Tensor A, torch::Tensor W, torch::Tensor SF, torch::Tensor GS, torch::Tensor sorted_ids,
               torch::Tensor expert_ids, torch::Tensor n_post_pad, torch::Tensor C, int64_t num_valid, int64_t top_k,
               int64_t max_blocks, bool gated) {
    int K = A.size(1);
    int N_out = C.size(1);
    dim3 grid(max_blocks, N_out / BN);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (gated) {
        w4a16_kernel<true><<<grid, THREADS, 0, stream>>>(
            (const __nv_bfloat16*)A.data_ptr(), A.stride(0), (const uint8_t*)W.data_ptr(), W.stride(0), W.stride(1),
            (const uint8_t*)SF.data_ptr(), SF.stride(0), SF.stride(1), (const float*)GS.data_ptr(),
            sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(), n_post_pad.data_ptr<int32_t>(),
            (__nv_bfloat16*)C.data_ptr(), C.stride(0), (int)num_valid, (int)top_k, K, N_out);
    } else {
        w4a16_kernel<false><<<grid, THREADS, 0, stream>>>(
            (const __nv_bfloat16*)A.data_ptr(), A.stride(0), (const uint8_t*)W.data_ptr(), W.stride(0), W.stride(1),
            (const uint8_t*)SF.data_ptr(), SF.stride(0), SF.stride(1), (const float*)GS.data_ptr(),
            sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(), n_post_pad.data_ptr<int32_t>(),
            (__nv_bfloat16*)C.data_ptr(), C.stride(0), (int)num_valid, (int)top_k, K, N_out);
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
                        " int64_t max_blocks, bool gated);",
            cuda_sources=_CUDA_SRC, functions=["w4a16_moe"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"],
            verbose=False)
    return _cuda


CUDA_MAX_TOKENS = 32  # above this, Marlin's larger M tiles win


def _experts_cuda(x, local_ids, g, st, top_k, count, I, e_off):
    """Small-batch path: our streaming W4A16 kernels on the raw NVFP4 weights."""
    M, H = x.shape
    dev = x.device
    pairs = M * top_k
    sorted_ids, expert_ids, n_post = moe_align_block_size(local_ids, 16, count)
    max_blocks = expert_ids.numel()
    inter = torch.empty((pairs, I), dtype=torch.bfloat16, device=dev)
    mod = _cuda_mod()
    w13, w13_sf, w2, w2_sf = st["raw_w13"], st["raw_w13_sf"], st["raw_w2"], st["raw_w2_sf"]
    mod.w4a16_moe(x, w13[e_off:], w13_sf[e_off:], st["raw_w13_gs"][e_off:], sorted_ids, expert_ids, n_post, inter,
                  pairs, top_k, max_blocks, True)
    y = torch.empty((pairs, H), dtype=torch.bfloat16, device=dev)
    mod.w4a16_moe(inter, w2[e_off:], w2_sf[e_off:], st["raw_w2_gs"][e_off:], sorted_ids, expert_ids, n_post, y,
                  pairs, 1, max_blocks, False)
    return y


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
            "raw_w13": weights["w13"].contiguous(), "raw_w13_sf": weights["w13_sf"].contiguous(),
            "raw_w13_gs": weights["w13_gs"].float().contiguous(),
            "raw_w2": weights["w2"].contiguous(), "raw_w2_sf": weights["w2_sf"].contiguous(),
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
        y = _experts_cuda(x, local_ids.contiguous(), g, st, top_k, count, I, e_off)
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
        y = _experts_cuda(x, local_ids, g, state, cfg["top_k"], cfg["local_count"], cfg["d_expert"], cfg["local_start"])
    else:
        y = _experts_marlin(x, local_ids, g, bank, state["workspace"], cfg["top_k"])
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    _finalize[(M,)](y, local_ids, g, state["post_norm_w"], out, POST_NORM_EPS, H=H, TOP_K=cfg["top_k"],
                    BLOCK_H=triton.next_power_of_2(H), num_warps=8)
    return out
