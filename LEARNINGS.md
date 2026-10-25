# LEARNINGS

Durable facts across sessions. Read before designing anything.

## Judge and environment (verified 2026-10-25)

- Official image `lmsysorg/sglang:dev-muse-spark-open` pulls fine on Modal (`gpu="H200"`).
  torch 2.14.1+cu130, triton 3.8.0, flashinfer 0.7.0.post1, sgl_kernel 0.4.9, deep_gemm
  0.2.1.post1+muse, tilelang 0.1.12, cutlass 4.8.0, nvcc 13.0, python 3.12. `ncu` and `nsys` present.
  H200 SXM 141 GB, SM clock 1980 MHz max.
- Judge timing = CUDA graph replay, cold L2 before each replay, paired A/B ratios, ~4 us replay
  cost subtracted. Every timed call is correctness-checked against the reference on fresh inputs.
- Scored shapes are public (`problems/*/meta.json`), including seeds. Tests are listed in
  `problems/*/tests.json` (seeds there are ours; the judge's differ).
- Final positions are medians of organizer reruns, on scored and nearby shapes. Kernels must be
  robust on nearby shapes too (no overfitting to the exact m, ctx).
- Overall board requires all four decode problems solved (every top row has solved=4).

## Problem intel

- proj: m in {1,8,16,32,128,256}; weights qkv_gate 2304x8192, o 8192x1024, pooler 4096x8192,
  unpooler 8192x4096 (only 1024 rows at m<=8), router 256x8192 fp32 (tol 1e-4, probe rows 1e-3).
  All decode shapes are HBM-bound weight streaming; SGLang (cuBLAS) is ~1.6x off at the top of the board.
- attention: 16 q heads / 2 kv heads / d 64, FP8 cache, page_size 1 in scored cases (64 in tests),
  ctx ~80K (+-7.5% agentic spread), sliding window 2048 with interleaved RoPE, full layers no RoPE,
  q_len 1 or 8. Sliding shapes carry weight 46 vs 16 for full; sliding floor is ~0 so launch count
  and latency dominate decode-1 loads. Tests include heads "dp" (128/16), ctx up to 1M, kv_scale 1/64,
  window-edge keys. Tolerance 3 to 15% rel L2; FP8 queries allowed.
- moe: 32 local experts of 256, H=I=4096, NVFP4 (E2M1 + fp8 e4m3 block-16 scales + fp32 global),
  top-8, M in {1,8,16,32,128,256}, output = sum_g g * rmsnorm((y*(1+w)).bf16()). Tol 1.5%.
  Every scored shape is weight-read bound (28 MB per touched expert). Baseline = Marlin W4A16.
- decode_layer: composition of the above plus fused norms/residuals/router; outputs
  (hidden bf16, stream fp32, topk_ids int32) and in-place cache writes. Router tolerance via
  routing check (ties within 0.05 allowed). Baseline uses a side stream for the router.
- prefill_layer: same layer at q_len 753 / 16384; grouped NVFP4 GEMM dominates (compute-bound);
  separate board, no overall credit.

## Judge replica calibration (2026-10-25 03:40)

- Baseline vs itself: proj x1.0036, moe x1.0012. Per-shape noise about +-2 to 3% at 10 reps; use >=20 reps
  for keep decisions under 5%. The replica is trustworthy.
- proj: SGLang/cuBLAS vs HBM floor at m=1: qkv_gate 18.3 vs 7.9 us, o 9.7 vs 3.5, pooler 29 vs 14,
  unpooler(1024 rows) 9.4 vs 1.7, router 13.0 vs 1.8 (fp32 F.linear), router hi/lo at m=8: 29.4 us (!).
  m=128/256 cuBLAS is only 1.5 to 2x off. Biggest single wins: router at every m, unpooler slice, o.
- moe: Marlin baseline bs1 85 us, bs8 161, bs16 190, bs32 225, bs128 446, bs256 631 us. Weight-read floor
  with the real routing (TASK.md) 11/37/53/71/101/109 us. Everything is 4 to 8x off roofline. #1 on the
  board (2.52x) is still ~3x off the floor. This is the largest lever on the overall board.
- Image has deep_gemm (muse fork) with nvfp4_mega_moe, m_grouped_fp4_gemm_nt_contiguous/_masked,
  fp8_fp4 variants, cublaslt_nvfp4_gemm_nt; flashinfer 0.7 with cute_dsl_fused_moe_nvfp4, trtllm_fp4_block_scale_moe
  (sm100 only, likely useless on H200), grouped_mm_fp4; sgl_kernel with cutlass_w4a8_moe_mm, ggml_moe_a8_vec
  (GEMV-style MoE for small M). Check which of these run on sm_90 before writing custom kernels.
- attention harness bug fixed (check must read the cache the kernel wrote). Recalibration in flight.
