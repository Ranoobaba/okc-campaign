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
