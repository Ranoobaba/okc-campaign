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
- attention replica calibration: baseline vs itself x0.9965 (ok). Baseline FA3 path: sliding b1 23.7 us (floor 0.1:
  pure launch latency of 5 kernels), full b1 37 us (floor 4.3), full b16 213 us (floor 63), full b32 399 us
  (floor 126, 3.2x off). Sliding shapes carry weight 46 vs 16, so a fused single-launch decode kernel is
  the whole game on decode-1 and the split-KV efficiency matters on b16/b32 full.

## Real judge vs replica (2026-10-25 04:05)

- proj v2: real judge x1.1075 [1.091, 1.124] vs replica x1.1019. The replica is faithful; trust it for keep/discard.
- The real judge returns a per-shape kernel profile (records event=profile) and 60 paired ratios per shape. Save every
  job JSON under results/<problem>/judge_<id>.json; it is free intel.
- Judge-timed shape time is roughly kernel GPU time + ~5 us of fixed per-launch cost inside the graph (our m=1
  qkv_gate gemv is 10.7 us of GPU time but times at ~16.7 us). So on tiny shapes, launch count is the lever:
  one kernel per shape, and no zero-fill or cast launches.
- prepare() took 134 s on the judge (our tuning loop); keep it under ~200 s.

## Attention decode findings (2026-10-25 07:20)

- Decode-1 sliding (2048 keys) costs ~15.5 us in our 3-launch kernel regardless of BLOCK_N (32/64/128), num_stages
  (1/2/3), split count (x4) or masking: fixed per-launch latency dominates, not tile work. Baseline = 5 launches
  = 24 us. The lever is launch count: one launch (in-kernel combine via an atomic counter) is the target.
- Fusing prep into the attention launch (v4) saved nothing on its own at 3 -> 2 launches for decode shapes and
  slowed the 64-row verify blocks (q-norm/RoPE recomputed per program, register pressure). Fuse only for 16-row blocks.
- 64-row (q_len 8) programs over full context are 3x slower than FA3; more warps (8) made it worse, per-token
  row split is 8x KV re-read. Still unexplained; candidates: fp8 trans layout conversions per tile, exp cost,
  or latency chains with too few programs per SM. Needs an ncu pass.
- FP8 x FP8 QK dot (queries rounded to E4M3) is accepted by the checker with ~4 to 7% error against a 11 to 15% bound.

## proj: what does not beat cuBLAS at m 8..256 (2026-10-25 06:40)

- Triton tl.dot skinny GEMM (any tile), Triton TMA-streamed GEMM, cuBLASLt heuristic algo search, and a hand
  CUDA mma.sync kernel (4 warps split K, cp.async 3 stages): all at or below cuBLAS. The CUDA kernel measured
  1.5x slower than cuBLAS (33.6 vs 22 us at m 16 qkv_gate): too little data in flight per SM.
- Judge-timed shape time ~= kernel GPU time + ~4.5 us per kernel launch. cuBLAS split-K = 2 kernels at ~4.7 TB/s.
  A single-kernel GEMM at >= 4.5 TB/s would give ~1.5x on m 8..32 shapes; that is the bar.
- The router (fp32 weights) is where we win: hi/lo bf16 Triton dot at 1.6 to 2.8x.

## attention: single launch needs an unrolled combine (2026-10-25 06:40)

- v5 (one launch, last split merges) was slow because the merge loop used a runtime `range` over splits: every
  partial load serialized (~1 us each). Use tl.static_range and cap splits so the unrolled loop stays small.

## ncu on the MoE CUDA kernel v4b, bs 1 (2026-10-25 06:55)

- Works on Modal H200 with `ncu --clock-control none` (clock locking is refused). Probe: harness/ncu_probe.py.
- GEMM1: 90 us, DRAM 13.8%, L1/TEX 58%, SM 26%, achieved occupancy 12% (7.7 warps/SM), 0.43 waves, 1.93 active
  warps per scheduler, 4.1 cycles per issued instruction, 56 regs, 20.8 KB static smem. Diagnosis: far too few
  resident warps and an LDS-heavy inner loop (1-byte scattered smem loads for the packed weights).
- Fix direction: 8 rows per warp (BN 32, 4 warps, ~1000 blocks), 4 stages (18 KB smem -> 11 blocks/SM), and a
  lossless repack so each lane's 8 bytes per chunk are contiguous (one 64-bit LDS instead of 8 byte loads).
- v7 (one launch, unrolled merge): sliding b1 12.8 us = identical to the 3-launch v6. The time is ~6 dependent
  global round trips, not launch count. Next: no split on sliding layers (one program per (b, h) streams the whole
  2048-key window with pipelined tile loads and writes the output itself): 4 round trips.
- ncu on the proj CUDA skinny kernel (m 16 qkv_gate): 26.5 us, DRAM 31.6% (1.5 TB/s), 144 blocks = 0.36 waves,
  achieved occupancy 7%, 1.1 active warps per scheduler. Same disease as the MoE kernel: too few warps and bytes
  in flight. For a 16-row-per-block streaming GEMM the fix is many K-split warps per block (16 warps, ~200 KB smem).
- ncu on v4f (split-K 4, contiguous tiles): GEMM1 64 us, SM throughput 69%, DRAM 19%, 4.3 active warps/scheduler,
  issue 0.74/cycle: now instruction-throughput bound (~150 warp-instructions per 64-k chunk per warp). Marlin at
  bs 1 is latency/occupancy bound instead (46 us for ~56 MB). To beat it: halve the instruction count (8-nibble
  prmt dequant ~17 ops per 8 values, SIMD scale conversion, hoisted cp.async pointers) and split K 8 ways.
- v4g ncu (bs 1): GEMM1 44 us (SM 66%, DRAM 28%, occupancy 42%), GEMM2 28 us. Still instruction-bound at roughly
  100 to 140 warp-instructions per 64-k chunk per warp; the prmt dequant (1.5/value), scale conversion and copy
  issue dominate. Realistic ceiling of this design ~25 us per GEMM = parity with Marlin + fused finalize at bs 1.
  Parked; the layer's launch count is the cheaper lever now (~22 launches x ~4.5 us fixed = ~100 of 155 us at bs 1).
