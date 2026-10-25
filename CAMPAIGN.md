# Open Kernel Challenge campaign ledger (append only)

Account: GitHub Ranoobaba. Kernel track closes 2026-11-04 23:59 PT.
Targets: all four decode boards entered; overall top 50 (serving priority); then climb each board.

## 2026-10-25 03:00 PT: setup

- Installed okc 0.7.1 (~/.okc/bin). NOT logged in yet: a person must run `okc login` (rules + GitHub device flow).
- Cached every problem's spec/baseline/reference/tests under problems/, boards under snapshots/.
- Board at setup: proj #1 1.610 / #50 1.444; attention #1 2.910 / #50 2.185; moe #1 2.523 / #50 2.126;
  decode_layer #1 2.285 / #50 1.885; prefill #1 1.888 / #50 1.626. 1077 teams overall. Queue idle (0 backlog, 112 GPUs).
- Built harness/judge.py (Modal H200 replica of the judge) and harness/board.py. Calibration runs in flight.

## Queue (ordered)

1. proj: Triton weight-streaming skinny GEMM, per-(m, layer) configs tuned in prepare(); router as fp32 GEMV
   (m<=8) and bf16 hi+lo dot (m>8); split-K only where blocks < 2 per SM.
2. attention: fused decode kernel (qk-norm + RoPE + cache store + split-KV flash-decode + combine with
   rms*sigmoid(gate)); FP8 K/V gathered through the page table; q_len 8 causal block; dp heads path.
3. moe: dispatch (count/scan/scatter) + NVFP4 dequant GEMM1 with fused SiLU + GEMM2 with fused
   post-norm weighted sum; decode shapes as weight-streaming.
4. decode_layer: assemble 1 to 3 with SGLang's fused norm/residual kernels (they are in the image).
5. prefill_layer: CUTLASS/CuTe mixed-input WGMMA grouped GEMM vs Marlin.

## 2026-10-25 04:40 PT: first kernels through the replica

- Replica calibrated on proj (x1.0036), moe (x1.0012), attention (x0.9965): trustworthy.
- proj v1 (Triton skinny GEMM, tuned in prepare): x1.0086. Router wins big (1.3 to 2.7x), m=1 gemv 1.1 to 1.3x,
  but tl.dot path at m>=8 loses to cuBLAS (down to 0.6x at m=128). DISCARD.
- proj v2 (same + per-(layer, m) race against cuBLAS in prepare, 3% margin): x1.1019, all tests pass. KEEP.
  Headroom: m=1 gemv only reaches ~2.2 TB/s; needs a proper bandwidth GEMV (split-K or CUDA) to reach 1.6x+.
- moe v1 compile error (Triton global constant); fixed, rerunning. attention v1 same bug; fixed, rerunning.
- Devpost/product subagent (Opus) running in background for the Cal Hacks 8:30 am deadline.

## 2026-10-25 05:10 PT: logged in, first real submission, attention/moe first results

- Rayyan authorized the device flow and accepted the rules (version 2026-10-22.1, email yes) at 03:59 PT. Token in .okc_token.
- proj v2 SUBMITTED scored: job f35dd5d974a44531 (first board entry; result pending).
- moe v1: correct (5/5 tests, 5e-3 error) but x0.022: per-element duplicated byte loads along contiguous K
  serialize into scalar loads. v2 (vectorized [BN, BK/2] byte tile, even/odd split dots, scale expansion via reshape) running.
- attention v1: 16/16 tests pass with ~1e-3 error (very accurate). x0.96 geomean: sliding 1.3 to 1.9x, full decode
  1.0 to 1.2x, verify full 0.31 to 0.68 (fp8->f32->bf16 conversions on 64-row tiles). v2 = fp8 x fp8 QK dot with
  scales folded into scalars, running. Next: fuse prep into attn (2 launches) for the latency-bound sliding shapes.
- Product subagent done: name Pitcrew, files in ~/inference-product (RESEARCH.md, DEVPOST.md, site/). Devpost
  auto-fill blocked (Chrome remote debugging off); Rayyan pastes or enables chrome://inspect remote debugging.

## 2026-10-25 05:35 PT: first board entry

- proj v2 on the REAL judge: x1.1075 [1.091, 1.124], rank #824/932 on muse_bf16_proj; overall #1019 (solved 1/4).
  Replica predicted 1.1019: calibration holds.
- moe v2 (vectorized loads): correct, x0.40 (175 us at bs 1 vs 85). Profiling the bs=1 shape to see gemm1/gemm2/finalize split.
- decode_layer v1 assembled (proj + attention + moe components + SGLang fused norms); first replica run in flight.

## 2026-10-25 06:50 PT: two boards entered

- moe v3 (Marlin + fused finalize) REAL JUDGE x1.1338 [1.117, 1.151] (job 9c1b770f3b684916). Replica said 1.152.
- decode_layer v1 assembled from components: 13/13 tests pass. Speed 0.59 only because it carried the slow Triton
  MoE; rebuilding with moe v3. Attention component still v2 (sliding 1.3 to 1.9x, verify full 0.3 to 0.7x).
- Triton NVFP4 dequant GEMM parked: v2 0.40x, v2b 0.35x. ALU-bound dequant; the path to beat Marlin at small M is a
  CUDA kernel with lop3/prmt dequant and mma.sync, or a smarter lossless repack.
- Attention diagnostics running: 8-variant microbench on verify/decode shapes; v4 fuses prep into the attention launch.

## 2026-10-25 04:50 PT: three boards entered (note: earlier timestamps in this file were written in UTC-ish error; local time is PDT)

- attention v6 REAL JUDGE x1.1813 [1.164, 1.199], #588/692. Replica 1.221 (slightly optimistic on this problem).
- Board: proj #826 (1.107), attention #588 (1.181), moe #700 (1.134). Overall #705/1081, solved 3/4.
- decode_layer v2 (attention v2 + moe v3) replica x1.0935, 13/13 tests; v3 with attention v6 running; submit on pass.
- Attention latency findings in LEARNINGS.md: launch fusion did not help; the per-kernel dependency chain does.

## 2026-10-25 05:15 PDT: all four decode boards entered

- decode_layer v3 REAL JUDGE x1.1220 [1.105, 1.139] (job fa8d57ba9f2746ca), #572/636. Overall #693/1083, solved 4/4.
- Board: proj #827 (1.107), attention #588 (1.181), moe #700 (1.134), layer #572 (1.122). Top 50 needs
  roughly proj 1.46, attention 2.19, moe 2.13, layer 1.89.
- proj v3 (TMA-streamed Triton skinny GEMM configs): x1.1023, no gain over cuBLAS. DISCARD.
- Real-judge profiles show ~5 us of fixed cost per kernel inside the timed graph: cuBLAS split-K (2 kernels) at
  m 8..32 pays it twice. Next proj experiment: per-shape cuBLASLt algorithm search favouring single-kernel algos.
- MoE Marlin variant sweep (atomic add, fp32 reduce, block sizes, workspace) running on the microbench.

## 2026-10-25 07:10 PDT: morning summary for Rayyan

Board (real judge): proj #828 x1.107, attention #591 x1.181, moe #705 x1.134, decode_layer #574 x1.122.
Overall #697/1085 with all four decode problems solved. Prefill not entered (no gain over baseline yet).

What was tried overnight after the four entries (all on the replica, all discarded unless noted):
- proj: Triton TMA configs (v3), cuBLASLt algo search (v4), CUDA skinny GEMM 4 warps (v5) and 8 K-split warps (v6):
  none beat cuBLAS at m 8..256. ncu: our CUDA kernel runs 1.5 TB/s at 7% occupancy. Working: v2 (x1.10).
- attention: fused prep (v4), one launch with atomic merge (v5, v7), unsplit window kernel (v8), tile prefetch +
  4x splits (v9): the sliding shapes are bound by ~5 dependent memory round trips, not launches; full-context shapes
  regress with register pressure. Working: v6 (x1.18 real).
- moe: CUDA W4A16 streaming kernel v4..v4h: correct from the first run, improved from 159 us to 78 us at bs 1
  (beats the SGLang baseline's 85 us) but still behind Marlin + fused finalize (62 us); ncu shows it is
  instruction-bound at ~3.5 instr/weight, parity with Marlin is the ceiling of the design. Parked at v4g.
  Working: v3b (Marlin + fused finalize + atomic-add at bs<=8, x1.16 replica; v3 is on the board at x1.134).
- decode_layer: v4 removes two launches (fused local-ids kernel, unpooler straight into a zeroed buffer); running.

Where the next real gains are (needs CUDA at FA3/Marlin quality, multi-day): attention full-context decode is 3x off
roofline for everyone including FA3; MoE needs <2 instr/weight dequant; the layer needs ~10 launches, not ~20.
Cheap remaining items: layer launch cuts, Marlin block-size tuning per shape, resubmitting v3b moe (+2%).
