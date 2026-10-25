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
