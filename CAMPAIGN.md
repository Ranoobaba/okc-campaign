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
