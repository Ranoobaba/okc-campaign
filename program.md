# Open Kernel Challenge campaign program

This file is authoritative for every agent session in this repo.
Read it first, then `LEARNINGS.md`, then the tail of `CAMPAIGN.md`.

## The game

- Site: https://openkernelchallenge.dev (spec cached in `spec/`, rules in `spec/llms.txt`).
- Kernel track closes Nov 4, 11:59 pm PT. Serving track (the $100k) runs Nov 5 to 19.
- The top 50 of the overall kernel board get serving queue priority. That is the first objective.
- Overall board = the four decode problems together (`muse_bf16_proj`, `muse_attention_decode`,
  `muse_moe_w4a16`, `muse_decode_layer`). `muse_prefill_layer` has its own board only.
- Score per problem = geomean over loads of (SGLang load time / ours). Final ranks come from the
  organizers' reruns (medians), so draw-farming does nothing here. Only real kernel speed counts.
- Per problem per day (UTC reset): 10 scored runs, 20 check runs. Same code twice returns the old result.
- One Python file per submission defining `kernel_fn` (and `prepare` where there are weights).
  First line must be `# okc: problem=<name>`.

## Hard rules (disqualification; never bend them)

- Capturable in a CUDA graph: no `.item()`, `.cpu()`, sync, printing tensor values, no
  allocation or branching on input values. Shapes and `config` may be branched on.
- Fresh inputs every call: never cache anything derived from `data`. No output replay.
- Weights keep their precision (NVFP4 stays NVFP4, BF16 stays BF16; lossless repacking is fine;
  router FP32 may be a BF16 hi+lo pair with FP32 accumulation). Activations BF16 where SGLang
  keeps BF16; KV cache FP8; queries may be FP8 E4M3 and no coarser.
- Attention attends every key in the window / context. No top-k, sparse, or skipping.
- No patching timing, no harness introspection, no subprocesses, no network, no imports of the reference.
- If something looks like a reward hack, it is. Withdraw it (`okc withdraw <job>`).

## Tools

- Local judge replica (Modal H200, official image): `modal run harness/judge.py --problem P --submission F`.
  It is the main development loop. It runs the tests, then times every scored shape exactly like the
  judge (graph replay, cold L2, paired ratios) and prints per-shape speedups with the roofline floor.
  Flags: `--reps N`, `--only "decode 1"`, `--skip-tests`, `--profile`, `--no-bench`, `--tag name`.
  A run costs about $0.30 to $1 of Modal credit. Use it freely but not wastefully.
- Board: `python3 harness/board.py` (snapshots to `snapshots/`). Run it at the start of every session.
- Submissions: `python3 harness/okc_api.py check|submit P FILE`, `status JOB --wait`, `limits`.
  Needs a token: `okc login` once by a person, or `OKC_TOKEN`.
- Modal CLI: `~/fib-moe-loop/.venv/bin/modal`.

## The loop (one experiment per session)

1. `python3 harness/board.py`. Note rank changes in CAMPAIGN.md.
2. Pick ONE problem and ONE idea from the queue in CAMPAIGN.md (evidence first, then leverage).
3. Make one coherent change to `problems/P/submission.py` (keep old versions in `problems/P/candidates/`).
4. Run the local judge. Correctness first (`--no-bench` while iterating), then timing.
5. Keep/discard rule: keep if the geomean improves beyond the reported interval on the loads
   that matter, with no test or check failure, and no shape regressing more than the gain.
6. Append one row to `results.tsv` and a dated entry to CAMPAIGN.md (what, why, number, verdict).
   Negative results are kept; they prune the search.
7. Promote to the real judge only when the local replica says it beats our current board entry:
   `okc_api.py check` first (free of attempts), then `submit`. Record job ids and results.
8. Write durable facts to LEARNINGS.md. Exit.

## Priorities

Leverage on the overall board follows SGLang's time per decode step:
decode_layer 22.6 ms > moe 13.4 ms > proj 4.8 ms > attention 3.8 ms. But the decode layer is
composed of the other three, so build the components first and assemble them in the layer.
Order: proj (fast to win, feeds the layer) -> attention decode (biggest headroom, 2.9x at #1) ->
moe (decode shapes are weight-streaming bound) -> decode_layer assembly -> prefill_layer last.
