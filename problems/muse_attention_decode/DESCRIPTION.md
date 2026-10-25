One decode step, or one speculative-decoding verify block, of Muse Spark Open's attention
for a batch of requests in a paged FP8 KV cache, on one H200: from the projections'
outputs to the input of the output projection.

Muse uses grouped-query attention (128 query and 16 key/value heads of 64; 16 and 2 per
GPU under TP8, as SGLang serves it on 8xH200) with two layer types. Three of every four
layers attend over a 2,048-token sliding window with RoPE; every fourth is full attention
over the whole context, with no RoPE, and the context runs to 1M tokens. Queries and keys
are RMS-normalized per head (no weight; queries then scaled by 5.473), and the attention
output is normalized per head and gated by sigmoid(gate). The KV cache is FP8 E4M3, as
SGLang uses for Muse on every GPU, with per-tensor scales: 1 in the scored cases (SGLang
leaves the released checkpoint's unset), 1/64 in some tests, so apply them.

The inputs are what SGLang's attention layer receives: `qkv` is the fused projection
output, [B, L, 2304] per token q | k | v | gate (16, 2, 2 and 16 heads of 64 under tp8;
128, 16, 16 and 128 heads in the dp tests), and the
step's metadata, which SGLang builds once per step for every layer: `positions` (each new
token's position), `cache_loc` (its flat cache row), `cache_seqlens` (seq_lens + L),
`cu_seqlens_q` (0, L, 2L, ...) and, on full-attention decode steps, `scheduler_metadata`
(FA3's tile schedule from get_scheduler_metadata, None otherwise).

Each step: normalize the queries and the block's new keys, apply RoPE on sliding layers
(interleaved pairs, as SGLang applies it to Muse), write the new keys and values into the
cache (FP8, at the rows in cache_loc: positions seq_lens[b] onwards), attend with scale 1/8 (each
token sees the cache and the block's tokens up to itself, within the window on sliding
layers), then return rmsnorm(attention) * sigmoid(gate) as bfloat16 [B, L, Hq*64]. L is 1
for a decode step and 8 for a verify block (SGLang's default serving mode drafts 8 tokens
per step). The scored cases use page size 1, one token per page, as SGLang serves Muse on
H200; some tests use 64-token pages, so read config["page_size"].

Correctness: relative L2 error vs. an FP32 reference (BF16 queries) within the case's
bound: 2.5x the reference's own error when its queries are rounded to FP8 E4M3, plus
0.015, between 3% and 15% (15%, the cap, on the scored sliding-window cases and the sharp
tests; about 8-11% on the scored full-attention cases). Queries may be attended in E4M3,
as FA3 does; nothing coarser. The new cache entries must be within 8e-2 of the
reference's, and since FP8 entries round differently by implementation, the output is
checked against the reference attending your own new entries.
Some tests plant keys at the window's edges, so an off-by-one window fails.
The scored shapes, and some tests, give each KV head's cached keys a shared direction, as
Muse's real keys have, so a query's weight spreads as in the real model: over tens of keys
in a sliding window, hundreds to thousands on a full layer at long context (not one-hot,
as i.i.d. keys would make it).
On full layers the scored shapes also give the values of each head's low-attention keys
a shared direction (as attention sinks do), so that long tail counts. The rules forbid
skipping values by their attention weight, and final review checks for it.

Shapes follow SGLang's H200 profile and LMSYS's agentic coding workload (contexts around
80K tokens): decode steps of 1, 16 and 32 requests and 8-token verify blocks for the
same, each on a sliding-window and a full-attention layer. 32 is the most requests the
serving recipe runs at once: what its KV pool holds at these contexts.
