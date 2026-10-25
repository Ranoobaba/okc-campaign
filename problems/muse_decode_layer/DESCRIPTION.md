One whole decoder layer of Muse Spark Open, as one H200 runs it when SGLang serves the
model on 8xH200 (TP8/EP8): everything the other three problems cover (the projections,
the routed experts, attention) and everything between them: QK-norm and RoPE, the cache
write, the gated output norm, both fused norm-residual boundaries, the router and its
top-8 routing, the pre- and post-expert norms. In SGLang's H200 profile that glue alone
is a quarter of a decode step with one request; this is the problem that rewards fusing
across ops.

Everything except cross-GPU communication. On 8 GPUs the output projection and the
experts end in an all-reduce, and up to 8 tokens the unpooler in an all-gather; here each
is the identity on this GPU's partial result, as if the other seven GPUs contributed
nothing. The partial sums pass through unchanged, and up to 8 tokens the all-gathered
unpooler output is this GPU's 1/8 slice (rank 0's 1024 columns, from the weight's first
1024 rows) followed by zeros. Above 8 tokens SGLang runs the whole unpooler on every GPU,
so its output is the full product.

The weights arrive once, in prepare(weights), which is untimed (repack them however you
like). Each timed call is kernel_fn(data, state), with T = B * L tokens
(request-major) and Muse's FP32 residual stream:
  qkv, gate = split(attn_input @ qkv_gate.T)          # q 16x64, k 2x64, v 2x64, gate 16x64
  q, k = rmsnorm(q) per head, rmsnorm(k) per head     # no weight, eps 1e-5
  q, k = rope(q, p), rope(k, p)                       # sliding layers; interleaved pairs, theta 5e5
  cache[p] = fp8(k / k_scale), fp8(v / v_scale)       # at p = seq_lens[b] + j (row cache_loc[t])
  a = softmax(5.473 * q @ K.T / 8) @ V                # causal; a 2,048-token window on sliding layers
  a = (rmsnorm(a) per head * sigmoid(gate)) @ o.T     # then all-reduce: identity
  stream = alpha1 * stream + beta1 * rmsnorm(a)       # eps 1e-8; alpha, beta from the residual gate
  h = rmsnorm(stream) * (1 + pre_ffn_norm)
  scores = sigmoid(0.17678 * h @ router.T)            # FP32 router
  experts = top8(scores + router_bias)                # weights: chosen scores / their sum
  x = rmsnorm(h @ pooler.T) * (1 + pre_expert_norm)
  m = sum over local experts e of weight_e * rmsnorm(expert_e(x) * (1 + post_expert_norm))
                                                      # this GPU's 32 experts; then all-reduce: identity
  f = m @ unpooler.T                                  # T > 8: replicated on every GPU
  f = [m @ unpooler[:1024].T, 0, ..., 0]              # T <= 8: this GPU's slice; all-gather: identity
  f = f * (1 + post_ffn_norm)
  stream = alpha2 * stream + beta2 * rmsnorm(f)       # eps 1e-8
  out = rmsnorm(stream) * (1 + next_pre_attn_norm)    # the next layer's attention input
where expert_e(x) = (silu(x @ w1.T) * (x @ w3.T)) @ w2.T with NVFP4 weights, and a residual
gate g gives beta = sigmoid(g / 0.3), alpha = sqrt(max(sigmoid(-g / 0.3) * (1 + beta), 1e-3)).
This GPU's 32 experts are local_start .. local_start + 31 (data["local_start"], an int32
tensor; their weights are w13[i] for expert local_start + i). A step waits for the
all-reduce, so its time is the busiest GPU's: each input's local_start is the range its
routing sends the most (token, expert) pairs to.
Return (out bfloat16 [T, 8192], stream float32 [T, 8192], experts int32 [T, 8] global ids),
and write the block's keys and values into the cache in place.

Correctness: the residual update (the output stream minus alpha1 * alpha2 times the input
stream) and the next attention input each within 2.5x the reference's own error with FP8
E4M3 queries (between 3% and 15%) relative L2 of an FP32 reference, no row off by more than
five times that, and the new cache entries within 8e-2. The layer inherits
muse_attention_decode's FP8 queries: on soft keys (the scored shapes) SGLang's path lands
at about 5%, on i.i.d. keys at up to 13.6%; a plain BF16 implementation at about 1%.
Routing: as on the released checkpoint, the router reads what attention adds to the
stream, so FP8 attention moves a token's near-tied experts, and Muse's routing
concentrates on a few experts per layer (the stream has a persistent direction the router
partly aligns with). Your experts pass if each one's biased score (sigmoid score +
router_bias) is within 5e-2 of the reference's 8th best; you keep each of the reference's
experts on this GPU whose biased score beats its 9th best by more than 0.03 (0.04 above
256 tokens); and you route no fewer tokens, nor fewer distinct experts, to this GPU's
experts than the reference does, beyond a small slack (8% + 4 routes; 15% + 2 experts).
The rest of the check then follows your experts. SGLang's FP8 path reroutes 9-46% of
tokens and stays within all three.

The scored shapes, and some tests, give each KV head's cached keys a shared direction, as
Muse's real keys have, so a query's weight spreads as in the real model: over tens of keys
in a sliding window, hundreds to thousands on a full layer at long context (not one-hot,
as i.i.d. keys would make it).
The step's metadata comes with the inputs, as SGLang builds it once per step for every
layer: positions, cache_loc (each new token's flat cache row), cache_seqlens (seq_lens + L),
cu_seqlens_q and, on full-attention decode steps, scheduler_metadata (FA3's tile schedule
from get_scheduler_metadata, None otherwise). The FP8 cache scales are 1 in the scored cases (SGLang leaves the released
checkpoint's unset) and 1/64 in a test.
The scored cases use page size 1, as SGLang serves Muse on H200; some tests use 64-token
pages, so read config["page_size"].
Shapes follow SGLang's H200 profile and LMSYS's agentic coding workload: decode steps of
1, 16 and 32 requests and 8-token verify blocks for the same, at contexts around 80K
tokens, each on a sliding-window and a full-attention layer. 32 is the most requests the
serving recipe runs at once: what its KV pool holds at these contexts.
