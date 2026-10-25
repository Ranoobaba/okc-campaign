The routed experts of Muse Spark Open's mixture-of-experts layer, with NVFP4 expert
weights and BF16 activations, on one H200, as SGLang serves it (Marlin W4A16 on Hopper).

Muse's MoE block projects the hidden state 8192 -> 4096, routes each token to the top 8
of 256 experts (intermediate 4096, SwiGLU, no shared expert) and projects back; this
problem is the experts in between. Each expert's output is normalized on its own
(RMSNorm, scaled first) before it is weighted and summed. As SGLang serves it on 8xH200
(EP8), this GPU holds experts 0-31 of 256. Every token arrives, routed over all 256, and
only its pairs with local experts are computed, and rows with no local expert are zero
(the GPUs' partial sums are then all-reduced). A step waits for that all-reduce, so its
time is the busiest GPU's: each routing is drawn over all 256 experts, then relabelled
by whole GPUs so that the GPU with the most pairs is this one (2.6 experts for one token,
about 14 pairs for 8 tokens, 50 for 32, 390 for 256).

The weights arrive once, in prepare(weights), which is untimed: repack them however you
like, as production does at load. Each timed call is kernel_fn(data, state).

`data`:
  hidden_states  bfloat16 [M, 4096]
  topk_ids       int32 [M, 8]    global expert ids
  topk_weights   float32 [M, 8]
  config         {"bs", "d_hidden", "d_expert", "n_experts", "local_start", "local_count", "top_k", "sharding"}
weights: w13 uint8 [256, 8192, 2048] (gate rows then up rows, packed E2M1), w13_sf
float8_e4m3fn [256, 8192, 256], w13_gs float32 [256], w2 uint8 [256, 4096, 2048], w2_sf
float8_e4m3fn [256, 4096, 256], w2_gs float32 [256], post_norm_w bfloat16 [4096].

For each token t and each (expert e, weight g) in its top-k row with e local:
  gate, up = split(hidden_states[t] @ dequant(w13[e]).T)
  y = (silu(gate) * up) @ dequant(w2[e]).T
  out[t] += g * rmsnorm(y * (1 + post_norm_w))      (eps 1e-8)

M is tokens per step: decode steps of 1, 16 and 32 requests, and 8-token speculative
verify blocks for 1, 16 and 32 requests (M = 8, 128, 256). The serving recipe runs at
most 32 requests at once: that is what its KV pool holds at agentic contexts (~80K).
With speculative decoding on (SGLang's default), the experts are the largest share of a
decode step from 16 requests up.

Correctness: relative L2 error vs. an FP32 reference <= 1.5e-2, no single token's row off
by more than five times that (7.5e-2, relative to the row's own norm or a tenth of the mean
row norm, whichever is larger), no NaN/Inf.
