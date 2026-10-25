The dense projections of Muse Spark Open, at the shapes one GPU runs when SGLang serves
the model on 8xH200 (TP8/EP8). The released NVFP4 checkpoint keeps every attention
projection and the MoE block's pooler and unpooler in BF16, and the router in FP32. In
SGLang's H200 profile they are the largest share of a decode step at low batch: about
a third of GPU time with one request.

The weights arrive once, in prepare(weights), which is untimed (repack them however you
like). Each timed call is kernel_fn(data, state): out[M, N] = x[M, K] @ W[:N, K]^T,
where W is the weight named by config["layer"] and N = config["n"] (all of W's rows,
except the unpooler's first 1024 up to 8 tokens):
  qkv_gate  2304x8192  BF16  fused query, key, value and output-gate projection (16/2/2/16 heads of 64)
  o         8192x1024  BF16  output projection
  pooler    4096x8192  BF16  MoE block: hidden -> expert width (replicated on every GPU)
  unpooler  8192x4096  BF16  MoE block: expert width -> hidden. Up to 8 tokens SGLang computes
                             only this GPU's 1/8 of the output (rows 0-1023) and all-gathers
                             it; above that, the whole output (replicated on every GPU)
  router     256x8192  FP32  expert logits; x is BF16, the output is FP32
BF16 projections return bfloat16; the router returns float32.

M is tokens per step: decode steps of 1, 16 and 32 requests, and 8-token speculative
verify blocks for 1, 16 and 32 requests (M = 8, 128, 256). The serving recipe runs at
most 32 requests at once: that is what its KV pool holds at agentic contexts (~80K).
Every load runs all five projections.

Correctness: relative L2 error vs. an FP64 reference <= 1e-2 (the router: 1e-4, since its
logits choose experts), no single row off by more than five times that, no NaN/Inf. The
router's FP32 weights may run as a BF16 hi + lo pair with FP32 accumulation at any M, as
SGLang does above 8 tokens; BF16 router weights alone fail. Otherwise the weights are
used at their own precision: about one row in 8 of x has a single nonzero entry, a power
of two, so its output is a weight column: exact with the original weights in any
accumulation order (split-K, BF16 partials), about 3e-6 off with the router's hi + lo
split. Such rows must be within 1e-3. Weights re-quantized to 8 bits miss that by 5-10x.
