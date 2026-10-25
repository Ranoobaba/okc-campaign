One whole decoder layer of Muse Spark Open at the prefill shapes of an agentic coding
turn, as one H200 runs it when SGLang serves the model on 8xH200 (TP8/EP8): the same
layer, inputs, outputs and reference as muse_decode_layer (whose files this problem
shares), with long query blocks instead of decode steps.

SGLang's layer at these shapes is mostly the routed experts as one large grouped NVFP4
GEMM, then the dense GEMMs and, on full layers at long context, chunked-prefill attention
over the paged FP8 cache.

Each case is one prefill forward: B requests, each adding L new tokens (causal within the
block) after seq_lens[b] cached ones, page size 1 as SGLang serves H200 (one test keeps
64-token pages, so read config["page_size"]). The block's keys and values are written
into the cache; everything else, including the correctness bounds and the routing check,
is muse_decode_layer's.

Loads: a later turn (753 new tokens on an 80K-token conversation), four conversations'
later turns in one forward, each on a sliding-window and a full-attention layer, and the
first turn's 16K-token chunks (SGLang's chunked-prefill size): full-attention chunks at
contexts 0, 32K and 64K, and one sliding-window chunk at 32K that stands for all three,
since a sliding layer attends only its 2,048-token window.
