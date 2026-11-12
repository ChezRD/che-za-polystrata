# LSA on the dense Qwen3.8-27B: the distilled map, and what using it takes

A design note from surveying **https://github.com/syv-ai/HyperQwen** (`lsa/`) against our dense `qwen35` family. It
answers one question: can we run the paper's **D-LSA (Flat)** — the variant measured at 72.4× prefill / 42.5× decode
per attention layer at 10M tokens on this exact model — or only the training-free fallback that the paper says
"trails" it?

**Answer: the distilled maps exist and are shipped as weights.** They are a 6.30 MB release asset of the
Prefrontal-AI repository, pinned by SHA-256, containing `W_Q_idx` and `W_K_idx` per attention layer.

Nothing was downloaded and nothing was installed. The size and the tensor names below are from the pin file, the
loader code and an HTTP `HEAD`; the tensor *values* have not been inspected.

## 1. Are the maps actually there?

**The trained tensors ship as weights — case one, the good case.** The chain is three files:

| where | what it says |
| --- | --- |
| `lsa/PIN` | `LSA_MAP_RELEASE=d-lsa-weights-v1`, `LSA_MAP_NAME=qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt`, `LSA_MAP_SHA256=24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b` |
| `lsa/install.sh` | fetches `https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1/<same name>` with `curl`, verifies the SHA-256, fails loudly on a mismatch |
| `lsa/python/hyperqwen_lsa/backend.py` | `torch.load(..., weights_only=True)`, then `pair["W_Q_idx"]`, `pair["W_K_idx"]` per layer, `.float()` |

`HEAD` on that release URL returns **302 → 200, `content-type: application/octet-stream`,
`content-length: 6302911`** — 6.30 MB. A per-layer map of `[768, 64]` fp16 is 96 KiB, so 32 of them (two per layer ×
16 layers) is 3.0 MB; at fp32 it is 6.0 MB. The 6.30 MB matches **32 tensors of `[768, 64]` in fp32**, which is what
`.float()` implies the blob stores.

**Tensor names and shapes.** `${layer}.W_Q_idx` and `${layer}.W_K_idx`, both `[768, 64]`. The loader *checks* the
shape and refuses a mismatch: `key_dim = num_kv_heads * (head_size - rotary_dim)` = 4 × 192 = **768**, and
`if tuple(self.lsa_w_k.shape) != (key_dim, core.INDEX_DIM): raise ValueError`. Verified per layer at first use
(`_lsa_setup`), not at load.

**Which layers.** Not 16, and not 64: **the layers whose decoder index is a key of the map**, and the map is built for
the full-attention layers only. `backend._lsa_setup` looks up `layer_index(name)` (the `layers.N.` in vLLM's module
name), and if `index not in maps or name.startswith("mtp.")` it logs `"LSA: %s is not in the map, it keeps the dense
Triton read"` and returns `None`. So a GDN layer, or the MTP block, is untouched — **16 layers out of 65**, matching
`full_attention_interval = 4` on 64 trunk layers. Everything else runs its dense path unchanged.

**No τ in the shipped map.** The paper's Table 2 has a per-layer temperature `τ` for the distillation *loss*
(Equation 2), and the loader reads only `W_Q_idx` and `W_K_idx`. `τ` is a training-time quantity; it is not applied at
inference, so nothing is missing.

**No training-free fallback in play.** `core.py` contains no PCA and no SVD; the only projections are the two loaded
tensors. The port is the distilled variant, and the training-free variant is the paper's separate ablation
("D-LSA (Training-Free)") that this repo does not ship.

## 2. The geometry, against our GGUF

Confirmed from the file's own header (`tools/gguf_reader.py`, 866 tensors): `key_length = 256`,
`rope.dimension_count = 64`, `attention.head_count = 24`, `attention.head_count_kv = 4`. The repo's own constants
match ours: `core.POOL_SIZE = 4`, `WINDOW = 512`, `SINK = 4`, `TOP_K = 4096`, `TOP_P = 1024`, `INDEX_DIM = 64`,
`BUCKET_SIZE = 8`, `BUCKET_CAP = 16`.

**The index key is the RoPE-free tail, `[64, 256)` per head.** `core.index_keys`:

```python
flat = key[:, :, rotary_dim:].reshape(t, -1).float()
return F.normalize(flat @ w_k, dim=-1)
```

`rotary_dim = head_size * partial_rotary_factor` = 256 × 0.25 = **64**, so the slice is dims 64..255: 192 values a
head, 4 heads side by side = 768 in, projected to 64. This is the *tail*, not `[64, 128)` — it matches what I derived
from `head_norm_rope_kernel`'s `if (i >= n_rot)` and what is already written into `docs/QWEN38.md`.

**The query side sums each KV group's query heads.** `core.index_queries`:

```python
grouped = query.view(t, kv_heads, group, dim)[..., rotary_dim:].sum(2, dtype=torch.float32)
return F.normalize(grouped.reshape(t, -1) @ w_q, dim=-1)
```

`group = 24 / 4 = 6`, so the six query heads that read KV head *g* are summed into that head's 192-dim slice, the four
slices concatenate to 768, and `W_Q` projects to 64. **One index query per position, shared by all 24 attention
heads** — which is exactly why a single index list `J_i` serves every head, and why the paper can afford it.

## 3. What it would take to use

### The maps in our terms

Two **fp32 `[768, 64]` matrices a layer**, 32 in total, 6.30 MB. In our tree they would become two `Dev` matrices a
`Qwen35::Block` (beside `wq`/`wk`/`wv`), read at load in `Qwen35::load_layer` for the attention layers only. fp32 is
the right storage: the index key is an L2-normalised projection and its precision feeds the selection, which is
discrete — a bucket chosen or not. 196 KB a layer of VRAM, 3.1 MB over the 16.

### Does `k[:, :, rot:]` match our layout?

**Yes, with no change to `attn_store`.** `Qwen35::attn_store` calls `head_norm_rope(kraw_, head_dim_, …, n_rot_,
theta_scale_, …)`, and `head_norm_rope_kernel` writes `k_` as the normed key for dims `[0, n_rot)` rotated and for
dims `[n_rot, head_dim)` **copied unrotated** (`if (i >= n_rot) { yr[i] = xr[i] * scale * w[i]; continue; }`). So the
buffer `attn_store` already produces has the RoPE-free tail in place at `[64, 256)`, in the same order and the same
per-head stride the repo assumes. The index key can be computed **from `k_` after `head_norm_rope`, before it is
stored** — no second read, no re-derivation.

### Does it need the attention keys, or a second buffer?

**A second buffer, and this is the whole memory question.** The index key is a *projection* of the RoPE-free slice,
not the slice itself: 768 values → 64. It cannot be recovered from the KV cache later without keeping the 768, which
is 1.5 KB a token a layer (fp16), i.e. nearly as much as the KV cache it indexes. So `attn_store` must write it:

- one `d_idx = 64` fp16 value a token a layer (the index key), plus
- one `d_idx = 64` fp16 centroid a bucket, and 16 `int16` member positions a bucket, every 8 tokens.

At the paper's `n_pool = 4` the *pool key* is the sum of four consecutive index keys, so the natural store is one
index key a token (128 B) and the pool is formed at clustering time. That is **128 B a token a layer**, against
k8v4's 1,696 B a cell — **7.5% on top of the KV cache**, which matches the repo's own "~7% of the KV pool". At our
61,440-token ceiling that is 7.9 MB a layer, 126 MB over the 16.

**So yes: `attn_store` gains a write, and it is the only attention function that does.** That is a change to a
function the LSA constraint says to leave alone — but it is an *addition of a store*, not a refactor of the read
path, and it can be a separate `if (lsa_)` block exactly like the existing `k8v4` / `q8_0` branches. The read path
(`attn`, `attn_rows`) needs the id list instead, which is the kernel work and is the follow-up.

### Where the selection's kernels would live

`strata-poly` links `strata_kernels` (`cmake/strata_poly.cmake`), and both engines see it. So the score/select/expand
kernels go in `src/kernels/` once, and `src/poly/qwen35.cpp` calls them — no duplication, and no new dependency for
the `strata` engine, which would simply not call them.

### Licence

- **Our code:** HyperQwen is Apache-2.0, the `lsa/` port is Apache-2.0 with a NOTICE that credits the paper and
  `Prefrontal-AI/latent-sparse-attention` (also Apache-2.0). Taking the *algorithm* and re-implementing it in our
  kernels needs the NOTICE and nothing else. Copying `core.py` verbatim would mean carrying its NOTICE text.
- **The maps:** the NOTICE says *"The distilled map … is an asset of that repository's release `d-lsa-weights-v1` and
  is **not redistributed here**."* So HyperQwen does not ship them, and we would fetch them from Prefrontal-AI at
  install time. The upstream repository is Apache-2.0, which covers its release assets, but **the map is derived from
  Qwen3.8-27B's weights** and that model's licence governs derivatives. That is the constraint to resolve before
  shipping: our artifact would gain a runtime fetch of a 6.3 MB derived asset, and our docs would have to name both
  licences. **Open question for you, not something I can settle.**

## 4. The cheapest accuracy check, and the number that stops it

Before any kernel: **compare the LSA selection against the model's own attention on one real prompt, in Python, on
the CPU or a single GPU.**

The recipe, which needs only what we already have plus the map:

1. Take a prompt of ~8,000 tokens (above `dense_below = 4,612`, so the budget actually bites, and affordable on this
   card). Run `strata-poly` with `--dump-layers`-style instrumentation, or a small harness, to capture for **one**
   attention layer: the post-RoPE `k_` (256 a head) and the queries, at a handful of query positions.
2. Compute the index keys `= normalize(k[:, 64:256] @ W_K)` and the index query
   `= normalize(sum over each group's 6 heads of q[:, 64:256] @ W_Q)` in numpy, exactly as `core.py` does.
3. Run the selection: pool by 4, Ward to 64 buckets a 512-key block, score centroids, exact top-1024, expand best
   first to 4096 skipping sinks and window.
4. Compute the attention two ways over the same keys: dense causal, and over the selected `J_i` only. Compare.

**The number that decides it: the cosine similarity of the two attention outputs, per query head.** The repo's own
gate is `cos > 0.99999` for the *kernel* against a reference (`tests/check_core.py`), which is a different question —
that one holds the port to the reference, and both use the same selection. What we need is the *selection's* effect
on the model's output, which no one has published for this checkpoint at our contexts.

My stop rule: **if the selected-keys attention is below `cos 0.99` against dense at 8k tokens for the median head,
stop** — the paper's own accuracy result is "within 5 points of full attention up to 1M", i.e. a small but real
change, and a selection that already diverges at 8k will not hold at 61k. Between 0.99 and 0.9999 it is a judgement
call for you; above 0.9999 it is safe to spend the kernel work. **The check is a day of Python, not a week of CUDA,
and it is the only thing that should gate the port.**

## 5. What this changes in the plan

`LSA-PLAN.md` §1.3 ruled out D-LSA because it "needs distillation (352M tokens)". That is true of *training* a map and
false of *using* one: the map for this checkpoint is a 6.30 MB download, and the repo that points at it also carries
the selection algorithm and its tests. The training-free PCA fallback is not needed, and the paper's own ablation is
why that matters — it "stays close" on Qwen but "trails" on GLM, so the distilled map is the stronger of the two and
it is available.

The three things that stay true and unchanged:

- **LSA cuts compute, not memory.** The KV cache is still 6.91 GiB at 256K with k8v4; 10.02 GiB of weights in
  VRAM plus that is 16.9 GiB, which does not fit 16 GB at 256K either way. What LSA removes is attention over
  262,144 keys where 4,612 will do.
- **The KV layout in `lsa/` is not ours to adopt** (see [HYPERQWEN.md](HYPERQWEN.md) Part 2): it packs index rows into
  KV slots, which would change the k8v4 row stride and the q8_0 scale positions. The index goes in its own allocation, keyed by
  position.
- **`attn_store`, `attn`, `attn_rows` were not touched by the dense port** — confirmed by inspecting the diff: the
  hunks are in `load_geometry`, `load_layer`, `layer_head`, and the four MoE stubs, and none in the attention entry
  points.
