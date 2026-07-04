# PROGRESS — gpt-from-scratch

Running log of the char-level GPT build on tinyshakespeare. One notebook per phase;
this file is the human-readable memory of what happened, what tripped me up, and what
stuck. Goal: **understand transformers**, not ship a model. End target: full transformer
(multi-head attention → blocks → scale up).

---

## Done so far — `gpt.ipynb` (Steps 1–6: data → trained bigram baseline)

| Step | What | Result |
|------|------|--------|
| 1 | Load & inspect tinyshakespeare | 1,115,394 chars |
| 2 | Char-level tokenizer (`stoi`/`itos`/`encode`/`decode`) | roundtrip passes |
| 3 | Encode corpus → tensor, 90/10 train/val split | `(1115394,)` int64; 1003854 / 111540 |
| 4 | Batching: `get_batch` → random `(B,T)` batches | `(4,8)` on MPS |
| 5 | Bigram model (`nn.Module`, forward+loss, generate) | init loss ≈ 4.17 ✓ |
| 6 | Training loop (AdamW + zero_grad/backward/step) | loss 4.6 → ~2.2 |

**Bottom line:** a trained bigram generates English-*flavored* gibberish (real spacing,
capitalization, apostrophes) but no real words — because it can only condition on the
**single current character**. That ~2.5 loss is the bigram ceiling. Attention is what
breaks it.

---

## Problems I ran into (and the fix)

1. **Tokenizer returned the wrong types.** `encode` wrapped ids in `str()` → `['46',...]`,
   `itos` was keyed by strings, and `decode` returned a *list* of chars, not a joined
   string. It *looked* right when printed but the sanity check `decode(encode(s)) == s`
   was actually `False`. Two bugs (string ids + string keys) were propping each other up.
   **Fix:** int-clean `encode` (`stoi[c]`), int-keyed `itos`, `''.join(...)` in `decode`.
   **Lesson:** a `print` that looks right ≠ a passing `assert`. Test the equality, not the eyes.

2. **Skipped verifying `.dtype`.** In Step 3 I printed `.shape` but not `.dtype`, even though
   dtype was the whole point. Also used `dtype=int` (works — maps to int64) before switching
   to the explicit `torch.int64`. **Lesson:** verify the thing the exercise is *about*, and
   prefer explicit torch dtypes over Python builtins.

3. **Off-by-one in the batch offset.** First used `randint(0, len-block_size-1)` — safe but
   one position too conservative. Tightened to `len - block_size` after reasoning through
   "what's the largest index `y` actually touches?" **Lesson:** `randint`'s `high` is
   exclusive; always ask which is the largest index you touch.

4. **Hardcoded shapes in the loss reshape.** Wrote `logits.view(block_size*batch_size, ...)`
   — worked only because the batch was exactly `(4,8)`. Would have silently broken in
   `generate`, which feeds `(1, T)` with growing `T`. **Fix:** `B, T, C = logits.shape`.
   **Lesson:** functions must read dims from their tensors, not from globals.

5. **Passed the class, not the instance, to the optimizer.** `AdamW(BigramLanguageModel.parameters(), ...)`
   instead of `AdamW(model.parameters(), ...)`. The optimizer needs the *actual* weight
   tensors, which live on the instance `model`, not the blueprint class. **Lesson:**
   class vs instance — `parameters()` walks a specific object's tensors.

6. **Backwards intuition about the init loss.** I expected the loss to be *off* from 4.17
   because init is random. Actually random init is *why* it's ~4.17: random weights →
   ≈uniform prediction → `-ln(1/65) ≈ 4.17`. It sits *slightly above* the floor because
   random ≠ perfectly uniform (occasionally confident-and-wrong, which is penalized more).

---

## Side questions I asked (worth keeping)

- **Tensors vs Python lists** — one fixed dtype + rigid shape buys vectorized math, broadcasting,
  GPU execution, and autograd. Slicing returns a *view* (shared memory), not a copy.
- **`.view`** — reinterprets the same memory with a new shape (no copy). Needs contiguous
  memory (else `.reshape`). `-1` infers a dim. Row-major order is *why* flattening logits and
  targets the same way keeps predictions aligned with answers.
- **`dim=` field** — an index into the shape tuple. For `softmax`/`sum` it's the axis
  normalized/collapsed; for `cat` it's the axis that grows; for `stack` a new axis is inserted.
- **Why "cross-entropy"?** Info theory. Self-information `= -log(p)` (surprise). Entropy
  `H(p) = -Σ p log p` (a distribution scored against itself). **Cross**-entropy `H(p,q) =
  -Σ p log q` — weight by the *true* dist `p`, score surprise with the *model* dist `q`: two
  distributions crossed. With a one-hot target it collapses to `-log q(correct)`.
- **Is that data science or math/physics?** Information theory — Shannon (1948), borrowing
  "entropy" from statistical mechanics (Boltzmann/Gibbs). Same `p log p` in a steam engine,
  a phone line, and this model.
- **What is softmax?** `exp(x_i)/Σ exp(x_j)` → turns logits into a probability distribution
  (positive, sums to 1); `exp` exaggerates the leader but keeps everyone nonzero (→ sampling variety).
- **Where are gradients stored?** In each parameter's `.grad`, *not* the optimizer. `backward()`
  *accumulates* into `.grad` (hence `zero_grad()` first); the optimizer just holds references.

---

## Concepts that stuck

- **Tokenizer = bijection text ↔ ints.** Ints because the model does math / embedding lookups.
- **`(B, T)` and `(B, T, C)`** — the shape vocabulary. Batch × Time, then × Channels(=vocab).
- **A length-`(block_size+1)` chunk packs `block_size` training examples** (predict at every position).
- **Char order is sacred; batch/example order is randomized.** Shuffle *which* chunks, never
  the chars inside them.
- **Embedding = learnable lookup table.** Bigram makes it `(vocab, vocab)` so a lookup *is* the logits.
- **cross_entropy wants `(N, C)` and `(N,)`** — class dim must be dim 1, so flatten `(B,T,C)→(B*T,C)`.
- **Sanity check by expected loss:** `ln(vocab) ≈ 4.17` at init catches reshape/vocab bugs instantly.
- **Autoregressive generation:** forward → take last step → softmax → multinomial (not argmax) →
  append along time → repeat. Same loop for every model; only `forward` changes later.
- **Training ritual (order matters):** `zero_grad → backward → step`. Gradients accumulate by default.
- **The bigram ceiling (~2.5):** can't see past the current char. This is the motivation for attention.

---

## In progress — `attention.ipynb` (Part 2)

Setup carried over from Part 1 (imports, data, tokenizer, split, `get_batch`).

| Step | What | Status |
|------|------|--------|
| 7 | "Average your predecessors" trick, three equivalent ways | ✅ done |
| 8 | Single-head self-attention (Q/K/V), toy tensors | ✅ done |
| 9 | Wrap the head into a reusable `nn.Module` (`Head`) | ✅ done |

**Step 7 — the mathematical trick (all three `allclose` → True):**
1. explicit double loop (ground truth): `xbow[b,t] = x[b,:t+1].mean(dim=0)`.
2. matmul with a normalized lower-triangular matrix: `wei = tril(ones); wei /= wei.sum(dim=1,keepdim=True); xbow2 = wei @ x`.
3. masked scores + softmax: `wei = zeros; wei.masked_fill(tril==0, -inf); softmax` → same weights.
   **The aha:** a causal weighted-average *is* a masked matrix multiply; the `-inf`→softmax→0
   trick is exactly how attention hides the future.

**Step 8 — real attention (done, toy tensors):** replaced the constant `wei` with scores
*computed from the tokens*. Each token projects to query/key/value; `wei = q @ k.transpose(-2,-1)`
(affinities), same mask+softmax, then `out = wei @ v` → `(B,T,head_size)`. Two bugs caught and
fixed along the way (below).

**Step 9 — head as an `nn.Module` (done):** wrapped the toy into `class Head(nn.Module)` — owns
its `query/key/value` `Linear`s, registers `tril` as a **buffer** (non-learned state that rides
the device / persists), scales affinities by `1/√head_size`, and crops the mask with
`self.tril[:T,:T]` so it stays correct when `T < block_size` at generation time. Smoke test:
`out (4,8,16)`, all finite.

**Bugs caught in Step 8 (both ran clean — no exception):**
1. *Transposed both operands.* `q.transpose(1,2) @ k.transpose(1,2)` → inner dims `8` vs `16`
   don't cancel → RuntimeError. Fix: transpose **only** `k`. Lesson: the transpose exists to make
   the two operands *complementary* (rows vs columns); rotating both re-aligns them. Debug by
   canceling inner dims, not by reading the traceback prose.
2. *Softmax over the wrong axis.* Carried `dim=1` over from the 2-D Version-3 code, but adding the
   batch dim shifted the key axis from `1`→`2`. Result: **columns** summed to 1, not rows — a valid-
   looking but wrong distribution, no error, right output shape. Caught it with a **row-sum check**
   (row 0 must be `[1,0,0,…]`). Fix: `dim=-1` (robust to added leading dims). This is *the* classic
   silent softmax bug.

### Side questions (Part 2)

- **What is `C` here?** The **embedding dimension** (`n_embd`), *not* vocab_size. In the bigram
  `C` equalled vocab only because that embedding doubled as logits. Generally decoupled.
- **Why are the toy values negative / not probabilities?** `torch.randn` → standard normal
  (mean 0). The stuff being *averaged* (`x`) is arbitrary features, any sign. Only the *weights*
  (`wei`) are probability-like (nonnegative, rows sum to 1). Two roles — payload vs weighting.
- **How does the toy relate to `batch_size`/`block_size`?** Same roles: `B`=batch, `T`=time.
  The toy feeds `x = torch.randn(B,T,C)` — *fake* data — to isolate the attention mechanism.
  In the real model `x` will be `embedding(idx)` from `get_batch`; this head drops into a model
  that keeps the Part 1 tokenizer/batching/training/generate unchanged. Swapping the engine,
  not rebuilding the car. (`T` = actual seq length ≤ `block_size` = its max — read `T` from the
  tensor, don't hardcode.)

### Concepts that stuck (Part 2)

- **Causal weighted-average = masked lower-triangular matmul.** `wei @ x`.
- **Softmax + `-inf` mask = the permanent scaffolding of attention.** Only the *source* of the
  scores changes (constant → `q·k`).
- **Q/K/V roles:** query = "what I'm looking for", key = "what I advertise", value = "what I
  hand over". q·k sets *how much* to attend; v sets *what flows*.
- **Blur analogy (mine):** box blur = fixed uniform kernel; attention = a content-dependent,
  causal, spatially-varying blur where each position computes its own kernel from `q·k`.

## Next up

- Finish Step 8 (single head), then multi-head → transformer block (residuals, LayerNorm,
  feed-forward) → scale up. Add the √head_size scaling refinement to `wei`.
- Assemble the full model: embedding → attention → logits, wired to Part 1's `get_batch`,
  training loop, and `generate`.
- `estimate_loss()` (averaged train/val) lands when we first train the attention model — the
  point where "did we actually beat 2.5?" becomes a real question.
