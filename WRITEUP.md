# Learning transformers by building one — with Claude Code as a tutor

> A working log turned essay. I set out to *understand* transformers by deriving a
> character-level GPT from scratch, and used Claude Code not as a code generator but as a
> Socratic tutor that refused to hand me answers. This is what that felt like, what I learned
> about PyTorch's sharp edges, and how it closed real gaps in my mental model of LLMs.

*Status: draft in progress. Written alongside the code in this repo (`gpt.ipynb`,
`attention.ipynb`). Sections marked 🚧 are stubs.*

---

## Why this repo exists

I didn't want another tutorial I could nod along to and forget. I wanted the struggle — the
part where you type the wrong thing, the shapes don't line up, and you have to actually
understand *why* before the error goes away. So the rule for the whole project: **the value is
in the derivation, not the finished model.** Every piece — tokenizer, batching, attention — is
built by hand from first principles.

The twist: I ran the whole thing with Claude Code configured as a **learning tutor, not an
autocomplete.** The project has a `CLAUDE.md` that explicitly overrides the assistant's default
"just fix it" behavior:

- Explain the *why* and the *what*, not just the *how*.
- Point out where things are wrong — and the *mechanism* of the bug, and its downstream
  consequence — but **don't write the fix.**
- Nudge with leading questions. Let me make the connection.
- Only give the answer if I *explicitly* ask.

That one config file changed the entire character of the collaboration. More on that below.

---

## Theme 1 — Claude Code as a learning tool

The default mode of an AI coding assistant is to *remove* friction: you describe what you want,
it produces working code. That's exactly wrong for learning, where the friction *is* the
lesson. The interesting discovery was how much the experience changed by inverting that default.

**What worked:**

- **Scaffold-and-nudge rhythm.** The tutor would set up a cell with a `TODO(human)` and a
  couple of hints, I'd fill in the actual logic, then say "check." It reviewed like a TA:
  intent → what's wrong → consequence → a *question* pointing at the fix. I wrote every load-
  bearing line myself.
- **Bugs explained, not patched.** When my tokenizer silently returned the wrong types, it
  didn't hand me corrected code — it explained that two bugs were propping each other up so the
  output *looked* right while the assertion was actually failing. I fixed it. It stuck.
- **Calibration is a real thing.** Early on it over-explained — full tables enumerating every
  PyTorch broadcasting rule — and I said so ("I felt my hand was being held"). It dialed back to
  intuition + the one subtlety that bites. Later I pushed the other way: "just give me the class
  and method names, let me figure it out." The fact that you can *steer the teaching altitude*
  mid-session turned out to matter more than any single explanation.

**The honest caveat:** it's easy to let the assistant do the thinking. The discipline has to
come from the config *and* from me actually resisting the urge to ask "just show me." When I
did ask for the answer, I got it — but I could feel the difference in how well it stuck.

🚧 *To expand: the specific `CLAUDE.md` rules, the "Learn by Doing" request format, and a couple
of side-by-side moments where a nudge beat a solution.*

---

## Theme 2 — PyTorch's intricacies (the stuff that actually bit me)

The math of a transformer is *less* subtle than the plumbing. A running list of the sharp edges
I only understood by hitting them:

- **A `print` that looks right ≠ a passing `assert`.** My tokenizer roundtripped visually but
  `decode(encode(s)) == s` was `False`. Test the equality, not your eyes.
- **`.view()` needs contiguous memory and row-major reasoning.** Flattening `(B,T,C) → (B*T,C)`
  for cross-entropy only keeps predictions aligned with targets because both flatten in the same
  order. Understanding *why* the reshape is safe was more valuable than the reshape.
- **`dim=` is an index into the shape tuple** — and the same argument means "normalize over
  this axis" for softmax but "grow along this axis" for cat. Getting it wrong doesn't crash.
- **The silent softmax bug.** I carried `dim=1` from a 2-D tensor to a 3-D one. Adding the batch
  dimension shifted the axis I actually wanted from `1` to `2`. Result: my attention weights
  normalized down the *columns* instead of across each row — a valid-looking, completely wrong
  distribution, no error, right output shape. Caught only by checking that each row summed to 1.
  `dim=-1` is the robust idiom precisely because it survives added leading dims.
- **Matmul is a grid of dot products; the transpose is just bookkeeping.** `q @ k.transpose(-2,-1)`
  computes every query·key affinity at once. Debug shape errors by *canceling the inner dims*,
  not by reading the traceback prose.
- **Buffers vs parameters.** A causal mask is state the module needs but shouldn't learn —
  `register_buffer` so it rides `.to(device)` and persists, without collecting a gradient.
- **Device placement is a landmine.** `torch.arange(T)` is born on the CPU; add it to an MPS
  tensor and you crash. `device=idx.device` is the portable fix.
- **Class vs instance.** Passing `Model.parameters()` (the class) instead of
  `model.parameters()` (the instance) to the optimizer — the weights live on the instance.

The pattern across all of these: **the errors that teach you the most are the ones that don't
throw.** Wrong-but-runs is where the real understanding gets forged.

🚧 *To expand: a shape-debugging cheat sheet, and the `(B, T, C)` vocabulary as the single most
useful mental model in the whole project.*

---

## Theme 3 — Closing the LLM knowledge gap

The gaps that closed weren't the famous ones ("attention is all you need"). They were the
quiet, load-bearing ones underneath:

- **Cross-entropy is information theory, not just "the loss function."** Self-information
  `-log(p)` is surprise; entropy is a distribution scored against itself; cross-entropy scores
  the *true* distribution's surprise using the *model's* probabilities. With a one-hot target it
  collapses to `-log(q_correct)`. The `ln(vocab_size) ≈ 4.17` sanity check at init falls straight
  out of this — and catches reshape/vocab bugs instantly.
- **The bigram ceiling is a *concept*, not a number.** A model that sees only the current
  character generates English-*flavored* gibberish — right spacing and capitalization, no real
  words — and plateaus around loss 2.5. That ceiling *is* the motivation for attention. Feeling
  the wall is different from being told about it.
- **Attention is a content-dependent blur.** Coming from a graphics background, this was the
  analogy that made it click: a box blur is a fixed uniform kernel; the causal averaging trick is
  a backward-only blur; *attention is a blur whose kernel each position computes for itself* from
  query·key. Communication between tokens; the feed-forward layer is the computation on what was
  gathered.
- **Position must be injected.** Attention is order-blind — the affinities are identical however
  you shuffle the tokens. A separate position embedding, *summed* with the token embedding, is
  the only thing that encodes sequence. This is obvious in retrospect and invisible until you ask
  "wait, where does order come from?"
- **Measurement is not the metric.** A single training batch of 32 tokens is a noise generator,
  not a loss. Comparing models by eyeballing the step loss is comparing dice rolls. This is why
  every real loop reports a smoothed eval loss.

🚧 *To expand: the full derivation chain from cross-entropy → the 4.17 sanity check, and the
Tier-1/Tier-2 map of what actually turns gibberish into Shakespeare (multi-head, feed-forward,
residuals + LayerNorm, depth, scale).*

---

## Where the code is

- `gpt.ipynb` — Part 1: data → tokenizer → tensor/split → batching → bigram baseline → training.
- `attention.ipynb` — Part 2: the averaging trick → single-head self-attention → wrapping it in
  a module → assembling a working attention language model. (In progress: multi-head, blocks,
  scale-up.)
- `PROGRESS.md` — the running technical log: every step, every bug, every side question.
- `CLAUDE.md` — the tutor configuration that made this a learning project instead of a codegen one.

🚧 *Closing section to come once the model actually generates convincing Shakespeare.*
