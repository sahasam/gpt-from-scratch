# GPT from scratch — materials

Companion to Andrej Karpathy, *"Let's build GPT: from scratch, in code, spelled out"*
Video: https://www.youtube.com/watch?v=kCc8FmEb1nY

## Core links (from the video description)
- Companion code repo (this video): https://github.com/karpathy/ng-video-lecture
  - `bigram.py` — the simple baseline model
  - `gpt.py` — the full transformer we build up to
  - `input.txt` — tinyshakespeare (same file as our `shakespeare.txt`)
- Google Colab used in the video: https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing
  *(reference only — we are building our own, not running this)*
- nanoGPT (the "production" version): https://github.com/karpathy/nanoGPT
- Neural Networks: Zero to Hero (full series): https://karpathy.ai/zero-to-hero.html

## Prerequisite videos (makemore series — build intuition first)
- The series playlist lives under Zero to Hero above. Relevant priors:
  - intro to language modeling / bigrams
  - backprop / `micrograd`
  - batchnorm, and the "math trick" for attention

## Referenced papers
- Attention Is All You Need (the Transformer): https://arxiv.org/abs/1706.03762
- OpenAI GPT-3 — Language Models are Few-Shot Learners: https://arxiv.org/abs/2005.14165
- OpenAI GPT-2 — Language Models are Unsupervised Multitask Learners:
  https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

## Our plan (build order)
1. Load & inspect the data
2. Build a char-level tokenizer (encode/decode)
3. Encode dataset → tensor, train/val split
4. Batched data loader (block_size context windows)
5. Bigram baseline model + training loop
6. The math trick: weighted aggregation as matrix multiply
7. Single self-attention head
8. Multi-head attention
9. Feedforward + Blocks + residual connections + LayerNorm
10. Scale up, dropout, and generate Shakespeare

> Building it ourselves, cell by cell. `shakespeare.txt` is downloaded. Reference
> `bigram.py`/`gpt.py` only if stuck — the point is to derive it.
