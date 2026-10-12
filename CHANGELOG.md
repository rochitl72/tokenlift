# Changelog

## v0.1.0 (first public release)

First public release of TokenLift, a lossless word-level speculative decoder for byte-level-BPE
language models, evaluated on Tamil and Hindi with Qwen3.

**What it does**
- Drafts whole words with a small head (about 10.7 M parameters, 21 MB in fp16) and verifies them
  with the target model's own argmax, so greedy output matches plain decoding.
- Needs no separate draft model and no change to the target model's weights.
- Command-line tool (`tokenlift`) and Python API (`tokenlift.TokenLift.from_pretrained`).

**Defaults and safety**
- Loads the model in float32 by default. float16 and bfloat16 raise a warning, because near-tied
  logits can make output differ across batch shapes.
- 14 unit tests pass.

**Measured behaviour (Qwen3-4B-Base, batch size 1)**
- 2.23 tokens per forward pass on Tamil and 1.98 on Hindi.
- A standard same-tokenizer self-speculation baseline (Qwen3-0.6B drafting for Qwen3-4B) reaches
  3.40 and 3.23, but needs 1.19 GB of extra weights, which is 56 times the memory of the head.

**Known limitations**
- You train one head per target model and language.
- Tested on Tamil and Hindi Wikipedia text and Qwen3-family tokenizers only.
- No KV-cache integration or wall-clock benchmark against other decoders yet.
- Medusa, unrestricted EAGLE and lookahead decoding were not benchmarked.
- There is no post-cutoff holdout, so training-data overlap with Wikipedia cannot be ruled out.
