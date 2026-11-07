# Reproducibility study: A-Mem on LoCoMo (DeepSeek backend)

This repository is a fork of [`WujiangXu/A-mem`](https://github.com/WujiangXu/A-mem), the
official evaluation code for **A-Mem: Agentic Memory for LLM Agents**
(Xu et al., NeurIPS 2025, arXiv:2502.12110). Everything under `repro/`, the
`--ablate_memory_evolution` switch, the token accounting and the DeepSeek
adaptation described below were added for FTEC5660 Individual Project 2
(Reproducibility).

## What is reproduced

The paper reports per-category token-level **F1 / BLEU-1** on the LoCoMo
long-conversation QA benchmark. For GPT-4o-mini the reported numbers are:

| Category | F1 | BLEU-1 |
|---|---|---|
| Multi-hop | 27.02 | 20.09 |
| Temporal | 45.85 | 36.67 |
| Open-domain | 12.14 | 12.00 |
| Single-hop | 44.65 | 37.06 |
| Adversarial | 50.03 | 49.47 |

We reproduce a **subset**: one LoCoMo conversation (`conv_1`, 369 turns,
105 questions) rather than all ten, as permitted by the assignment brief.

## Model substitution (controlled variable)

The paper's experiments use OpenAI GPT-4o / GPT-4o-mini. Those endpoints are not
available to this student account, so the model is replaced with
**DeepSeek V4.1 Flash** (`deepseek-flash`, OpenAI-compatible endpoint
`https://api.deepseek.com/v1`). Recorded settings:

| Item | Value |
|---|---|
| model | `deepseek-flash` (DeepSeek-V4.1-Flash) |
| base URL | `https://api.deepseek.com/v1` |
| decoding | `temperature` as configured per stage (0.7 default, 0.5 for category-5 answers) |
| max tokens | 4096 (`AMEM_MAX_TOKENS`) |
| thinking | **disabled** (`thinking={"type":"disabled"}`) |
| prompt changes | none |

Two adaptations were required and are documented as such:

1. **`max_tokens` raised from 1000 to 4096.** DeepSeek counts reasoning tokens
   against the completion budget, so at 1000 tokens some structured answers came
   back empty (`finish_reason=length`). With 4096 tokens and thinking disabled,
   a smoke run had zero truncated/failed calls.
2. **Thinking disabled.** With the default reasoning mode each call took ~9 s and
   produced ~1.2k completion tokens for a ~400-token prompt; with
   `thinking={"type":"disabled"}` the same call took ~1.5 s and ~220 tokens. The
   paper's GPT-4o-mini baseline is a single-shot, non-reasoning call pattern, so
   disabling thinking is the closer match, and it keeps the run affordable.

**Results are therefore not directly comparable to the paper**: the underlying
LLM differs. Tasks, metrics and evaluation script are identical.

## Setup

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# HuggingFace is unreachable from this network; use a mirror for the
# sentence-transformers encoder (all-MiniLM-L6-v2, ~90 MB)
export HF_ENDPOINT=https://hf-mirror.com

# DeepSeek credentials (the harness talks to any OpenAI-compatible endpoint)
export OPENAI_API_KEY=<deepseek key>
export OPENAI_BASE_URL=https://api.deepseek.com/v1
```

The harness downloads NLTK `punkt`/`wordnet` on first import.

## Running

```bash
python repro/split_dataset.py --indices 1      # -> repro/datasets/conv_1.json

# baseline (memory evolution ON)
./repro/run_conv.sh 1

# ablation "w/o ME": link generation kept, neighbour rewrite skipped
./repro/run_conv.sh 1 --no-me
```

Outputs land in `repro/out/`; each results JSON also carries
`llm_usage` (calls / prompt / completion / cached tokens) and `llm_failures`.

## The modification

`--ablate_memory_evolution` removes exactly one step: the third LLM call in
`RobustAgenticMemorySystem.process_memory`, the one that rewrites the context and
tags of neighbouring notes ("memory evolution"). Note construction and link
generation are untouched, so this is the paper's **w/o ME** ablation, not
"w/o LG & ME".

Baseline and ablation use separate memory-cache directories
(`cached_memories_robust_openai_deepseek-flash` vs `..._noME`) so no state is
shared between arms.

## Repository layout

```
repro/split_dataset.py   split locomo10.json into one file per conversation
repro/run_conv.sh        run one conversation (baseline or --no-me)
repro/stats.py           per-conversation turn / question counts
test_advanced_robust.py  upstream harness (+ --ablate_memory_evolution, usage output)
memory_layer_robust.py   upstream memory layer (+ ablation flag, token accounting,
                         configurable max_tokens / thinking)
```
