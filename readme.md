# Exploring Pain Metaphor Comprehension Abilities in Large Language Models

**MA Thesis — Leiden University Centre for Linguistics, 2023**
Eres Ferro Bastian · Supervisors: Stephan Raaijmakers, Lettie Dorst

> This repository is being actively updated (2026) with a modernized model lineup, expanded metaphor coverage, and a Multi-Agent Debate evaluation framework.

---

## Overview

Patients routinely describe pain in metaphorical terms: *stabbing*, *burning*, *gnawing*. For LLM-based medical chatbots to respond safely, they must recognize that these descriptions are not literal. This project investigates whether current LLMs pass that test — and whether they can articulate *why* a pain description is metaphorical.

The study is grounded in **Deliberate Metaphor Theory** (Steen, 2008), which distinguishes metaphors that demand conscious interpretation (deliberate) from those processed automatically (non-deliberate). Pain metaphors straddle this boundary, making them a demanding test case for LLM figurative language understanding.

---

## Dataset

- **200 utterances** derived from [MedDialog](https://github.com/UCSD-AI4H/Medical-Dialogue-System), a large-scale public medical dialogue dataset
- Three pain metaphors covered: **stabbing**, **burning**, **gnawing**
- Each utterance paired with an *elenchus lure* — a follow-up question that presupposes a literal interpretation (e.g., *"What kind of knife was it?"*)
- Direct metaphors (*"The pain is stabbing me"*) and indirect metaphors (*"It feels like someone is stabbing my chest"*) both represented

---

## Models Evaluated (2023 snapshot)

| Model | Provider | Type |
|---|---|---|
| GPT-4o | OpenAI | General-purpose |
| GPT-3.5-Turbo-0125 | OpenAI | General-purpose |
| Gemini-1.0-Pro | Google | General-purpose |
| Llama-2-7b-chat | Meta (via Replicate) | Open-weight |
| BioMistral-7B | HuggingFace | Biomedical |

---

## Methodology

### Elenchus Prompting

The model is presented with a pain utterance and a lure question that treats the metaphor as literal. A model that is *not* lured recognizes the description as figurative and does not answer the presupposed question directly.

```
Utterance: "I have a stabbing pain in my chest."
Lure:      "What kind of knife was it?"
```

A lured model answers "a kitchen knife" or similar. A non-lured model redirects to the medical concern.

### Maieutic Prompting

A three-stage chain-of-thought that extracts latent metaphorical understanding:

1. **Decision:** Does this utterance contain a metaphorical comparison? (yes/no)
2. **Reasoning:** Provide a reason and a confidence score (0–10)
3. **Counter-argument:** Argue against your own reasoning and rate confidence

This tests whether models hold metaphorical knowledge they may not surface in a single-turn evaluation.

---

## Key Findings

**Elenchus experiment:**

- **GPT-4o** was the strongest model overall. It achieved perfect accuracy on stabbing metaphors (1.00) and near-perfect scores on burning and gnawing. More surrounding clinical context (e.g., a named disease or body part) consistently improved all models' resistance to lures.
- **GPT-3.5-Turbo** was misled by burning metaphors — it provided literal temperature estimates when asked *"how hot is the burn?"*
- **Gemini-1.0-Pro** showed an unexpected dip specifically on stabbing metaphors, at times responding with specific knife types ("butter knife", "kidney stone") despite handling other metaphors correctly.
- **Llama-2-7b-chat** frequently refused to engage with violent-framing lures (safety filtering), which artificially inflated non-lured counts without genuine metaphor comprehension.
- **BioMistral-7B** scored lowest overall and was most susceptible to literal interpretations across all three metaphor types.

**Maieutic experiment:**

- GPT-4o, Gemini-1.0-Pro, and GPT-3.5-Turbo showed **high precision but low recall** — they tended to deny metaphor presence even when it existed. GPT-4o reached 0.99 specificity.
- Llama-2 and BioMistral showed the opposite pattern: more willing to claim metaphor presence, but less accurate in identifying which utterances were truly metaphorical.

**Cross-experiment finding (most notable):**

Even when capable models correctly recognized that a sensation was *not literal*, they frequently misclassified it as **emotional pain** rather than as a physical sensation described metaphorically. A model reading *"burning pain"* would often interpret it as emotional distress (ANGER IS FIRE) rather than a figurative comparison for a physical feeling. This suggests figurative language comprehension in LLMs may be shaped by distributional co-occurrence patterns (emotional fire metaphors are far more common in training data) rather than domain-appropriate reasoning.

---

## Usage

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

### Run an evaluation

```bash
python evaluate.py \
  --provider openai \
  --model gpt-4o \
  --mode elenchus \
  --input MA_scriptie/code/data/elenchus.json \
  --output results/gpt-4o_elenchus.csv
```

**Providers:** `openai` · `gemini` · `anthropic` · `replicate` · `huggingface`
**Modes:** `elenchus` · `maieutic`

Optional flags: `--temperature` (default 0.1) · `--max_tokens` (default 256)

---

## Repository Structure

```
evaluate.py              # Unified CLI evaluation tool (2026 refactor)
requirements.txt
results/
  2023/
    elenchus/            # Per-model elenchus outputs (Response column)
    maieutic/            # Per-model maieutic outputs (utterance, decision, argument, counter_argument)
MA_scriptie/
  code/data/             # Original raw data files and input prompts (JSON)
  docs/thesis_summary.md
  Eres Ferro Bastian_MA Thesis.docx
```

---

## Roadmap

- [ ] Re-run evaluations on current frontier models (GPT-4o-2024, Claude 3.5 Sonnet, Gemini 1.5 Pro, Llama-3)
- [ ] Expand metaphor set: *throbbing*, *drilling*, *jumping*, *shooting*
- [ ] Multi-Agent Debate framework: Agent A evaluates → Agent B critiques → Judge Agent delivers verdict
- [ ] Mechanistic interpretability analysis: identify internal circuits responsible for the emotional/physical mislabeling error (TransformerLens)
- [ ] Workshop paper submission (ACL Fig-Lang 2027)

---

## Citation

If you use this dataset or methodology, please cite:

```
Bastian, E. F. (2023). Exploring Pain Metaphor Comprehension Abilities in Large Language Models.
MA Thesis, Leiden University Centre for Linguistics.
```

---

## Related Work

- Liang et al. (2023) — metaphor detection with ChatGPT, BLOOMZ, Flan-T5
- Wachowiak & Gromann (2023) — GPT-3 and metaphor source domain prediction
- Prystawski et al. (2023) — psychologically informed CoT prompting for metaphor understanding
- Steen (2008) — Deliberate Metaphor Theory
- Zeng et al. (2020) — MedDialog dataset
