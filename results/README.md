# Results

Raw model outputs from all evaluation runs. Each subdirectory is a year; within each year, outputs are split by prompting strategy.

## 2023 — Original MA Thesis

**Dataset:** 200 utterances (stabbing / burning / gnawing pain metaphors) derived from MedDialog.

### `2023/elenchus/`

One file per model. Each file has a single `Response` column — the model's answer to the lure question.

| File | Model |
|---|---|
| `gpt-4o_elenchus.csv` | GPT-4o (2024-05-13 snapshot) |
| `gpt-3.5-turbo-0125_elenchus.csv` | GPT-3.5-Turbo-0125 |
| `gemini-1.0-pro_elenchus.csv` | Gemini-1.0-Pro |
| `llama-2-7b-chat_elenchus.csv` | Llama-2-7b-chat (via Replicate) |
| `medicine-chat_elenchus.csv` | BioMistral-7B / Medicine-LLM |
| `elenchus_output_rated.csv` | Consolidated file with `lure` + `Response` columns; `lured?` column reserved for manual annotation |

### `2023/maieutic/`

One file per model. Each file has four columns: `utterance`, `decision`, `argument`, `counter_argument`. Split-run files (part_1 / part_2) have been merged into single-model files.

| File | Model |
|---|---|
| `gpt-4o_maieutic.csv` | GPT-4o (merged from two runs) |
| `gpt-3.5-turbo-0125_maieutic.csv` | GPT-3.5-Turbo-0125 |
| `gemini-1.0-pro_maieutic.csv` | Gemini-1.0-Pro (merged from two runs) |
| `llama-2-7b-chat_maieutic.csv` | Llama-2-7b-chat (merged from two runs) |
| `medicine-chat_maieutic.csv` | BioMistral-7B / Medicine-LLM |

---

*Future runs (2026+) will be added as `results/2026/` with the same elenchus/maieutic structure.*
