# Repository for Eres's MA Thesis: Pain Metaphor Comprehension

Topic: Exploring Pain Metaphor Comprehension Abilities in Large Language Models

Supervisors: Stephan Raaijmakers, Lettie Dorst

MA Linguistics, Leiden University

## Unified Evaluation Tool

The `evaluate.py` script provides a unified interface for evaluating different LLM providers using two prompting strategies: Elenchus and Maieutic.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your environment variables:
   Copy `.env.example` to `.env` and add your API keys.

### Usage

```bash
python evaluate.py --provider [openai|gemini|anthropic|replicate|huggingface] \
                  --model [model_name] \
                  --mode [elenchus|maieutic] \
                  --input path/to/input.json \
                  --output path/to/output.csv
```

Example (Elenchus):
```bash
python evaluate.py --provider openai --model gpt-4o --mode elenchus --input MA_scriptie/code/data/elenchus.json --output results.csv
```

Example (Maieutic):
```bash
python evaluate.py --provider gemini --model gemini-1.5-pro --mode maieutic --input MA_scriptie/code/data/maieutic.json --output results.csv
```
