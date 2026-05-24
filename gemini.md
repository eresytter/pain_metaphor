# Project Context: MA_scriptie (LLM Pain Metaphor Evaluation)

> **Note to future agents:** Please read this document carefully before making any changes or starting new tasks. It contains critical context about the repository's purpose, structure, and the user's research goals.

## Research Overview
This repository contains the codebase and data for a Master's thesis investigating how capable Large Language Models (LLMs) are at detecting and understanding pain metaphors (e.g., "stabbing", "burning", "gnawing" pain) in patient-doctor dialogue contexts. 

### Core Research Questions
1. How do different LLMs (GPT-3.5, GPT-4o, Gemini-1.0-Pro, Llama-2, BioMistral, etc.) perform in deliberate and non-deliberate pain metaphor testing?
2. Can these models elaborate on *why* a sentence is considered to contain a metaphorical expression?

### Methodology
The research employs two main prompting strategies using LangChain:
- **Elenchus Prompting:** Luring the model into thinking a pain description is literal (e.g., asking "How big was the hammer?" when a patient says "It feels as if someone hit my heel with a hammer.").
- **Maieutic Prompting:** A three-step recursive chain-of-thought method. It asks the model to decide if there's a metaphor, provide a reason, and then formulate a counter-argument against its own reason. This tests if the model has latent metaphorical understanding.

### Key Findings So Far
- **GPT-4o** is highly accurate and consistent.
- **GPT-3.5-Turbo** and **Llama-2** are decent but more susceptible to literal lures.
- **Gemini-1.0-Pro** and **BioMistral** struggle heavily and are easily lured.
- A common flaw across capable models is mistaking a physical pain metaphor (like "burning") for *emotional pain*, rather than recognizing it as a figurative comparison for a physical sensation.

## Tech Stack & Architecture
- **Language:** Python
- **Core Library:** LangChain (used to build the prompting pipelines)
- **Providers Configured:** OpenAI, Google GenAI, Replicate (Llama), HuggingFace Endpoint (MedicineChat).

### Repository Structure
- `/data/`: Contains the datasets (`.json` files with `lure` and `utterance` keys) and the output `.csv` evaluation files.
- `/docs/`: Contains summary documents, such as `thesis_summary.md`.
- `/code/evaluate_model.py`: The unified CLI tool for running evaluations. It takes arguments for `--provider`, `--model`, `--mode` (elenchus or maieutic), `--input`, and `--output`.

## Future Work / Immediate Next Steps
The research is expanding towards Multi-Agent Debate logic (Plan 2):
- Developing a system where multiple LLMs debate the metaphor data points to reach a consensus.
- Example flow: Agent A evaluates the utterance, Agent B critiques Agent A, and a Judge Agent provides a final verdict based on the transcript.

## Guidelines for Agents
1. **Maintain Python purity:** Use `argparse` for CLI tools and abstract repetitive logic (DRY).
2. **LangChain Usage:** For pipelines, stick to `RunnablePassthrough`, `ChatPromptTemplate`, and `StrOutputParser` as established in `evaluate_model.py`.
3. **Execution Safety:** When running evaluation scripts, be mindful of API keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc.) which the user will need to provide. Ensure imports that require third-party libraries are scoped properly or fail gracefully.
