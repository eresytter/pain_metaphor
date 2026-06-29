import re
import argparse
import os
import json
import csv
from typing import List, Dict
from dotenv import load_dotenv

from evaluate import get_llm

load_dotenv()

PROMPT_AGENT_A = """You are evaluating a patient's description of their pain.

Utterance: "{utterance}"

Does this utterance contain a metaphorical expression — that is, is the patient describing their pain figuratively rather than literally? State your position (yes or no) and explain your reasoning."""

PROMPT_AGENT_B = """You are a critical reviewer in a structured debate about figurative language in medical contexts.

Utterance: "{utterance}"

Agent A has given the following assessment:
{agent_a_eval}

Your task is to critique Agent A's reasoning. Challenge any assumptions, expose weaknesses in the argument, and make the strongest possible case for the opposing position. Be specific."""

PROMPT_JUDGE = """You are a judge in a structured debate about how a patient's pain description should be interpreted.

Utterance: "{utterance}"

Agent A's assessment:
{agent_a_eval}

Agent B's critique:
{agent_b_critique}

Based on the full debate, classify this utterance into exactly one of these three categories:
- PHYSICAL_METAPHOR: the patient uses figurative language to describe a physical sensation (e.g. "stabbing pain" means pain that feels like stabbing, not literal stabbing)
- EMOTIONAL_METAPHOR: the patient uses figurative language tied to an emotional state (e.g. "burning with anger")
- LITERAL: the description is not metaphorical — it refers to an actual physical event or sensation without figurative comparison

Begin your response with exactly one of these labels on its own line: PHYSICAL_METAPHOR, EMOTIONAL_METAPHOR, or LITERAL.
Then rate your confidence from 0 to 10.
Then explain your reasoning."""


def parse_judge_class(verdict: str) -> str:
    """Return lowercase snake_case label by scanning the first two lines with word-boundary regex."""
    lines = verdict.strip().split("\n")
    first_two = "\n".join(lines[:2]).upper()
    if re.search(r"\bPHYSICAL_METAPHOR\b", first_two):
        return "physical_metaphor"
    if re.search(r"\bEMOTIONAL_METAPHOR\b", first_two):
        return "emotional_metaphor"
    if re.search(r"\bLITERAL\b", first_two):
        return "literal"
    return "unknown"


def parse_confidence(verdict: str) -> str:
    """Extract the first integer 0-10 from the verdict; return as string or '' if absent."""
    m = re.search(r"\b(10|[0-9])\b", verdict)
    return m.group(1) if m else ""


def run_debate(
    llm_a, llm_b, llm_judge,
    prompts: List[Dict],
    file_output: str,
):
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt_a     = ChatPromptTemplate.from_template(PROMPT_AGENT_A)
    prompt_b     = ChatPromptTemplate.from_template(PROMPT_AGENT_B)
    prompt_judge = ChatPromptTemplate.from_template(PROMPT_JUDGE)

    chain = (
        RunnablePassthrough.assign(agent_a_eval=prompt_a | llm_a | StrOutputParser())
        | RunnablePassthrough.assign(agent_b_critique=prompt_b | llm_b | StrOutputParser())
        | RunnablePassthrough.assign(judge_verdict=prompt_judge | llm_judge | StrOutputParser())
    )

    print(f"Running debate on {len(prompts)} utterances...")
    results = chain.batch(prompts, return_exceptions=True, config={"max_concurrency": 8})

    os.makedirs(os.path.dirname(file_output) or ".", exist_ok=True)

    fieldnames = ["utterance", "agent_a_eval", "agent_b_critique", "judge_verdict", "judge_class", "judge_confidence"]
    with open(file_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt, result in zip(prompts, results):
            if isinstance(result, Exception):
                exc = result
                err_str = f"ERROR: {type(exc).__name__}: {exc}"
                writer.writerow({
                    "utterance":        prompt.get("utterance", ""),
                    "agent_a_eval":     err_str,
                    "agent_b_critique": err_str,
                    "judge_verdict":    err_str,
                    "judge_class":      "error",
                    "judge_confidence": "",
                })
            else:
                result["judge_class"]      = parse_judge_class(result["judge_verdict"])
                result["judge_confidence"] = parse_confidence(result["judge_verdict"])
                writer.writerow({k: result[k] for k in fieldnames})

    n_unknown = sum(1 for p, r in zip(prompts, results)
                    if not isinstance(r, Exception) and parse_judge_class(r["judge_verdict"]) == "unknown")
    print(f"Unknown judge_class: {n_unknown}/{len(results)}")
    print(f"Results saved to {file_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate: two LLMs debate a pain metaphor, a judge delivers a verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experimental conditions (Paper 2 locked design):
  Agent A and Judge are fixed to openai/gpt-4.1-mini.
  Only Agent B varies.

  # Homogeneous debate (same-model self-critique)
  python debate.py \\
    --agent-b-provider openai --agent-b-model gpt-4.1-mini \\
    --input MA_scriptie/code/data/maieutic.json \\
    --output results/2026/debate/homogeneous.csv

  # Cross-model debate — Anthropic critic
  python debate.py \\
    --agent-b-provider anthropic --agent-b-model claude-haiku-4-5 \\
    --input MA_scriptie/code/data/maieutic.json \\
    --output results/2026/debate/cross_anthropic.csv

  # Cross-model debate — Google critic
  python debate.py \\
    --agent-b-provider gemini --agent-b-model gemini-2.0-flash \\
    --input MA_scriptie/code/data/maieutic.json \\
    --output results/2026/debate/cross_google.csv

Override Agent A or Judge for future experiments:
  python debate.py \\
    --agent-a-provider anthropic --agent-a-model claude-haiku-4-5 \\
    --agent-b-provider openai    --agent-b-model gpt-4.1-mini \\
    --judge-provider  gemini     --judge-model   gemini-2.0-flash \\
    --input MA_scriptie/code/data/maieutic.json \\
    --output results/2026/debate/custom.csv
""",
    )

    # Agent A — locked to openai/gpt-4.1-mini in the experimental design; overridable
    parser.add_argument("--agent-a-provider", dest="agent_a_provider", default="openai")
    parser.add_argument("--agent-a-model",    dest="agent_a_model",    default="gpt-4.1-mini")

    # Agent B — the only role that varies across experimental conditions
    parser.add_argument("--agent-b-provider", dest="agent_b_provider", required=True)
    parser.add_argument("--agent-b-model",    dest="agent_b_model",    required=True)

    # Judge — locked to openai/gpt-4.1-mini in the experimental design; overridable
    parser.add_argument("--judge-provider", dest="judge_provider", default="openai")
    parser.add_argument("--judge-model",    dest="judge_model",    default="gpt-4.1-mini")

    parser.add_argument("--input",      required=True,  help="Path to input JSONLines file (utterance key)")
    parser.add_argument("--output",     required=True,  help="Path to save the output CSV")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens",  type=int,   default=512)

    args = parser.parse_args()

    provider_a, model_a = args.agent_a_provider, args.agent_a_model
    provider_b, model_b = args.agent_b_provider, args.agent_b_model
    provider_j, model_j = args.judge_provider,   args.judge_model

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        return

    # Load prompts and track actual file line numbers for validation
    prompts_with_lineno = []
    with open(args.input, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                prompts_with_lineno.append((lineno, json.loads(line)))

    if not prompts_with_lineno:
        print("Error: no prompts found in input file.")
        return

    # Validate every entry has a non-empty 'utterance' key (before spending any tokens)
    for lineno, entry in prompts_with_lineno:
        if not entry.get("utterance", "").strip():
            print(f"Error: line {lineno}: missing or empty 'utterance' key.")
            return

    prompts = [entry for _, entry in prompts_with_lineno]

    print(f"Agent A : {provider_a} / {model_a}")
    print(f"Agent B : {provider_b} / {model_b}")
    print(f"Judge   : {provider_j} / {model_j}")

    try:
        llm_a     = get_llm(provider_a, model_a, args.temperature, args.max_tokens)
        llm_b     = get_llm(provider_b, model_b, args.temperature, args.max_tokens)
        llm_judge = get_llm(provider_j, model_j, args.temperature, args.max_tokens)
    except Exception as e:
        print(f"Error initialising LLMs: {e}")
        return

    run_debate(llm_a, llm_b, llm_judge, prompts, args.output)


if __name__ == "__main__":
    main()
