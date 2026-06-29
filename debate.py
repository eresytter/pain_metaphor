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

PROMPT_JUDGE = """You are a judge in a structured debate about whether a patient's pain description contains a metaphorical expression.

Utterance: "{utterance}"

Agent A's assessment:
{agent_a_eval}

Agent B's critique:
{agent_b_critique}

Based on the full debate above, deliver your final verdict. State whether this utterance contains a pain metaphor (yes or no), explain your reasoning, and rate your confidence from 0 to 10."""


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
    results = chain.batch(prompts)

    os.makedirs(os.path.dirname(file_output) or ".", exist_ok=True)

    fieldnames = ["utterance", "agent_a_eval", "agent_b_critique", "judge_verdict"]
    with open(file_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Results saved to {file_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate: two LLMs debate a pain metaphor, a judge delivers a verdict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Homogeneous debate — all three roles use the same model
  python debate.py --provider openai --model gpt-4o \\
    --input MA_scriptie/code/data/maieutic.json \\
    --output results/2026/debate/gpt-4o_homogeneous.csv

  # Cross-provider debate — different model per role
  python debate.py \\
    --agent-a-provider openai     --agent-a-model gpt-4o \\
    --agent-b-provider anthropic  --agent-b-model claude-sonnet-4-6 \\
    --judge-provider  gemini      --judge-model   gemini-1.5-pro \\
    --input MA_scriptie/code/data/maieutic.json \\
    --output results/2026/debate/gpt4o-vs-claude-judge-gemini.csv
""",
    )

    # Homogeneous shorthand
    parser.add_argument("--provider", help="Provider for all three roles (overridden by per-role flags)")
    parser.add_argument("--model",    help="Model for all three roles (overridden by per-role flags)")

    # Per-role overrides
    for role in ("agent-a", "agent-b", "judge"):
        parser.add_argument(f"--{role}-provider", dest=f"{role.replace('-', '_')}_provider")
        parser.add_argument(f"--{role}-model",    dest=f"{role.replace('-', '_')}_model")

    parser.add_argument("--input",      required=True,  help="Path to input JSONLines file (utterance key)")
    parser.add_argument("--output",     required=True,  help="Path to save the output CSV")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens",  type=int,   default=512)

    args = parser.parse_args()

    def resolve(role_provider_attr, role_model_attr):
        provider = getattr(args, role_provider_attr) or args.provider
        model    = getattr(args, role_model_attr)    or args.model
        if not provider or not model:
            parser.error(
                f"Specify either --provider/--model (homogeneous) or "
                f"--{role_provider_attr.replace('_', '-')} and "
                f"--{role_model_attr.replace('_', '-')} (per-role)."
            )
        return provider, model

    provider_a, model_a = resolve("agent_a_provider", "agent_a_model")
    provider_b, model_b = resolve("agent_b_provider", "agent_b_model")
    provider_j, model_j = resolve("judge_provider",   "judge_model")

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        return

    prompts = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    if not prompts:
        print("Error: no prompts found in input file.")
        return

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
