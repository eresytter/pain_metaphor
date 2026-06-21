import argparse
import os
import json
import csv
from typing import List, Dict
from dotenv import load_dotenv

# Try to load environment variables from .env file
load_dotenv()

def get_llm(provider: str, model: str, temperature: float = 0.1, max_tokens: int = 256):
    """
    Factory function to get the appropriate LLM instance.
    """
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif provider == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)
    
    elif provider == 'replicate':
        from langchain_community.llms import Replicate
        return Replicate(model=model, model_kwargs={"temperature": temperature, "max_tokens": max_tokens})
    
    elif provider == 'huggingface':
        from langchain_community.llms import HuggingFaceEndpoint
        return HuggingFaceEndpoint(repo_id=model, temperature=temperature, max_new_tokens=max_tokens)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

def run_elenchus(llm, prompts: List[Dict], file_output: str):
    """
    Runs the Elenchus prompting strategy.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    template = """
Your task is to answer the question below.

Question: {lure}
Answer:
"""
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | llm | StrOutputParser()

    print(f"Running elenchus mode on {len(prompts)} prompts...")
    responses = chain.batch(prompts)

    with open(file_output, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Response'])
        writer.writerows([[res] for res in responses])
    
    print(f"Results saved to {file_output}")

def run_maieutic(llm, prompts: List[Dict], file_output: str):
    """
    Runs the Maieutic (Chain-of-Thought with self-critique) prompting strategy.
    """
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Step 1: Initial decision
    prompt_one = ChatPromptTemplate.from_template(
        "Decide if \"{utterance}\" contains a metaphorical comparison or not. Limit response to yes or no."
    )
    
    # Step 2: Reasoning and confidence
    prompt_two = ChatPromptTemplate.from_template(
        "You have said {decision} if \"{utterance}\" contains a metaphorical comparison or not. "
        "Provide a reason and rate how confident you are from 0 to 10."
    )
    
    # Step 3: Counter-argument
    prompt_three = ChatPromptTemplate.from_template(
        "Provide a counter-argument against \"{argument}\". Rate how confident you are from 0 to 10."
    )

    chain = (
        RunnablePassthrough.assign(decision=prompt_one | llm | StrOutputParser())
        | RunnablePassthrough.assign(argument=prompt_two | llm | StrOutputParser())
        | RunnablePassthrough.assign(counter_argument=prompt_three | llm | StrOutputParser())
    )

    print(f"Running maieutic mode on {len(prompts)} prompts...")
    results = chain.batch(prompts)

    fieldnames = ['utterance', 'decision', 'argument', 'counter_argument']
    with open(file_output, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Ensure we only write the fields we want
            writer.writerow({k: row[k] for k in fieldnames})
            
    print(f"Results saved to {file_output}")

def main():
    parser = argparse.ArgumentParser(description="Unified Pain Metaphor Evaluation Tool")
    parser.add_argument("--provider", type=str, required=True, 
                        choices=['openai', 'gemini', 'anthropic', 'replicate', 'huggingface'],
                        help="LLM provider")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name/identifier (e.g., 'gpt-4o', 'gemini-1.5-pro')")
    parser.add_argument("--mode", type=str, required=True, 
                        choices=['elenchus', 'maieutic'],
                        help="Prompting strategy")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input JSONLines file")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to save the output CSV")
    parser.add_argument("--temperature", type=float, default=0.1, 
                        help="Model temperature (default: 0.1)")
    parser.add_argument("--max_tokens", type=int, default=256, 
                        help="Max tokens to generate (default: 256)")

    args = parser.parse_args()

    # Load prompts
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    prompts = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    if not prompts:
        print("Error: No prompts found in the input file.")
        return

    # Initialize LLM
    try:
        llm = get_llm(args.provider, args.model, args.temperature, args.max_tokens)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return

    # Execute selected mode
    if args.mode == 'elenchus':
        run_elenchus(llm, prompts, args.output)
    elif args.mode == 'maieutic':
        run_maieutic(llm, prompts, args.output)

if __name__ == "__main__":
    main()
