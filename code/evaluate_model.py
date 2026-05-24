import argparse
import os
import getpass
import json
import csv

def get_llm(provider, model):
    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key: ")
        return ChatOpenAI(model=model, temperature=0.1, max_tokens=256)
    elif provider == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Google API key: ")
        return ChatGoogleGenerativeAI(model=model, temperature=0.1, max_tokens=256)
    elif provider == 'replicate':
        from langchain_community.llms import Replicate
        if "REPLICATE_API_TOKEN" not in os.environ:
            os.environ["REPLICATE_API_TOKEN"] = getpass.getpass("Replicate API key: ")
        return Replicate(model=model, temperature=0.1, max_tokens=256)
    elif provider == 'medicinechat':
        from langchain.llms import HuggingFaceEndpoint
        # Make sure HUGGINGFACEHUB_API_TOKEN is set
        endpoint_url = "endpoint_url"
        return HuggingFaceEndpoint(
            endpoint_url=endpoint_url,
            huggingface_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

def run_elenchus(llm, prompts, file_output):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    template = """
Your task is to answer the question below.

Question: {lure}
Answer:
"""
    prompt_template = ChatPromptTemplate.from_template(template)
    runnable = prompt_template | llm | StrOutputParser()

    print(f"Running elenchus mode on {len(prompts)} prompts...")
    response = runnable.batch(prompts)

    with open(file_output, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Response'])
        writer.writerows([[item] for item in response])
    print(f"Output saved to {file_output}")

def run_maieutic(llm, prompts, file_output):
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    prompt_one = ChatPromptTemplate.from_template(
        "Decide if {utterance} contains a metaphorical comparison or not. Limit response to yes or no."
    )
    runnable_one = prompt_one | llm | StrOutputParser()
    chain_one = RunnablePassthrough.assign(decision=runnable_one)

    prompt_two = ChatPromptTemplate.from_template(
        "You have said {decision} if \"{utterance}\" contains a metaphorical comparison or not. Provide a reason and rate how confident you are from 0 to 10."
    )
    runnable_two = prompt_two | llm | StrOutputParser()
    chain_two = chain_one | RunnablePassthrough.assign(argument=runnable_two)

    prompt_three = ChatPromptTemplate.from_template(
        "Provide a counter-argument against \"{argument}\". Rate how confident you are from 0 to 10."
    )
    runnable_three = prompt_three | llm | StrOutputParser()
    chain_three = chain_two | RunnablePassthrough.assign(counter_argument=runnable_three)

    print(f"Running maieutic mode on {len(prompts)} prompts...")
    response = chain_three.batch(prompts)

    fieldnames = ['utterance', 'decision', 'argument', 'counter_argument']
    with open(file_output, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in response:
            writer.writerow(row)
    print(f"Output saved to {file_output}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on pain metaphors.")
    parser.add_argument("--provider", choices=['openai', 'gemini', 'replicate', 'medicinechat'], required=True, help="LLM Provider")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt-4o, gemini-1.0-pro)")
    parser.add_argument("--mode", choices=['elenchus', 'maieutic'], required=True, help="Prompting mode")
    parser.add_argument("--input", required=True, help="Input JSONLines file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()

    # Load prompts
    prompts = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(json.loads(line))

    # Get LLM
    llm = get_llm(args.provider, args.model)

    # Run Mode
    if args.mode == 'elenchus':
        run_elenchus(llm, prompts, args.output)
    elif args.mode == 'maieutic':
        run_maieutic(llm, prompts, args.output)

if __name__ == "__main__":
    main()
