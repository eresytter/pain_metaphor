# Thesis Summary: LLMs and Pain Metaphors

## Core Research Question and Methodology
This thesis investigates the capabilities of Large Language Models (LLMs) in comprehending pain metaphors.

**Research Questions:**
- **RQ1:** How does GPT-3.5, BioMedLM, and Medicine-LLM perform in deliberate and non-deliberate pain metaphor testing?
- **RQ2:** Can GPT-3.5, BioMedLM, and Medicine-LLM elaborate on why a sentence is considered to contain a metaphorical expression?

**Methodology:**
- An empirical study utilizing a dataset of 200 utterances containing "stabbing", "burning", and "gnawing" pain metaphors.
- Uses *elenchus prompting* (luring models into thinking the pain is literal).
- Uses *maieutic prompting* (a three-step chain-of-thought method) to extract and evaluate if LLMs can elaborate on their metaphorical understanding through counterarguments.

## Key Findings on LLM Perception of Pain Metaphors
- **GPT-4o** was the most consistent and accurate model (0.92 accuracy) across both direct and indirect metaphors.
- **GPT-3.5-Turbo** and **Llama-2** performed decently but were more easily lured into literal interpretations compared to GPT-4o.
- **Gemini-1.0-Pro** and **BioMistral-7B** struggled significantly, showing a high susceptibility to being lured by "stabbing pain" metaphors into viewing them literally.
- A critical observation: Even when capable models (like GPT-4o, GPT-3.5, and Gemini) recognized that sensations like "stabbing" or "burning" were not literal acts of violence or fire, they frequently misinterpreted them as *emotional pain* rather than a metaphorical comparison for a physical sensation.

## Known Limitations from the Study
- **Data Scarcity:** Medical dialogues are highly protected due to PII. There were no existing datasets focusing on pain metaphors, requiring a custom, limited dataset.
- **Scope Limitations:** Due to time and resource constraints, the dataset is limited to only 200 utterances and specifically covers "stabbing", "burning", and "gnawing" pain.
- **Excluded Metaphors:** Many other common pain metaphors such as "throbbing", "drilling", "jumping", and "shooting" were not covered.
