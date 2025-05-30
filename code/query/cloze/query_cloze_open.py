#!/usr/bin/env python
# coding: utf-8

import os
import torch
import random
import pandas as pd
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import outlines

# Define Hugging Face authentication token (leave blank if not using authentication)
token_hf = ""
login(token_hf)

# File paths
explica_path = "../../data/explica/explica.csv"
outfolder = "../../data/res"

# List of pre-trained NLP models to be used
models = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-7b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it"
]

# Define choice ratings and connectives
connective_list = ['so', 'then', 'because', 'after']

# Example sentences with correct connective answers for reference
bullet_points = [
    '- Sentence 1: "Jude walked under the rain for an hour."\n   Sentence 2: "Jude got sick."\n   Answer: "so"\n',
    '- Sentence 1: "Mary bought some flowers."\n   Sentence 2: "Mary wants to give a present to her mom."\n   Answer: "because"\n',
    '- Sentence 1: "The girl put her books in the backpack."\n   Sentence 2: "The girl finished her homework."\n   Answer: "after"\n',
    '- Sentence 1: "James took the phone."\n   Sentence 2: "James called Clara."\n   Answer: "then"\n'
]

# Define an Outlines prompt template for multiple-choice evaluation
@outlines.prompt
def multiple_choice(sentence_1, sentence_2, connectivesstring, bullet_points_string):
    """Select the word that best describes the relationship between the events in these two sentences.
    
    Use this template: event in sentence 1 <word> event in sentence 2.
    Choose from: ['{{connectivesstring}}'].
    Provide only one word, no explanation.
    
    Examples:
    {{bullet_points_string}}
    
    Sentences:
    - Sentence 1: "{{sentence_1}}"
    - Sentence 2: "{{sentence_2}}"
    
    Answer: """

# Function to generate multiple-choice prompt using randomized bullet points
def multiple_choice_prompt(x, choices, bullet_points):
    random.shuffle(bullet_points)  # Shuffle example bullet points for variety
    bullet_points_string = "\n".join(bullet_points)
    connectivesstring = "', '".join(random.sample(choices, len(choices)))  # Randomize connective choices
    return multiple_choice(x['Sentence_A'], x['Sentence_B'], connectivesstring, bullet_points_string)

# Load dataset
df = pd.read_csv(explica_path, sep=",")

# Generate prompts for each sentence pair
df['prompt'] = df.apply(multiple_choice_prompt, axis=1, choices=connective_list, bullet_points=bullet_points)

# Process each model
for model_id in models:
    print(f"Processing with model: {model_id}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
        ).to("cuda")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda")
    
    # Initialize Outlines model for text generation
    outlines_model = outlines.models.Transformers(model, tokenizer)
    generator = outlines.generate.choice(outlines_model, connective_list)
    
    # Process prompts in batches
    batch_size = 16
    prompts = df['prompt'].tolist()
    results = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        responses = generator(batch_prompts)
        results.extend(responses)
    
    # Store results in DataFrame
    df['generated_answer_outlines'] = results

    # Cleanup memory
    del outlines_model, generator
    gc.collect()
    torch.cuda.empty_cache()

    # Move model to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Generate responses using greedy decoding
    results = []
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, num_return_sequences=1, do_sample=False, temperature=0, max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:])[0]
        results.append(gen_text)
    
    # Store results in DataFrame
    df['generated_answer_greedy'] = results

    # Save results to TSV file
    model_name = model_id.split("/")[1]
    df.to_csv(f"{outfolder}/cloze_res_{model_name}.tsv", sep="\t", index=False)
    
    # Cleanup memory
    gc.collect()
    torch.cuda.empty_cache()
