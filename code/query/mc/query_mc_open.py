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
outfolder = "../../data/res/"

# List of pre-trained NLP models
models = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-7b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it"
]

# Define choice ratings and connectives
connectives = ["so", "then", "because", "after"]
choices = ["A", "B", "C", "D"]
batch_size = 8

bullet_point_blocks = [
    "- 1. Sentence A: \"Jude walked under the rain for an hour.\"\n"
    "  2. Sentence B: \"Jude got sick.\"\n"
    "  3. Words:\n"
    "  - A. then\n"
    "  - B. after\n"
    "  - C. because\n"
    "  - D. so\n"
    "  4. Answer: D",
    
    "- 1. Sentence A: \"Mary bought some flowers.\"\n"
    "  2. Sentence B: \"Mary wants to give a present to her mom.\"\n"
    "  3. Words:\n"
    "  - A. because\n"
    "  - B. then\n"
    "  - C. after\n"
    "  - D. so\n"
    "  4. Answer: A",
    
    "- 1. Sentence A: \"The girl put her books in the backpack.\"\n"
    "  2. Sentence B: \"The girl finished her homework.\"\n"
    "  3. Words:\n"
    "  - A. because\n"
    "  - B. after\n"
    "  - C. so\n"
    "  - D. then\n"
    "  4. Answer: B",
    
    "- 1. Sentence A: \"James took the phone.\"\n"
    "  2. Sentence B: \"James called Clara.\"\n"
    "  3. Words:\n"
    "  - A. after\n"
    "  - B. because\n"
    "  - C. then\n"
    "  - D. so\n"
    "  4. Answer: C"
]

# Define an Outlines prompt template for multiple-choice evaluation
@outlines.prompt
def multiple_choice(sentence_a, sentence_b, choicestring, connectivesstring, multiple_choices, bullet_points_string):
    """Task Description:
    
    You are given two sentences, Sentence A and Sentence B, and a list of words. Your task is to select the most appropriate word to connect the two sentences logically and coherently. The chosen word should fit grammatically and contextually.

    Format:
    1. Sentence A: "{{sentence_a}}"
    2. Sentence B: "{{sentence_b}}"
    3. Words:  
    {{multiple_choices}}
    4. Answer: """

# Function to generate multiple-choice prompt using randomized bullet points
def multiple_choice_prompt(x, choices, bullet_point_blocks):
    random.shuffle(bullet_point_blocks)  # Shuffle example bullet points for variety
    bullet_points_string = "\n".join(bullet_point_blocks)
    connectives_shuffled = random.sample(connectives, len(connectives))  # Shuffle connectives for variety
    
    # Create answer mapping
    choice_conn_dict = {choice: conn for choice, conn in zip(choices, connectives_shuffled)}
    multiple_choices = "\n".join([f"- {choice}. {conn}" for choice, conn in zip(choices, connectives_shuffled)])
    choicestring = ", ".join(choices)
    connectivesstring = ", ".join(connectives_shuffled)
    
    # Identify correct answer
    correct_answer = next((key for key, value in choice_conn_dict.items() if value == x['connective']), "none")
    
    prompt = multiple_choice(x['Sentence_A'], x['Sentence_B'], choicestring, connectivesstring, multiple_choices, bullet_points_string)
    return prompt, correct_answer, choice_conn_dict

# Load dataset
df = pd.read_csv(explica_path, sep=",")

# Generate prompts for each sentence pair
df[['prompt', 'correct_answer', 'answers_dict']] = df.apply(
    multiple_choice_prompt, result_type='expand', axis=1, choices=choices, bullet_point_blocks=bullet_point_blocks
)

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
    generator = outlines.generate.choice(outlines_model, choices)

    # Process prompts in batches
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

    # Save results to CSV file
    model_name = model_id.split("/")[1]
    df.to_csv(f"{outfolder}/mc_res_{model_name}.csv", sep="\t", index=False)

    # Cleanup memory
    gc.collect()
    torch.cuda.empty_cache()
