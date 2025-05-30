#!/usr/bin/env python
# coding: utf-8

import os
import torch
import random
import pandas as pd
import gc
from tqdm import tqdm
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from outlines.models.openai import OpenAIConfig
import outlines

# Define OpenAI API key
open_api_key = ""
os.environ['OPENAI_API_KEY'] = open_api_key

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths for input and output files
explica_path = "../../data/explica/explica.csv"
outfolder = "../../data/res/"

# Define choices for relation types
choices = ["A", "B", "C", "D"]

# Define connectives
connectives = ["because", "so", "after", "then"]

# Define example sentence pairs with labeled relations for model evaluation
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
]

# Define models to be used
models = ["gpt-4o-mini", "gpt-4o"]
num_gen_per_prompt = 1

# Define an Outlines prompt template for multiple-choice evaluation
@outlines.prompt
def multiple_choice(sentence_a, sentence_b, choicestring, connectivesstring, multiple_choices, bullet_points_string):
    """Task Description:
    You are given two sentences, Sentence A and Sentence B, and a list of words. Your task is to select the most appropriate word to connect the two sentences logically and coherently.
    
    Instructions:
    1. Read Sentence A and Sentence B carefully.
    2. Review the list of words provided.
    3. Select the word that best connects the two sentences.
    
    Examples:
    {{bullet_points_string}}
    
    Sentence Connection Task:
    1. Sentence A: "{{sentence_a}}"
    2. Sentence B: "{{sentence_b}}"
    3. Words:\n    {{multiple_choices}}
    4. Answer: """

# Function to generate multiple-choice prompts using shuffled bullet points
def multiple_choice_prompt(x, choices, bullet_point_blocks):
    random.shuffle(connectives)
    random.shuffle(bullet_point_blocks)
    bullet_points_string = "\n".join(bullet_point_blocks)
    
    correct_idx = connectives.index(x.get('true_connective', random.choice(connectives)))
    correct_answer = choices[correct_idx]
    choice_conn_dict = {choice: conn for choice, conn in zip(choices, connectives)}
    multiple_choices = "\n".join([f"- {choice}. {conn}" for choice, conn in zip(choices, connectives)])
    
    choicestring = ", ".join(choices)
    connectivesstring = ", ".join(connectives)
    
    prompt = multiple_choice(x['Sentence_A'], x['Sentence_B'], choicestring, connectivesstring, multiple_choices, bullet_points_string)
    return prompt, correct_answer, choice_conn_dict

# Function to check if a file exists
def check_file_exists(file_path):
    return Path(file_path).exists()

# Load dataset
df_in = pd.read_csv(explica_path, sep=",")

# Generate prompts for each sentence pair
df_in[['prompt', 'correct_answer', 'answers_dict']] = df_in.apply(multiple_choice_prompt, result_type='expand', axis=1, choices=choices, bullet_point_blocks=bullet_point_blocks)

# Process each model
for model_name in models:
    outpath = f"{outfolder}/mc_res_{model_name}.tsv"
    print(f"Processing with model: {model_name}")
    
    df_out = pd.read_csv(outpath, sep="\t") if check_file_exists(outpath) else pd.DataFrame()
    if not df_out.empty:
        print(f"File '{outpath}' exists. Filtering unprocessed rows.")
        df_in = df_in[~df_in.index.isin(df_out.index)]
    
    config = OpenAIConfig(
        temperature=0.0,
        max_tokens=50,
        n=num_gen_per_prompt,
        seed=0,
    )
    
    model = outlines.models.openai(model_name, config, api_key=os.environ["OPENAI_API_KEY"])
    generator = outlines.generate.choice(model, choices)
    
    for i, row in enumerate(tqdm(df_in.itertuples(index=False))):
        current_row = row._asdict()
        prompt = current_row['prompt']
        out_out = generator(prompt)
        
        completion = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
            n=num_gen_per_prompt
        )
        out_norm = completion.choices[0].message.content
        
        current_row['generated_answer_normal'] = out_norm
        current_row['generated_answer_outlines'] = out_out
        
        pd.DataFrame([current_row]).to_csv(outpath, sep="\t", index=False, header=not check_file_exists(outpath), mode='a')
    
    gc.collect()
    torch.cuda.empty_cache()
    del model, generator
