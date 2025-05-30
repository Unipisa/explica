#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import outlines
from openai import OpenAI
from outlines.models.openai import OpenAIConfig
from pathlib import Path

# Set OpenAI API key
open_api_key = ""  # Ensure this is filled
os.environ['OPENAI_API_KEY'] = open_api_key

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths for input and output files
explica_path = "../../data/explica/explica.csv"
outfolder = "../../data/res/"

# Define choices for relation types
choices = ['so', 'then', 'because', 'after']

# Example sentence pairs with labeled relations for model evaluation
bullet_points = [
    '- Sentence 1: "Jude walked under the rain for an hour."\n   Sentence 2: "Jude got sick."\n   Answer: "so"\n',
    '- Sentence 1: "Mary bought some flowers."\n   Sentence 2: "Mary wants to give a present to her mom."\n   Answer: "because"\n',
    '- Sentence 1: "The girl put her books in the backpack."\n   Sentence 2: "The girl finished her homework."\n   Answer: "after"\n',
    '- Sentence 1: "James took the phone."\n   Sentence 2: "James called Clara."\n   Answer: "then"\n'
]

# Define the columns used for comparison to filter unprocessed rows
compare_columns = ['Sentence_A', 'Sentence_B', 'relation', 'order', 'conn_in_list']

# Define the models to be used
models = ["gpt-4o", "gpt-4o-mini"]

# Define an Outlines prompt template for multiple-choice evaluation
@outlines.prompt
def multiple_choice(sentence_1, sentence_2, connectivesstring, bullet_points_string):
    """
    Select the word that best describes the relationship between the events in these two sentences.

    Use this template: event in sentence 1 <word> event in sentence 2.
    Choose from: ['{{connectivesstring}}'].
    Provide only one word, no explanation.

    Examples:
    {{bullet_points_string}}

    Sentences:
    - Sentence 1: "{{sentence_1}}"
    - Sentence 2: "{{sentence_2}}"

    Answer:
    """

# Function to generate multiple-choice prompts using shuffled bullet points
def multiple_choice_prompt(row, choices, bullet_points):
    random.shuffle(bullet_points)
    bullet_points_string = "\n".join(bullet_points)
    connectivesstring = ", ".join(random.sample(choices, len(choices)))

    return multiple_choice(row['Sentence_A'], row['Sentence_B'], connectivesstring, bullet_points_string)

# Function to check if a token exists in the tokenizer's vocabulary
def is_token_in_vocab(token, tokenizer):
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id != tokenizer.unk_token_id  # Returns True if token exists in vocabulary

# Function to filter rows that haven't been processed yet
def filter_unprocessed_rows(input_df, output_df, compare_columns):
    # Ensure the compare_columns exist in both dataframes
    missing_columns = [col for col in compare_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in output data: {missing_columns}")

    # Identify rows that are in input_df but not yet processed in output_df
    merged_df = pd.merge(input_df, output_df[compare_columns], on=compare_columns, how='left', indicator=True)
    return merged_df[merged_df['_merge'] == 'left_only'][input_df.columns]

# Function to check if a file exists
def check_file_exists(file_path):
    return Path(file_path).exists()

# Load dataset
try:
    df_in = pd.read_csv(explica_path, sep=",")
    print(f"Dataset loaded: {explica_path} ({df_in.shape[0]} rows)")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise SystemExit

# Apply the multiple-choice prompt function correctly using lambda
df_in['prompt'] = df_in.apply(lambda row: multiple_choice_prompt(row, choices, bullet_points), axis=1)

# Number of generations per prompt
num_gen_per_prompt = 1

# Iterate over each model
for model_name in models:
    outpath = f"{outfolder}/cloze_res_{model_name}.tsv"
    print(f"\nProcessing with model: {model_name}")

    # Check if output file exists
    if check_file_exists(outpath):
        print(f"File '{outpath}' exists. Filtering unprocessed rows.")
        df_out = pd.read_csv(outpath, sep="\t")
        df_in = filter_unprocessed_rows(df_in, df_out, compare_columns)
        is_first_row = False
    else:
        print(f"File '{outpath}' does not exist. Processing all rows.")
        is_first_row = True

    # Initialize results storage
    results_out = [np.nan] * df_in.shape[0]
    results_nor = [np.nan] * df_in.shape[0]

    # Configure OpenAI model settings
    config = OpenAIConfig(
        temperature=0.0,
        max_tokens=50,
        n=num_gen_per_prompt,
        seed=0
    )

    # Initialize Outlines model
    try:
        model = outlines.models.openai(model_name, config, api_key=os.environ["OPENAI_API_KEY"])
        generator = outlines.generate.choice(model, choices)
    except Exception as e:
        print(f"Error loading OpenAI model {model_name}: {e}")
        continue  # Skip to the next model if there's an error

    # Process each prompt
    for i, row in enumerate(tqdm(df_in.itertuples(index=False))):
        current_row = row._asdict()  # Convert named tuple to dictionary
        prompt = current_row['prompt']

        try:
            # Generate response using Outlines
            out_out = generator(prompt)

            # Generate response using OpenAI API
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                n=num_gen_per_prompt
            )
            out_norm = completion.choices[0].message.content

            # Store results
            current_row['generated_answer_normal'] = out_norm
            current_row['generated_answer_outlines'] = out_out

            del current_row['prompt']  # Remove prompt before saving

            # Append result to file
            pd.DataFrame([current_row]).to_csv(
                outpath,
                sep="\t",
                index=False,
                header=is_first_row,  # Write header only if it's the first row
                mode='a'  # Append mode
            )

            is_first_row = False  # Ensure header is written only once

        except Exception as e:
            print(f"Error processing prompt {i} for model {model_name}: {e}")

    # Memory cleanup
    del model, generator
    torch.cuda.empty_cache()

print("\nProcessing completed for all models.")
