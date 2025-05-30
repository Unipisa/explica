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
from pathlib import Path  # To check for existing files

# Set OpenAI API key
open_api_key = ""  # Ensure this is filled
os.environ['OPENAI_API_KEY'] = open_api_key

# Paths for input and output files
explica_path = "../../data/explica/explica_freq_4800.csv"
outfolder = "../../data/res/"

# Define choice ratings from 1 to 10
choices = [str(x) for x in range(1, 11)]

# List of connectives used for evaluating relationships between events
connective_list = ['so', 'then', 'because', 'after']

# Example sentences with ratings for model evaluation
bullet_point_blocks = [
    '- So (effect): "Jude walked under the rain for an hour, so Jude got sick." (Rating: 10)',
    '- Because (cause): "Mary bought some flowers, because Jean went to the dentist." (Rating: 1)',
    '- After (preceding event): "The girl finished her homework, after the girl put her books in the backpack." (Rating: 1)',
    '- Then (following event): "James took the phone, then James called Clara." (Rating: 10)'
]

# Models to use
models = ["gpt-4o", "gpt-4o-mini"]
num_gen_per_prompt = 1  # Number of generations per prompt

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define an Outlines prompt template for multiple-choice evaluation
@outlines.prompt
def multiple_choice(sentence, connectivesstring, bullet_points_string):
    """
    Evaluate the acceptability of sentences that describe two events linked by connectives.

    Rate each sentence on a scale from 1 to 10 based on how well the connective expresses the relationship between the events.

    Examples:
    {{bullet_points_string}}

    Sentence: {{sentence}}
    Rating:
    """

# Function to generate multiple-choice prompt using random bullet points
def multiple_choice_prompt(x, choices, bullet_points):
    random.shuffle(bullet_points)  # Shuffle examples for diversity
    bullet_points_string = "\n".join(bullet_points)
    connectivesstring = ", ".join(random.sample(choices, len(choices)))  # Randomize connective choices

    return multiple_choice(x['connected_sent'], connectivesstring, bullet_points_string)

# Function to check if a file exists
def check_file_exists(file_path):
    return Path(file_path).exists()

# Function to filter out already processed rows
def filter_unprocessed_rows(input_df, output_df, compare_columns):
    """
    Filters rows from input_df that have not yet been processed in output_df.
    """
    # Ensure all comparison columns exist
    missing_columns = [col for col in compare_columns if col not in output_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in output data: {missing_columns}")

    # Identify rows that are in input_df but not in output_df
    merged_df = pd.merge(input_df, output_df[compare_columns], on=compare_columns, how='left', indicator=True)
    
    # Keep only the unprocessed rows
    return merged_df[merged_df['_merge'] == 'left_only'][input_df.columns]

# Load dataset
try:
    df = pd.read_csv(explica_path, sep=",")
    print(f"Dataset loaded: {explica_path} ({df.shape[0]} rows)")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise SystemExit

# Apply the multiple-choice prompt function correctly using lambda
df['prompt'] = df.apply(lambda row: multiple_choice_prompt(row, connective_list, bullet_point_blocks), axis=1)

# Columns used for checking unprocessed rows
compare_columns = ['connected_sent']  # You may expand this list if needed

# Loop through each model
for model_name in models:
    print(f"Processing with model: {model_name}")

    # Define the output file path
    output_file = f"{outfolder}/accept_res_{model_name}.tsv"

    # Check for existing output file and filter unprocessed rows
    if check_file_exists(output_file):
        print(f"File '{output_file}' exists. Filtering unprocessed rows...")
        df_out = pd.read_csv(output_file, sep="\t")
        df = filter_unprocessed_rows(df, df_out, compare_columns)

    # If there are no unprocessed rows left, skip model processing
    if df.empty:
        print(f"All rows have been processed for {model_name}. Skipping...")
        continue

    # Initialize empty results
    results_out = [np.nan] * df.shape[0]
    results_nor = [np.nan] * df.shape[0]

    # Configure OpenAI API model settings
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
    for i, prompt in enumerate(tqdm(df['prompt'])):
        try:
            # Outlines model generation
            out_out = generator(prompt)
            results_out[i] = str(out_out)

            # OpenAI API call
            completion = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
                n=num_gen_per_prompt
            )

            # Extract response
            choices_res = completion.choices[0].message.content
            results_nor[i] = choices_res

        except Exception as e:
            print(f"Error processing prompt {i} for model {model_name}: {e}")
            results_out[i] = "ERROR"
            results_nor[i] = "ERROR"

    # Store results in DataFrame
    df['generated_answer_normal'] = results_nor
    df['generated_answer_outlines'] = results_out

    # Save results to a new file
    df_to_write = df.drop(columns=["prompt"])  # Remove prompt column before saving
    
    try:
        df_to_write.to_csv(output_file, sep="\t", index=False, mode='a', header=not check_file_exists(output_file))
        print(f"Results appended: {output_file}")
    except Exception as e:
        print(f"Error saving results for model {model_name}: {e}")

    # Memory cleanup
    del model
    torch.cuda.empty_cache()

print("\nProcessing completed for all models.")
