#!/usr/bin/env python
# coding: utf-8

import os

# Importing necessary libraries for NLP and data handling
from transformers import AutoTokenizer, AutoModelForCausalLM  # For loading and tokenizing models
import torch  # PyTorch for deep learning
from tqdm import tqdm  # For displaying progress bars
import outlines  # A library for structured text generation
import random  # For randomization
import pandas as pd  # For data manipulation
import gc  # Garbage collection to free up memory
from huggingface_hub import notebook_login, login, logout  # For accessing Hugging Face models

# Authenticate with Hugging Face (this will prompt for login credentials)
notebook_login()

# List of pre-trained NLP models to be used
models = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-7b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct"
]

# Paths for input and output files
explica_path = "../../data/explica/explica_freq_4800.csv"
outfolder = "../../data/res/"
token_hf = ""  # Placeholder for Hugging Face API token


# Define choice ratings from 1 to 10
choices = [str(x) for x in range(1, 11)]

# List of connectives used for evaluating relationships between events
connective_list = ['so', 'then', 'because', 'after']

# Example sentences with ratings, used to guide model evaluation
bullet_point_blocks = [
    '- So (effect): "Jude walked under the rain for an hour, so Jude got sick." (Rating: 10)',
    '- Because (cause): "Mary bought some flowers, because Jean went to the dentist." (Rating: 1)',
    '- After (preceding event): "The girl finished her homework, after the girl put her books in the backpack." (Rating: 1)',
    '- Then (following event): "James took the phone, then James called Clara." (Rating: 10)'
]

# Login to Hugging Face with provided token
login(token_hf)

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
    random.shuffle(bullet_points)  # Shuffle example bullet points for variety
    bullet_points_string = "\n".join(bullet_points)
    connectivesstring = ", ".join(random.sample(choices, len(choices)))  # Randomize connective choices

    prompt = multiple_choice(x['connected_sent'], connectivesstring, bullet_points_string)
    print(f"Prompt: {prompt}")  # Debugging print statement
    return prompt

# Function to check if a specific token exists in the tokenizer's vocabulary
def is_token_in_vocab(token, tokenizer):
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id != tokenizer.unk_token_id  # Returns True if token exists in vocabulary

# Load dataset from TSV file
df = pd.read_csv(explica_path, sep=",")



# Apply the multiple-choice prompt function to the dataset
df['prompt'] = df.apply(multiple_choice_prompt, result_type='expand', axis=1, choices=connective_list, bullet_points=bullet_point_blocks)

# Loop through each model to process prompts
for model_id in models:
    print(f"Processing with model: {model_id}")

    # Load tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Attempt to load model with Flash Attention, fallback if unsupported
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to("cuda")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")

    # Initialize Outlines model for text generation
    outlines_model = outlines.models.Transformers(model, tokenizer)
    generator = outlines.generate.choice(outlines_model, choices)

    # Batch processing setup
    batch_size = 16
    prompts = df['prompt'].tolist()
    results = []

    # Generate responses in batches
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        responses = generator(batch_prompts)
        results.extend(responses)

    # Store generated responses in DataFrame
    df['generated_answer_outlines'] = results

    # Cleanup: Remove model and generator from memory
    del outlines_model
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    # Move model to available device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate responses using greedy decoding
    results = []
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Generate text using the model (without sampling)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_return_sequences=1,
                do_sample=False,  # Greedy decoding
                temperature=0,
                max_new_tokens=5,  # Generate up to 5 new tokens
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode generated tokens
        gen_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:])[0]
        results.append(gen_text)

    # Store results in DataFrame
    df['generated_answer_greedy'] = results

    # Save results to TSV file
    model_name = model_id.split("/")[1]
    df.to_csv(f"{outfolder}/accept_res_{model_name}.tsv", sep="\t", index=False)

    # Cleanup memory
    gc.collect()
    torch.cuda.empty_cache()
