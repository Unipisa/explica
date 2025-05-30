import os
import torch
import pandas as pd
import argparse
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, logout


# Device configuration (use CUDA if available, otherwise fallback to CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for input and output files
explica_path = "../../data/explica/explica_freq_4800.csv"
outfolder = "../../data/res/"
token_hf = ""  # Placeholder for Hugging Face API token

# List of models to evaluate
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



# Login to Hugging Face Hub
login(token_hf)

# Load dataset and sort by 'pair_id'
data = pd.read_csv(explica_path, sep=",")
data = data.sort_values(by='pair_id').reset_index(drop=True)

def calculate_perplexity(sentence):
    """
    Calculate the perplexity of a given sentence using the loaded model.
    """
    inputs = tokenizer(sentence, return_tensors='pt')
    input_ids = inputs.input_ids.to(device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity

def run_test(x):
    """
    Compute perplexity for a given input sentence.
    """
    prompt = x.connected_sent
    return calculate_perplexity(prompt)

def rank_perplexity(df):
    """
    Rank the data based on 'pair_id' and 'perplexity'.
    """
    df['rank'] = df.groupby(['Sentence_A', 'Sentence_B'])['perplexity'].rank(method='dense', ascending=True).astype(int)
    return df

# Iterate over each model and perform perplexity computation
for model_name in models:
    print(f"Processing model: {model_name}")
    outputs = []
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply perplexity calculation using tqdm progress bar
    tqdm.pandas()
    results = data.progress_apply(run_test, axis=1)
    
    # Store results and rank them
    data["perplexity"] = results
    df = rank_perplexity(data)
    
    # Save results to a CSV file
    outfile_name = f"{outfolder}/perplexity/{model_name.split('/')[-1]}.csv"
    df.to_csv(outfile_name, sep="\t", index=False)
    print(f"Results saved to {outfile_name}")
