# Model evaluation

Create a Python 3 environment and install the requirements.

## Query

The query folder contains the Python script to query the open models and Gpt models. Variables are at the beginning of the script. Among those, the *GPT Api key* and *Hugging Face Token* must be inserted.
The provided code is only for few-shot learning. Commenting the "Examples" part of the prompt template to query the model in zero-shot setting is sufficient.


## Answer analysis

The answer analysis consists of two steps:
* Answer cleaning: must be applied to answers returned by the models in the greedy search setup
* outline_accuracy and greedy_accuracy, allow for computing accuracy in these two setups along with perplexity.
  
The code for the analyses is provided in Jupyter Notebooks. In each notebook, paths to files are defined in a Python dictionary.

## Other

This folder contains the notebooks for other analyses and plots: frequency, distribution & correlation, incremental model size, and error analysis.


