# Re-import libraries and redefine everything after code execution reset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Models and task definitions
models = ["GPT4o-mini", "GPT4o", "Falcon", "Gemma", "Llama", "Mistral", "Qwen"]
task_order = ["APS", "Acceptability", "Cloze", "MC"]
task_colors = {
    "Acceptability": "#1f77b4",
    "Cloze": "#2ca02c",
    "MC": "#ff7f0e",
    "APS": "#9467bd"
}

# Data for Greedy Search
greedy_data = {
    "Acceptability_FEW": [0.71, 0.78, 0.20, 0.51, 0.33, 0.25, 0.48],
    "Acceptability_ZERO": [0.63, 0.69, 0.19, 0.59, 0.52, 0.29, 0.54],
    "Cloze_FEW": [0.54, 0.69, 0.24, 0.02, 0.38, 0.49, 0.56],
    "Cloze_ZERO": [0.52, 0.55, 0.25, 0.20, 0.39, 0.41, 0.37],
    "MC_FEW": [0.00, 0.59, 0.19, 0.00, 0.00, 0.40, 0.50],
    "MC_ZERO": [0.00, 0.54, 0.00, 0.00, 0.00, 0.39, 0.50],
}

# Data for Outlines
outlines_data = {
    "Acceptability_FEW": [0.64, 0.78, 0.26, 0.33, 0.24, 0.19, 0.23],
    "Acceptability_ZERO": [0.47, 0.77, 0.26, 0.57, 0.26, 0.22, 0.28],
    "Cloze_FEW": [0.54, 0.70, 0.24, 0.52, 0.26, 0.43, 0.29],
    "Cloze_ZERO": [0.53, 0.54, 0.29, 0.53, 0.28, 0.36, 0.31],
    "MC_FEW": [0.32, 0.67, 0.24, 0.51, 0.26, 0.41, 0.28],
    "MC_ZERO": [0.29, 0.63, 0.27, 0.50, 0.26, 0.40, 0.28],
}

# APS results
aps_overall = {
    "GPT4o-mini": np.nan,
    "GPT4o": np.nan,
    "Falcon": 0.66,
    "Gemma": 0.62,
    "Llama": 0.65,
    "Mistral": 0.65,
    "Qwen": 0.59,
}

# Build dataframes and add APS column
df_greedy = pd.DataFrame(greedy_data, index=models)
df_outlines = pd.DataFrame(outlines_data, index=models)
df_greedy["APS"] = [aps_overall[m] for m in models]
df_outlines["APS"] = [aps_overall[m] for m in models]

# Plot setup
x = np.arange(len(models))
width = 0.12
offsets = [-3, -2, -1, 0, 1, 2, 3]

fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# GREEDY PLOT
for i, task in enumerate(task_order):
    if task == "APS":
        axs[0].bar(x + offsets[i]*width, df_greedy["APS"], width, color=task_colors["APS"])
    else:
        axs[0].bar(x + offsets[i*2 - 1]*width, df_greedy[f"{task}_FEW"], width, label=f"{task} FEW", color=task_colors[task])
        axs[0].bar(x + offsets[i*2]*width, df_greedy[f"{task}_ZERO"], width, label=f"{task} ZERO", color=task_colors[task], alpha=0.5)

axs[0].set_title("Model Performance with Greedy Search and APS", fontsize=23)
axs[0].set_ylabel("Score", fontsize=21)
axs[0].tick_params(axis='y', labelsize=19)
axs[0].set_ylim(0, 1)

# OUTLINES PLOT
for i, task in enumerate(task_order):
    if task == "APS":
        axs[1].bar(x + offsets[i]*width, df_outlines["APS"], width, color=task_colors["APS"])
    else:
        axs[1].bar(x + offsets[i*2 - 1]*width, df_outlines[f"{task}_FEW"], width, color=task_colors[task])
        axs[1].bar(x + offsets[i*2]*width, df_outlines[f"{task}_ZERO"], width, color=task_colors[task], alpha=0.5)

axs[1].set_title("Model Performance with Outlines and APS", fontsize=23)
axs[1].set_ylabel("Score", fontsize=21)
axs[1].set_xticks(x)
axs[1].set_xticklabels(models, rotation=0, fontsize=21)
axs[1].tick_params(axis='y', labelsize=19)
axs[1].set_ylim(0, 1)

# Legend
legend_labels = [
    "Acceptability FEW", "Acceptability ZERO",
    "Cloze FEW", "Cloze ZERO",
    "MC FEW", "MC ZERO",
    "APS"
]
legend_handles = [
    plt.Rectangle((0,0),1,1, color=task_colors["Acceptability"]),
    plt.Rectangle((0,0),1,1, color=task_colors["Acceptability"], alpha=0.5),
    plt.Rectangle((0,0),1,1, color=task_colors["Cloze"]),
    plt.Rectangle((0,0),1,1, color=task_colors["Cloze"], alpha=0.5),
    plt.Rectangle((0,0),1,1, color=task_colors["MC"]),
    plt.Rectangle((0,0),1,1, color=task_colors["MC"], alpha=0.5),
    plt.Rectangle((0,0),1,1, color=task_colors["APS"])
]

fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=False, fontsize=20, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()
