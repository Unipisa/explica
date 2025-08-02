# ExpliCa
## Evaluating Explicit Causal Reasoning in Large Language Models

ExpliCa (Explicit Causality) is a dataset designed to evaluate LLMs on commonsense causal reasoning through causal discovery tasks, more specifically via Pairwise Causal Discovery (PCD), which focuses on determining the existence of a causal relationship between two events and establishing the causal direction, that is, identifying which event serves as the cause and which as the effect.
In ExpliCa PCD is formulated to take into account also the entanglement of causal and temporal relations between events. 

### Repo description

This repository provides insights into model behavior and evaluation methodologies, offering valuable benchmarks for future research:
* The folder code contains Python scripts and notebooks used to query the models and analyze the answers
* The folder data/res/ contains the models' answers to each task. 
* The folder imgs contains some plots describing the analysis computed over ExpliCa by evaluating the LLMs.

## The dataset
ExpliCa contains 600 sentence pairs, each presented in both possible orders. Each sentence pair has been joined in a single sentence through connectives that explicitly indicate causal or temporal relations, as well as the direction of those relations, i.e., iconic (i.e., when the cause/antecedent event in sentence A linguistically appears  before  the effect/the subsequent event in sentence B) and anti-iconic (effect-cause; subsequent-antecedent):
* A causes B > “so”
* B causes A > “because”
* A precede B > “then”
* B precede A > “after”

### Human ratings 
We collected human acceptability ratings concerning the connective in each sentence via crowdsourcing. Subjects rated on a scale from 1 to 10 how acceptable the connective is to express the relation between the events in the sentences.
For example:
* Jude walked under the rain for an hour, so Jude got sick: Rating 10 (highly acceptable)
* Mary bought some flowers, because Jean went to the dentist: Rating 1 (not acceptable)
Each sentence was rated by 15 English native speakers


### Structure
* Pair_ID: ID of the sentence pair
* Sentence_A: sentence A
* Sentence_B: sentence B
* rating_iconic_causal: the average rating given to sentence pairs connected by a causal connective, where the sentence order follows an iconic direction, i.e., A SO B.
* rating_anticonic_causal: the average rating given to sentence pairs connected by a causal connective, where the sentence order follows an anti-iconic direction, i.e., A BECAUSE B.
* rating_iconic_temporal: the average rating given to sentence pairs connected by a temporal connective, where the sentence order follows an iconic direction, i.e., A THEN B.
* rating_anticonic_temporal: the average rating given to sentence pairs connected by a temporal connective, where the sentence order follows an anti-iconic direction, i.e., A AFTER B.
* human_preferred_connective: connective for which humans provided the highest average rating
* human_preferred_connective_desc: indication of order (iconic vs anti-iconic) and relation (causal vs temporal) of the connective for which humans provided the highest average rating

#### Additional columns in the 4800 version

* origin: to which dataset the sentence pair originally belonged
* frequency: the frequency of the triplets \{Sent1verb, connective, Sent2verb\} computed on enTenTen21
* freq_cat: the frequency category assigned to the item according to the frequency quartile ranges
* human_preferred_connective_unrel: if, according to the applied threshold on human ratings, the item belongs to the unrelated group
* tested_relation: the relation type expressed by the connective in the item (causal vs temporal)
* tested_order: the order expressed by the connective in the item (iconic vs anti-iconic)
* relation_human: the relation type expressed by the item according to humans
* order_human: the order of the relation expressed by the item according to humans
* condition_human: relation type and order expressed by the item according to humans
* tested_connective: connective used to join the sentence pair in the item
* additional_dimension: values are 'socially_challenging' (see the dedicated section of the readme) or 'none'. 

### Dataset License

The dataset is made publicly available under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license, which allows redistribution, adaptation, and reuse for non-commercial purposes, provided proper attribution is given and derivative works are shared under the same terms. However, the dataset cannot be used for training artificial intelligence models or other machine learning systems.

### Socially challenging items

Among the causal sentence pairs in ExpliCa, 50 are labeled as 'socially challenging' under the 'additional_dimension' column. This annotation indicates that the item content touches on sensitive or potentially offensive topics such as religion, abortion, immigration, gender identity, drug abuse, and bribery. 
*Some items may be offensive to certain groups.*
The themes contained in these items were added to evaluate whether bias-mitigation strategies in LLMs would impact PCD performance. We plan to explore such aspects more in-depth in future works.


# The Evaluation

## Models
Human ratings served as ground truth for the evaluation of seven LLMs: 
* Mistral-7B-Instruct-v0.3 
* falcon-7b-instruct
* Meta-Llama-3.1-8B-Instruct 
* gemma-2-9b-it 
* Qwen2.5-7B-Instruct (with also versions of 0.5, 1.5, 3, 7, 14, 32B for acceptability ratings in few-shot settings)
* gpt4o 
* gpt4o-mini

## Tasks
The evaluation covers four key tasks:
* Three prompting tasks (Acceptability rating task, cloze test, and multiple-choice) under different conditions (Few-shot and zero-shot setups, Greedy search vs. the Outlines framework) for response generation
* Perplexity evaluation

## Evaluation metrics
For performance assessment, we used accuracy as the primary metric.

## Other analyses
Finally, we analyzed the models' acceptability rating distributions and compared them to human ratings, assessing their correlation with human judgment.


## Reference
If you use any material from this repository, please cite this paper:
```
@inproceedings{miliani-etal-2025-explica,
    title = "{E}xpli{C}a: Evaluating Explicit Causal Reasoning in Large Language Models",
    author = "Miliani, Martina  and
      Auriemma, Serena  and
      Bondielli, Alessandro  and
      Chersoni, Emmanuele  and
      Passaro, Lucia  and
      Sucameli, Irene  and
      Lenci, Alessandro",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.891/",
    pages = "17335--17355",
    ISBN = "979-8-89176-256-5"
}
```
