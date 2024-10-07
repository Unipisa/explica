# ExpliCa


## Context
ExpliCa (Explicit Causality) is a dataset designed to evaluate models on commonsense causal reasoning through causal discovery tasks, more specifically via Pairwise Causal Discovery (PCD), which focuses on determining the existence of a causal relationship between two events and establishing the causal direction, that is, identifying which event serves as the cause and which as the effect (Gao et al., 2023a).


## Stimuli
ExpliCa contains 600 sentence pairs, each presented in both possible orders, resulting in a total of 1200 sentence pairs. Each pair of sentences has been joined in a single sentence through connectives that explicitly indicate causal or temporal relations, as well as the direction of those relations, i.e., iconic (i.e., when the cause/antecedent event in sentence A linguistically appears  before  the effect/the subsequent event in sentence B) and and anti-iconic (effect-cause; subsequent-antecedent):
* A causes B > “so”
* B causes A > “because”
* A precede B > “then”
* B precede A > “after”

## Human ratings 
We collected human acceptability ratings concerning the connective in each sentence via crowdsourcing. Subjects rated on a scale from 1 to 10 how acceptable the connective is to express the relation between the events in the sentences.
For example:
* Jude walked under the rain for an hour, so Jude got sick: Rating 10 (highly acceptable)
* Mary bought some flowers, because Jean went to the dentist: Rating 1 (not acceptable)
Each sentence was rated by 15 English native speakers


## Structure
* Pair_ID: ID of the sentence pair
* Sentence_A: sentence A
* Sentence_B: sentence B
* rating_iconic_causal: the average rating given to sentence pairs connected by a causal connective, where the sentence order follows an iconic direction, i.e., A SO B.
* rating_anticonic_causal: the average rating given to sentence pairs connected by a causal connective, where the sentence order follows an anti-iconic direction, i.e., A BECAUSE B.
* rating_iconic_temporal: the average rating given to sentence pairs connected by a temporal connective, where the sentence order follows an iconic direction, i.e., A THEN B.
* rating_anticonic_temporal: the average rating given to sentence pairs connected by a temporal connective, where the sentence order follows an anti-iconic direction, i.e., A AFTER B.
* human_preferred_connective: connective for which humans provided the highest average rating
* human_preferred_connective_desc: indication of order (iconic vs anti-iconic) and relation (causal vs temporal) of the connective for which humans provided the highest average rating


##Acknowledgements

This research was funded by PNRR—M4C2—Inv. 1.3, Partenariato Esteso PE00000013—“FAIR—Future Artificial Intelligence Research”—Spoke 1 “Human-centered AI,” funded by the European Commission under the NextGeneration EU programme.
