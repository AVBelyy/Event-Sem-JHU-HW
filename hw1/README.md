# HW1: Semantic Role Labeling

The Github repository link is: https://github.com/AVBelyy/Event-Sem-JHU-HW

* The first homework is at `hw1`
* The definitions of the semantic roles are at `hw1/sem_roles.sparql`

## Data collection

1. Apart from the `AGENT` and `PATIENT` roles, I chose to model `THEME`, `INSTRUMENT`, and `BENEFICIARY`. The `THEME` role is similar to `PATIENT` in definition (differing only in whether the argument undergoes change of state), and so it would be interesting to compare and contrast those two similar roles differing only in one proto-role property. `INSTRUMENT` and `BENEFICIARY` seemed promising for proto-role based modeling, in that they are largely expressible via a specific proto-role property (namely, `was_used` for `INSTRUMENT` and `was_for_benefit` for `BENEFICIARY`).
2. The definition of `PATIENT`, as well as `INSTRUMENT` and `BENEFICIARY` were motivated by Section 2 in [1]. For the `PATIENT` role, I also considered that it needs to be disjoint from `AGENT` (hence the `(volition < 0) ∧ (instigation < 0)` term) and that the role undergoes change of state. The `THEME` role is defined similarly to the `PATIENT` role, without the requirement to change state, according to [2]. The full definitions of all roles can be found [here](https://github.com/AVBelyy/Event-Sem-JHU-HW/blob/main/hw1/sem_roles.sparql).
3. The frequency table (in the form of # positives / # negatives) for all the roles is shown below. We see that the distributions of roles are skewed: while the `AGENT` role is present in around 60% of sentences, the `PATIENT` and `BENEFICIARY` roles are present in less than 10% of sentences. This could be (at least, partially) explained by varying granularity and selectional restrictions of roles: for instance, the `PATIENT` role requires the argument to undergo change of state, have no volition and no instigation, while the definition of `AGENT` is more lax. The `BENEFICIARY` role usually requires a for-prepositional phrase or a bi-transitive verb, which also narrows down the set of contexts where this role may be observed.

|             | Train       | Dev       | Test      |
| ----------- | ----------- | --------- | --------- |
| AGENT       | 2827 / 2047 | 380 / 252 | 373 / 209 |
| PATIENT     | 343 / 4531  | 42 / 590  | 18 / 564  |
| THEME       | 732 / 4142  | 95 / 537  | 73 / 509  |
| INSTRUMENT  | 1234 / 3640 | 156 / 476 | 133 / 449 |
| BENEFICIARY | 401 / 4473  | 63 / 569  | 42 / 540  |


## Modeling

1. The model is comprised of 1) a sequence-to-sequence encoder (in my case, a bi-directional LSTM with 300-dimensional GloVe embeddings as inputs) and a two-layer perceptron with an intermediate Dropout layer. I tuned parameters such dimensionality of a seq2seq encoder, MLP, and the dropout rate using the dev sets of the `AGIENT` role. Furthermore,since the roles have unequal class balance, I tuned the weight of the positive class for the CrossEntropyLoss for each class individually, choosing it from `{0.5, 1.0, 2.0, 5.0, 10.0}`.
2. The model uses the standard feature representation, using embeddings of the predicate and argument heads as features for the two-level perceptron.
3. The performance of each binary classifier model on the test set is shown below. The performance is definitely corelated with the label frequency, even after a (partial) remedy in the form of re-weighting the CrossEntropyLoss.

|             | Precision | Recall | F1   |
| ----------- | --------- | ------ | ---- |
| AGENT       | 0.75      | 0.86   | 0.80 |
| PATIENT     | 0.22      | 0.11   | 0.15 |
| THEME       | 0.23      | 0.58   | 0.33 |
| INSTRUMENT  | 0.30      | 0.71   | 0.42 |
| BENEFICIARY | 0.14      | 0.10   | 0.11 |


## Exploration

1. Looking at the training set for the `INSTRUMENT` role, many of the errors could be attributed to the errors in the proto-role annotations: in particular, the arg in "Guerrillas killed(pred) an engineer(arg) , Asi Ali , from Tikrit . " or in "We(arg) all know(pred) ... what happened in Chernobyl ..." or in "S. and I have an acquaintance who has hosted(pred) several(arg) of these children ..." should not have had `was_used > 0`). To modify the definition, I would perhaps include the syntactic and morphological properties of the arguments (e.g. a with-prepositional phrase in English or an instrumental case in Russian could be a good indicator of the `INSTRUMENT` role). Unfortunately, these properties are all outside the proto role criteria captured by the UDS dataset.
2. Using the fact that `PATIENT` and `THEME` roles differ only in that an argument has to undergo change of state in one but not the other, and that we have a binary classifier for both roles, we construct the following minimal pair. In both of the examples, our model preferred the `THEME` label in both cases, which is not correct in the first case. This preference could in part be explained by the `PATIENT` label being less frequent than the `THEME` label in the training set.

|                                                         | True role | Predicted role |
| ------------------------------------------------------- | --------- | -------------- |
| Ben opened(pred) the door(arg) with a key .             | `PATIENT` | `THEME`        |
| Ben opened(pred) the event(arg) with a keynote speech . | `THEME`   | `THEME`        | 


## Extension

1. The first way would be to look at the roles based on their proto role definitions (prior to training any models). The other way could be based on the trained models' score. In that way, it is easy to see, for instance, that the `AGENT` and `PATIENT` roles are completely disjoint based purely on proto role definitions, yet the trained models currently could still label a single argument both as an `AGENT` and as a `PATIENT`.
2. A multi-label classifier could capture inter-label dependencies by modeling a joint distribution that does not necessarily assume pairwise independece of labels (which is the case for binary one-vs-all classifiers).
3. We've already seen the interplay between syntax and semantics in class [1, 3]; arguably, sequence-to-sequence models could better capture syntax (by e.g. accounting for the signed distance between the predicate and arguments, which could be used a proxy for syntactic argument annotations) which gives sequence-to-sequence models additional form of signal and they are thus expected to perform better than the bag-of-words models.
4. As discussed in various parts of the report, explicit syntactic annotations such as 1) syntactic edge label, 2) (pred - arg) distance, or 3) labels of ancestral syntactic nodes (e.g. with-PP, for-PP, bi-transitive-VP) could all provide the currently missing syntactic information, which could capitalize on the aforementioned interplay between syntax and semantics and provide useful signal for SRL models.

[1] Schuler, Karin Kipper. "VerbNet: A broad-coverage, comprehensive verb lexicon." (2005).

[2] https://en.wikipedia.org/wiki/Thematic_relation

[3] Reisinger, Drew, et al. "Semantic proto-roles." Transactions of the Association for Computational Linguistics 3 (2015): 475-488.
