# SEQ2SEQ using LSTM BY SHAELESH A

## 1. Overview

The project implements a three-stage NLU pipeline for a calendar assistant. Each stage processes the calendar query for a different purpose:

1. **Task 1:** Intent Classification using a unidirectional vanilla LSTM.
2. **Task 2:** Slot Filling / Entity Tagging using a BiLSTM.
3. **Task 3:** Temporal String Normalization using an LSTM Encoder–Decoder without attention.

The models are trained separately, while their outputs can ultimately be combined into the required:

```
<INTENT>|<BIO TAGS>|<CANONICAL DATETIME>
```

# 2. Task 1 — Intent Classification

### Evaluation

For this classification task, **accuracy, precision, recall, F1-score, and confusion matrix** are appropriate because the output is a single categorical label.

### Results

The model achieved:

**Validation accuracy: 99.86%**

**Test accuracy: 100%**

| Intent          | Precision |   Recall |       F1 | Support |
| --------------- | --------: | -------: | -------: | ------: |
| CREATE_EVENT    |      1.00 |     1.00 |     1.00 |     320 |
| SET_REMINDER    |      1.00 |     1.00 |     1.00 |     211 |
| QUERY_FREE_TIME |      1.00 |     1.00 |     1.00 |     134 |
| CANCEL          |      1.00 |     1.00 |     1.00 |      60 |
| **Overall**     |  **1.00** | **1.00** | **1.00** | **725** |

The confusion matrix contained **zero misclassifications** across all four intent classes.

# 3. Task 2 — Slot Filling / Entity Tagging

### Evaluation

Token-level accuracy alone is not sufficient for sequence tagging, so **precision, recall and F1-score**, along with **per-tag performance**, were used.

### Results

The model achieved:

* **Token Accuracy:** 100%
* **Precision:** 100%
* **Recall:** 100%
* **F1-score:** 100%

| Tag      | Precision | Recall |   F1 | Support |
| -------- | --------: | -----: | ---: | ------: |
| B-DATE   |      1.00 |   1.00 | 1.00 |     551 |
| B-EVENT  |      1.00 |   1.00 | 1.00 |     532 |
| B-PERSON |      1.00 |   1.00 | 1.00 |     191 |
| B-TIME   |      1.00 |   1.00 | 1.00 |     338 |
| I-DATE   |      1.00 |   1.00 | 1.00 |     633 |
| I-EVENT  |      1.00 |   1.00 | 1.00 |     227 |
| O        |      1.00 |   1.00 | 1.00 |    3846 |

There were **6318 tokens** in the test set, all correctly tagged.


# 4. Task 3 — String Normalization

### Input representation

Instead of feeding the complete sentence to the normalization model, the implementation extracts the relevant `DATE`/`TIME` span using the BIO tags before encoding it. 

The target is represented using five slots:

```text
[YEAR, MONTH, DAY, HOUR, MINUTE]
```

with `NA` used for missing components. 

This provides a structured output representation while still using the required LSTM Encoder–Decoder architecture.

### Evaluation

For this task, **exact-match accuracy** is the most important metric because the final output is required to be an exact canonical string. A prediction such as:

```
Target:     2026-06-22 15:15
Prediction: 2026-06-22 15:30
```

should be considered incorrect even though most of the characters are correct.

Therefore, performance was also broken down by target type.

### Results

**Overall test accuracy: 84.28%**

```
Correct: 611 / 725
```

| Target type | Correct |   Total | Exact Match |
| ----------- | ------: | ------: | ----------: |
| DATE ONLY   |     255 |     272 |  **93.75%** |
| DATE + TIME |     198 |     279 |  **70.97%** |
| TIME ONLY   |      43 |      59 |  **72.88%** |
| NA          |     115 |     115 |    **100%** |
| **Overall** | **611** | **725** |  **84.28%** |


# 5. Overall Comparison

| Task                       | Architecture                       | Main Metric                   | Test Result |
| -------------------------- | ---------------------------------- | ----------------------------- | ----------: |
| **Task 1 — Intent**        | Uni-LSTM, many-to-one              | Accuracy / F1                 |    **100%** |
| **Task 2 — Slot Filling**  | BiLSTM                             | Token F1 / Precision / Recall |    **100%** |
| **Task 3 — Normalization** | LSTM Encoder–Decoder, no attention | Exact Match Accuracy          |  **84.28%** |

### Overall conclusion

The first two tasks achieve near-perfect performance on the test set, indicating that the dataset's intent and entity patterns are highly learnable with the required LSTM architectures. Task 3 is more challenging because it requires **sequence-to-sequence transformation** rather than classification or per-token prediction.

In particular, the normalization model performs very strongly on `DATE ONLY` (**93.75%**) and `NA` (**100%**), but `DATE + TIME` remains the main source of errors (**70.97%**). This difference highlights the difficulty of generating an exact structured sequence from a fixed-size encoder representation without attention.



The three outputs can then be combined into the required final calendar-assistant representation.
