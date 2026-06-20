# Part-of-Speech Tagger — BiLSTM vs Transformer

MSc Applied AI | COMP634 Assessment 3 | University of Liverpool (2024/25)
Group Project — Akheel Gogeri, Md Omer Rehan, Raaki Ravi, Udayakumar Yaragola Venkateshappa

---

## Overview

A comparative study of two neural sequence labelling architectures for 33-class Part-of-Speech (POS) tagging, implemented entirely from scratch in pure PyTorch — no external NLP libraries. Both a Bidirectional LSTM (BiLSTM) and a Transformer encoder were built, trained, and evaluated under identical conditions. The BiLSTM outperformed the Transformer, achieving 91% token-level accuracy and 0.831 macro-F1 on the held-out test set.

---

## Task

Assign grammatical POS tags (NN, VBD, IN, DT, etc.) to each token in a sentence. A 33-class sequence labelling problem across 3,914 sentences in CoNLL-style format.

---

## Dataset

- **Total sentences:** 3,914 (avg. 21 tokens per sentence)
- **Unique tokens:** 11,943 | **Vocabulary (training only):** 10,577
- **POS tags:** 33 classes (imbalanced — nouns and prepositions dominate)
- **Split:** 80% train / 10% validation / 10% test — sentence-level, fixed random seed
- **OOV rate:** ~9.6% on validation, ~8.4% on test

---

## Implementation

Everything built from scratch in PyTorch — no HuggingFace, spaCy, or NLTK:

- Custom vocabulary construction (training data only) with `<PAD>` and `<UNK>` tokens
- Custom PyTorch Dataset class returning encoded token sequences, tag sequences, and lengths
- Custom collation function with padding, boolean masks, and length retention
- Separate DataLoaders for train/val/test; shuffling on training only

---

## Models

### BiLSTM Tagger
- Trainable embedding layer (dim=128)
- 2-layer Bidirectional LSTM (hidden=128, dropout=0.5)
- Forward + backward hidden states concatenated → Dropout → Linear classifier
- Packed sequences used — padded positions do not influence recurrent computation
- Earlier experiments with hidden=256 and dropout=0.3 caused overfitting → capacity reduced

### Transformer Tagger
- Token embeddings + learned positional embeddings
- 3 Transformer encoder layers with multi-head self-attention (8 heads, dim=256, dropout=0.2)
- Padding mask applied during attention
- Linear classifier on encoder output

### Training
- **Optimiser:** AdamW with weight decay
- **Loss:** CrossEntropyLoss with padding ignored + label smoothing (0.1)
- **Gradient clipping:** max norm 1.0
- **Early stopping:** patience=3, based on validation macro-F1
- **Learning rates:** BiLSTM 1e-3, Transformer 1e-4 (higher LR caused instability for Transformer)

---

## Results

### Validation Performance
| Model | Validation Macro-F1 |
|---|---|
| BiLSTM | **0.826** |
| Transformer | 0.728 |

### Test Set Performance
| Model | Accuracy | Macro F1 | Weighted F1 | Sentence EM |
|---|---|---|---|---|
| **BiLSTM** | **0.910** | **0.831** | **0.911** | **0.225** |
| Transformer | 0.851 | 0.769 | 0.848 | 0.110 |

The BiLSTM's sentence-level exact match of 0.225 means roughly 1 in 4 sentences were tagged with zero errors — a strict metric that confirms strong sequence-level accuracy.

### Why BiLSTM Won
- Converged faster and more stably than the Transformer
- Sequential inductive bias is well-suited to POS tagging where local context dominates
- Transformer confusion matrix shows more dispersed errors, particularly on low-frequency tags
- With limited training data (~3,100 sentences), the Transformer's data requirements were not met

---

## Tech Stack

- Python, PyTorch (pure — no external NLP libraries)
- Custom tokenisation, vocabulary, and encoding pipelines
- matplotlib, seaborn (loss curves, confusion matrices)

---

## Repository Structure

```
pos-tagger-nlp/
├── notebook.ipynb       # Full implementation: preprocessing, models, training, evaluation
├── report.pdf           # Group report with analysis and discussion
└── README.md
```

---

## Academic Context

- **Module:** COMP634 — Applied AI (Assessment 3)
- **Institution:** University of Liverpool, Department of Computer Science
- **Year:** 2024/25
- **Group project** — 4 members

---

## Author (Group Member)

**Akheel R Gogeri**
MSc Artificial Intelligence with Data Science — University of Liverpool
[LinkedIn](https://linkedin.com/in/akheel-gogeri) | [GitHub](https://github.com/AkheelGogeri)

