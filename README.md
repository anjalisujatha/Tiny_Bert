# Tiny_Bert

A lightweight implementation of the [BERT](https://arxiv.org/abs/1810.04805/) (Bidirectional Encoder Representations from Transformers) architecture, designed for educational purposes. This project breaks down the complex architecture of BERT into a "Tiny" version to make it easy to understand, modify, and experiment with.

##  Features
This repository implements the core BERT architecture and applies it to four distinct Natural Language Processing (NLP) tasks:
1.  **Sentiment Analysis:** A standard sequence classification task (Positive/Negative).
2.  **Named Entity Recognition (NER):** A token-level classification task identifying entities (Person, Location, etc.).
3.  **Next Sentence Prediction (NSP):** A pre-training objective that predicts if two sentences are sequentially connected.
4.  **Multitask Learning:** A unified model that shares the BERT encoder across multiple tasks (Sentiment + Paraphrase).

##  File Structure

* `bert.py`: The core `TinyBERT` model implementation. Supports Token, Position, and Segment embeddings.
* `sentiment_classifier.py`: Implementation of a Movie Review sentiment classifier using a `SimpleTokenizer`.
* `ner_task.py`: Named Entity Recognition model (`TinyBERTNER`) and training loop.
* `nsp_task.py`: Next Sentence Prediction task (`TinyBERTNSP`) with valid/fake sentence pair generation.
* `multitask_model.py`: A model architecture (`MultitaskTinyBERT`) with shared backbone and separate heads for different tasks.
* `train_multitask.py`: Script to train the multitask model.

##  Installation

The only major dependency is PyTorch.

\`\`\`bash
pip install torch
\`\`\`

##  Usage

### 1. Sentiment Analysis
Train a model to classify movie reviews as Positive or Negative.
\`\`\`bash
python3 sentiment_classifier.py
\`\`\`

### 2. Named Entity Recognition (NER)
Train a model to identify names and locations in text (e.g., "John" -> \`B-PER\`, "Paris" -> \`B-LOC\`).
\`\`\`bash
python3 ner_task.py
\`\`\`

### 3. Next Sentence Prediction (NSP)
Train the model to understand the relationship between two sentences (Is Sentence B the true successor of Sentence A?).
\`\`\`bash
python3 nsp_task.py
\`\`\`

### 4. Multitask Learning
Run the unified model that handles multiple tasks simultaneously.
\`\`\`bash
python3 train_multitask.py
\`\`\`

##  References

This implementation is based on the concepts introduced in the original BERT paper:
* **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Devlin et al., 2018).
* Paper PDF: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)