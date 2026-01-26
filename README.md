# Tiny_Bert

A lightweight implementation of the BERT (Bidirectional Encoder Representations from Transformers) architecture, designed for educational purposes and sentiment analysis tasks. This repository contains a custom BERT model implementation, tools for loading pretrained weights, and a sentiment classifier.

## 🚀 Features

- **Custom BERT Architecture**  
  A modular implementation of the BERT model including multi-head attention and transformer layers.

- **Pretrained Weights Support**  
  Functionality to load and utilize existing BERT weights for transfer learning.

- **Sentiment Classification**  
  A downstream application showing how to fine-tune or use the model for sentiment analysis.

- **Testing Suite**  
  Scripts to verify the model architecture and performance.

## 📁 Repository Structure

- `bert.py`: The core architecture of the Tiny BERT model.
- `pretrained_bert.py`: Handles the loading and mapping of pretrained BERT parameters.
- `sentiment_classifier.py`: A classifier head and training loop for sentiment analysis tasks.
- `test_bert.py`: Unit tests and verification scripts for the BERT implementation.

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/anjalisujatha/Tiny_Bert.git
cd Tiny_Bert
