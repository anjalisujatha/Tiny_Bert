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
Install dependencies (recommended to use a virtual environment):

bash
Copy code
pip install torch transformers
💻 Usage
1. Model Initialization
You can initialize the BERT model using the configuration defined in bert.py:

python
Copy code
from bert import BertModel

# Initialize with custom config or default parameters
model = BertModel(config)
2. Running the Sentiment Classifier
To use the model for sentiment analysis:

bash
Copy code
python sentiment_classifier.py
3. Testing
To ensure everything is working correctly, run the provided test script:

bash
Copy code
python test_bert.py
📖 How it Works
Encoder
The model utilizes a stack of Transformer encoders that process input tokens bidirectionally.

Pooling
The [CLS] token representation is pooled to provide a fixed-length vector for classification tasks.

Fine-tuning
The sentiment_classifier.py script adds a linear layer on top of the encoder to categorize text into sentiment classes.
