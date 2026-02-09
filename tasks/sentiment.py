"""
Real sentiment analysis using Tiny BERT
Classifies movie reviews as positive or negative
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tiny_bert import TinyBERT, TinyBERTClassifier, SimpleTokenizer

# Resolve paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")


# ============================================
# 1. CREATE DATASET
# ============================================

class SentimentDataset(Dataset):
    """Movie review dataset"""

    def __init__(self, tokenizer, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(_DATA_DIR, "reviews.csv")
        # Load reviews from CSV (only positive and negative for training)
        self.texts = []
        self.labels = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['category'] in ('positive', 'negative'):
                    self.texts.append(row['text'])
                    self.labels.append(int(row['label']))

        self.tokenizer = tokenizer
        # "Teach" the tokenizer all words
        print("Building vocabulary...")
        for text in self.texts:
            self.tokenizer.encode(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        word_ids = self.tokenizer.encode(text)

        return {
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


# ============================================
# 2. TRAINING FUNCTION
# ============================================

def train_model(model, train_loader, num_epochs=10):
    """Train the sentiment classifier"""

    print("\n" + "=" * 60)
    print("TRAINING SENTIMENT CLASSIFIER")
    print("=" * 60 + "\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            word_ids = batch['word_ids']
            labels = batch['label']

            # Forward pass
            optimizer.zero_grad()
            outputs = model(word_ids)
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total * 100
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {accuracy:.1f}%")

    print("\n Training complete!")


# ============================================
# 3. PREDICTION FUNCTION
# ============================================

def predict_sentiment(model, tokenizer, text):
    """Predict sentiment for new text"""

    model.eval()

    # Tokenize
    word_ids = torch.tensor([tokenizer.encode(text, add_new_words=False)])

    # Predict
    with torch.no_grad():
        output = model(word_ids)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)

    sentiment = "POSITIVE" if prediction.item() == 1 else "NEGATIVE"
    confidence = probabilities[0][prediction.item()].item() * 100

    return sentiment, confidence


# ============================================
# 4. MAIN PROGRAM
# ============================================

def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS WITH TINY BERT")
    print("=" * 60)

    # 1. Create tokenizer
    print("\n1. Creating tokenizer...")
    tokenizer = SimpleTokenizer()

    # 2. Create dataset
    print("2. Creating dataset...")
    dataset = SentimentDataset(tokenizer)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"   Dataset size: {len(dataset)} reviews")
    print(f"   Vocabulary size: {len(tokenizer.word_to_id)} words")

    # 3. Create model
    print("\n3. Creating model...")
    bert = TinyBERT(vocab_size=len(tokenizer.word_to_id), hidden_size=32, num_layers=2)
    model = TinyBERTClassifier(bert, num_classes=2)

    # 4. Train
    train_model(model, train_loader, num_epochs=20)

    # 5. Save the trained BERT encoder for visualization
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(_OUTPUT_DIR, "trained_bert.pt")
    torch.save(bert.state_dict(), save_path)
    print(f"\nSaved trained BERT encoder to '{save_path}'")

    # 6. Test on new reviews
    print("\n" + "=" * 60)
    print("TESTING ON NEW REVIEWS")
    print("=" * 60 + "\n")

    test_reviews = [
        "this is an amazing movie",
        "terrible waste of time",
        "absolutely brilliant",
        "very disappointing film",
    ]

    for review in test_reviews:
        sentiment, confidence = predict_sentiment(model, tokenizer, review)
        print(f"Review: '{review}'")
        print(f"  → {sentiment} (confidence: {confidence:.1f}%)\n")

    print("=" * 60)
    print("DONE! You've built a working sentiment classifier!")
    print("="* 60)


if __name__ == "__main__":
    main()
