"""
Real sentiment analysis using Tiny BERT
Classifies movie reviews as positive or negative
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bert import TinyBERT, TinyBERTClassifier


# ============================================
# 1. CREATE SIMPLE TOKENIZER
# ============================================

class SimpleTokenizer:
    """Convert words to numbers"""

    def __init__(self):
        self.word_to_id = {"[PAD]": 0, "[UNK]": 1}
        self.id_to_word = {0: "[PAD]", 1: "[UNK]"}
        self.next_id = 2

    def add_word(self, word):
        """Add a new word to vocabulary"""
        if word not in self.word_to_id:
            self.word_to_id[word] = self.next_id
            self.id_to_word[self.next_id] = word
            self.next_id += 1

    def encode(self, text, max_length=10):
        """Convert text to word IDs"""
        words = text.lower().split()

        # Add words to vocabulary
        for word in words:
            self.add_word(word)

        # Convert to IDs
        ids = [self.word_to_id.get(word, 1) for word in words]

        # Pad or truncate
        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        return ids

    def decode(self, ids):
        """Convert IDs back to text"""
        words = [self.id_to_word.get(id, "[UNK]") for id in ids if id != 0]
        return " ".join(words)


# ============================================
# 2. CREATE DATASET
# ============================================

class SentimentDataset(Dataset):
    """Movie review dataset"""

    def __init__(self, tokenizer):
        # Sample movie reviews
        self.texts = [
            "this movie is amazing and wonderful",
            "terrible film waste of time",
            "absolutely loved it great acting",
            "boring and predictable bad movie",
            "fantastic storyline brilliant performance",
            "worst movie ever very disappointed",
            "enjoyed it thoroughly highly recommend",
            "awful terrible horrible experience",
            "masterpiece one of the best films",
            "disaster poorly written badly acted",
        ]

        # Labels: 1 = Positive, 0 = Negative
        self.labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

        self.tokenizer = tokenizer

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
# 3. TRAINING FUNCTION
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
# 4. PREDICTION FUNCTION
# ============================================

def predict_sentiment(model, tokenizer, text):
    """Predict sentiment for new text"""

    model.eval()

    # Tokenize
    word_ids = torch.tensor([tokenizer.encode(text)])

    # Predict
    with torch.no_grad():
        output = model(word_ids)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)

    sentiment = "POSITIVE" if prediction.item() == 1 else "NEGATIVE"
    confidence = probabilities[0][prediction.item()].item() * 100

    return sentiment, confidence


# ============================================
# 5. MAIN PROGRAM
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

    # 5. Test on new reviews
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