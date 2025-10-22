from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset


# ============================================
# 1. SIMPLE DATASET
# ============================================

class SimpleDataset(Dataset):
    """Wrapper for our data"""

    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# ============================================
# 2. MAIN PROGRAM
# ============================================

def main():
    print("=" * 60)
    print("USING PRE-TRAINED BERT")
    print("=" * 60)

    # 1. Load pre-trained BERT
    print("\n1. Loading pre-trained BERT...")
    print("   (This will download ~400MB on first run)")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    print("   ✓ Model loaded!")
    print(f"   Parameters: {model.num_parameters() / 1e6:.1f}M")

    # 2. Prepare data
    print("\n2. Preparing data...")

    train_texts = [
        "This movie is absolutely fantastic and amazing!",
        "Terrible film, complete waste of my time.",
        "Loved every minute of it, highly recommended!",
        "Boring and predictable, very disappointing.",
        "Masterpiece! One of the best I've seen.",
        "Awful movie, poorly acted and written.",
    ]

    train_labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

    train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)

    print(f"   Training samples: {len(train_texts)}")

    # 3. Setup training
    print("\n3. Setting up training...")

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 4. Train
    print("\n4. Training...")
    print("-" * 60)

    trainer.train()

    print("-" * 60)
    print("✓ Training complete!")

    # 5. Test predictions
    print("\n5. Testing predictions...")
    print("=" * 60)

    test_texts = [
        "This is an amazing film!",
        "Very bad movie, don't watch",
        "Absolutely brilliant performance!",
    ]

    model.eval()

    for text in test_texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment = torch.argmax(predictions, dim=-1)

        sentiment_label = "POSITIVE" if sentiment.item() == 1 else "NEGATIVE"
        confidence = predictions[0][sentiment.item()].item() * 100

        print(f"\nText: '{text}'")
        print(f"  → {sentiment_label} ({confidence:.1f}% confident)")

    print("\n" + "=" * 60)
    print("SUCCESS! You're using real pre-trained BERT!")
    print("=" * 60)
    print("\nThis BERT:")
    print("  ✓ Was trained on billions of words")
    print("  ✓ Understands context deeply")
    print("  ✓ Works great even with little data")
    print("  ✓ Is production-ready!")


if __name__ == "__main__":
    main()