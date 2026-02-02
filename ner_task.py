import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bert import TinyBERT
# Import the tokenizer from your existing sentiment file
from sentiment_classifier import SimpleTokenizer


# ==========================================
# 1. THE NER MODEL (Token-Level)
# ==========================================
class TinyBERTNER(nn.Module):
    def __init__(self, bert, num_classes):
        super().__init__()
        self.bert = bert
        # A linear layer that maps Hidden_Size -> Num_Classes
        # This applies to EVERY token position automatically
        self.classifier = nn.Linear(bert.word_embed.embedding_dim, num_classes)

    def forward(self, word_ids):
        # 1. Get sequence embeddings from BERT
        # Shape: (Batch_Size, Seq_Len, Hidden_Size)
        bert_output = self.bert(word_ids)

        # 2. Classify EVERY token position
        # Shape: (Batch_Size, Seq_Len, Num_Classes)
        logits = self.classifier(bert_output)

        return logits


# ==========================================
# 2. THE NER DATASET
# ==========================================
class NERDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Labels: 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC
        self.data = [
            ("john lives in new york", [1, 0, 0, 3, 4]),
            ("alice goes to london", [1, 0, 0, 3]),
            ("bob is a developer", [1, 0, 0, 0]),
            ("visit paris france", [0, 3, 3]),
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels = self.data[idx]

        # Encode text (adds padding to length 10)
        word_ids = self.tokenizer.encode(text)

        # Pad labels to match text length (10)
        # We use -100 so the loss function ignores these padded spots
        pad_len = 10 - len(labels)
        padded_labels = labels + [-100] * pad_len

        return {
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels[:10], dtype=torch.long)
        }


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_ner(model, loader):
    print("\nTraining NER Model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ignore_index=-100 prevents the model from learning "O" for padding
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(20):
        total_loss = 0
        for batch in loader:
            word_ids = batch['word_ids']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model(word_ids)  # Shape: (Batch, Seq, Class)

            # Flatten to (Batch * Seq, Class) for CrossEntropy
            outputs_flat = outputs.view(-1, outputs.shape[-1])
            labels_flat = labels.view(-1)

            loss = loss_fn(outputs_flat, labels_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup Data
    tokenizer = SimpleTokenizer()
    dataset = NERDataset(tokenizer)

    # TEACH TOKENIZER: Register all words in dataset
    for text, _ in dataset.data:
        tokenizer.encode(text)

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Setup Model
    bert = TinyBERT(vocab_size=len(tokenizer.word_to_id), hidden_size=32)
    model = TinyBERTNER(bert, num_classes=5)

    # 3. Run Training
    train_ner(model, loader)

    # 4. Test Prediction
    print("\nTesting: 'John lives in New York'")
    test_ids = torch.tensor([tokenizer.encode("john lives in new york")])
    with torch.no_grad():
        logits = model(test_ids)
        preds = torch.argmax(logits, dim=2)

    id_to_label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
    print("Tokens:", tokenizer.decode(test_ids[0].tolist()))
    print("Preds :", [id_to_label.get(p.item(), "PAD") for p in preds[0] if p.item() in id_to_label])