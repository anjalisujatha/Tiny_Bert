import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from bert import TinyBERT
from sentiment_classifier import SimpleTokenizer


# ==========================================
# 1. THE NSP MODEL HEAD
# ==========================================
class TinyBERTNSP(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        # Binary Classifier: IsNext (1) vs NotNext (0)
        # We use the [CLS] token representation C for NSP
        self.nsp_classifier = nn.Linear(bert.word_embed.embedding_dim, 2)

    def forward(self, word_ids, segment_ids):
        # 1. Run BERT with Segment IDs
        bert_output = self.bert(word_ids, segment_ids)

        # 2. Get the [CLS] token (first token) representation
        cls_embedding = bert_output[:, 0, :]

        # 3. Predict IsNext/NotNext
        logits = self.nsp_classifier(cls_embedding)
        return logits


# ==========================================
# 2. THE NSP DATASET GENERATOR
# ==========================================
class NSPDataset(Dataset):
    def __init__(self, tokenizer, documents):
        self.tokenizer = tokenizer
        self.data = []

        # Generate Training Pairs
        for doc in documents:
            for i in range(len(doc) - 1):
                sentence_a = doc[i]

                # 50% chance: IsNext (Actual next sentence)
                if random.random() > 0.5:
                    sentence_b = doc[i + 1]
                    label = 1  # True Next
                # 50% chance: NotNext (Random sentence from random doc)
                else:
                    random_doc = random.choice(documents)
                    sentence_b = random.choice(random_doc)
                    label = 0  # Random/Fake Next

                self.data.append((sentence_a, sentence_b, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent_a, sent_b, label = self.data[idx]

        # --- BUILD INPUT: [CLS] A [SEP] B [SEP] ---
        # 1. Tokenize A and B
        ids_a = self.tokenizer.encode(sent_a, max_length=100)
        ids_b = self.tokenizer.encode(sent_b, max_length=100)

        # Strip padding (0) to pack them tight
        ids_a = [i for i in ids_a if i != 0]
        ids_b = [i for i in ids_b if i != 0]

        # Get Special Token IDs
        CLS = self.tokenizer.word_to_id["[CLS]"]
        SEP = self.tokenizer.word_to_id["[SEP]"]

        # 2. Construct Word IDs sequence
        # Paper format: [CLS] Sentence A [SEP] Sentence B [SEP]
        word_ids = [CLS] + ids_a + [SEP] + ids_b + [SEP]

        # 3. Construct Segment IDs (0 for A, 1 for B)
        # 0s for A (including CLS and first SEP), 1s for B (including final SEP)
        segment_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)

        # 4. Pad both to fixed length (e.g., 20)
        max_len = 20
        pad_len = max_len - len(word_ids)

        if pad_len > 0:
            word_ids += [0] * pad_len
            segment_ids += [0] * pad_len
        else:
            word_ids = word_ids[:max_len]
            segment_ids = segment_ids[:max_len]

        return {
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_nsp():
    print("\n--- Training Next Sentence Prediction ---")

    # 1. Setup Tokenizer & Add Special Tokens
    tokenizer = SimpleTokenizer()
    tokenizer.add_word("[CLS]")  # Special start token
    tokenizer.add_word("[SEP]")  # Special separator token

    # 2. Create Dummy "Documents"
    documents = [
        ["the dog walked", "he found a bone", "he ate the bone"],
        ["machine learning is cool", "bert is a transformer", "transformers are powerful"],
        ["i like apples", "apples are red", "red is a color"],
    ]

    # Teach tokenizer all words
    for doc in documents:
        for sent in doc: tokenizer.encode(sent)

    dataset = NSPDataset(tokenizer, documents)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 3. Setup Model
    bert = TinyBERT(vocab_size=len(tokenizer.word_to_id), hidden_size=32)
    model = TinyBERTNSP(bert)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Train
    for epoch in range(10):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # PASS BOTH WORD IDs AND SEGMENT IDs
            logits = model(batch['word_ids'], batch['segment_ids'])

            loss = loss_fn(logits, batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")


if __name__ == "__main__":
    train_nsp()