"""
TINY BERT - Simplest BERT implementation
"""

import torch
import torch.nn as nn


class TinyBERT(nn.Module):
    """Ultra-simple BERT - just the essentials!"""

    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2):
        super().__init__()
        print(f"Creating Tiny BERT:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Number of layers: {num_layers}")

        # Convert words to vectors
        self.word_embed = nn.Embedding(vocab_size, hidden_size)
        self.position_embed = nn.Embedding(100, hidden_size)

        #  Segment Embeddings
        # 0 = Sentence A, 1 = Sentence B
        self.segment_embed = nn.Embedding(2, hidden_size)

        # Attention layers (words look at each other)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=2, batch_first=True)
            for _ in range(num_layers)
        ])

        # Processing layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
            for _ in range(num_layers)
        ])

        # Normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers * 2)
        ])

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}\n")

    def forward(self, word_ids, segment_ids=None):
        """
        Args:
            word_ids: (Batch, Seq_Len) -> Word indices
            segment_ids: (Batch, Seq_Len) -> 0 for Sent A, 1 for Sent B
        """
        # Handle missing segment_ids (defaults to all 0s/Sentence A)
        if segment_ids is None:
            segment_ids = torch.zeros_like(word_ids)
        seq_len = word_ids.size(1)
        positions = torch.arange(seq_len, device=word_ids.device)

        #  Add Segment Embeddings to the sum
        # Input representation is sum of Token + Segment + Position
        x = self.word_embed(word_ids) + \
            self.position_embed(positions) + \
            self.segment_embed(segment_ids)

        # Process through each layer
        for i, (attn, ffn) in enumerate(zip(self.attention_layers, self.ffn_layers)):
            # Attention step
            attn_out, _ = attn(x, x, x)
            x = self.norms[i * 2](x + attn_out)

            # Feed-forward step
            ffn_out = ffn(x)
            x = self.norms[i * 2 + 1](x + ffn_out)

        return x


class TinyBERTClassifier(nn.Module):
    """Use Tiny BERT for classification"""

    def __init__(self, bert, num_classes=2):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.word_embed.embedding_dim, num_classes)

    def forward(self, word_ids):
        bert_output = self.bert(word_ids)
        first_word = bert_output[:, 0, :]  # Use first word
        logits = self.classifier(first_word)
        return logits


def test_tiny_bert():
    """Test that everything works!"""
    print("=" * 60)
    print("TESTING TINY BERT")
    print("=" * 60 + "\n")

    # 1. Create model
    print("Step 1: Creating model...")
    bert = TinyBERT(vocab_size=1000, hidden_size=64, num_layers=2)
    classifier = TinyBERTClassifier(bert, num_classes=2)

    # 2. Create fake data
    print("Step 2: Creating sample data...")
    # Pretend we have 3 sentences, each with 5 words
    word_ids = torch.tensor([
        [45, 123, 67, 89, 234],  # Sentence 1
        [12, 456, 78, 90, 111],  # Sentence 2
        [99, 333, 22, 44, 555],  # Sentence 3
    ])
    print(f"  Input shape: {word_ids.shape} (3 sentences, 5 words each)\n")

    # 3. Test forward pass
    print("Step 3: Running model...")
    with torch.no_grad():
        output = classifier(word_ids)

    print(f"  Output shape: {output.shape} (3 sentences, 2 classes)")
    print(f"  Output values:\n{output}\n")

    # 4. Make predictions
    print("Step 4: Making predictions...")
    predictions = torch.argmax(output, dim=1)
    print(f"  Predictions: {predictions}")
    print(f"  (0 = Negative, 1 = Positive)\n")

    print("=" * 60)
    print("SUCCESS! Tiny BERT is working!")
    print("=" * 60)


if __name__ == "__main__":
    test_tiny_bert()