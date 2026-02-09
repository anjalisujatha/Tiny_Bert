import torch
from tiny_bert import TinyBERT, TinyBERTClassifier


def test_different_sizes():
    """Test BERT with different configurations"""

    print("\n" + "=" * 60)
    print("TESTING DIFFERENT BERT SIZES")
    print("=" * 60 + "\n")

    configs = [
        ("Micro", 500, 32, 1),
        ("Tiny", 1000, 64, 2),
        ("Small", 2000, 128, 4),
    ]

    for name, vocab, hidden, layers in configs:
        print(f"\n{name} BERT:")
        bert = TinyBERT(vocab, hidden, layers)

        # Test with sample data
        sample = torch.randint(0, vocab, (2, 5))
        with torch.no_grad():
            output = bert(sample)

        print(f"✓ {name} BERT works! Output shape: {output.shape}\n")


def test_batch_sizes():
    """Test with different batch sizes"""

    print("\n" + "=" * 60)
    print("TESTING DIFFERENT BATCH SIZES")
    print("=" * 60 + "\n")

    bert = TinyBERT()
    classifier = TinyBERTClassifier(bert)

    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        # Create batch
        word_ids = torch.randint(0, 1000, (batch_size, 5))

        with torch.no_grad():
            output = classifier(word_ids)

        print(f"Batch size {batch_size}: ✓ Works! Output shape: {output.shape}")

    print("\n✓ All batch sizes work!")


def test_sequence_lengths():
    """Test with different sentence lengths"""

    print("\n" + "=" * 60)
    print("TESTING DIFFERENT SEQUENCE LENGTHS")
    print("=" * 60 + "\n")

    bert = TinyBERT()

    seq_lengths = [3, 5, 10, 20]

    for seq_len in seq_lengths:
        word_ids = torch.randint(0, 1000, (2, seq_len))

        with torch.no_grad():
            output = bert(word_ids)

        print(f"Sequence length {seq_len}: ✓ Works! Output shape: {output.shape}")

    print("\n✓ All sequence lengths work!")


if __name__ == "__main__":
    test_different_sizes()
    test_batch_sizes()
    test_sequence_lengths()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
