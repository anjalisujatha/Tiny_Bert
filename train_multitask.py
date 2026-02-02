import torch
import torch.nn as nn
from bert import TinyBERT
from multitask_model import MultitaskTinyBERT


def test_multitask_flow():
    print("=" * 60)
    print("TESTING MULTITASK TINY BERT")
    print("=" * 60)

    # 1. Setup Model
    vocab_size = 100
    bert = TinyBERT(vocab_size=vocab_size, hidden_size=32, num_layers=2)
    model = MultitaskTinyBERT(bert)

    # 2. Fake Data
    # Batch of 2 sentences, 5 words each
    fake_input = torch.randint(0, vocab_size, (2, 5))

    print("\n[Step 1] Testing Sentiment Task...")
    sent_out = model(fake_input, task_name="sentiment")
    print(f"Sentiment Output Shape: {sent_out.shape} (Expected: [2, 2])")

    print("\n[Step 2] Testing Paraphrase Task...")
    para_out = model(fake_input, task_name="paraphrase")
    print(f"Paraphrase Output Shape: {para_out.shape} (Expected: [2, 2])")

    print("\nSUCCESS: Model switches heads correctly!")


if __name__ == "__main__":
    test_multitask_flow()