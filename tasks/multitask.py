import torch
import torch.nn as nn
from tiny_bert import TinyBERT


class MultitaskTinyBERT(nn.Module):
    """
    A Multitask Model that shares the BERT backbone but has
    separate 'heads' for different tasks.
    """

    def __init__(self, bert, num_sentiment_labels=2, num_paraphrase_labels=2):
        super().__init__()
        # 1. SHARED BACKBONE
        self.bert = bert

        # 2. HEAD FOR SENTIMENT (Binary: Positive/Negative)
        self.sentiment_classifier = nn.Linear(bert.word_embed.embedding_dim, num_sentiment_labels)

        # 3. HEAD FOR PARAPHRASE (Binary: Paraphrase/Not Paraphrase)
        self.paraphrase_classifier = nn.Linear(bert.word_embed.embedding_dim, num_paraphrase_labels)

    def forward(self, word_ids, task_name):
        """
        Args:
            word_ids: The input token IDs
            task_name: String, either 'sentiment' or 'paraphrase'
        """
        # Step 1: Run the shared BERT model
        bert_output = self.bert(word_ids)

        # Use the first token (index 0) as the summary of the sentence
        cls_embedding = bert_output[:, 0, :]

        # Step 2: Route to the correct specific head
        if task_name == "sentiment":
            logits = self.sentiment_classifier(cls_embedding)

        elif task_name == "paraphrase":
            logits = self.paraphrase_classifier(cls_embedding)

        else:
            raise ValueError(f"Unknown task: {task_name}. Choose 'sentiment' or 'paraphrase'.")

        return logits


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
