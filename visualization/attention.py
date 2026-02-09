import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tiny_bert import TinyBERT, SimpleTokenizer

# Resolve paths relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")


# Words the model learned as positive/negative from training
POSITIVE_WORDS = {
    "amazing", "wonderful", "loved", "great", "fantastic", "brilliant",
    "masterpiece", "best", "incredible", "outstanding", "beautiful",
    "superb", "excellent", "inspiring", "remarkable", "stunning",
    "delightful", "heartwarming", "perfect",
}
NEGATIVE_WORDS = {
    "terrible", "boring", "predictable", "bad", "worst", "awful",
    "horrible", "disaster", "poorly", "dull", "disappointing", "painful",
    "ugly", "dreadful", "weak", "hated", "poor", "lazy", "waste",
}


def get_word_color(word):
    """Color by semantic role: green=positive, red=negative, gray=neutral."""
    if word in POSITIVE_WORDS:
        return "#2ecc71"
    elif word in NEGATIVE_WORDS:
        return "#e74c3c"
    return "#95a5a6"


def visualize_attention(bert_model, tokenizer, text, layer=0):
    """Plot attention heatmap for a single sentence."""
    word_ids = tokenizer.encode(text, max_length=20, add_new_words=False)
    tokens = text.lower().split()

    input_ids = torch.tensor([word_ids])

    bert_model.eval()
    with torch.no_grad():
        bert_model(input_ids)

    attn = bert_model.attention_maps[layer][0].numpy()

    num_real = min(len(tokens), 20)
    attn = attn[:num_real, :num_real]
    tokens = tokens[:num_real]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        square=True,
    )
    plt.title(f"Semantic Attention Map (Layer {layer + 1})\n\"{text}\"")
    plt.xlabel("Key (Attending To)")
    plt.ylabel("Query (Token)")
    plt.tight_layout()
    return plt.gcf()


def run_visualization():
    # Load training texts to rebuild vocabulary
    training_texts = []
    csv_path = os.path.join(_DATA_DIR, "reviews.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] in ("positive", "negative"):
                training_texts.append(row["text"])

    tokenizer = SimpleTokenizer()
    for text in training_texts:
        tokenizer.encode(text)

    # Load the fine-tuned BERT encoder
    bert_model = TinyBERT(vocab_size=len(tokenizer.word_to_id), hidden_size=32, num_layers=2)
    model_path = os.path.join(_OUTPUT_DIR, "trained_bert.pt")
    bert_model.load_state_dict(torch.load(model_path, weights_only=True))
    bert_model.eval()
    print(f"Loaded fine-tuned BERT encoder from '{model_path}'")

    sentence = "the movie was amazing and brilliant"

    output_dir = os.path.join(_OUTPUT_DIR, "attention_maps")
    os.makedirs(output_dir, exist_ok=True)

    # Attention heatmaps for both layers
    for layer in range(2):
        fig = visualize_attention(bert_model, tokenizer, sentence, layer=layer)
        filename = os.path.join(output_dir, f"attention_layer{layer + 1}.png")
        fig.savefig(filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close(fig)

    # Semantic Space: t-SNE of word embeddings for this sentence
    print("\nGenerating semantic space visualization...")

    word_embeddings = bert_model.word_embed.weight.detach().numpy()

    # Get unique words from the sentence and their embeddings
    tokens = sentence.lower().split()
    seen = set()
    unique_words = []
    for w in tokens:
        if w not in seen and w in tokenizer.word_to_id:
            unique_words.append(w)
            seen.add(w)

    word_ids = [tokenizer.word_to_id[w] for w in unique_words]
    embeddings = word_embeddings[word_ids]

    # t-SNE: 32 dimensions -> 2 dimensions
    perp = min(5, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(embeddings)

    # Color each word by sentiment role
    token_colors = [get_word_color(w) for w in unique_words]

    plt.figure(figsize=(12, 9))
    for i, word in enumerate(unique_words):
        plt.scatter(reduced[i, 0], reduced[i, 1], c=token_colors[i], s=200,
                    edgecolor="black", alpha=0.85, zorder=2)
        plt.annotate(word, (reduced[i, 0] + 1.5, reduced[i, 1] + 1.5),
                     fontsize=12, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Positive", markerfacecolor="#2ecc71", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Negative", markerfacecolor="#e74c3c", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Neutral", markerfacecolor="#95a5a6", markersize=12),
    ]
    plt.legend(handles=legend_elements, loc="best", fontsize=11)

    plt.title(f"Semantic Space — Word Embeddings (t-SNE)\n\"{sentence}\"", fontsize=13)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    tsne_path = os.path.join(output_dir, "semantic_space.png")
    plt.savefig(tsne_path, dpi=150)
    print(f"Saved: {tsne_path}")
    plt.close()

    # Cosine similarity heatmap for the sentence tokens (using contextual hidden states)
    print("Generating cosine similarity heatmap...")
    word_ids_sent = tokenizer.encode(sentence, max_length=20, add_new_words=False)
    input_ids = torch.tensor([word_ids_sent])
    with torch.no_grad():
        hidden_states = bert_model(input_ids)

    sent_tokens = sentence.lower().split()
    num_real = min(len(sent_tokens), 20)
    sent_tokens = sent_tokens[:num_real]
    sent_embeddings = hidden_states[0, :num_real, :].numpy()

    norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
    cosine_sim = (sent_embeddings @ sent_embeddings.T) / (norms @ norms.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cosine_sim,
        xticklabels=sent_tokens,
        yticklabels=sent_tokens,
        cmap="RdYlGn",
        annot=True,
        fmt=".2f",
        square=True,
        vmin=-1, vmax=1,
    )
    plt.title(f"Cosine Similarity Between Token Embeddings\n\"{sentence}\"")
    plt.tight_layout()
    cosine_path = os.path.join(output_dir, "cosine_similarity.png")
    plt.savefig(cosine_path, dpi=150)
    print(f"Saved: {cosine_path}")
    plt.close()


if __name__ == "__main__":
    run_visualization()
