import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tiny_bert import TinyBERT, SimpleTokenizer


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "outputs")

# Words to visualize
POSITIVE_WORDS = {
    "amazing", "wonderful", "loved", "great", "fantastic", "brilliant",
    "masterpiece", "best", "excellent"
}
NEGATIVE_WORDS = {
    "terrible", "boring", "predictable", "bad", "worst", "awful",
    "horrible", "disaster", "disappointed"
}


def get_word_color(word):
    if word in POSITIVE_WORDS:
        return "#2ecc71"
    elif word in NEGATIVE_WORDS:
        return "#e74c3c"
    return "#95a5a6"


def visualize_attention(bert_model, tokenizer, text, layer=0):
    """Plotting attention heatmap"""
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
    print("1. Rebuilding Tokenizer from reviews.csv")
    tokenizer = SimpleTokenizer()
    csv_path = os.path.join(_DATA_DIR, "reviews.csv")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] in ("positive", "negative"):
                tokenizer.encode(row["text"])
    print(f"   Vocabulary size: {len(tokenizer.word_to_id)}")

    # Load the fine-tuned BERT encoder
    print("2. Loading Model")
    bert_model = TinyBERT(vocab_size=len(tokenizer.word_to_id), hidden_size=32, num_layers=2)
    model_path = os.path.join(_OUTPUT_DIR, "trained_bert.pt")

    if not os.path.exists(model_path):
        print(f"ERROR: '{model_path}' not found.")
        print("Run 'python -m tasks.sentiment' first to train and save the model.")
        return

    bert_model.load_state_dict(torch.load(model_path, weights_only=True))
    bert_model.eval()
    print(f"   Loaded fine-tuned BERT from '{model_path}'")

    # Attention Heatmaps
    sentence = "the movie was amazing and brilliant"
    print(f"\n3. Generating Attention Maps for: '{sentence}'")

    output_dir = os.path.join(_OUTPUT_DIR, "attention_maps")
    os.makedirs(output_dir, exist_ok=True)

    for layer in range(2):
        fig = visualize_attention(bert_model, tokenizer, sentence, layer=layer)
        filename = os.path.join(output_dir, f"attention_layer{layer + 1}.png")
        fig.savefig(filename, dpi=150)
        print(f"   Saved: {filename}")
        plt.close(fig)

    semantic_sentence = "the movie was amazing but the story was terrible"
    print(f"\n4. Generating Semantic Space for: '{semantic_sentence}'")

    tokens = semantic_sentence.lower().split()
    word_embeddings = bert_model.word_embed.weight.detach().numpy()


    embeddings = []
    for w in tokens:
        wid = tokenizer.word_to_id.get(w, 1)
        embeddings.append(word_embeddings[wid])
    embeddings = np.array(embeddings)

    # Build display labels
    word_counts = {}
    display_labels = []
    subscripts = "₁₂₃₄₅₆₇₈₉"
    for word in tokens:
        word_counts[word] = word_counts.get(word, 0) + 1
    occurrence = {}
    for word in tokens:
        if word_counts[word] > 1:
            occurrence[word] = occurrence.get(word, 0) + 1
            display_labels.append(f"{word}{subscripts[occurrence[word] - 1]}")
        else:
            display_labels.append(word)

    # PCA: 32-d  →  2-d
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    np.random.seed(42)
    seen_positions = {}
    for i, word in enumerate(tokens):
        key = tokenizer.word_to_id.get(word, 1)
        if key in seen_positions:
            reduced[i] += np.random.uniform(-0.15, 0.15, size=2)
        seen_positions[key] = i

    # Plotting
    token_colors = [get_word_color(w) for w in tokens]

    plt.figure(figsize=(14, 10))
    for i, (word, label) in enumerate(zip(tokens, display_labels)):
        plt.scatter(reduced[i, 0], reduced[i, 1], c=token_colors[i], s=300,
                    edgecolor="black", linewidth=1.2, alpha=0.85, zorder=2)
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=13, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Positive", markerfacecolor="#2ecc71", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Negative", markerfacecolor="#e74c3c", markersize=12),
        Line2D([0], [0], marker="o", color="w", label="Neutral / Function word", markerfacecolor="#95a5a6", markersize=12),
    ]
    plt.legend(handles=legend_elements, loc="best", fontsize=11)

    plt.title(f"Semantic Space — Contextual Token Embeddings (PCA)\n\"{semantic_sentence}\"", fontsize=13)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "semantic_space.png")
    plt.savefig(save_path, dpi=150)
    print(f"   Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_visualization()
