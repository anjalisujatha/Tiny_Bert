import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from bert import TinyBERT
from sentiment_classifier import SimpleTokenizer


def run_visualization():
    # 1. Load all reviews from CSV
    sample_texts = []
    labels = []
    training_texts = []
    with open("reviews.csv", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_texts.append(row['text'])
            labels.append(int(row['label']))
            # Track training-only texts (positive/negative) for building vocabulary
            if row['category'] in ('positive', 'negative'):
                training_texts.append(row['text'])

    # 2. Build tokenizer with the training vocabulary
    tokenizer = SimpleTokenizer()
    for text in training_texts:
        tokenizer.encode(text)

    # Load the fine-tuned TinyBERT encoder
    bert_model = TinyBERT(vocab_size=len(tokenizer.word_to_id), hidden_size=32, num_layers=2)
    bert_model.load_state_dict(torch.load("trained_bert.pt", weights_only=True))
    bert_model.eval()
    print("Loaded fine-tuned BERT encoder from 'trained_bert.pt'")

    # 3. Extraction: Get the [CLS] (thought vector) for each review
    embeddings = []

    print("\nExtracting embeddings...")
    with torch.no_grad():
        for text in sample_texts:
            word_ids = tokenizer.encode(text, add_new_words=False)
            input_ids = torch.tensor([word_ids])

            # Forward pass through TinyBERT
            last_hidden_state = bert_model(input_ids)

            cls_embedding = last_hidden_state[0, 0, :].numpy()
            embeddings.append(cls_embedding)

    embeddings = np.array(embeddings)

    # 4. Reduction: Use t-SNE to squash dimensions to 2D
    tsne = TSNE(n_components=2, perplexity=20, random_state=42, init='pca', learning_rate='auto')
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 5. Plotting using Seaborn
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Color map: 0=Negative(blue), 1=Positive(red), 2=Mixed(green)
    color_map = {0: 'blue', 1: 'red', 2: 'green'}
    colors = [color_map[l] for l in labels]

    # Scatter plot
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=colors,
        s=150,
        edgecolor='black',
        alpha=0.7
    )


    plt.title("TinyBERT Embeddings: Positive vs Negative vs Mixed", fontsize=15)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Mixed', markerfacecolor='green', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig("embedding_visualization.png", dpi=150)
    print("Saved plot to 'embedding_visualization.png'")
    plt.show()


if __name__ == "__main__":
    run_visualization()