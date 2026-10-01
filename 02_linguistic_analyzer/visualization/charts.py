import matplotlib.pyplot as plt

def plot_pos_distribution(pos_distribution):
    labels = list(pos_distribution.keys())
    values = list(pos_distribution.values())

    fig, ax = plt.subplots(figsize=(10,6))

    ax.barh(labels, values)

    ax.set_title("POS Distribution")
    ax.set_xlabel("Part of Speech")
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig

def plot_ngram_distribution(ngram_counts, title):
    ngrams = [
        " ".join(ngram)
        for ngram, count in ngram_counts.most_common(10)
    ]
    frequencies = [
        count
        for ngram, count in ngram_counts.most_common(10)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(ngrams, frequencies)

    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("N-Gram")

    for bar, value in zip(bars, frequencies):
        ax.text(
            value + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center"
        )
    fig.tight_layout()

    return fig