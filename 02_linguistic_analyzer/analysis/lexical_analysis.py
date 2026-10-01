from collections import Counter

def lexical_analysis(tokens):
    words = [
        token.lower()
        for token in tokens
        if token.isalpha()
    ]
    total_words = len(words)
    unique_words = set(words)
    vocabulary_size = len(unique_words)

    lexical_diversity = (
        vocabulary_size / total_words
        if total_words > 0
        else 0
    )
    average_word_length = (
        sum(len(word) for word in words) / total_words
        if total_words > 0
        else 0
    )

    longest_word = max(words, key=len) if words else ""
    shortest_word = min(words, key=len) if words else ""
    frequency = Counter(words)

    rare_words = [
        word
        for word, count in frequency.items()
        if count == 1
    ]

    return {
        "total_words": total_words,
        "vocabulary_size": vocabulary_size,
        "lexical_diversity": lexical_diversity,
        "average_word_length": average_word_length,
        "longest_word": longest_word,
        "shortest_word": shortest_word,
        "rare_words": rare_words
    }