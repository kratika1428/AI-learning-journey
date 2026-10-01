def calculate_statistics(text, sentences, tokens):
    words = [
        token 
        for token in tokens
        if token.isalpha()
    ]
    unique_words = set(words)
    word_count = len(words)
    unique_word_count = len(unique_words)
    sentence_count = len(sentences)

    average_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0
        else 0
    )
    average_word_length = (
        sum(len(word) for word in words) / word_count
        if word_count > 0
        else 0
    )

    return {
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": average_sentence_length,
        "average_word_length": average_word_length
    }