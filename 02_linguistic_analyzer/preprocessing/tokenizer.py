from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize_words(text):
    return word_tokenize(text)

def tokenize_sentences(text):
    return sent_tokenize(text)