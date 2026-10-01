from nltk.util import ngrams
from collections import Counter

def generate_ngrams(tokens, n):
    ngram_list = list(ngrams(tokens, n))
    return Counter(ngram_list)