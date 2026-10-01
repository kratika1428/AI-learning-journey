from nltk import pos_tag
from nltk.tokenize import word_tokenize

from collections import Counter

def get_pos_tags(text):
    tokens = word_tokenize(text)
    words = [
        token for token in tokens
        if token.isalpha()
    ]
    tagged_words = pos_tag(words)
    return tagged_words

def get_pos_description(tag):
    descriptions = {
        "NN": "Noun",
        "NNS": "Noun",
        "NNP": "Proper Noun",
        "NNPS": "Proper Noun",

        "VB": "Verb",
        "VBD": "Verb",
        "VBG": "Verb",
        "VBN": "Verb",
        "VBP": "Verb",
        "VBZ": "Verb",

        "JJ": "Adjective",
        "JJR": "Adjective",
        "JJS": "Adjective",

        "RB": "Adverb",
        "RBR": "Adverb",
        "RBS": "Adverb",

        "PRP": "Pronoun",
        "PRP$": "Possessive Pronoun",

        "DT": "Determiner",

        "IN": "Preposition/Conjunction",

        "CC": "Conjunction"
    }
    return descriptions.get(tag, "Other")

def get_pos_distribution(tagged_words):
    pos_categories = []
    for word, tag in tagged_words:
        category = get_pos_description(tag)
        pos_categories.append(category)
    return Counter(pos_categories)