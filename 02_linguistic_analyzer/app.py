import streamlit as st

from preprocessing.cleaning import clean_text
from preprocessing.tokenizer import tokenize_sentences, tokenize_words

from analysis.statistics import calculate_statistics
from analysis.pos_analysis import get_pos_tags, get_pos_distribution
from analysis.ngram_analysis import generate_ngrams
from analysis.lexical_analysis import lexical_analysis

from visualization.charts import plot_pos_distribution, plot_ngram_distribution

st.set_page_config(
    page_title="linguistic analyzer",
    layout="centered"
)

st.title("Linguistic Analyzer")
st.write("Analyze the linguistic structure of the text")
st.divider()

text = st.text_area(
    "Enter your text",
    height=200,
    placeholder="Type or paste your text here..."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text...")
    else:
        cleaned_text = clean_text(text)
        sentences = tokenize_sentences(text)
        tokens = tokenize_words(cleaned_text)
        lexical_data = lexical_analysis(tokens)
        statistics = calculate_statistics(
            cleaned_text,
            sentences,
            tokens
        )

        st.subheader("Text Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Sentences",
                statistics["sentence_count"]
            )
        with col2:
            st.metric(
                "Words",
                statistics["word_count"]
            )
        with col3:
            st.metric(
                "Unique words",
                statistics["unique_word_count"]
            )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Average Sentence Length",
                f"{statistics['average_sentence_length']:.2f}"
            )
        with col2:
            st.metric(
                "Average Word Length",
                f"{statistics['average_word_length']:.2f}"
            )
        
        st.subheader("Sentence Analysis")
        for i, sentence in enumerate(sentences, start=1):
            st.write(f"**Sentence {i}:** {sentence}")
        
        st.subheader("Lexical Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Total Words",
                lexical_data["total_words"]
            )
        with col2:
            st.metric(
                "Vocabulary Size",
                lexical_data["vocabulary_size"]
            )
        with col3: 
            st.metric(
                "Lexical Diversity",
                f"{lexical_data['lexical_diversity']:.2f}"
            )
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric(
                "Average Word Length",
                f"{lexical_data['average_word_length']:.2f}"
            )
        with col5:
            st.metric(
                "Longest Word",
                lexical_data["longest_word"]
            )
        with col6:
            st.metric(
                "Shortest Word",
                lexical_data["shortest_word"]
            )
        
        st.write("### Rare Words")
        if lexical_data["rare_words"]:
            st.write(", ".join(lexical_data["rare_words"]))
        else:
            st.write("No rare words found.")

        unigrams = generate_ngrams(tokens, 1)

        bigrams = generate_ngrams(tokens, 2)
        st.subheader("Bigram Analysis")
        for ngram, frequency in bigrams.most_common(10):
            phrase = " ".join(ngram)
            st.write(f"**{phrase}** → {frequency}")
        st.subheader("Bigram Visualization")
        fig = plot_ngram_distribution(
            bigrams,
            "Top 10 Bigrams"
        )
        st.pyplot(fig)
        st.subheader("Bigram Analysis")
        for ngram, frequency in bigrams.most_common(10):
            phrase = " ".join(ngram)
            st.write(f"**{phrase}** → {frequency}")

        trigrams = generate_ngrams(tokens, 3)
        st.subheader("Trigram Analysis")
        for ngram, frequency in trigrams.most_common(10):
            phrase = " ".join(ngram)
            st.write(f"**{phrase}** → {frequency}")
        st.subheader("Trigram Visualization")
        fig = plot_ngram_distribution(
            trigrams,
            "Top 10 Trigrams"
        )
        st.pyplot(fig)
        st.subheader("Trigram Analysis")
        for ngram, frequency in trigrams.most_common(10):
            phrase = " ".join(ngram)
            st.write(f"**{phrase}** → {frequency}")

        tagged_words = get_pos_tags(cleaned_text)
        pos_distribution = get_pos_distribution(tagged_words)
        st.subheader("POS Distribution")
        for category, count in pos_distribution.items():
            st.write(f"**{category}:** {count}")

        st.subheader("POS Distribution")
        fig = plot_pos_distribution(pos_distribution)
        st.pyplot(fig)