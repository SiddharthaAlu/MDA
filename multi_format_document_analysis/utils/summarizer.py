import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

def professional_summary(text, ratio=0.15, max_sentences=4):
    sentences = sent_tokenize(text)

    # Very small documents → return as-is
    if len(sentences) <= max_sentences:
        return text

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1).A1

    # Rank sentences by importance
    ranked_idx = scores.argsort()[::-1]

    # Decide how many sentences to keep
    top_n = min(max_sentences, max(2, int(len(sentences) * ratio)))

    selected = sorted(ranked_idx[:top_n])
    summary = " ".join(sentences[i] for i in selected)

    return summary

