import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

def professional_summary(text, ratio=0.3):
    sentences = sent_tokenize(text)

    if len(sentences) < 5:
        return text

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(sentences)
    scores = np.array(X.sum(axis=1)).ravel()

    ranked_idx = scores.argsort()[::-1]
    top_n = max(5, int(len(sentences) * ratio))

    selected = sorted(ranked_idx[:top_n])
    summary = " ".join([sentences[i] for i in selected])

    return summary
