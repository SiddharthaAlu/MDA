import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import textstat

nltk.download("punkt")
nltk.download("vader_lexicon")

def keyword_scores(text, top_n=15):
    tfidf = TfidfVectorizer(stop_words="english", max_features=800)
    X = tfidf.fit_transform([text])
    scores = X.toarray()[0]
    words = tfidf.get_feature_names_out()
    pairs = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]

def topics(text, n_topics=4):
    tfidf = TfidfVectorizer(stop_words="english", max_features=600)
    X = tfidf.fit_transform([text])

    if X.shape[1] < n_topics:
        return []

    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(X)

    words = tfidf.get_feature_names_out()
    topic_list = []

    for topic in nmf.components_:
        topic_list.append([words[i] for i in topic.argsort()[:-6:-1]])

    return topic_list

def sentiment_scores(text):
    sia = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    return [sia.polarity_scores(s)["compound"] for s in sentences]

def readability(text):
    return {
        "Reading Ease": round(textstat.flesch_reading_ease(text), 2),
        "Grade Level": round(textstat.flesch_kincaid_grade(text), 2)
    }
