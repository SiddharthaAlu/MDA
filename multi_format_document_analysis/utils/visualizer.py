import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def wordcloud_plot(text):
    wc = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    return fig


def keyword_plot(keywords):
    fig, ax = plt.subplots()
    ax.bar(keywords, range(len(keywords)))
    ax.set_title("Keyword Importance")
    return fig
