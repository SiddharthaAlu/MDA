import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from utils.extractor import extract_text
from utils.summarizer import professional_summary
from utils.nlp_analysis import keyword_scores, topics, sentiment_scores, readability

st.set_page_config("📘 Advanced Multi-Format Document Intelligence", layout="wide", page_icon="📄")

st.markdown("<h1 style='text-align:center; color: #2E86C1;'>📘 Multi-Format Document Intelligence</h1>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload your PDF, DOCX, or TXT document", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file, file_type)

    if not text:
        st.error("No readable content found.")
        st.stop()

    # ---------------- KPIs ----------------
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
    col1.metric("Words", len(text.split()))
    col2.metric("Sentences", len(text.split(".")))
    col3.metric("Characters", len(text))
    col4.metric("Avg. Word Length", round(sum(len(w) for w in text.split()) / len(text.split()),2))
    col5.metric("Format", file_type.upper())

    st.markdown("---")

    # ---------------- Tabs ----------------
    tabs = st.tabs(["📝 Summary", "📊 Keywords", "🧠 Topics", "📈 Sentiment", "📚 Readability"])

    # -------- SUMMARY --------
    with tabs[0]:
        st.subheader("Professional Summary")
        summary = professional_summary(text)
        st.success(summary)

        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )

    # -------- KEYWORDS --------
    with tabs[1]:
        st.subheader("Keyword Analysis")
        keywords = keyword_scores(text)
        df_keywords = pd.DataFrame(keywords, columns=["Keyword", "Score"])

        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("**Keyword Frequency Bar Chart**")
            fig, ax = plt.subplots(figsize=(5, max(3, len(df_keywords)*0.3)))
            ax.barh(df_keywords["Keyword"], df_keywords["Score"], color="#3498DB")
            ax.invert_yaxis()
            st.pyplot(fig)

        with col2:
            st.markdown("**Keyword WordCloud**")
            wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(keywords))
            st.image(wc.to_array(), use_column_width=True)

    # -------- TOPICS --------
    with tabs[2]:
        st.subheader("Discovered Topics")
        topic_list = topics(text)
        for i, topic in enumerate(topic_list,1):
            with st.expander(f"Topic {i}"):
                st.write(", ".join(topic))

    # -------- SENTIMENT --------
    with tabs[3]:
        st.subheader("Sentiment Analysis")
        scores = sentiment_scores(text)
        df_sentiment = pd.DataFrame(scores, columns=["Compound Score"])
        st.line_chart(df_sentiment)
        st.write("Positive sentences:", len([s for s in scores if s>0]))
        st.write("Negative sentences:", len([s for s in scores if s<0]))
        st.write("Neutral sentences:", len([s for s in scores if s==0]))

    # -------- READABILITY --------
    with tabs[4]:
        st.subheader("Readability Metrics")
        r = readability(text)
        st.metric("Reading Ease", r["Reading Ease"])
        st.metric("Grade Level", r["Grade Level"])
