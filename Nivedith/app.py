import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Optional, Tuple

# sklearn core
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Optional heavy libs (import safely)
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

# transformers summarizer (optional)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# gensim coherence (optional)
try:
    import gensim
    from gensim import corpora
    GENSIM_AVAILABLE = True
except Exception:
    GENSIM_AVAILABLE = False

# ---------------------------
# Minimal safe defaults
# ---------------------------
if NLTK_AVAILABLE:
    SID = SentimentIntensityAnalyzer()
    STOPWORDS = set(stopwords.words("english"))
else:
    SID = None
    STOPWORDS = {"the", "a", "and", "is", "it", "of", "to", "in"}  # fallback tiny set

# spaCy model loader (cached)
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    if not SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

nlp = load_spacy_model()

# summarizer loader
@st.cache_resource(show_spinner=False)
def load_summarizer():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        try:
            return pipeline("summarization")
        except Exception:
            return None

SUMMARIZER = load_summarizer()

# ---------------------------
# Preprocessing utilities
# ---------------------------
RE_NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9\s']+")
RE_EXTRA_WS = re.compile(r"\s+")

def clean_text_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r", " ").replace("\n", " ")
    t = RE_NON_ALPHANUM.sub(" ", t)
    t = RE_EXTRA_WS.sub(" ", t).strip()
    return t

def simple_tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = re.findall(r"\b\w[\w']*\b", text.lower())
    if remove_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens

@st.cache_data(show_spinner=False)
def build_bigram_detector(docs: List[str], min_count: int = 5) -> dict:
    pairs = Counter()
    for d in docs:
        tokens = simple_tokenize(d, remove_stopwords=False)
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i+1])] += 1
    bigrams = {"_".join(k): v for k, v in pairs.items() if v >= min_count}
    return bigrams

def apply_bigrams(text: str, bigrams: dict) -> str:
    if not bigrams:
        return text
    tokens = simple_tokenize(text, remove_stopwords=False)
    i = 0
    out = []
    while i < len(tokens):
        if i < len(tokens) - 1:
            pair = f"{tokens[i]}_{tokens[i+1]}"
            if pair in bigrams:
                out.append(pair)
                i += 2
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)

def lemmatize_doc(text: str, nlp_model) -> str:
    if not nlp_model:
        return text
    doc = nlp_model(text)
    lemmas = [tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space]
    return " ".join(lemmas)

def preprocess_documents(docs: List[str],
                         use_lemmatize: bool = True,
                         min_bigram_count: int = 5,
                         extra_stopwords: Optional[set] = None) -> List[str]:
    extra_stopwords = extra_stopwords or set()
    cleaned = [clean_text_basic(d) for d in docs]
    bigrams = build_bigram_detector(cleaned, min_count=min_bigram_count)
    processed = []
    for txt in cleaned:
        txt2 = apply_bigrams(txt, bigrams)
        if use_lemmatize and nlp:
            txt2 = lemmatize_doc(txt2, nlp)
        tokens = simple_tokenize(txt2)
        tokens = [t for t in tokens if t not in extra_stopwords]
        processed.append(" ".join(tokens))
    return processed

# ---------------------------
# Vectorizers and feature helpers
# ---------------------------
@st.cache_resource(show_spinner=False)
def make_tfidf(max_features=3000, ngram_range=(1,2)):
    return TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=ngram_range)

@st.cache_resource(show_spinner=False)
def make_count(max_features=3000, ngram_range=(1,1)):
    return CountVectorizer(stop_words="english", max_features=max_features, ngram_range=ngram_range)

def compute_tfidf_matrix(corpus: List[str], vect: TfidfVectorizer):
    if not corpus:
        return None, None
    X = vect.fit_transform(corpus)
    return X, vect

def extract_top_tfidf_terms(tfidf_matrix, feature_names, top_n=15):
    if tfidf_matrix is None:
        return []
    col_sum = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    top_ix = col_sum.argsort()[::-1][:top_n]
    return [(feature_names[i], float(col_sum[i])) for i in top_ix]

# ---------------------------
# Robust NMF fit (safe)
# ---------------------------
def fit_nmf_safe(tfidf_matrix, n_topics=6, random_state=42):
    """
    Safe NMF: ensures n_components <= min(n_samples, n_features), returns (model,W,H) or (None,None,None)
    """
    if tfidf_matrix is None:
        st.warning("TF-IDF matrix is None — skipping NMF.")
        return None, None, None
    try:
        n_samples, n_features = tfidf_matrix.shape
    except Exception as e:
        st.warning(f"Invalid TF-IDF matrix shape: {e}")
        return None, None, None

    if n_samples < 1 or n_features < 1:
        st.warning("Not enough data to run NMF.")
        return None, None, None

    safe_max = min(n_samples, n_features)
    if n_topics > safe_max:
        st.info(f"Requested {n_topics} topics, but safe max is {safe_max}. Reducing to {safe_max}.")
        n_topics = safe_max
    if n_topics < 1:
        st.warning("n_topics resolved to < 1 — skipping NMF.")
        return None, None, None

    try:
        model = NMF(n_components=n_topics, init="nndsvda", random_state=random_state, max_iter=400)
        W = model.fit_transform(tfidf_matrix)
        H = model.components_
        return model, W, H
    except Exception as e:
        st.error(f"NMF failed: {e}")
        return None, None, None

# ---------------------------
# LDA fit (safe)
# ---------------------------
def fit_lda_safe(count_matrix, n_topics=6, random_state=42):
    if count_matrix is None:
        st.warning("Count matrix is None — skipping LDA.")
        return None, None, None
    try:
        n_samples, n_features = count_matrix.shape
    except Exception as e:
        st.warning(f"Invalid Count matrix shape: {e}")
        return None, None, None
    safe_max = min(n_samples, n_features)
    n_topics = max(1, min(n_topics, safe_max))
    try:
        model = LDA(n_components=n_topics, random_state=random_state, learning_method="batch", max_iter=20)
        W = model.fit_transform(count_matrix)
        components = model.components_
        return model, W, components
    except Exception as e:
        st.error(f"LDA failed: {e}")
        return None, None, None

# ---------------------------
# Topic helpers & coherence (gensim optional)
# ---------------------------
def get_topic_terms(components, feature_names, top_n=10):
    topics = []
    for idx, comp in enumerate(components):
        top_idx = comp.argsort()[::-1][:top_n]
        topics.append((idx+1, [feature_names[i] for i in top_idx]))
    return topics

def compute_coherence_gensim(tokenized_texts, topics_terms):
    if not GENSIM_AVAILABLE:
        return None
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    from gensim.models import CoherenceModel
    coherences = []
    for _, terms in topics_terms:
        cm = CoherenceModel(topics=[terms], texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherences.append(float(cm.get_coherence()))
    return coherences

# ---------------------------
# Visualization helpers
# ---------------------------
def plot_top_keywords_bar(keywords: List[Tuple[str, float]]):
    if not keywords:
        st.write("No keywords to plot.")
        return
    terms, scores = zip(*keywords)
    fig, ax = plt.subplots(figsize=(8, max(3, len(terms)*0.35)))
    y = np.arange(len(terms))
    ax.barh(y, scores[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(list(terms)[::-1])
    ax.set_xlabel("TF-IDF score (sum)")
    ax.set_title("Top Keywords")
    st.pyplot(fig)

# ---------------------------
# Document chunking (for single long doc)
# ---------------------------
def chunk_document(text: str, chunk_size_chars: int = 2500) -> List[str]:
    if not isinstance(text, str) or len(text) == 0:
        return []
    chunks = []
    for i in range(0, len(text), chunk_size_chars):
        chunks.append(text[i:i+chunk_size_chars])
    return chunks

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Advanced Text Analysis — Fixed", layout="wide")
st.title("Advanced Dynamic Text Analysis — Fixed & Hardened")

# Sidebar: Inputs and options
with st.sidebar:
    st.header("1) Data input")
    data_input = st.radio("Input mode", ["Paste text", "Upload .txt file(s)", "Upload CSV (col: text)"])
    docs = []
    if data_input == "Paste text":
        pasted = st.text_area("Paste one or many documents (separate with blank line)")
        if pasted and pasted.strip():
            # split on blank lines into docs
            docs = [p.strip() for p in re.split(r"\n\s*\n", pasted.strip()) if p.strip()]
    elif data_input == "Upload .txt file(s)":
        uploaded_files = st.file_uploader("Select .txt files (multi)", type=["txt"], accept_multiple_files=True)
        if uploaded_files:
            for f in uploaded_files:
                try:
                    docs.append(f.read().decode("utf-8"))
                except Exception:
                    try:
                        docs.append(f.read().decode("latin-1"))
                    except Exception:
                        st.warning(f"Could not decode {f.name}")
    else:
        csv_file = st.file_uploader("CSV (must contain 'text' column)", type=["csv"])
        if csv_file:
            df_in = pd.read_csv(csv_file)
            if "text" in df_in.columns:
                docs = df_in["text"].astype(str).tolist()
            else:
                st.error("CSV must contain a column named 'text'.")

    st.markdown("---")
    st.header("2) Preprocessing")
    use_lemmatize = st.checkbox("Lemmatize (spaCy) [optional]", value=False, disabled=not bool(nlp))
    min_bigram_count = st.number_input("Min bigram count to join", min_value=2, max_value=50, value=5)
    extra_stop = st.text_input("Extra stopwords (comma-separated)", value="")
    extra_stopwords = set([s.strip().lower() for s in extra_stop.split(",") if s.strip()])

    st.markdown("---")
    st.header("3) Modeling options")
    topic_method = st.selectbox("Topic method", ["NMF (TF-IDF)", "LDA (CountVectorizer)"])
    n_topics = st.slider("Number of topics", 1, 20, 6)
    top_n_terms = st.slider("Top terms per topic", 3, 20, 8)

    st.markdown("---")
    st.header("4) Clustering & Visualization")
    light_mode = st.checkbox("Light mode (reduce RAM/CPU)", value=True,
                             help="Enable to disable heavy ops (t-SNE) and lower vector sizes.")
    do_cluster = st.checkbox("Do KMeans clustering", value=not light_mode)
    n_clusters = st.number_input("KMeans clusters", 2, 20, 4)
    do_tsne = st.checkbox("Show 2D t-SNE plot (heavy)", value=False, disabled=light_mode)

    st.markdown("---")
    st.header("5) Summarization (optional)")
    do_summary = st.checkbox("Generate extractive summary (transformers)", value=False, disabled=not TRANSFORMERS_AVAILABLE)
    summary_max_chars = st.number_input("Summary max chars", 50, 3000, 400)

    st.markdown("---")
    st.header("6) Execute")
    run_button = st.button("Run analysis")

# show minimal stats
col_main, col_stats = st.columns([3,1])
with col_stats:
    st.metric("Docs loaded", len(docs))
    st.metric("spaCy available", "Yes" if nlp else "No")
    st.metric("Transformers", "Yes" if SUMMARIZER else "No")

if not run_button:
    st.info("Configure options in the sidebar, then press **Run analysis**.")
    st.stop()

if not docs:
    st.error("No documents provided. Upload or paste text and try again.")
    st.stop()

# Preprocessing
with st.spinner("Preprocessing..."):
    # If a single very long doc and topics requested, chunk into pseudo-documents
    if len(docs) == 1 and (topic_method is not None):
        if len(docs[0]) > 4000:
            docs = chunk_document(docs[0], chunk_size_chars=2500)
            st.info(f"Single long document split into {len(docs)} chunks for topic modeling.")

    processed_docs = preprocess_documents(docs, use_lemmatize=use_lemmatize, min_bigram_count=min_bigram_count, extra_stopwords=extra_stopwords)

# Build vectorizers
tfidf_max = 1500 if light_mode else 4000
count_max = 1500 if light_mode else 4000

tfidf_vect = make_tfidf(max_features=tfidf_max, ngram_range=(1,2))
count_vect = make_count(max_features=count_max, ngram_range=(1,1))

tfidf_matrix, tfidf_vect = compute_tfidf_matrix(processed_docs, tfidf_vect)
count_matrix = count_vect.fit_transform(processed_docs)
feature_names_tfidf = tfidf_vect.get_feature_names_out() if tfidf_vect else []
feature_names_count = count_vect.get_feature_names_out() if count_vect else []

# Top keywords
top_keywords = extract_top_tfidf_terms(tfidf_matrix, feature_names_tfidf, top_n=20)

st.header("Results")

# Top keywords + bar
st.subheader("Top Keywords (corpus-level)")
plot_top_keywords_bar(top_keywords[:15])
df_kw = pd.DataFrame(top_keywords, columns=["Keyword", "Score"])
st.dataframe(df_kw.head(50))

# Sentiment (VADER)
st.subheader("Corpus Sentiment (VADER)")
if SID:
    # operate on whole corpus as single string and per-doc
    whole = " ".join(docs)
    scores = SID.polarity_scores(whole)
    st.write("Corpus-level scores:", scores)
    # per doc
    per_doc_scores = [SID.polarity_scores(d) for d in docs]
    df_sent = pd.DataFrame(per_doc_scores)
    st.dataframe(df_sent.head(50))
else:
    st.info("NLTK/VADER not available — install nltk to enable sentiment.")

# Topics
st.subheader("Topic Modeling")
if topic_method == "NMF (TF-IDF)":
    if tfidf_matrix is None:
        st.error("TF-IDF matrix is empty.")
        nmf_model = W = H = None
    else:
        nmf_model, W, H = fit_nmf_safe(tfidf_matrix, n_topics=n_topics)
        if nmf_model is None:
            st.warning("NMF not computed (insufficient data or error).")
        else:
            topics = get_topic_terms(H, feature_names_tfidf, top_n=top_n_terms)
            for tid, terms in topics:
                st.markdown(f"**Topic {tid}:** " + ", ".join(terms))
            doc_topics = np.argmax(W, axis=1) + 1
else:
    if count_matrix is None:
        st.error("Count matrix is empty.")
        lda_model = W = components = None
    else:
        lda_model, W, components = fit_lda_safe(count_matrix, n_topics=n_topics)
        if lda_model is None:
            st.warning("LDA not computed.")
        else:
            topics = get_topic_terms(components, feature_names_count, top_n=top_n_terms)
            for tid, terms in topics:
                st.markdown(f"**Topic {tid}:** " + ", ".join(terms))
            doc_topics = np.argmax(W, axis=1) + 1

# Coherence (optional)
if GENSIM_AVAILABLE and ('topics' in locals()):
    st.subheader("Topic Coherence (gensim c_v)")
    tokenized = [simple_tokenize(d, remove_stopwords=False) for d in processed_docs]
    coherences = compute_coherence_gensim(tokenized, topics)
    if coherences:
        for i, c in enumerate(coherences):
            st.write(f"Topic {i+1} coherence: {c:.3f}")
else:
    st.info("Gensim not available — skipping coherence (optional).")

# Document-level results
st.subheader("Document-level Results")
df_docs = pd.DataFrame({"original": docs, "processed": processed_docs})
if 'doc_topics' in locals():
    df_docs["topic"] = doc_topics
else:
    df_docs["topic"] = np.nan
st.dataframe(df_docs.head(200))

# Representative doc per topic (if available)
if 'doc_topics' in locals():
    st.subheader("Representative document per topic")
    unique_topics = sorted(set(doc_topics))
    for t in unique_topics:
        idxs = np.where(doc_topics == t)[0]
        if len(idxs) == 0:
            continue
        chosen = idxs[0]
        st.markdown(f"**Topic {t} — Doc {chosen}**")
        st.write(docs[chosen][:800] + ("..." if len(docs[chosen]) > 800 else ""))

# Clustering & visualization
if do_cluster:
    st.subheader("Clustering (KMeans on SVD-reduced TF-IDF)")
    try:
        n_svd = 20 if light_mode else min(50, tfidf_matrix.shape[1]-1)
        svd = TruncatedSVD(n_components=max(2, min(n_svd, tfidf_matrix.shape[1]-1)), random_state=42)
        docvecs = svd.fit_transform(tfidf_matrix)
        k = min(n_clusters, max(2, docvecs.shape[0]-1))
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(docvecs)
        st.write("Cluster sizes:", dict(Counter(clusters)))
        # show 2 examples per cluster
        for c in sorted(set(clusters)):
            st.markdown(f"Cluster {c} (size={sum(clusters==c)})")
            idxs = np.where(clusters == c)[0][:2]
            for i in idxs:
                st.write(f"- Doc {i}: {docs[i][:180]}...")
    except Exception as e:
        st.warning(f"Clustering error: {e}")

    # optional 2D plot (t-SNE asked explicitly)
    if do_tsne:
        st.subheader("2D Visualization (SVD→t-SNE)")
        try:
            # t-SNE is expensive — do it on docvecs (already reduced)
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, init="pca", random_state=42, perplexity=min(30, max(5, docvecs.shape[0]-1)))
            coords = tsne.fit_transform(docvecs)
            df_plot = pd.DataFrame(coords, columns=["x","y"])
            df_plot['cluster'] = clusters
            fig, ax = plt.subplots(figsize=(8,6))
            scatter = ax.scatter(df_plot['x'], df_plot['y'], c=df_plot['cluster'], cmap="tab10", alpha=0.8)
            ax.set_title("Document clusters (t-SNE)")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"t-SNE failed or too heavy: {e}")

# Wordcloud
st.subheader("Word Cloud")
try:
    from wordcloud import WordCloud
    wc_text = " ".join(processed_docs)
    if wc_text.strip():
        wc = WordCloud(width=900, height=400, background_color="white", max_words=150).generate(wc_text)
        fig, ax = plt.subplots(figsize=(10,4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.write("No text for wordcloud.")
except Exception:
    st.info("wordcloud not installed or failed — install 'wordcloud' to enable this feature.")

# Summaries (optional)
if do_summary:
    st.subheader("Summaries")
    if SUMMARIZER is None:
        st.error("Transformers summarizer not available. Install transformers to enable.")
    else:
        for i, d in enumerate(docs):
            st.markdown(f"Doc {i} summary:")
            try:
                text_for_sum = d[:5000] if len(d) > 5000 else d
                res = SUMMARIZER(text_for_sum, max_length=min(200, summary_max_chars//2), min_length=30)
                st.write(res[0]["summary_text"])
            except Exception as e:
                st.warning(f"Summarization failed for doc {i}: {e}")

# Export results
st.markdown("---")
st.subheader("Export")
results_df = df_docs.copy()
csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv, file_name="analysis_results.csv", mime="text/csv")
st.success("Analysis complete ✅")
