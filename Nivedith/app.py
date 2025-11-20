import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import zipfile
import os
import tempfile
import matplotlib.pyplot as plt
import gc
from collections import Counter
from typing import List, Optional, Tuple

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.cluster import KMeans

# Optional libraries (graceful)
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

try:
    import docx  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except Exception:
    OPENPYXL_AVAILABLE = False

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

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import gensim
    GENSIM_AVAILABLE = True
    from gensim import corpora
except Exception:
    GENSIM_AVAILABLE = False

# -------------------------
# Basic configs & loaders
# -------------------------
if NLTK_AVAILABLE:
    SID = SentimentIntensityAnalyzer()
    STOPWORDS = set(stopwords.words("english"))
else:
    SID = None
    STOPWORDS = {"the", "a", "and", "is", "it", "of", "to", "in"}

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    if not SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

nlp = load_spacy_model()

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

# -------------------------
# File parsing utilities
# -------------------------
def read_txt_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    # last resort
    return b.decode("latin-1", errors="ignore")

def extract_text_from_pdf_bytes(b: bytes) -> str:
    if not PYPDF2_AVAILABLE:
        raise RuntimeError("PyPDF2 not installed")
    text_parts = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(b))
        for p in reader.pages:
            try:
                page_text = p.extract_text() or ""
            except Exception:
                page_text = ""
            text_parts.append(page_text)
    except Exception as e:
        # fallback: try older api if present
        raise RuntimeError(f"PDF read error: {e}")
    return "\n".join(text_parts)

def extract_text_from_docx_bytes(b: bytes) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx not installed")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(b)
        tmp.flush()
        tmp_name = tmp.name
    try:
        doc = docx.Document(tmp_name)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass

def extract_text_from_json_bytes(b: bytes) -> List[str]:
    """
    Attempts to find text-like fields in JSON content.
    Returns list of text entries (documents).
    """
    try:
        txt = read_txt_bytes(b)
        obj = pd.io.json.loads(txt)
    except Exception:
        import json
        try:
            obj = json.loads(b.decode("utf-8", errors="ignore"))
        except Exception:
            raise RuntimeError("JSON decode failed")
    texts = []

    def recurse(o):
        if o is None:
            return
        if isinstance(o, str):
            if o.strip():
                texts.append(o.strip())
        elif isinstance(o, list):
            for item in o:
                recurse(item)
        elif isinstance(o, dict):
            for k, v in o.items():
                # heuristics: keys likely to contain textual data
                if isinstance(v, (str, list, dict)):
                    recurse(v)
                elif isinstance(v, (int, float)):
                    continue
        else:
            return

    recurse(obj)
    # if nothing found, return stringified JSON
    if not texts:
        texts = [str(obj)]
    return texts

def extract_text_from_excel_bytes(b: bytes) -> Tuple[List[str], List[str]]:
    """
    Returns (sheets, extracted_texts)
    For each sheet, concatenates row values into text rows, returns as documents.
    """
    if not OPENPYXL_AVAILABLE:
        raise RuntimeError("openpyxl not installed")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(b)
        tmp.flush()
        tmp_name = tmp.name
    try:
        xls = pd.read_excel(tmp_name, sheet_name=None, dtype=str)
        docs = []
        sheets = []
        for sname, df in xls.items():
            sheets.append(sname)
            # combine rows into textual documents
            for _, row in df.fillna("").iterrows():
                row_text = " ".join([str(x) for x in row.values if str(x).strip()])
                if row_text.strip():
                    docs.append(row_text)
        return sheets, docs
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass

def extract_texts_from_zip_bytes(b: bytes) -> List[str]:
    texts = []
    try:
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            for name in z.namelist():
                # skip directories
                if name.endswith('/'):
                    continue
                try:
                    ext = os.path.splitext(name)[1].lower()
                    data = z.read(name)
                    if ext == ".txt":
                        texts.append(read_txt_bytes(data))
                    elif ext == ".csv":
                        try:
                            df = pd.read_csv(io.BytesIO(data))
                            # try to find textual column
                            for col in df.columns:
                                if df[col].dtype == object:
                                    texts.extend(df[col].dropna().astype(str).tolist())
                                    break
                        except Exception:
                            texts.append(read_txt_bytes(data))
                    elif ext in (".json",):
                        try:
                            texts.extend(extract_text_from_json_bytes(data))
                        except Exception:
                            texts.append(read_txt_bytes(data))
                    elif ext in (".pdf",):
                        if PYPDF2_AVAILABLE:
                            try:
                                texts.append(extract_text_from_pdf_bytes(data))
                            except Exception:
                                texts.append("")
                    elif ext in (".docx",):
                        if DOCX_AVAILABLE:
                            try:
                                texts.append(extract_text_from_docx_bytes(data))
                            except Exception:
                                texts.append("")
                    # ignore other file types
                except Exception:
                    continue
    except Exception as e:
        raise RuntimeError(f"Zip read failed: {e}")
    return texts

# -------------------------
# Text processing & analysis (kept memory-safe)
# -------------------------
RE_NON_ALNUM = re.compile(r"[^a-zA-Z0-9\s'-]+")
RE_MULTI_WS = re.compile(r"\s+")

def clean_text_basic(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = RE_NON_ALNUM.sub(" ", s)
    s = RE_MULTI_WS.sub(" ", s).strip()
    return s

def tokenize(text: str, remove_stopwords=True):
    toks = re.findall(r"\b\w[\w']*\b", text.lower())
    if remove_stopwords:
        return [t for t in toks if t not in STOPWORDS]
    return toks

def preprocess_documents(docs: List[str], extra_stop: Optional[set] = None) -> List[str]:
    extra_stop = extra_stop or set()
    processed = []
    for d in docs:
        dclean = clean_text_basic(d)
        toks = tokenize(dclean)
        toks = [t for t in toks if t not in extra_stop]
        processed.append(" ".join(toks))
    return processed

def compute_tfidf_safe(docs: List[str], max_features=2000):
    if not docs:
        return None, None
    vect = TfidfVectorizer(stop_words="english", max_features=max_features, ngram_range=(1,2))
    X = vect.fit_transform(docs)
    return X, vect

def fit_nmf_safe(tfidf_matrix, n_topics=6):
    if tfidf_matrix is None:
        return None, None, None
    n_samples, n_features = tfidf_matrix.shape
    safe_max = min(max(1, n_samples), max(1, n_features))
    n_topics = max(1, min(n_topics, safe_max))
    try:
        model = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=400)
        W = model.fit_transform(tfidf_matrix)
        H = model.components_
        return model, W, H
    except Exception as e:
        st.error(f"NMF failed: {e}")
        return None, None, None

# -------------------------
# Plot helpers (memory-friendly)
# -------------------------
def plot_horizontal(items: List[Tuple[str, float]], title="Top items", max_items=12):
    items = items[:max_items]
    if not items:
        st.write("No items to plot.")
        return
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7, 0.6*len(labels) + 1.2), dpi=80)
    y = np.arange(len(labels))
    ax.barh(y, vals, color="#2b8cbe", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    gc.collect()

# -------------------------
# UI: Accept many formats
# -------------------------
st.set_page_config(page_title="Dynamic Text Analysis — All Formats", layout="wide")
st.title("Dynamic Text Analysis — Accepts TXT, CSV, XLSX, JSON, PDF, DOCX, ZIP, Paste")

col1, col2 = st.columns([3, 1])
with col1:
    st.write("Upload files or paste text. The app will attempt to extract textual content from many formats.")
with col2:
    st.write("Optional libs:")
    st.write(f"PyPDF2: {'Yes' if PYPDF2_AVAILABLE else 'No'}")
    st.write(f"python-docx: {'Yes' if DOCX_AVAILABLE else 'No'}")
    st.write(f"openpyxl: {'Yes' if OPENPYXL_AVAILABLE else 'No'}")

st.markdown("---")

# Input area
input_mode = st.radio("Input type", ("Paste text", "Upload file(s)"))
docs: List[str] = []

if input_mode == "Paste text":
    pasted = st.text_area("Paste text here (separate documents with a blank line)", height=240)
    if pasted and pasted.strip():
        docs = [p.strip() for p in re.split(r"\n\s*\n", pasted.strip()) if p.strip()]
else:
    uploaded = st.file_uploader("Upload one or more files", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            name = f.name.lower()
            data = f.read()
            ext = os.path.splitext(name)[1]
            try:
                if ext in (".txt", ".text"):
                    docs.append(read_txt_bytes(data))
                elif ext == ".csv":
                    try:
                        df = pd.read_csv(io.BytesIO(data))
                        # try to find best text column heuristically
                        text_cols = [c for c in df.columns if df[c].dtype == object]
                        if text_cols:
                            col = st.selectbox(f"Choose text column for {f.name}", text_cols, key=f"name_col_{name}")
                            docs.extend(df[col].dropna().astype(str).tolist())
                        else:
                            # fallback: concat each row
                            docs.extend(df.apply(lambda r: " ".join([str(x) for x in r.values if str(x).strip()]), axis=1).tolist())
                    except Exception:
                        docs.append(read_txt_bytes(data))
                elif ext in (".xls", ".xlsx"):
                    if OPENPYXL_AVAILABLE:
                        try:
                            sheets, sheet_docs = extract_text_from_excel_bytes(data)
                            # let user select sheet(s) if more than one
                            if len(sheets) > 1:
                                sel = st.multiselect(f"Select sheets to include from {f.name}", sheets, default=sheets)
                                # include only selected
                                with pd.ExcelFile(io.BytesIO(data)) as xls:
                                    for s in sel:
                                        df = pd.read_excel(xls, sheet_name=s, dtype=str)
                                        for _, row in df.fillna("").iterrows():
                                            row_text = " ".join([str(x) for x in row.values if str(x).strip()])
                                            if row_text.strip():
                                                docs.append(row_text)
                            else:
                                docs.extend(sheet_docs)
                        except Exception as e:
                            st.warning(f"Could not parse Excel {f.name}: {e}. Treating as binary text.")
                            docs.append(read_txt_bytes(data))
                    else:
                        st.warning(f"openpyxl not installed — cannot parse {f.name}. Install openpyxl to enable Excel parsing.")
                        docs.append(read_txt_bytes(data))
                elif ext == ".json":
                    try:
                        jtexts = extract_text_from_json_bytes(data)
                        # if many keys, let user decide later
                        docs.extend(jtexts)
                    except Exception as e:
                        st.warning(f"JSON parse failed for {f.name}: {e}")
                        docs.append(read_txt_bytes(data))
                elif ext == ".pdf":
                    if PYPDF2_AVAILABLE:
                        try:
                            pdf_text = extract_text_from_pdf_bytes(data)
                            docs.append(pdf_text)
                        except Exception as e:
                            st.warning(f"PDF read failed for {f.name}: {e}")
                    else:
                        st.warning(f"PyPDF2 not installed — cannot parse PDF {f.name}.")
                elif ext == ".docx":
                    if DOCX_AVAILABLE:
                        try:
                            docx_text = extract_text_from_docx_bytes(data)
                            docs.append(docx_text)
                        except Exception as e:
                            st.warning(f"DOCX read failed for {f.name}: {e}")
                    else:
                        st.warning(f"python-docx not installed — cannot parse {f.name}")
                elif ext == ".zip":
                    try:
                        ztexts = extract_texts_from_zip_bytes(data)
                        docs.extend(ztexts)
                    except Exception as e:
                        st.warning(f"ZIP extract failed for {f.name}: {e}")
                else:
                    # unknown extension: try to decode as text
                    try:
                        docs.append(read_txt_bytes(data))
                    except Exception:
                        st.warning(f"Unknown file type {f.name} — skipped or binary.")
            except Exception as e:
                st.warning(f"Failed to process {f.name}: {e}")

# Sidebar options
st.sidebar.header("Analysis options")
extra_stop = st.sidebar.text_input("Extra stopwords (comma-separated)", value="")
extra_stopwords = set([s.strip().lower() for s in extra_stop.split(",") if s.strip()])
tfidf_max = st.sidebar.selectbox("TF-IDF max features", [500, 1000, 2000, 4000], index=1)
n_topics = st.sidebar.slider("Topics (NMF)", 1, 20, 6)
top_n_terms = st.sidebar.slider("Top terms per topic", 3, 12, 8)
light_mode = st.sidebar.checkbox("Light mode (reduce memory)", True)
do_cluster = st.sidebar.checkbox("Clustering (KMeans)", value=not light_mode)
do_summarize = st.sidebar.checkbox("Summarize (transformers)", value=False, disabled=not bool(SUMMARIZER))
run = st.sidebar.button("Run analysis")

if not run:
    st.info("Upload files or paste text and press Run.")
    st.stop()

if not docs:
    st.error("No textual content extracted. Try a different file or paste text.")
    st.stop()

# Preprocess docs
with st.spinner("Preprocessing..."):
    docs = [d for d in docs if isinstance(d, str) and d.strip()]
    # if a single very long doc and user wants topics, chunk
    if len(docs) == 1 and len(docs[0]) > 6000:
        chunk_size = 3000 if light_mode else 5000
        docs = [docs[0][i:i+chunk_size] for i in range(0, len(docs[0]), chunk_size)]
        st.info(f"Single long document chunked into {len(docs)} parts for analysis.")
    processed = preprocess_documents(docs, extra_stop=extra_stopwords)

# Compute TF-IDF
tfidf_max_used = 1000 if light_mode else tfidf_max
tfidf_matrix, tfidf_vect = compute_tfidf_safe(processed, max_features=tfidf_max_used)
feature_names = tfidf_vect.get_feature_names_out() if tfidf_vect else []

# Display simple overview
st.header("Overview")
st.metric("Documents (extracted)", len(docs))
st.metric("Processed (after cleaning)", len(processed))
if SID:
    whole = " ".join(docs)
    st.subheader("Corpus Sentiment (VADER)")
    st.json(SID.polarity_scores(whole))

# Top keywords & plot
st.subheader("Top Keywords (TF-IDF)")
if tfidf_matrix is not None:
    col_sum = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    top_ix = col_sum.argsort()[::-1][:30]
    top_kw = [(feature_names[i], float(col_sum[i])) for i in top_ix]
else:
    top_kw = []

plot_horizontal(top_kw, title="Top TF-IDF keywords", max_items=(12 if light_mode else 20))
st.dataframe(pd.DataFrame(top_kw, columns=["Keyword", "Score"]).head(200))

# Topic modeling (NMF safe)
st.subheader("Topics (NMF)")
nmf_model, W, H = fit_nmf_safe(tfidf_matrix, n_topics=n_topics)
if nmf_model is None:
    st.warning("NMF could not be computed (insufficient data or error).")
else:
    topics = []
    for idx, row in enumerate(H):
        top_idx = row.argsort()[::-1][:top_n_terms]
        terms = [feature_names[i] for i in top_idx]
        topics.append((idx+1, terms))
        st.markdown(f"**Topic {idx+1}:** " + ", ".join(terms))

# Optional clustering
if do_cluster:
    st.subheader("Clustering (KMeans on SVD-reduced TF-IDF)")
    try:
        svd_k = 10 if light_mode else min(50, max(2, tfidf_matrix.shape[1]-1))
        svd = TruncatedSVD(n_components=min(svd_k, tfidf_matrix.shape[1]-1), random_state=42)
        docvecs = svd.fit_transform(tfidf_matrix)
        k = min( max(2, int(len(docs)/2)), 10)  # conservative default
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(docvecs)
        st.write("Cluster counts:", dict(Counter(clusters)))
        # show example docs per cluster
        for c in sorted(set(clusters)):
            st.markdown(f"**Cluster {c}**")
            idxs = np.where(clusters == c)[0][:2]
            for i in idxs:
                st.write(f"- Doc {i}: {docs[i][:200]}...")
    except Exception as e:
        st.warning(f"Clustering error or OOM: {e}")

# Word cloud (safe)
st.subheader("Word Cloud")
try:
    from wordcloud import WordCloud
    wc_text = " ".join(processed)
    if wc_text.strip():
        wc = WordCloud(width=720, height=300, background_color="white", max_words=(80 if light_mode else 150)).generate(wc_text)
        fig, ax = plt.subplots(figsize=(9, 3.5), dpi=80)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
        gc.collect()
    else:
        st.write("No text for word cloud.")
except Exception:
    st.info("Install 'wordcloud' to enable this feature.")

# Summarization (optional)
if do_summarize and SUMMARIZER:
    st.subheader("Summaries (transformers)")
    for i, d in enumerate(docs):
        snippet = d[:5000] if len(d) > 5000 else d
        try:
            out = SUMMARIZER(snippet, max_length=150, min_length=40)
            st.markdown(f"**Doc {i} summary:** {out[0]['summary_text']}")
        except Exception as e:
            st.warning(f"Summary failed for doc {i}: {e}")
elif do_summarize:
    st.info("Summarizer unavailable (transformers missing).")

# Export
st.subheader("Export results")
df_export = pd.DataFrame({"original": docs, "processed": processed})
csv = df_export.to_csv(index=False).encode("utf-8")
st.download_button("Download processed CSV", data=csv, file_name="processed_texts.csv", mime="text/csv")

st.success("Done — analysis complete ✅")
