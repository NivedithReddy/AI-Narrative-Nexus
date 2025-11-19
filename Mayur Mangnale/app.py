import streamlit as st
import pandas as pd
from datetime import datetime
from docx import Document
import PyPDF2
import io
import re
from collections import Counter

# Page Config
st.set_page_config(
    page_title="AnalyzerNexus",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal Dark CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .stTextArea textarea {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #404040;
    }
    
    .stButton>button {
        background-color: #404040;
        color: #ffffff;
        border: 1px solid #606060;
    }
    
    .stButton>button:hover {
        background-color: #505050;
    }
    
    h1, h2, h3 {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'history' not in st.session_state:
    st.session_state.history = []

# ==================== TEXT PROCESSING FUNCTIONS ====================

def clean_text(text):
    """Remove special characters and extra spaces"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    """Remove common stopwords"""
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very'
    ])
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(filtered_words)

def tokenize(text):
    """Split text into tokens"""
    return text.split()

def analyze_sentiment(text):
    """Simple sentiment analysis based on word counts"""
    positive_words = set([
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'positive', 'love', 'happy', 'joy', 'success', 'best', 'perfect',
        'beautiful', 'brilliant', 'awesome', 'outstanding', 'satisfied'
    ])
    negative_words = set([
        'bad', 'terrible', 'horrible', 'awful', 'poor', 'negative',
        'hate', 'sad', 'failure', 'worst', 'ugly', 'disappointing',
        'problem', 'issue', 'difficult', 'hard', 'pain'
    ])
    
    words = text.lower().split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    total = positive_count + negative_count
    
    if total == 0:
        return 'Neutral', 0.5
    
    sentiment_score = positive_count / total
    
    if sentiment_score > 0.6:
        return 'Positive', sentiment_score
    elif sentiment_score < 0.4:
        return 'Negative', sentiment_score
    else:
        return 'Neutral', sentiment_score

def extract_topics(text, num_topics=3):
    """Extract most frequent words as topics"""
    words = text.split()
    word_freq = Counter(words)
    
    topics = []
    for word, freq in word_freq.most_common(num_topics * 3):
        if len(word) > 4:
            topics.append({'word': word, 'frequency': freq})
        if len(topics) == num_topics:
            break
    
    return topics

def extractive_summarization(text, num_sentences=2):
    """Extract key sentences from text"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= num_sentences:
        return sentences
    
    words = tokenize(clean_text(text))
    word_freq = Counter(words)
    
    sentence_scores = []
    for sentence in sentences:
        score = sum(word_freq.get(word, 0) for word in tokenize(clean_text(sentence)))
        sentence_scores.append((sentence, score))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in sentence_scores[:num_sentences]]

def generate_insights(sentiment, topics, word_count):
    """Generate insights and recommendations"""
    insights = []
    recommendations = []
    
    if sentiment == 'Positive':
        insights.append("Overall positive sentiment detected in the text.")
        recommendations.append("Leverage this positive tone in communications.")
    elif sentiment == 'Negative':
        insights.append("Negative sentiment detected in the text.")
        recommendations.append("Address negative aspects and investigate root causes.")
    else:
        insights.append("Neutral sentiment detected in the text.")
        recommendations.append("Monitor for changes in future data.")
    
    if topics:
        top_topic = topics[0]['word']
        insights.append(f"Main focus area: '{top_topic}' appears most frequently.")
        recommendations.append(f"Explore '{top_topic}' for more details.")
    
    if word_count < 50:
        insights.append("Brief text with limited content.")
    elif word_count > 500:
        insights.append("Comprehensive text with rich details.")
    
    return insights, recommendations

def parse_file(uploaded_file):
    """Parse uploaded files"""
    text = ""
    try:
        if uploaded_file.type == 'text/plain':
            text = uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            text = ' '.join(df.astype(str).values.flatten())
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            text = ' '.join([para.text for para in doc.paragraphs])
        elif uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
    return text

# ==================== MAIN APP ====================

def main():
    st.title("AnalyzerNexus")
    st.write("Advanced Text Analysis Platform")
    st.divider()
    
    # Input Section
    st.header("Upload or Enter Text")
    
    uploaded_file = st.file_uploader("Choose file (.txt, .csv, .docx, .pdf)", type=['txt', 'csv', 'docx', 'pdf'])
    
    text_input = ""
    file_name = "Direct Input"
    
    if uploaded_file:
        text_input = parse_file(uploaded_file)
        file_name = uploaded_file.name
        st.success(f"File loaded: {file_name}")
    else:
        text_input = st.text_area("Or paste text here", height=150)
    
    if st.button("Analyze"):
        if not text_input or len(text_input.strip()) < 10:
            st.error("Please provide valid text")
        else:
            with st.spinner("Analyzing..."):
                # Processing
                cleaned_text = clean_text(text_input)
                processed_text = remove_stopwords(cleaned_text)
                tokens = tokenize(processed_text)
                
                # Analysis
                sentiment, sentiment_score = analyze_sentiment(processed_text)
                topics = extract_topics(processed_text, 3)
                summary = extractive_summarization(text_input, 2)
                insights, recommendations = generate_insights(sentiment, topics, len(tokens))
                
                # Store
                st.session_state.analysis_data = {
                    'file_name': file_name,
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score,
                    'topics': topics,
                    'summary': summary,
                    'insights': insights,
                    'recommendations': recommendations,
                    'total_words': len(tokens),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.session_state.history.append({
                    'timestamp': st.session_state.analysis_data['timestamp'],
                    'file': file_name,
                    'words': len(tokens),
                    'sentiment': sentiment
                })
            
            st.success("Analysis Complete!")
    
    st.divider()
    
    # Results
    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        
        st.header("Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Words", data['total_words'])
        col2.metric("Sentiment", data['sentiment'])
        col3.metric("Topics", len(data['topics']))
        
        st.divider()
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Topics", "Summary", "Insights"])
        
        with tab1:
            st.write(f"**Sentiment:** {data['sentiment']}")
            st.write(f"**Score:** {data['sentiment_score']:.2%}")
            st.progress(data['sentiment_score'])
        
        with tab2:
            st.write("**Key Topics:**")
            for topic in data['topics']:
                st.write(f"- {topic['word']}: {topic['frequency']} mentions")
        
        with tab3:
            st.write("**Summary:**")
            for i, sent in enumerate(data['summary'], 1):
                st.write(f"{i}. {sent}")
        
        with tab4:
            st.write("**Insights:**")
            for insight in data['insights']:
                st.write(f"• {insight}")
            st.write("\n**Recommendations:**")
            for rec in data['recommendations']:
                st.write(f"• {rec}")
        
        st.divider()
        
        # Export
        report = f"""
ANALYZERNEXUS - ANALYSIS REPORT
Generated: {data['timestamp']}
Source: {data['file_name']}

SUMMARY
Total Words: {data['total_words']}
Sentiment: {data['sentiment']} (Score: {data['sentiment_score']:.2%})

TOPICS
{chr(10).join([f"- {t['word']}: {t['frequency']} mentions" for t in data['topics']])}

SUMMARY
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(data['summary'])])}

INSIGHTS
{chr(10).join([f"- {i}" for i in data['insights']])}

RECOMMENDATIONS
{chr(10).join([f"- {r}" for r in data['recommendations']])}
"""
        st.download_button("Download Report", data=report, file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")
    
    # History
    st.divider()
    st.header("History")
    
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No analysis history yet.")

if __name__ == "__main__":
    main()
