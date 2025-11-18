import streamlit as st

# Title
st.title("🧠 Dynamic Text Analysis Platform")

# Subtitle
st.subheader("Upload and analyze your text data")

# Upload or paste text
option = st.radio("Choose input method:", ["Upload File", "Paste Text"])

text_data = ""

if option == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])
    if uploaded_file is not None:
        text_data = uploaded_file.read().decode("utf-8")
        st.text_area("File Content:", text_data, height=200)
else:
    text_data = st.text_area("Paste your text below:", "", height=200)

# Analyze Button
if st.button("🔍 Analyze Text"):
    if text_data.strip() == "":
        st.warning("Please provide some text to analyze.")
    else:
        # Simple analysis: word count, character count
        word_count = len(text_data.split())
        char_count = len(text_data)
        st.success("✅ Analysis Complete!")
        st.write(f"**Word Count:** {word_count}")
        st.write(f"**Character Count:** {char_count}")
