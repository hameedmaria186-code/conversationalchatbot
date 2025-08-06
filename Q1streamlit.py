import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os
from dotenv import load_dotenv


# ---- SETUP GEMINI ----
# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# ---- SESSION STATE INIT ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# ---- UTILS ----
def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def chunk_text(text, max_len=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current, count = [], "", 0
    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks

def search_relevant_chunks(query, chunks, top_k=3):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for score, chunk in scored if score > 0][:top_k]

def format_chat_history(history):
    formatted = ""
    for speaker, msg in history:
        if speaker == "user":
            formatted += f"User: {msg}\n"
        else:
            formatted += f"Bot: {msg}\n"
    return formatted.strip()

# ---- APP UI ----
st.set_page_config(page_title="📚 Gemini Chatbot", layout="centered")
st.title("📚 Gemini Chatbot with File Support")

# Tone selector
tone = st.selectbox("Choose the bot's tone:", ["Friendly", "Formal", "Technical"])

# File upload
uploaded_file = st.file_uploader("Upload a .pdf or .txt file (optional)", type=["pdf", "txt"])
if uploaded_file and not st.session_state.file_uploaded:
    with st.spinner("Processing file..."):
        raw_text = extract_text_from_file(uploaded_file)
        chunks = chunk_text(raw_text)
        st.session_state.file_chunks = chunks
        st.session_state.file_uploaded = True
    st.success("✅ File uploaded and processed!")

# Chat input
user_input = st.chat_input("Ask something...")

# Display chat like ChatGPT
for speaker, msg in st.session_state.chat_history:
    with st.chat_message("user" if speaker == "user" else "assistant"):
        st.markdown(msg)

# Chat handling
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Build context
            tone_instruction = {
                "Friendly": "Answer in a warm, conversational, and helpful tone.",
                "Formal": "Answer in a professional, concise, and respectful manner.",
                "Technical": "Answer using precise technical language assuming user has some background knowledge.",
            }[tone]

            doc_context = ""
            if st.session_state.file_chunks:
                relevant_chunks = search_relevant_chunks(user_input, st.session_state.file_chunks)
                doc_context = "\n\n".join(relevant_chunks)

            prompt = f"""{tone_instruction}

Refer to this document content if relevant:
{doc_context}

Conversation so far:
{format_chat_history(st.session_state.chat_history)}

User: {user_input}
Bot:"""

            try:
                response = model.generate_content(prompt)
                reply = response.text.strip()
            except Exception as e:
                reply = "❌ Error getting response from Gemini."

            st.markdown(reply)
            st.session_state.chat_history.append(("bot", reply))


# Summarize button
if st.button("📄 Summarize This Chat"):
    with st.spinner("Summarizing conversation..."):
        summary_prompt = f"""Summarize the following conversation between a user and a bot in bullet points:

{format_chat_history(st.session_state.chat_history)}
"""
        try:
            summary_response = model.generate_content(summary_prompt)
            summary = summary_response.text.strip()
        except Exception as e:
            summary = "❌ Error summarizing conversation."

    st.markdown("### 📝 Summary")
    st.markdown(summary)

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.file_chunks = []
    st.session_state.file_uploaded = False
    st.experimental_rerun()

