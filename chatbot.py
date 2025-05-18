# app.py
import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from ingest_database import parse_pdf_from_path, text_to_docs, build_faiss_index, load_faiss_index

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

INDEX_PATH = "faiss_index/faiss.index"
DATA_PATH = "data"

st.set_page_config(page_title="Sherubtse RAG Chatbot")
st.title("Sherubtse College Chatbot 🤖")

st.markdown("""
    <style>
    body { background-color: #000; color: white; }
    .chat-bubble {
        background-color: #111;
        border: 1px solid #1f51ff;
        padding: 1rem;
        border-radius: 1rem;
        margin: 1rem 0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("### Example questions to try:")
for q in [
    "What programs are offered at Sherubtse College?",
    "Who is the Dean of Academic Affairs?",
    "When does the academic session begin?"
]:
    st.markdown(f"<div class='chat-bubble'>{q}</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📁 Upload PDF(s)", type="pdf", accept_multiple_files=True)

def process_and_index(files):
    os.makedirs(DATA_PATH, exist_ok=True)
    all_docs = []
    for file in files:
        file_path = os.path.join(DATA_PATH, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        text, filename = parse_pdf_from_path(file_path)
        docs = text_to_docs(text, filename)
        all_docs.extend(docs)
    return build_faiss_index(all_docs, openai_api_key, index_path=INDEX_PATH)

@st.cache_resource
def load_or_create_vectordb(files):
    if files:
        with st.spinner("🔍 Indexing uploaded PDFs..."):
            return process_and_index(files)
    else:
        try:
            return load_faiss_index(openai_api_key, INDEX_PATH)
        except FileNotFoundError:
            return None

vectordb = load_or_create_vectordb(uploaded_files)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

fallback_responses = [
    "Kuzu Zangpola! Please refer to Sherubtse’s website for more details!",
    "I couldn’t find that info here. Please check with the college!",
    "Not in the PDFs. Maybe contact your academic coordinator.",
    "I looked but didn’t find that. Please try a different question?"
]

user_input = st.chat_input("💬 Ask anything about Sherubtse College...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    if not vectordb:
        with st.chat_message("assistant"):
            st.markdown("⚠️ Please upload PDF(s) to start chatting.")
        st.stop()

    # ✨ SMARTER SEARCH: use similarity_search_with_score
    results_with_scores = vectordb.similarity_search_with_score(user_input, k=5)

    # Filter high-quality matches only
    relevant_docs = [doc for doc, score in results_with_scores if score < 0.3]   # score closer to 0 = more similar

    if relevant_docs:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = "\n\n".join([doc.page_content for doc, _ in results_with_scores])

    prompt = (
        f"Use the following Sherubtse College information to answer the user's question.\n\n"
        f"{context}\n\n"
        f"Question: {user_input}\n\n"
        f"If no relevant info, say 'Kuzu Zangpola!' and suggest checking the official website."
    )

    response = []
    with st.chat_message("assistant"):
        bot_placeholder = st.empty()
        typing_indicator = st.empty()
        typing_indicator.markdown("<div class='blinking'>...</div>", unsafe_allow_html=True)

        for chunk in client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                response.append(delta)
                bot_placeholder.markdown(
                    f"<div class='chat-bubble'>{''.join(response)}</div>",
                    unsafe_allow_html=True
                )

        typing_indicator.empty()

    final_response = "".join(response)
    st.session_state.chat_history.append({"role": "assistant", "content": final_response})
