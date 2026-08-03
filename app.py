import shutil
import streamlit as st

from src.pdf_loader import PDFLoader
from src.text_splitter import TextSplitter
from src.vector_db import VectorDatabase
from src.rag_pipeline import RAGPipeline
from src.config import (
    UPLOAD_DIR,
    VECTORSTORE_DIR
)

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="PDF QA Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Question Answering Chatbot")
st.caption("LangChain • Hugging Face • FAISS • Streamlit")

# ---------------------------------
# Session State
# ---------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# ---------------------------------
# Sidebar
# ---------------------------------

with st.sidebar:

    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process = st.button(
        "⚙ Process PDFs",
        use_container_width=True
    )

    clear_chat = st.button(
        "🗑 Clear Chat",
        use_container_width=True
    )

# ---------------------------------
# Clear Chat
# ---------------------------------

if clear_chat:
    st.session_state.chat_history = []
    st.rerun()

# ---------------------------------
# Process PDFs
# ---------------------------------

if process:

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    # Clear previous uploaded PDFs
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for file in UPLOAD_DIR.glob("*.pdf"):
        try:
            file.unlink()
        except Exception:
            pass

    # Clear previous vector database
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    for file in VECTORSTORE_DIR.glob("*"):
        try:
            file.unlink()
        except Exception:
            pass

    pdf_paths = []

    progress = st.progress(0)

    # Save uploaded PDFs
    for i, uploaded_file in enumerate(uploaded_files):

        save_path = UPLOAD_DIR / uploaded_file.name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pdf_paths.append(save_path)

        progress.progress((i + 1) / len(uploaded_files))

    
    with st.spinner("Reading PDFs..."):
        documents = PDFLoader.load_multiple_pdfs(pdf_paths)

    
    with st.spinner("Splitting text..."):
        chunks = TextSplitter().split_documents(documents)

    
    if len(chunks) == 0:
        st.error("❌ No text could be extracted from the uploaded PDF.")
        st.stop()

    with st.spinner("Creating FAISS index..."):
        db = VectorDatabase()
        db.create_vectorstore(chunks)
        db.save()

    st.session_state.db_ready = True

    st.success("✅ PDFs processed successfully!")

# ---------------------------------
# Chat Area
# ---------------------------------

st.divider()

st.subheader("💬 Ask Questions")

# ---------------------------------
# Chat Interface
# ---------------------------------

if not st.session_state.db_ready:
    st.info("Upload PDFs and click 'Process PDFs'.")
    st.stop()

# User Input
question = st.chat_input("Ask a question about your PDF...")

# Display previous chat
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:

            st.markdown("### 📚 Sources")

            for source in message["sources"]:

                st.markdown(
                    f"- **{source['source']}** (Page {source['page']})"
                )

# Handle Question
if question:

    # Show User Message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Get AI Response
    with st.spinner("Thinking..."):

        rag = RAGPipeline()

        answer, sources = rag.ask(question)

        # Show AI Message
    with st.chat_message("assistant"):

        st.markdown(answer)

        st.markdown("### 📚 Sources")

        unique_sources = []
        seen = set()

        for source in sources:
            key = (source["source"], source["page"])

            if key not in seen:
                seen.add(key)
                unique_sources.append(source)

        for source in unique_sources:
            st.markdown(
                f"📄 **{source['source']}** — Page **{source['page']}**"
            )

    # Save Assistant Message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": unique_sources
    })

    st.rerun()