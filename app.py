import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import time

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1a5276;
    }
    .source-box {
        background: #eaf4fb;
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #444;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = {}


# ── Helper functions ──────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_pdf(uploaded_file) -> tuple:
    """Process uploaded PDF and return vectorstore + metadata."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.unlink(tmp_path)

    metadata = {
        "pages": len(documents),
        "chunks": len(chunks),
        "filename": uploaded_file.name
    }

    return vectorstore, metadata


def ask_question(vectorstore, question: str, api_key: str) -> tuple:
    """Query vectorstore and return answer + sources."""
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        groq_api_key=api_key
    )

    prompt_template = """Use the following context to answer the question accurately and concisely.
If you cannot find the answer in the context, say "I couldn't find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]

    return answer, sources


# ── UI ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📄 RAG Document Q&A</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a PDF and ask questions about its content using AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.divider()
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("Processing document..."):
                vectorstore, metadata = process_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_metadata = metadata
                st.session_state.chat_history = []
            st.success("✅ Document ready!")

    st.divider()

    if st.session_state.doc_metadata:
        st.header("📊 Document Info")
        meta = st.session_state.doc_metadata
        st.metric("Pages", meta.get("pages", 0))
        st.metric("Chunks", meta.get("chunks", 0))
        st.caption(f"📎 {meta.get('filename', '')}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Main area
if not api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
elif not st.session_state.vectorstore:
    st.info("👈 Upload a PDF and click 'Process Document' to begin.")
else:
    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f'<div class="source-box">📄 Source {i} — Page {source.metadata.get("page", "?")+1}<br>{source.page_content[:300]}...</div>', unsafe_allow_html=True)

    # Input
    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer, sources = ask_question(st.session_state.vectorstore, question, api_key)
            st.write(answer)
            with st.expander("📚 View Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f'<div class="source-box">📄 Source {i} — Page {source.metadata.get("page", "?")+1}<br>{source.page_content[:300]}...</div>', unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
