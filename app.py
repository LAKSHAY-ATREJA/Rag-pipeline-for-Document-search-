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
import logging

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────
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
    """Load and cache the HuggingFace embedding model."""
    try:
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        raise RuntimeError(
            f"Could not load the embedding model. "
            f"Ensure sentence-transformers is installed. Details: {e}"
        ) from e


def process_pdf(uploaded_file) -> tuple:
    """
    Parse an uploaded PDF, split it into chunks, embed them and
    return a FAISS vectorstore together with document metadata.

    Args:
        uploaded_file: A Streamlit UploadedFile object for a PDF.

    Returns:
        A (vectorstore, metadata) tuple where metadata contains page
        count, chunk count and the original filename.

    Raises:
        ValueError: If the PDF contains no extractable text.
        RuntimeError: If embedding or indexing fails.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise ValueError(
                "No text could be extracted from the PDF. "
                "The file may be scanned or image-only."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Text was extracted but produced no usable chunks.")

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        metadata = {
            "pages": len(documents),
            "chunks": len(chunks),
            "filename": uploaded_file.name,
        }
        logger.info(
            "Processed '%s': %d pages, %d chunks",
            uploaded_file.name, len(documents), len(chunks),
        )
        return vectorstore, metadata

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.exception("Unexpected error while processing PDF")
        raise RuntimeError(f"Failed to process the PDF: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def build_qa_chain(vectorstore, api_key: str) -> RetrievalQA:
    """
    Construct a LangChain RetrievalQA chain backed by Groq.

    Args:
        vectorstore: A FAISS vectorstore to retrieve context from.
        api_key: A valid Groq API key.

    Returns:
        A configured RetrievalQA chain.
    """
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        groq_api_key=api_key,
        max_retries=2,
    )

    prompt_template = (
        "Use the following context extracted from a document to answer the question "
        "accurately and concisely.\n"
        "If the answer is not contained in the context, respond with: "
        "\"I couldn't find this information in the document.\"\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask_question(vectorstore, question: str, api_key: str) -> tuple:
    """
    Query the vectorstore and return an answer with its source chunks.

    Args:
        vectorstore: FAISS vectorstore built from the uploaded document.
        question: The natural-language question to answer.
        api_key: Groq API key used for LLM inference.

    Returns:
        A (answer, sources) tuple where answer is a string and sources
        is a list of LangChain Document objects.

    Raises:
        ValueError: If the question string is empty.
        RuntimeError: If the LLM or retrieval call fails.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    try:
        qa_chain = build_qa_chain(vectorstore, api_key)
        result = qa_chain.invoke({"query": question})
        return result["result"], result["source_documents"]
    except ValueError:
        raise
    except Exception as e:
        logger.exception("Error while answering question")
        raise RuntimeError(
            f"Failed to get an answer. Check your API key and network connection. "
            f"Details: {e}"
        ) from e


# ── UI ────────────────────────────────────────────────────────
st.markdown('<p class="main-header">RAG Document Q&A</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a PDF and ask questions about its content using AI</p>',
    unsafe_allow_html=True,
)

# Sidebar — configuration and upload
with st.sidebar:
    st.header("Configuration")

    # Allow API key from environment variable as a default
    env_key = os.environ.get("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com",
    )

    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("Process Document", use_container_width=True):
            with st.spinner("Processing document — this may take a moment..."):
                try:
                    vectorstore, metadata = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.doc_metadata = metadata
                    st.session_state.chat_history = []
                    st.success("Document ready!")
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
    elif uploaded_file and not api_key:
        st.warning("Enter your Groq API key before processing.")

    st.divider()

    if st.session_state.doc_metadata:
        st.header("Document Info")
        meta = st.session_state.doc_metadata
        col1, col2 = st.columns(2)
        col1.metric("Pages", meta.get("pages", 0))
        col2.metric("Chunks", meta.get("chunks", 0))
        st.caption(meta.get("filename", ""))

    if st.session_state.chat_history:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ── Main area ─────────────────────────────────────────────────
if not api_key:
    st.info("Enter your Groq API key in the sidebar to get started.")
elif not st.session_state.vectorstore:
    st.info("Upload a PDF and click 'Process Document' to begin.")
else:
    # Render existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        page_num = source.metadata.get("page")
                        page_label = (page_num + 1) if page_num is not None else "?"
                        st.markdown(
                            f'<div class="source-box">'
                            f"Source {i} — Page {page_label}<br>"
                            f"{source.page_content[:300]}..."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # New question input
    question = st.chat_input("Ask a question about your document...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    answer, sources = ask_question(
                        st.session_state.vectorstore, question, api_key
                    )
                except (ValueError, RuntimeError) as e:
                    answer = f"Error: {e}"
                    sources = []
                except Exception as e:
                    answer = f"An unexpected error occurred: {e}"
                    sources = []

            st.write(answer)

            if sources:
                with st.expander("View Sources"):
                    for i, source in enumerate(sources, 1):
                        page_num = source.metadata.get("page")
                        page_label = (page_num + 1) if page_num is not None else "?"
                        st.markdown(
                            f'<div class="source-box">'
                            f"Source {i} — Page {page_label}<br>"
                            f"{source.page_content[:300]}..."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
