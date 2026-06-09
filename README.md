# 📄 RAG Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and have an intelligent conversation about its content.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-orange)

## 🚀 Features

- **PDF Upload** — Drag and drop any PDF document
- **Intelligent Q&A** — Ask questions in natural language
- **Source Attribution** — See exactly which page and passage the answer came from
- **Conversation History** — Full chat interface with memory within the session
- **Document Metrics** — View page count and chunk statistics

## 🏗️ Architecture

```
PDF Upload → PyPDF Parser → Text Chunking (500 tokens, 50 overlap)
    → HuggingFace Embeddings (all-MiniLM-L6-v2)
    → FAISS Vector Store
    → Similarity Search (top-3 chunks)
    → Groq LLM (Llama3-8b) → Answer + Sources
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (Llama3-8b-8192) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| PDF Parsing | PyPDF |
| Orchestration | LangChain |

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/LAKSHAY-ATREJA/rag-document-qa
cd rag-document-qa

# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

Then open http://localhost:8501, enter your Groq API key and upload a PDF.

## 🔑 Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Paste it into the app sidebar

## 💡 Use Cases

- Query research papers and academic documents
- Extract information from legal contracts
- Summarise lengthy business reports
- Answer questions from technical documentation
- Analyse policy documents
