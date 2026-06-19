# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that lets you upload any PDF document and ask questions about it in natural language. The system retrieves the most relevant passages from your document and passes them as context to a large language model, producing grounded answers with source citations.

---

## How the pipeline works

### 1. Document ingestion

When you upload a PDF, the application writes it to a temporary file and uses PyPDF to extract text page by page. Each page becomes a LangChain Document object carrying the page number and filename as metadata.

### 2. Text chunking

The extracted text is split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter. The default configuration produces chunks of up to 500 characters with a 50-character overlap between adjacent chunks. Overlap ensures that sentences spanning a chunk boundary are not lost from either side.

### 3. Embedding

Each chunk is encoded into a dense 384-dimensional vector by the all-MiniLM-L6-v2 sentence transformer model from HuggingFace. This model runs entirely on the local CPU and does not require an API key. Embeddings are normalised to unit length so that cosine similarity reduces to a dot product, which speeds up retrieval.

### 4. Vector indexing

The embeddings and their corresponding text chunks are stored in a FAISS flat index held in memory. FAISS (Facebook AI Similarity Search) provides exact nearest-neighbour search over the index using highly optimised BLAS routines.

### 5. Retrieval

When you submit a question, the same embedding model encodes the question into a vector. FAISS returns the three index entries whose vectors are closest to the question vector. Closeness in embedding space corresponds to semantic similarity between the question and the stored chunk.

### 6. Answer generation

The three retrieved chunks are assembled into a context block and inserted into a structured prompt. The prompt instructs the model to answer only from the provided context and to acknowledge when the information is not present. This prompt is sent to Groq's hosted Llama3-8b-8192 model, which returns a concise, cited answer. The application then displays both the answer and expandable source excerpts.

---

## Features

- Upload any PDF and query it in seconds
- Source attribution shows the exact page and passage for each answer
- Conversation history is maintained within the session
- API key can be entered in the sidebar or pre-loaded from a GROQ_API_KEY environment variable
- Full error handling at every stage: missing text, empty chunks, API failures
- An offline demo script (demo.py) runs the full pipeline against an inline corpus with no PDF upload required
- Deployable to Render or any platform that supports Python web processes

---

## Installation

The application requires Python 3.9 or later. A virtual environment is recommended.

```bash
git clone https://github.com/LAKSHAYATREJA/Rag-pipeline-for-Document-search-
cd Rag-pipeline-for-Document-search-

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The sentence-transformers library will download the all-MiniLM-L6-v2 model on first run (approximately 90 MB). Subsequent runs load it from the local HuggingFace cache.

---

## Running locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

1. Paste your Groq API key into the sidebar (see below for how to obtain one).
2. Upload a PDF using the file uploader.
3. Click "Process Document" and wait for the index to be built.
4. Type a question in the chat input at the bottom of the page.

---

## Running the demo

demo.py exercises the full pipeline without a browser or PDF file. It builds a FAISS index from three short inline documents (covering neural networks, RAG, and vector databases) and runs five example queries against it.

```bash
python3 demo.py
```

If GROQ_API_KEY is set in your environment, the script also calls the Groq LLM and prints generated answers. If the key is not set, it prints the retrieved context chunks, which demonstrates the retrieval half of the pipeline independently.

Example output (retrieval only):

```
------------------------------------------------------------------------
RAG Pipeline Demo
------------------------------------------------------------------------
Corpus: 3 inline documents covering neural networks, RAG, and vector databases.

Loading embedding model (all-MiniLM-L6-v2)...
Split corpus into 14 chunks.
Building FAISS index...
Vector index ready.

------------------------------------------------------------------------
Query 1: What is backpropagation and why is it used?

Retrieved context chunks:
  [1] Introduction to Neural Networks: Training a neural network means adjusting
      the weights on connections between neurons so that the network's output
      matches desired targets on a training set. This is done by computing a
      loss function ...
  [2] Introduction to Neural Networks: A neural network is a computational model
      loosely inspired by the structure of the human brain ...
  [3] Retrieval-Augmented Generation: Retrieval-Augmented Generation (RAG) is a
      framework that combines a retrieval system ...
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| GROQ_API_KEY | Yes (for LLM) | API key for Groq inference. See below. |
| HUGGINGFACE_TOKEN | No | Needed only for gated HuggingFace models. |

Copy .env.example to .env and fill in your values:

```bash
cp .env.example .env
```

The Streamlit app reads GROQ_API_KEY from the environment automatically and pre-fills the sidebar field. You can still override it at runtime by typing a different key into the sidebar.

### Obtaining a Groq API key

1. Go to https://console.groq.com and create a free account.
2. Navigate to API Keys and click Create API Key.
3. Copy the key (it starts with gsk_) and add it to your .env file or paste it into the app sidebar.

Groq's free tier provides generous rate limits suitable for development and personal use.

---

## Project structure

```
Rag-pipeline-for-Document-search-/
├── app.py            Main Streamlit application
├── demo.py           Offline demonstration of the full RAG pipeline
├── requirements.txt  Python dependencies with minimum version pins
├── .env.example      Template for required environment variables
├── Procfile          Process declaration for Heroku-compatible platforms
├── render.yaml       Deployment manifest for Render free tier
└── README.md         This file
```

---

## Deployment

### Render (free tier)

The repository includes a render.yaml manifest. To deploy:

1. Push the repository to GitHub.
2. Log in to https://render.com and click New > Web Service.
3. Connect your GitHub repository.
4. Render will detect render.yaml and configure the service automatically.
5. Add a GROQ_API_KEY environment variable in the Render dashboard under Environment.
6. Click Deploy.

The application will be available at a render.app subdomain within a few minutes.

### Heroku

```bash
heroku create your-app-name
heroku config:set GROQ_API_KEY=gsk_...
git push heroku main
```

The Procfile declares the web process. Heroku will install dependencies from requirements.txt and start Streamlit on the assigned PORT.

### Any Linux server

```bash
export GROQ_API_KEY=gsk_...
pip install -r requirements.txt
streamlit run app.py --server.port 8080 --server.address 0.0.0.0 --server.headless true
```

---

## Technology stack

| Component | Technology |
|---|---|
| Web interface | Streamlit |
| LLM | Groq Llama3-8b-8192 |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector search | FAISS (CPU) |
| PDF parsing | PyPDF |
| Orchestration | LangChain |
