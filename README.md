# ContextMint — PDF RAG Pipeline

ContextMint is an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that turns a folder of PDF documents into a queryable knowledge base. It ingests PDFs, chunks and embeds them, stores the vectors in ChromaDB, retrieves the most relevant passages for a question, and generates grounded answers with an LLM — complete with similarity filtering, citations, summaries, and conversation history.

## Features

- **PDF ingestion** — recursively loads every PDF in a directory using LangChain's `PyPDFLoader`, preserving source metadata.
- **Smart chunking** — splits documents into overlapping chunks with `RecursiveCharacterTextSplitter` to retain context across boundaries.
- **Local embeddings** — generates dense vectors with `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim), no API key required.
- **Persistent vector store** — stores embeddings, text, and metadata in a persistent **ChromaDB** collection.
- **Semantic retrieval** — embeds queries and returns ranked results with similarity scores and distances.
- **Simple RAG** — straightforward retrieve-then-answer flow using Groq's `llama-3.1-8b-instant`.
- **Advanced RAG** — adds top-k + minimum-score filtering, source attribution, and confidence scoring to reduce hallucinations.
- **Stateful pipeline** — `AdvancedRAGPipeline` adds streaming output, inline citations, answer summaries, and query history.

## Architecture

```
PDFs ──▶ Load (PyPDFLoader) ──▶ Chunk (RecursiveCharacterTextSplitter)
     ──▶ Embed (SentenceTransformer) ──▶ Store (ChromaDB)
     ──▶ Retrieve (semantic search) ──▶ Generate (Groq LLM) ──▶ Answer + Citations
```

## Project structure

```
contextmint-rag/
├── notebooks/
│   └── ContextMint.ipynb     # The full pipeline, step by step
├── data/                     # Put your PDFs here (git-ignored)
│   └── vector_store/         # Persistent ChromaDB store (auto-created, git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/ritamjulie1984-cloud/contextmint-rag.git
cd contextmint-rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure your API key

The answer-generation step uses [Groq](https://console.groq.com/). Copy the example env file and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Add your PDFs

Drop one or more PDF files into the `data/` folder. The notebook scans `../data` recursively from the `notebooks/` directory.

### 4. Run the notebook

```bash
jupyter lab    # or: jupyter notebook
```

Open `notebooks/ContextMint.ipynb` and run the cells top to bottom. The pipeline will:

1. Load and chunk your PDFs
2. Generate embeddings and persist them to ChromaDB
3. Let you query the knowledge base with simple and advanced RAG

## Example usage

```python
# Simple RAG
answer = rag_simple("What are the advantages of using FAISS for similarity search?", rag_retriever, llm)
print(answer)

# Advanced RAG with sources and confidence
result = rag_advanced("Text Chunking Algorithm", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
print(result["answer"], result["sources"], result["confidence"])

# Stateful pipeline with streaming, citations, summary, and history
adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
result = adv_rag.query("representational capacity of vector embeddings", stream=True, summarize=True)
print(result["answer"])
```

## Tech stack

| Component        | Library                                   |
|------------------|-------------------------------------------|
| Document loading | `langchain-community` (PyPDFLoader)       |
| Chunking         | `langchain-text-splitters`                |
| Embeddings       | `sentence-transformers` (all-MiniLM-L6-v2)|
| Vector store     | `chromadb`                                |
| LLM              | `langchain-groq` (llama-3.1-8b-instant)   |
| Config           | `python-dotenv`                           |

## Notes

- Embeddings run locally on CPU; the first run downloads the `all-MiniLM-L6-v2` model (~90 MB).
- The ChromaDB store persists to `data/vector_store/` so you don't have to re-embed on every run.
- Both `data/` contents and the vector store are git-ignored to keep the repo lightweight and private.

## License

Released under the MIT License. See [LICENSE](LICENSE).
