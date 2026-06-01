# contextmint-rag
Jupyter notebook for ContextMint
Data Ingestion ──┬──► Chunk 1 ─┐
(PDF/HTML/Excel/DB)   Chunk 2  ├──► Embedding Manager ──► VectorStore
                 ├──► Chunk 3  │     (Text → Vectors)      (chunks +
                 ├──► Chunk 4  │                            embeddings)
                 └──► Chunk 5 ─┘
