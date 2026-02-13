# AIRMAN — Aviation Document AI Chat (RAG)

An Aviation Document AI Chat that answers questions **strictly and only** from provided aviation documents (PPL/CPL/ATPL textbooks, SOPs, manuals). Built with Retrieval-Augmented Generation (RAG) to prevent hallucinations and enforce grounded, traceable answers.

## Features

- **Document ingestion** — Load PDFs, chunk text, embed, and store in FAISS
- **Grounded chat** — Answers with citations; refuses when information is not in the documents
- **Hybrid retrieval** — Vector (FAISS) + BM25 for improved recall
- **Evaluation suite** — 50-question set with retrieval hit-rate, faithfulness, hallucination metrics

## Tech Stack

- **Language:** Python 3.10+
- **API:** FastAPI
- **Vector store:** FAISS
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **LLM:** Anthropic Claude (Haiku)

## Setup

1. **Clone and create virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate   # Linux/macOS
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   Copy `.env.example` to `.env` and set your Anthropic API key:

   ```bash
   copy .env.example .env   # Windows
   # cp .env.example .env   # Linux/macOS
   ```

4. **Place aviation PDFs in `data/`**

   Example: `data/airman_data.pdf`

5. **Ingest documents**

   ```bash
   # Option 1: API
   uvicorn app.main:app --reload
   # Then: POST http://localhost:8000/ingest

   # Option 2: Script
   python -m app.ingest
   ```

6. **Run the API**

   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

| Method | Endpoint   | Description                                      |
|--------|------------|--------------------------------------------------|
| GET    | `/health`  | Health check                                     |
| POST   | `/ingest`  | Ingest PDFs from `data/` into the vector store   |
| POST   | `/ask`     | Ask a question; returns answer, citations, chunks |

### POST /ask

**Request:**
```json
{
  "question": "What is indicated airspeed?",
  "debug": true
}
```

**Response:**
```json
{
  "answer": "...",
  "citations": ["airman_data.pdf page 42"],
  "chunks": []
}
```

- `debug: true` — Includes top retrieved chunks in the response.
- If the answer cannot be supported by the documents, the system responds with:
  > "This information is not available in the provided document(s)."

## Chunking Strategy

| Setting   | Value | Rationale                                                                 |
|-----------|-------|---------------------------------------------------------------------------|
| Chunk size| 500   | Balances context length with retrieval precision; fits typical definitions and procedures. |
| Overlap   | 50    | Prevents splitting concepts across boundaries; maintains continuity.      |

Chunking is performed **per page** — each PDF page is extracted, then split into overlapping segments. This preserves page-level locality for accurate citations.

## Evaluation

1. Start the API: `uvicorn app.main:app --reload`
2. Run evaluation:

   ```bash
   cd evaluation
   python evaluate.py
   ```

3. View the report: `report.md`

The evaluation uses 50 questions:
- 20 factual (definitions, lookups)
- 20 applied (scenario-based, procedures)
- 10 reasoning (multi-step, trade-offs)

Metrics:
- **Retrieval hit-rate** — Did retrieved chunks contain the answer?
- **Faithfulness** — Is the answer grounded in retrieved text?
- **Hallucination rate** — Unsupported claims
- **Qualitative** — 5 best and 5 worst answers with explanations

## Project Structure

```
airman-rag/
├── app/
│   ├── chunker.py     # Text chunking
│   ├── embeddings.py  # sentence-transformers
│   ├── ingest.py      # PDF → FAISS pipeline
│   ├── llm.py         # Claude generation
│   ├── main.py        # FastAPI app
│   ├── rag_pipeline.py
│   └── retriever.py   # FAISS + BM25
├── data/              # Place PDFs here
├── evaluation/
│   ├── questions.json
│   └── evaluate.py
├── vector_store/      # FAISS index + metadata
├── requirements.txt
├── .env.example
└── README.md
```

## License

For AIRMAN technical assignment use.
