
# Dr. E | RiteBridge | Champions – AI Agent (RAG)

A FastAPI service that answers questions about
**Dr. Ekaette Joseph-Isang**, **RiteBridge Consulting**, and **Champions Coaching Academy**
using Retrieval-Augmented Generation (RAG) over a JSON knowledge base.

## Features
- RAG with FAISS over JSON Q&A
- Formal + friendly tone, grounded answers
- Safety prompts (no private info; medical disclaimer)
- Minimal web chat UI (static HTML/JS)
- Rebuildable index via `/reload`

## Quickstart
1. Install & configure
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # add your OPENAI_API_KEY
   ```
2. Run the server
   ```bash
   uvicorn app:app --reload --port 8000
   ```
3. Open http://localhost:8000

## API
- `POST /ask`
  ```json
  { "question": "What does Dr. E specialize in?", "top_k": 5, "temperature": 0.2 }
  ```
  Returns `{ "answer": "...", "sources": ["..."] }`

- `POST /reload`
  Rebuilds the vector index from `data/knowledge.json`.

## Add/Update Knowledge
- Edit `data/knowledge.json` to add more Q&A.
- Then call `POST /reload` or restart.

## Models
- Chat: `gpt-4o-mini` (override with `OPENAI_CHAT_MODEL`)
- Embeddings: `text-embedding-3-large` (override with `OPENAI_EMBED_MODEL`)

## Production Notes
- Add logging, analytics, HTTPS, and auth for public deployments.
- Consider a managed vector DB (Pinecone/Weaviate/pgvector) when scaling.
