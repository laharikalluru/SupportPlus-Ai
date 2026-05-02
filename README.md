# SupportPlus AI — Smart Knowledge Enhancer

> **Production-ready AI support system** combining a FAQ knowledge base (RAG), real-time web search (Firecrawl), user memory, and **Gemini-powered** responses.

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│               FastAPI  (/ask)               │
└────┬────────────┬───────────────┬───────────┘
     │            │               │
     ▼            ▼               ▼
 RAG Service  Web Service    Memory Service
(ChromaDB +  (Firecrawl or  (SQLite / Redis)
 SentTrans.)    mock)
     │            │               │
     └────────────┴───────────────┘
                  │
                  ▼
            LLM Service
          (Google Gemini 1.5 Flash)
                  │
                  ▼
          Structured Response
```

---

## 📁 Project Structure

```
supportplus-ai/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app + all endpoints
│   ├── config.py        ← Centralised settings via .env
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_service.py     ← FAQ chunking, embeddings, Chroma retrieval
│   │   ├── memory_service.py  ← Interaction history (SQLite / Redis)
│   │   ├── web_service.py     ← Firecrawl web search + mock fallback
│   │   └── llm_service.py     ← Gemini prompt builder + response generator
│   ├── data/
│   │   └── faqs.txt           ← Sample FAQ knowledge base
│   └── db/
│       ├── vector_store/      ← Chroma persistent store (auto-created)
│       └── memory.db          ← SQLite memory DB (auto-created)
├── requirements.txt
├── .env                       ← Your API keys (DO NOT commit)
├── .env.example               ← Template for .env
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & enter the project

```bash
cd supportplus-ai
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` sentence transformer model (~90 MB).

### 4. Configure environment variables

```bash
# Copy the template
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

Edit `.env` and set at minimum:

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Your Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey)) |
| `FIRECRAWL_API_KEY` | Optional | Firecrawl key for live web search (mock used if blank) |

### 5. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000**

Interactive API docs: **http://localhost:8000/docs**

---

## 🧪 API Usage

### POST `/ask` — Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I reset my password?",
    "user_id": "user_123"
  }'
```

**Response:**
```json
{
  "user_id": "user_123",
  "query": "How do I reset my password?",
  "response": "To reset your password, visit the login page and click 'Forgot Password'...",
  "sources": {
    "faq": [
      {"question": "How do I reset my password?", "section": "Account & Billing", "score": 0.94}
    ],
    "web": [
      {"title": "SupportPlus AI Docs", "url": "https://docs.supportplus.ai", "score": 0.85}
    ],
    "confidence": {
      "faq_top_score": 0.94,
      "web_top_score": 0.85,
      "combined_confidence": 0.913
    }
  },
  "latency_ms": 1420.5
}
```

---

### GET `/history` — Get user history

```bash
curl "http://localhost:8000/history?user_id=user_123&limit=5"
```

**Response:**
```json
{
  "user_id": "user_123",
  "total": 2,
  "interactions": [
    {
      "id": 1,
      "query": "How do I reset my password?",
      "response": "To reset your password...",
      "created_at": "2025-01-15T10:30:00+00:00"
    }
  ]
}
```

---

### DELETE `/history` — Delete user history (GDPR)

```bash
curl -X DELETE "http://localhost:8000/history?user_id=user_123"
```

---

### POST `/reindex` — Re-index FAQ knowledge base

```bash
curl -X POST http://localhost:8000/reindex
```

Use this after updating `app/data/faqs.txt`.

---

### GET `/health` — System health check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "components": {
    "rag":    {"status": "ok", "chunks_indexed": 42, "embedding_model": "all-MiniLM-L6-v2"},
    "memory": {"status": "ok", "backend": "sqlite", "total_interactions": 15},
    "web":    {"status": "ok", "web_search_enabled": true, "firecrawl_configured": false},
    "llm":    {"status": "ok", "model": "gemini-1.5-flash", "api_key_set": true}
  }
}
```

---

## ⚙️ Configuration Reference

All settings are controlled via `.env`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | – | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`) |
| `GEMINI_MAX_TOKENS` | `1024` | Max response tokens |
| `GEMINI_TEMPERATURE` | `0.3` | Response creativity (0–1) |
| `FIRECRAWL_API_KEY` | – | Firecrawl API key (optional) |
| `WEB_SEARCH_ENABLED` | `true` | Enable/disable web search |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `RAG_TOP_K` | `4` | FAQ chunks to retrieve per query |
| `CHUNK_SIZE` | `400` | Characters per FAQ chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |  
| `MEMORY_BACKEND` | `sqlite` | `sqlite` or `redis` |
| `MEMORY_MAX_HISTORY` | `10` | Max stored interactions per user |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## 🔄 Memory Backend Upgrade

### Switch to Redis

```bash
# 1. Install Redis driver
pip install redis

# 2. Update .env
MEMORY_BACKEND=redis
REDIS_URL=redis://localhost:6379/0

# 3. Start Redis (Docker)
docker run -d -p 6379:6379 redis:alpine
```

---

## 📖 Extending the FAQ

Edit `app/data/faqs.txt` using this format:

```
====================
SECTION: Your Section Name
====================

Q: Your question here?
A: Your detailed answer here.
```

Then call `POST /reindex` to rebuild the vector store.

---

## 🛠 Development Tips

- Enable debug logging: `LOG_LEVEL=DEBUG` in `.env`
- The vector store persists in `app/db/vector_store/` — delete it to force a fresh index
- Run without web search: `WEB_SEARCH_ENABLED=false`
- Use `GET /docs` for the interactive Swagger UI

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| LLM | Google Gemini 1.5 Flash (via `google-generativeai`) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent) |
| Memory | SQLite (upgradeable to Redis) |
| Web search | Firecrawl API (with mock fallback) |
| Config | python-dotenv |
