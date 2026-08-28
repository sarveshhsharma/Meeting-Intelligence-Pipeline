# Automated Enterprise Meeting Intelligence & Action-Item Pipeline

A full-stack RAG system that converts raw meeting audio into structured, searchable, and actionable knowledge — automatically transcribing meetings, extracting decisions and action items, and answering natural-language questions about any indexed meeting.

Built with **OpenAI Whisper**, **LangChain**, **ChromaDB**, and **FastAPI**.

---

## Overview

Organizations record hundreds of meetings but rarely have an automated way to turn hours of audio into searchable, structured knowledge. Decisions, owners, and deadlines mentioned verbally get buried in raw recordings, and manually preparing meeting minutes is slow and error-prone.

This project automates that pipeline end-to-end:

1. **Transcribe** meeting audio (`.mp3`, `.wav`, `.m4a`) using OpenAI Whisper.
2. **Extract** structured meeting intelligence — executive summary, key decisions, action items (task/assignee/due date/priority), and sentiment — using schema-constrained LLM generation with Pydantic.
3. **Index** transcript chunks as vector embeddings in ChromaDB for semantic retrieval.
4. **Query** individual meetings conversationally through a context-filtered **Map-Reduce RAG** pipeline, or search globally across all indexed meetings.
5. **Serve** everything through a FastAPI backend and an interactive Streamlit dashboard.

---

## Key Features

- 🎙️ **Automated Speech-to-Text** — Whisper model loaded once at startup for low-latency transcription.
- 🧩 **Schema-Constrained Extraction** — LLM outputs validated at runtime via Pydantic models (`MeetingSummary`, `ActionItem`), eliminating free-form/hallucinated response formats.
- 🔍 **Map-Reduce RAG** — Each retrieved chunk is filtered for relevance (*Map*) before being combined into a condensed context for final answer generation (*Reduce*) — reducing noise and hallucination vs. naive "stuff everything into the prompt" RAG.
- 🗂️ **Per-Meeting & Global Search** — Ask questions scoped to a single meeting (metadata-filtered ChromaDB query) or semantically search across the entire meeting repository.
- 💬 **Conversational Frontend** — Streamlit chat interface with persistent session state per selected meeting.
- 🐳 **Containerized & Modular** — Clear separation between frontend, API, AI services, and data layers; ready for Docker Compose deployment.

---

## Architecture

```
┌─────────────────────────┐
│  Presentation Layer      │  Streamlit
│  - Audio Upload           │
│  - Meeting Explorer       │
│  - Meeting-specific Chat  │
│  - Task/Priority Analytics│
└───────────┬───────────────┘
            │ REST
┌───────────▼───────────────┐
│  API Gateway               │  FastAPI
│  POST /api/v1/process-meeting
│  GET  /api/v1/meetings
│  POST /api/v1/meeting-chat
│  POST /api/v1/search
└───────────┬───────────────┘
            │
┌───────────▼───────────────┐
│  AI Execution Core          │
│  - STT: OpenAI Whisper        │
│  - LLM: LangChain + GPT-4o-mini│
│  - Structured Output: Pydantic │
│  - Chunking: RecursiveCharacterTextSplitter │
│  - Vector Store: ChromaDB       │
│  - Embeddings: text-embedding-3-small │
└─────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Speech-to-Text | OpenAI Whisper |
| LLM Orchestration | LangChain (GPT-4o-mini) |
| Structured Output Validation | Pydantic |
| Text Chunking | `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) |
| Vector Store | ChromaDB (persistent client) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker / Docker Compose |

---

## Repository Structure

```
meeting-intelligence-pipeline/
├── .env
├── .gitignore
├── docker-compose.yml
├── README.md
│
├── data/
│   ├── uploads/
│   └── chromadb/
│
├── notebooks/
│   ├── 01_whisper_test.ipynb
│   └── 02_langchain_rag.ipynb
│
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── pages/
│       ├── 1_Upload_Audio.py
│       ├── 2_Meeting_History.py
│       └── 3_Analytics.py
│
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py
    ├── api/
    │   └── routes.py
    ├── services/
    │   ├── transcription.py
    │   ├── llm_engine.py
    │   ├── dummy_engine.py
    │   └── vector_store.py
    └── models/
        └── schemas.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/meetings` | Retrieve identifiers and filenames of all indexed meetings |
| `POST` | `/api/v1/process-meeting` | Upload audio, transcribe, extract structured intelligence, and index it |
| `POST` | `/api/v1/meeting-chat` | Run the Map-Reduce RAG pipeline against a selected meeting |
| `POST` | `/api/v1/search` | Global semantic search across all indexed meetings |

---

## Data Contracts

Structured outputs are enforced with Pydantic schemas to prevent format drift and hallucinated fields:

```python
class ActionItem(BaseModel):
    task: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    priority: str = "Normal"

class MeetingSummary(BaseModel):
    executive_summary: str
    key_decisions: List[str]
    action_items: List[ActionItem]
    overall_sentiment: str

class TranscriptionResponse(BaseModel):
    filename: str
    transcript_text: str
    duration_seconds: float
```

---

## How the RAG Pipeline Works

Rather than stuffing the entire transcript (or every retrieved chunk) into a single prompt, the system uses a two-stage **Map-Reduce** approach:

1. **Map** — Each of the top-k retrieved chunks (via ChromaDB similarity search, filtered to the selected meeting) is passed independently to the LLM, which extracts *only* the information relevant to the user's question and discards the rest.
2. **Reduce** — The extracted, relevance-filtered snippets are combined into a condensed context, which the LLM uses to generate the final answer — strictly grounded in retrieved content.

This reduces irrelevant context, controls generation more tightly, and lowers the risk of hallucination compared to naive whole-transcript or whole-chunk-set prompting.

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd meeting-intelligence-pipeline

# Configure environment variables
cp .env.example .env   # add your OPENAI_API_KEY

# Run with Docker Compose
docker-compose up --build
```

Or run services individually:

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## Engineering Notes

- **Hosted vs. open-source LLMs**: Early experiments with Mistral-7B / Zephyr-7B via hosted inference introduced integration friction between chat-completion and text-generation APIs; the project standardized on `gpt-4o-mini` for reliable structured-output support.
- **Environment variable ordering**: `load_dotenv()` is called at the application entry point *before* importing dependent services, since ChromaDB and OpenAI clients may read the API key at import/initialization time.
- **Mock service layer**: A `dummy_engine.py` mirrors the production LLM engine's Pydantic schemas, allowing frontend/API development without incurring live LLM calls.
- **Defensive ChromaDB parsing**: Query results are extracted with safe defaults (`.get(..., [])`) to guard against nested-structure `KeyError`/`IndexError`.

---

## Roadmap

- [ ] **Multi-speaker diarization** via `pyannote.audio`
- [ ] **Parallel Map-stage processing** using `asyncio.gather` to reduce RAG latency
- [ ] **Enterprise integrations** — Slack, Microsoft Teams, Jira for auto-distributing summaries and creating action-item tickets
- [ ] **Improved retrieval** — metadata filtering, hybrid search, reranking
- [ ] **Production hardening** — authentication, monitoring, logging, rate limiting, orchestration

---
