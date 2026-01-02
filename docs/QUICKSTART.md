# Quickstart

## Prerequisites

- Python 3.10+
- (Optional) Groq API key for answer generation

## Install

```bash
pip install -r requirements.txt
```

## Configure

Set values in `.env` (repo root):

- `GROQ_API_KEY` (optional)
- `HUGGINGFACE_MODEL` (optional)
- `FAISS_INDEX_PATH` (optional)

## Run

### Streamlit UI (primary)

```bash
streamlit run app/app.py
```

Open the printed URL (usually `http://localhost:8501`).

### FastAPI (optional REST API)

```bash
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

## Use the UI

### Ingest a PDF

1. Go to **📄 Ingest**
2. Use **Upload PDF**
3. Confirm text is extracted into the **Content** box
4. Click **🚀 Ingest**

### Ask a question

1. Go to **🔎 Query**
2. Enter a question
3. Click **🔍 Search**
4. (Optional) Enable Groq in the sidebar to generate an answer
