# 🚀 How to Run Your RAG Platform

## Two Ways to Use Your Platform

### 1️⃣ **Streamlit UI** (For Humans) 👥
Interactive web dashboard with visual interface

### 2️⃣ **FastAPI REST API** (For Machines) 🤖
Programmatic access via HTTP requests

---

## 🖥️ Running Streamlit UI

**Terminal 1:**
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run Streamlit
streamlit run app/app.py
```

**Opens at:** http://localhost:8501

**Features:**
- ✅ Document ingestion UI
- ✅ Real-time search
- ✅ LLM answer generation
- ✅ Visual configuration

---

## 🌐 Running FastAPI REST API

**Terminal 2:**
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run FastAPI
uvicorn app.main:app --reload --port 8000
```

**Opens at:** http://localhost:8000

**API Docs:** http://localhost:8000/docs

**Features:**
- ✅ POST /api/embed - Ingest documents
- ✅ POST /api/search - Search & query
- ✅ GET /healthz - Health check
- ✅ GET /diag - Diagnostics

---

## 🚀 Run BOTH Together

**PowerShell - Open 2 terminals:**

**Terminal 1 (Streamlit):**
```powershell
.venv\Scripts\activate
streamlit run app/app.py
```

**Terminal 2 (FastAPI):**
```powershell
.venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

---

## 🐳 Run with Docker (Both at Once)

```bash
docker-compose up --build
```

**Access:**
- Streamlit UI: http://localhost:8501
- FastAPI: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📊 Usage Examples

### **Streamlit UI** (Visual)
1. Open http://localhost:8501
2. Click "🚀 Initialize" in sidebar
3. Go to "📄 Ingest" tab
4. Paste text and click "🚀 Ingest"
5. Go to "🔎 Query" tab
6. Ask questions!

### **FastAPI** (Programmatic)

**Ingest Document:**
```bash
curl -X POST "http://localhost:8000/api/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here",
    "chunk_size": 1000,
    "overlap": 200
  }'
```

**Search & Query:**
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "summarize": true
  }'
```

---

## 🔧 Configuration

**Required Environment Variables (.env):**
```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/faiss_index
```

---

## ✅ Verify Setup

**Test Imports:**
```bash
python -c "from core import EmbeddingService; print('✓ OK')"
```

**Check API Health:**
```bash
curl http://localhost:8000/healthz
```

**Check Streamlit:**
Open http://localhost:8501 in browser

---

## 🎯 Quick Start (First Time)

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
# Edit .env file with your GROQ_API_KEY

# 3. Run Streamlit
streamlit run app/app.py

# 4. (Optional) Run API in another terminal
uvicorn app.main:app --reload
```

---

## 🌟 What's Running?

| Service | Port | Purpose | URL |
|---------|------|---------|-----|
| **Streamlit** | 8501 | Visual UI | http://localhost:8501 |
| **FastAPI** | 8000 | REST API | http://localhost:8000 |
| **API Docs** | 8000 | Swagger UI | http://localhost:8000/docs |

---

## 🔄 Typical Workflow

1. **Start Streamlit** → Visual interface
2. **Initialize system** → Click button in sidebar
3. **Ingest documents** → Use Ingest tab
4. **Query system** → Use Query tab
5. **(Optional) Use API** → For automation/scripts

---

## 🆘 Troubleshooting

**Streamlit won't start?**
```bash
pip install streamlit
streamlit run app/app.py
```

**FastAPI won't start?**
```bash
pip install fastapi uvicorn
uvicorn app.main:app --reload
```

**Import errors?**
```bash
# Make sure you're in project root
cd "C:\Users\TatsatPandey\Documents\Learnings\Ask Rag"
python -c "from core import config; print('OK')"
```

**Port already in use?**
```bash
# Streamlit on different port
streamlit run app/app.py --server.port 8502

# FastAPI on different port
uvicorn app.main:app --port 8001
```

---

## 🎉 You're Ready!

Your platform has:
- ✅ **Streamlit UI** for interactive use
- ✅ **FastAPI REST API** for automation
- ✅ Both share the same core services
- ✅ No duplicate UI code!

**One UI (Streamlit), One API (FastAPI), Infinite Possibilities! 🚀**
