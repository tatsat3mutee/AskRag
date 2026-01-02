import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Optional

# Optional dotenv support for local development. If python-dotenv is installed
# and a .env file exists in the repo root, load it so env vars are available.
try:
    from dotenv import load_dotenv
    if os.path.exists('.env'):
        load_dotenv('.env')
        logger = logging.getLogger(__name__)
        logger.info('Loaded .env file for local development')
except Exception:
    # dotenv is optional; if it's not available, fall back to environment vars.
    pass

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, FileResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import config
from core.embedding_service import EmbeddingService
from core.llm_service import ask_with_context

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

ROOT_PATH = os.environ.get("ROOT_PATH", "")
app = FastAPI(
    title="RAG Platform API",
    version="1.0.0",
    description="REST API for RAG Platform - Use Streamlit UI at port 8501",
    root_path=ROOT_PATH
)

# Limit concurrent heavy operations (PDF ingest, Mongo insert, vector search)
try:
    _max_concurrency = int(os.getenv("ASKDOC_MAX_CONCURRENCY", "4"))
except Exception:
    _max_concurrency = 4
CONCURRENCY_SEM = asyncio.Semaphore(_max_concurrency)


def _parse_bool(val, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).lower() in {"1", "true", "yes", "y", "on"}


def _check_credentials() -> None:
    """Check if required API keys are configured."""
    groq_key = config.get('GROQ_API_KEY')
    hf_model = config.get('HUGGINGFACE_MODEL')
    
    if not groq_key:
        raise RuntimeError('GROQ_API_KEY must be set in environment variables')
    if not hf_model:
        raise RuntimeError('HUGGINGFACE_MODEL must be set in environment variables')


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info("cid=%s start %s %s", request_id, request.method, request.url.path)
    response = await call_next(request)
    logger.info("cid=%s end %s status=%s", request_id, request.url.path, response.status_code)
    return response


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root endpoint - redirects to API documentation"""
    return {
        "message": "RAG Platform REST API",
        "version": "1.0.0",
        "docs": "/docs",
        "streamlit_ui": "http://localhost:8501",
        "endpoints": {
            "embed": "POST /api/embed - Ingest documents",
            "search": "POST /api/search - Search and query",
            "health": "GET /healthz - Health check",
            "diagnostics": "GET /diag - System diagnostics"
        }
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.post("/api/embed")
async def api_embed(request: Request, file: UploadFile | None = File(None)):
    # verify credentials early to avoid long-running work when missing
    try:
        _check_credentials()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    embed_svc = EmbeddingService()

    payload = {}
    form_data = None
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form_data = await request.form()
    else:
        try:
            payload = await request.json()
        except Exception:
            payload = {}

    # Domain parsing removed; rely solely on local chunking.
    chunk_size = int(
        (form_data.get("chunk_size") if form_data else None)
        or payload.get("chunk_size")
        or config['DEFAULT_CHUNK_SIZE']
    )
    overlap = int(
        (form_data.get("overlap") if form_data else None)
        or payload.get("overlap")
        or config['DEFAULT_CHUNK_OVERLAP']
    )
    store_mongo = _parse_bool(
        (form_data.get("store_mongo") if form_data else None)
        or payload.get("store_mongo"),
        True,
    )
    mongo_uri = (form_data.get("mongo_uri") if form_data else None) or payload.get("mongo_uri") or config.get('MONGO_URI')
    db_name = config['MONGO_DB']
    coll_name = config['MONGO_COLLECTION']
    max_chunks = payload.get("max_chunks") if payload else config.max_chunks()
    max_chunks = int(max_chunks) if max_chunks is not None else None

    try:
        if file is not None:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file selected")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                content = await file.read()
                tf.write(content)
                tmp_path = tf.name
            try:
                async with CONCURRENCY_SEM:
                    # Guard long-running ingest with timeout
                    try:
                        async with asyncio.timeout(300):
                            summary = await asyncio.to_thread(
                                embed_svc.ingest_pdf_and_store,
                                tmp_path,
                                mongo_uri,
                                chunk_size,
                                overlap,
                                db_name,
                                coll_name,
                                True,
                                store_mongo,
                                max_chunks,
                            )
                    except TimeoutError:
                        raise HTTPException(status_code=504, detail="Ingest timed out")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    logger.warning("Failed to delete temp file from /api/embed")
            return JSONResponse({"summary": summary})

        text = payload.get("text") or (form_data.get("text") if form_data else "")
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        # Chunk locally and batch embed to reduce per-request overhead
        texts = list(embed_svc._chunk_text(text, chunk_size=chunk_size, overlap=overlap))
        embeddings = embed_svc.batch_get_embeddings(texts)
        chunks = [
            {"text": t, "embedding": e}
            for t, e in zip(texts, embeddings)
        ]
        inserted = 0
        if store_mongo and chunks:
            async with CONCURRENCY_SEM:
                try:
                    async with asyncio.timeout(120):
                        inserted = await asyncio.to_thread(
                            embed_svc.store_chunks_to_mongo,
                            chunks,
                            mongo_uri,
                            db_name,
                            coll_name,
                        )
                except TimeoutError:
                    raise HTTPException(status_code=504, detail="Mongo insert timed out")
        embedded_count = sum(1 for c in chunks if c.get("embedding"))
        summary = {
            "chunks": len(chunks),
            "embedded": embedded_count,
            "stored": inserted,
        }
        return JSONResponse({"summary": summary})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("/api/embed failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/search")
async def api_search(request: Request):
    # verify credentials early
    try:
        _check_credentials()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    embed_svc = EmbeddingService()
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    query = payload.get("query") or ""
    if not query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    top_k = int(payload.get("top_k") or 5)
    num_candidates = int(payload.get("num_candidates") or 100)
    source_filter = payload.get("source_filter")
    # Default to summarizing unless explicitly disabled
    summarize = not (str(payload.get("summarize")).lower() in {"false","0","no","n","off"}) if ("summarize" in payload) else True
    system_prompt = payload.get("system_prompt")
    mongo_uri = payload.get("mongo_uri") or config.get('MONGO_URI')
    db_name = config['MONGO_DB']
    coll_name = config['MONGO_COLLECTION']
    try:
        async with CONCURRENCY_SEM:
            try:
                async with asyncio.timeout(60):
                    results = await asyncio.to_thread(
                        embed_svc.vector_search,
                        query,
                        mongo_uri,
                        config['ATLAS_INDEX_NAME'],
                        db_name,
                        coll_name,
                        top_k,
                        num_candidates,
                        source_filter,
                    )
            except TimeoutError:
                raise HTTPException(status_code=504, detail="Search timed out")
        answer = None
        if summarize and results:
            contexts = [r.get("text","") for r in results if r.get("text")]
            try:
                async with asyncio.timeout(45):
                    answer = await asyncio.to_thread(ask_with_context, query, contexts, system_prompt)
            except TimeoutError:
                answer = "Summarization timed out"
        return JSONResponse({"results": results, "answer": answer})
    except Exception as exc:
        logger.exception("/api/search failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/diag")
async def diag():
    """Lightweight diagnostics to aid troubleshooting and capacity planning."""
    creds_ok = True
    try:
        _check_credentials()
    except RuntimeError:
        creds_ok = False
    return JSONResponse({
        "status": "ok",
        "creds": creds_ok,
        "embedding_url_set": bool(getattr(config, "EMBEDDING_URL", "")),
        "domain_parser_set": bool(getattr(config, "DOMAIN_PARSER_URL", "")),
        "mongo_uri_set": bool(getattr(config, "MONGO_URI", "")),
        "atlas_index": getattr(config, "ATLAS_INDEX_NAME", None),
        "max_concurrency": _max_concurrency,
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
