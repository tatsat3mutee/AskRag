"""Streamlit Dashboard for RAG Platform

Production-ready web interface using existing infrastructure.
"""
import streamlit as st
import logging
import time
from datetime import datetime

import tempfile
from pathlib import Path

try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:  # pragma: no cover
    PyPDFLoader = None

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import from core
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config
from core.embedding_service import EmbeddingService
from core.vector_store import FAISSVectorStore
from core.llm_service import ask_with_context

# Page config
st.set_page_config(
    page_title="RAG Platform",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; padding: 1rem 0;}
    .success-box {background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# Session state and auto-initialization
def init_session_state():
    if 'embedding_service' not in st.session_state:
        st.session_state.embedding_service = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'doc_count' not in st.session_state:
        st.session_state.doc_count = 0
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

init_session_state()

# Auto-initialize on first load
if not st.session_state.initialized:
    try:
        with st.spinner("🚀 Initializing RAG Platform..."):
            st.session_state.embedding_service = EmbeddingService()
            st.session_state.vector_store = FAISSVectorStore(
                index_path=config['FAISS_INDEX_PATH'],
                dimension=st.session_state.embedding_service.embedding_dim
            )
            st.session_state.doc_count = len(st.session_state.vector_store.metadata)
            st.session_state.initialized = True
        st.success("✅ System ready!")
    except Exception as e:
        st.error(f"⚠️ Initialization failed: {e}")
        st.info("💡 Check your .env file has GROQ_API_KEY set")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("Embedding Model")
    st.info(f"**{config['HUGGINGFACE_MODEL']}**\n\nFREE HuggingFace model")
    
    st.subheader("LLM Generator")
    use_groq = st.checkbox("Enable Groq", value=bool(config.get('GROQ_API_KEY')))
    
    if use_groq:
        if config.get('GROQ_API_KEY'):
            st.success(f"✅ **{config['GROQ_MODEL']}**")
        else:
            groq_key = st.text_input("Groq API Key", type="password")
            if groq_key:
                config['GROQ_API_KEY'] = groq_key
                st.rerun()
    
    st.subheader("Settings")
    chunk_size = st.slider("Chunk Size", 200, 2000, config['DEFAULT_CHUNK_SIZE'], 100)
    chunk_overlap = st.slider("Overlap", 0, 500, config['DEFAULT_CHUNK_OVERLAP'], 50)
    top_k = st.slider("Top-K", 1, 10, 5)
    
    st.subheader("System Status")
    if st.session_state.initialized:
        st.success(f"✅ Ready | {st.session_state.doc_count} chunks indexed")
        if st.button("🔄 Reinitialize"):
            st.session_state.initialized = False
            st.rerun()
    else:
        st.warning("⚠️ System not initialized")
        if st.button("🚀 Initialize Now", type="primary"):
            with st.spinner("Initializing..."):
                try:
                    st.session_state.embedding_service = EmbeddingService()
                    st.session_state.vector_store = FAISSVectorStore(
                        index_path=config['FAISS_INDEX_PATH'],
                        dimension=st.session_state.embedding_service.embedding_dim
                    )
                    st.session_state.doc_count = len(st.session_state.vector_store.metadata)
                    st.session_state.initialized = True
                    st.success("✅ Ready!")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

st.markdown('<div class="main-header">🔍 RAG Platform</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📄 Ingest", "🔎 Query"])

# Ingest
with tab1:
    st.header("📄 Document Ingestion")
    
    if not st.session_state.embedding_service:
        st.warning("⚠️ Initialize system first")
    else:
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)
        extracted_text = ""
        if uploaded_pdf is not None:
            try:
                # LangChain loader expects a file path, so write to a temp file.
                suffix = Path(uploaded_pdf.name).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    tmp_path = tmp.name

                if PyPDFLoader is not None:
                    docs = PyPDFLoader(tmp_path).load()
                    extracted_text = "\n\n".join((d.page_content or "") for d in docs).strip()
                elif PdfReader is not None:
                    reader = PdfReader(tmp_path)
                    pages_text: list[str] = []
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            pages_text.append(page_text)
                    extracted_text = "\n\n".join(pages_text).strip()
                else:
                    st.error("PDF support is not available. Install langchain-community or PyPDF2 and restart.")

                if extracted_text:
                    st.success("✅ PDF text extracted")
                else:
                    st.warning("⚠️ Could not extract text from this PDF (it may be scanned images).")
            except Exception as e:
                st.error(f"❌ Failed to read PDF: {e}")
            finally:
                try:
                    if 'tmp_path' in locals() and tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        content = st.text_area(
            "Content",
            height=200,
            value=extracted_text,
            placeholder="Paste document...",
        )
        
        if st.button("🚀 Ingest"):
            if content.strip():
                with st.spinner("Processing..."):
                    try:
                        chunks = list(st.session_state.embedding_service._chunk_text(
                            content, chunk_size, chunk_overlap
                        ))
                        embeddings = st.session_state.embedding_service.batch_get_embeddings(chunks)
                        docs = [{"text": c, "chunk_id": f"c_{int(time.time())}_{i}"} 
                               for i, c in enumerate(chunks)]
                        
                        st.session_state.vector_store.add_documents(embeddings, docs)
                        st.session_state.vector_store.save()
                        st.session_state.doc_count += len(chunks)
                        
                        st.success(f"✅ Added {len(chunks)} chunks")
                        
                        with st.expander("Preview"):
                            for i, c in enumerate(chunks[:3]):
                                st.text(f"{i+1}. {c[:150]}...")
                    except Exception as e:
                        st.error(f"❌ {e}")

# Query
with tab2:
    st.header("🔎 Search & Query")
    
    if not st.session_state.initialized:
        st.warning("⚠️ System initializing... Please wait or click 'Initialize Now' in sidebar")
    elif st.session_state.doc_count == 0:
        st.info("💡 No documents indexed yet. Add some documents in the 'Ingest' tab first!")
    else:
        query = st.text_input("Question", placeholder="What is...?")
        
        if st.button("🔍 Search"):
            if query.strip():
                with st.spinner("Searching..."):
                    try:
                        q_emb = st.session_state.embedding_service.get_embedding_from_text(query)
                        results = st.session_state.vector_store.search(q_emb, top_k)
                        
                        st.subheader("📝 Results")
                        for i, r in enumerate(results, 1):
                            with st.expander(f"{i}. Score: {r['score']:.3f}"):
                                st.text(r['text'])
                        
                        if use_groq and config.get('GROQ_API_KEY'):
                            with st.spinner("Generating..."):
                                try:
                                    contexts = [r['text'] for r in results]
                                    answer = ask_with_context(query, contexts)
                                    st.subheader("💡 Answer")
                                    st.markdown(f'<div class="success-box">{answer}</div>', unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Generation failed: {e}")
                    except Exception as e:
                        st.error(f"❌ {e}")

st.divider()
st.caption("RAG Platform v1.0 | HuggingFace + FAISS + Groq")
