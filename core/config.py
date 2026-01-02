import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Centralized configuration defaults with environment overrides.
# Use env vars in production to avoid hardcoding secrets.

# Embedding / chunking defaults
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
_max_chunks_str = os.environ.get("MAX_CHUNKS")
MAX_CHUNKS = int(_max_chunks_str) if _max_chunks_str else None

# Mongo / Atlas defaults
# Use a local Mongo instance by default for development. In production set
# the `MONGO_URI` env var to your Atlas or managed Mongo connection string.
MONGO_URI = os.environ.get("MONGO_URI")
MONGO_DB = os.environ.get("MONGO_DB", "rag_db")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "chunks")
ATLAS_INDEX_NAME = os.environ.get("ATLAS_INDEX_NAME", "embedding_vector_index")

# HuggingFace Configuration
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.environ.get("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Options: "sentence-transformers/all-MiniLM-L6-v2" (384 dim, fast)
#          "sentence-transformers/all-mpnet-base-v2" (768 dim, better quality)
#          "BAAI/bge-small-en-v1.5" (384 dim, good performance)

# Groq LLM Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "mixtral-8x7b-32768")
# Available models: mixtral-8x7b-32768, llama2-70b-4096, gemma-7b-it
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "800"))
GROQ_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.7"))

# Vector Store Configuration
VECTOR_STORE = os.environ.get("VECTOR_STORE", "faiss")  # Options: "mongodb", "faiss"
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./faiss_index")

# Secret key fallback for dev only (not used by FastAPI templates but kept for compatibility)
SECRET_FALLBACK = os.environ.get("SECRET_FALLBACK") or os.urandom(32)


def max_chunks() -> Optional[int]:
    return MAX_CHUNKS


# Export config as dictionary for easier access
config = {
    'DEFAULT_CHUNK_SIZE': DEFAULT_CHUNK_SIZE,
    'DEFAULT_CHUNK_OVERLAP': DEFAULT_CHUNK_OVERLAP,
    'MAX_CHUNKS': MAX_CHUNKS,
    'MONGO_URI': MONGO_URI,
    'MONGO_DB': MONGO_DB,
    'MONGO_COLLECTION': MONGO_COLLECTION,
    'ATLAS_INDEX_NAME': ATLAS_INDEX_NAME,
    'HUGGINGFACE_API_KEY': HUGGINGFACE_API_KEY,
    'HUGGINGFACE_MODEL': HUGGINGFACE_MODEL,
    'GROQ_API_KEY': GROQ_API_KEY,
    'GROQ_API_URL': GROQ_API_URL,
    'GROQ_MODEL': GROQ_MODEL,
    'GROQ_MAX_TOKENS': GROQ_MAX_TOKENS,
    'GROQ_TEMPERATURE': GROQ_TEMPERATURE,
    'VECTOR_STORE': VECTOR_STORE,
    'FAISS_INDEX_PATH': FAISS_INDEX_PATH,
    'SECRET_FALLBACK': SECRET_FALLBACK,
}
