import logging
import os
import time
import datetime
from typing import Optional, List
from uuid import uuid4
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from .config import config

logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None


class EmbeddingService:
    """Embedding service using HuggingFace sentence-transformers models."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding service with a HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace model to use. If None, uses config default.
        """
        self.model_name = model_name or config.get('HUGGINGFACE_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f'Initializing HuggingFace embedding model: {self.model_name}')
        
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f'Loaded model {self.model_name} with embedding dimension: {self.embedding_dim}')
        except Exception as e:
            logger.error(f'Failed to load HuggingFace model {self.model_name}: {e}')
            raise RuntimeError(f'Failed to load embedding model: {e}')
        
        self._mongo_client: Optional[MongoClient] = None

    def _mongo(self, mongo_uri: str) -> MongoClient:
        """Get or create MongoDB client."""
        if self._mongo_client is None:
            self._mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            logger.info('Created Mongo client')
        return self._mongo_client

    # --- Mongo index helpers ---
    def ensure_indexes(self, mongo_uri: str, db_name: str = 'rag_db', collection_name: str = 'chunks') -> None:
        """Create helpful indexes for chunks collection."""
        logger.info('Ensuring indexes on %s.%s', db_name, collection_name)
        coll = self._mongo(mongo_uri)[db_name][collection_name]
        coll.create_index([('chunk_id', 1)], unique=True)
        coll.create_index([('source', 1), ('page', 1), ('chunk_index', 1)])
        coll.create_index([('created_on', -1)])
        logger.info('Ensured indexes on %s.%s', db_name, collection_name)

    # --- Chunking helpers ---
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200):
        """Fallback simple chunker (used if langchain splitter unavailable)."""
        logger.debug('Chunking text: chunk_size=%d, overlap=%d', chunk_size, overlap)
        start = 0
        L = len(text)
        while start < L:
            end = start + chunk_size
            chunk = text[start:end]
            yield chunk
            start = max(end - overlap, end)

    def batch_get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Generate embeddings for a list of texts using HuggingFace model.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings (as lists of floats) aligned with input texts
        """
        if not texts:
            logger.info('batch_get_embeddings called with empty texts list; returning []')
            return []
        
        logger.info(f'Generating embeddings for {len(texts)} texts using {self.model_name}')
        start_time = time.perf_counter()
        
        try:
            # Generate embeddings in batches for efficiency
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=batch_size
                )
                # Convert numpy arrays to lists
                all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
                logger.debug(f'Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}')
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f'Generated {len(all_embeddings)} embeddings in {elapsed_ms:.1f}ms')
            
            return all_embeddings
            
        except Exception as exc:
            logger.exception(f'Failed to generate embeddings: {exc}')
            # Return None for each text on failure
            return [None] * len(texts)

    def get_embedding_from_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding as a list of floats, or None on failure
        """
        if not text or not text.strip():
            logger.warning('Skipping embedding request because text is empty')
            return None
        
        logger.info(f'Generating embedding for text (len={len(text)})')
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f'Error generating embedding: {e}')
            return None

    # --- PDF chunking + embedding ---
    def get_embeddings_from_pdf_chunks(
        self, 
        pdf_path: str, 
        chunk_size: int = 1000, 
        overlap: int = 200, 
        max_pages: Optional[int] = None,
        use_tokenizer_splitter: bool = True
    ) -> List[dict]:
        """Use LangChain's PyPDFLoader + RecursiveCharacterTextSplitter for robust PDF chunking.

        Returns list of dicts: {text, embedding, metadata}
        metadata contains page, chunk_index, and source (file path).
        """
        if PyPDFLoader is None or RecursiveCharacterTextSplitter is None:
            logger.error('langchain with PyPDFLoader and RecursiveCharacterTextSplitter is required.')
            raise RuntimeError('langchain with PyPDFLoader and RecursiveCharacterTextSplitter is required.')

        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(pdf_path)

        logger.info('Loading PDF for chunking: %s', pdf_path)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        splitter = None
        if use_tokenizer_splitter and hasattr(RecursiveCharacterTextSplitter, 'from_tiktoken_encoder'):
            try:
                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=chunk_size, 
                    chunk_overlap=overlap
                )
                logger.info('Using tokenizer-aware splitter')
            except Exception as e:
                logger.warning('Falling back to character splitter: %s', e)
        
        if splitter is None:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=overlap
            )
        
        chunks_with_meta = []
        for d in docs:
            page_num = d.metadata.get('page', None)
            subdocs = splitter.split_documents([d])
            for idx, sd in enumerate(subdocs):
                chunks_with_meta.append({
                    'text': sd.page_content,
                    'metadata': {
                        'page': page_num,
                        'chunk_index': idx,
                        'source': pdf_path
                    }
                })

        logger.info('Created %d chunks from PDF', len(chunks_with_meta))
        
        # Generate embeddings for all chunks
        texts = [c['text'] for c in chunks_with_meta]
        embeddings = self.batch_get_embeddings(texts)
        
        result = []
        for c, e in zip(chunks_with_meta, embeddings):
            if e is None:
                logger.warning(
                    'Skipping chunk with missing embedding for source=%s page=%s idx=%s',
                    c['metadata'].get('source'),
                    c['metadata'].get('page'),
                    c['metadata'].get('chunk_index')
                )
                continue
            
            result.append({
                'text': c['text'],
                'embedding': e,
                'metadata': c['metadata']
            })
        
        return result

    # --- Mongo storage ---
    def store_chunks_to_mongo(
        self, 
        chunks: List[dict], 
        mongo_uri: str, 
        db_name: str = 'rag_db', 
        collection_name: str = 'chunks', 
        batch_size: int = 500
    ) -> int:
        """Insert chunk documents into MongoDB in batches. Returns number of inserted docs."""
        logger.info('Connecting to MongoDB to store %d chunks', len(chunks))
        coll = self._mongo(mongo_uri)[db_name][collection_name]
        
        if batch_size <= 0:
            batch_size = 500

        docs_batch: List[dict] = []
        total_inserted = 0

        def flush(batch_docs: List[dict]):
            nonlocal total_inserted
            if not batch_docs:
                return
            try:
                res = coll.insert_many(batch_docs, ordered=False)
                inserted_count = len(res.inserted_ids)
                total_inserted += inserted_count
                logger.info(
                    'Inserted %d documents into %s.%s (running total=%d)',
                    inserted_count, db_name, collection_name, total_inserted
                )
            except Exception as e:
                logger.exception('Failed to insert batch into MongoDB (size=%d): %s', len(batch_docs), e)
                raise

        for idx, c in enumerate(chunks):
            emb = c.get('embedding')
            if emb is None:
                logger.warning('Skipping chunk idx=%d due to missing embedding', idx)
                continue
            if not isinstance(emb, list) or not emb:
                logger.warning('Skipping chunk idx=%d due to invalid embedding type', idx)
                continue

            doc = {
                'text': c.get('text'),
                'embedding': emb,
                'created_on': datetime.datetime.utcnow(),
            }
            meta = c.get('metadata') or {}
            if 'page' in meta:
                doc['page'] = meta['page']
            if 'chunk_index' in meta:
                doc['chunk_index'] = meta['chunk_index']
            if 'source' in meta:
                doc['source'] = meta['source']
            doc['chunk_id'] = f"{doc.get('source','')}_{idx}"
            docs_batch.append(doc)

            if len(docs_batch) >= batch_size:
                flush(docs_batch)
                docs_batch = []

        # flush remainder
        flush(docs_batch)

        if total_inserted == 0:
            logger.info('No documents were inserted after validation')
        return total_inserted

    def ingest_pdf_and_store(
        self, 
        pdf_path: str, 
        mongo_uri: str, 
        chunk_size: int = 1000, 
        overlap: int = 200,
        db_name: str = 'rag_db', 
        collection_name: str = 'chunks', 
        ensure_index: bool = True,
        store_mongo: bool = True, 
        max_chunks: Optional[int] = None
    ) -> dict:
        """Chunk a PDF, embed all chunks, store to Mongo, and return summary."""
        if ensure_index:
            self.ensure_indexes(mongo_uri, db_name=db_name, collection_name=collection_name)

        # Use local chunking + embedding
        chunks = self.get_embeddings_from_pdf_chunks(
            pdf_path, 
            chunk_size=chunk_size, 
            overlap=overlap
        )
        
        if max_chunks is not None and len(chunks) > max_chunks:
            logger.info('Truncating chunks from %d to max_chunks=%d', len(chunks), max_chunks)
            chunks = chunks[:max_chunks]

        inserted = 0
        if store_mongo and chunks:
            inserted = self.store_chunks_to_mongo(
                chunks, 
                mongo_uri, 
                db_name=db_name, 
                collection_name=collection_name
            )
        else:
            logger.info('store_mongo=%s; skipping Mongo storage (chunks=%d)', store_mongo, len(chunks))

        embedded_count = sum(1 for c in chunks if c.get('embedding'))
        return {
            'chunks': len(chunks),
            'embedded': embedded_count,
            'stored': inserted
        }

    def embedding_dimension(self, mongo_uri: str, db_name: str = 'rag_db', collection_name: str = 'chunks') -> int:
        """Get embedding dimension from a stored document."""
        logger.info('Getting embedding dimension from %s.%s', db_name, collection_name)
        client = MongoClient(mongo_uri)
        doc = client[db_name][collection_name].find_one(
            {'embedding': {'$exists': True}}, 
            {'embedding': 1}
        )
        if not doc or 'embedding' not in doc:
            raise RuntimeError('No embedding found in collection to infer dimension')
        return len(doc['embedding'])

    def vector_search(
        self, 
        query_text: str, 
        mongo_uri: str, 
        index_name: str = 'embedding_vector_index',
        db_name: str = 'rag_db', 
        collection_name: str = 'chunks',
        limit: int = 5, 
        num_candidates: int = 100, 
        source_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> list:
        """Perform a vector search using Atlas $vectorSearch (requires Atlas Search index)."""
        if not query_text or not query_text.strip():
            logger.warning('Vector search skipped because query is empty')
            return []
        
        if top_k is not None:
            limit = top_k
        
        request_id = str(uuid4())
        logger.info(
            'Vector search start cid=%s query="%s" limit=%d source_filter=%s',
            request_id, query_text, limit, source_filter
        )
        
        start = time.perf_counter()
        query_vec = self.get_embedding_from_text(query_text)
        
        if query_vec is None:
            logger.error('Failed to generate query embedding')
            return []
        
        coll = self._mongo(mongo_uri)[db_name][collection_name]
        search_stage = {
            'index': index_name,
            'path': 'embedding',
            'queryVector': query_vec,
            'numCandidates': num_candidates,
            'limit': limit
        }
        
        if source_filter:
            search_stage['filter'] = {'source': source_filter}

        pipeline = [
            {'$vectorSearch': search_stage},
            {'$project': {
                '_id': 0,
                'text': 1,
                'source': 1,
                'page': 1,
                'chunk_index': 1,
                'score': {'$meta': 'vectorSearchScore'}
            }}
        ]
        
        results = list(coll.aggregate(pipeline))
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info('Vector search done cid=%s results=%d elapsed_ms=%.1f', request_id, len(results), elapsed_ms)
        
        return results


if __name__ == '__main__':
    # Example: ingest a PDF and store chunks
    svc = EmbeddingService()
    pdf_path = os.environ.get('PDF_PATH', 'sample.pdf')
    mongo_uri = os.environ.get('MONGO_URI')
    
    if mongo_uri and os.path.isfile(pdf_path):
        summary = svc.ingest_pdf_and_store(pdf_path, mongo_uri)
        print('Ingest summary:', summary)
    else:
        print('Set MONGO_URI and ensure PDF_PATH exists to ingest.')
        print(f'Embedding service initialized with model: {svc.model_name}')
        print(f'Embedding dimension: {svc.embedding_dim}')

# Made with Bob
