import os
import pickle
import logging
from typing import List, Dict, Optional
import numpy as np
import faiss
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for fast similarity search."""
    
    def __init__(self, index_path: str = "./faiss_index", dimension: int = 384):
        """Initialize FAISS vector store.
        
        Args:
            index_path: Directory to store FAISS index and metadata
            dimension: Embedding dimension (must match your model)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        
        self.index_file = self.index_path / "index.faiss"
        self.metadata_file = self.index_path / "metadata.pkl"
        
        # Initialize or load index
        if self.index_file.exists():
            self.load()
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.metadata = []
            logger.info(f"Created new FAISS index with dimension {dimension}")
    
    def add_documents(self, embeddings: List[List[float]], documents: List[Dict]) -> int:
        """Add documents with their embeddings to the index.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of document metadata dicts (must include 'text' field)
            
        Returns:
            Number of documents added
        """
        if not embeddings or not documents:
            logger.warning("No documents to add")
            return 0
        
        if len(embeddings) != len(documents):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents")
        
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Validate dimensions
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Add to index
        start_id = len(self.metadata)
        self.index.add(vectors)
        
        # Store metadata with IDs
        for i, doc in enumerate(documents):
            doc_with_id = doc.copy()
            doc_with_id['id'] = start_id + i
            self.metadata.append(doc_with_id)
        
        logger.info(f"Added {len(documents)} documents to FAISS index (total: {len(self.metadata)})")
        return len(documents)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if not self.metadata:
            logger.warning("Index is empty")
            return []
        
        # Convert query to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Validate dimension
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_vector.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Search
        top_k = min(top_k, len(self.metadata))
        distances, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                doc = self.metadata[idx].copy()
                # Convert L2 distance to similarity score (inverse)
                # Lower distance = higher similarity
                doc['score'] = float(1.0 / (1.0 + dist))
                doc['distance'] = float(dist)
                results.append(doc)
        
        logger.info(f"Found {len(results)} results for query")
        return results
    
    def save(self) -> None:
        """Save index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def load(self) -> None:
        """Load index and metadata from disk."""
        try:
            if not self.index_file.exists() or not self.metadata_file.exists():
                raise FileNotFoundError(f"Index files not found in {self.index_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.dimension = data['dimension']
            
            logger.info(f"Loaded FAISS index from {self.index_path} ({len(self.metadata)} documents)")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def clear(self) -> None:
        """Clear the index and metadata."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info("Cleared FAISS index")
    
    def delete(self) -> None:
        """Delete index files from disk."""
        try:
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info(f"Deleted FAISS index from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to delete FAISS index: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            'total_documents': len(self.metadata),
            'dimension': self.dimension,
            'index_type': 'IndexFlatL2',
            'index_path': str(self.index_path)
        }
    
    def __len__(self) -> int:
        """Return number of documents in index."""
        return len(self.metadata)


if __name__ == '__main__':
    # Example usage
    print("Testing FAISS Vector Store...")
    
    # Create store
    store = FAISSVectorStore(dimension=384)
    
    # Add some dummy documents
    embeddings = [
        [0.1] * 384,
        [0.2] * 384,
        [0.3] * 384,
    ]
    documents = [
        {'text': 'Document 1', 'source': 'test.pdf', 'page': 1},
        {'text': 'Document 2', 'source': 'test.pdf', 'page': 2},
        {'text': 'Document 3', 'source': 'test.pdf', 'page': 3},
    ]
    
    added = store.add_documents(embeddings, documents)
    print(f"Added {added} documents")
    
    # Search
    query_emb = [0.15] * 384
    results = store.search(query_emb, top_k=2)
    print(f"Search results: {len(results)}")
    for r in results:
        print(f"  - {r['text']} (score: {r['score']:.4f})")
    
    # Save
    store.save()
    print(f"Saved to {store.index_path}")
    
    # Stats
    print(f"Stats: {store.get_stats()}")

# Made with Bob
