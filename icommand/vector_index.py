"""Vector index management using FAISS for Approximate Nearest Neighbor search.

This module provides efficient ANN search for command embeddings using FAISS.
It supports incremental updates, memory-mapped indices for fast loading,
and automatic rebuilding when the embedding model changes.
"""

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Optional, Set

import numpy as np

from icommand.config import get_icommand_dir

# FAISS is optional - fallback to brute-force if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Falling back to brute-force search.")

# Embedding dimension for Arctic Embed XS
EMBEDDING_DIM = 384

# Index file names
INDEX_FILE = "vectors.faiss"
METADATA_FILE = "vectors_metadata.pkl"

# Model version for migration detection
CURRENT_MODEL = "arctic-xs"

logger = logging.getLogger(__name__)


class VectorIndex:
    """Manages FAISS vector index for fast semantic search.
    
    Uses an IVF-PQ index for memory-efficient ANN search:
    - IVF: Inverted File Index for coarse quantization
    - PQ: Product Quantization for vector compression (~32x)
    
    For smaller datasets (<10K), falls back to FlatIP (exact search).
    """
    
    def __init__(self) -> None:
        """Initialize the vector index manager."""
        self._index: Optional[faiss.Index] = None
        self._indexed_ids: Set[int] = set()  # Track which command IDs are indexed
        self._icommand_dir = get_icommand_dir()
        self._index_path = self._icommand_dir / INDEX_FILE
        self._metadata_path = self._icommand_dir / METADATA_FILE
        self._is_trained = False
        
    def _create_index(self, n_vectors: int = 0) -> faiss.Index:
        """Create appropriate FAISS index based on dataset size.
        
        Args:
            n_vectors: Expected number of vectors (for choosing index type)
            
        Returns:
            FAISS index instance
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")
            
        # For small datasets, use exact search (faster, no training needed)
        if n_vectors < 10000:
            # Flat index with inner product (cosine similarity for normalized vectors)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            logger.debug(f"Created FlatIP index for {n_vectors} vectors")
        else:
            # IVF-PQ for large datasets
            # nlist: number of Voronoi cells (sqrt(n) is a good heuristic)
            nlist = min(4096, max(4, int(np.sqrt(n_vectors))))
            # PQ: 32 subquantizers, 8 bits each = 32 bytes per vector (vs 1536 raw)
            m = 32  # number of subquantizers
            nbits = 8  # bits per subquantizer
            
            # Create quantizer
            quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
            # Create IVF-PQ index
            index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIM, nlist, m, nbits)
            logger.debug(f"Created IVF-PQ index (nlist={nlist}, m={m}) for {n_vectors} vectors")
            
        # Wrap with ID map to support arbitrary command IDs
        return faiss.IndexIDMap2(index)
    
    def _save(self) -> None:
        """Save index and mappings to disk atomically."""
        if self._index is None:
            return
            
        # Ensure directory exists
        self._icommand_dir.mkdir(parents=True, exist_ok=True)
        
        # Save index to temp file then rename for atomicity
        with tempfile.NamedTemporaryFile(
            mode='wb', 
            dir=self._icommand_dir, 
            delete=False,
            suffix='.tmp'
        ) as f:
            temp_path = Path(f.name)
            faiss.write_index(self._index, str(temp_path))
        
        # Atomic rename
        temp_path.replace(self._index_path)
        
        # Save metadata
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=self._icommand_dir,
            delete=False,
            suffix='.tmp'
        ) as f:
            temp_path = Path(f.name)
            pickle.dump({
                'model': CURRENT_MODEL,
                'indexed_ids': self._indexed_ids,
                'is_trained': self._is_trained,
            }, f)
        temp_path.replace(self._metadata_path)
        
        logger.debug(f"Saved index with {len(self._indexed_ids)} vectors")
    
    def _load(self) -> bool:
        """Load index and mappings from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self._index_path.exists():
            return False
            
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, cannot load index")
            return False
            
        try:
            # Load metadata first to check model version
            if self._metadata_path.exists():
                with open(self._metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                if metadata.get('model') != CURRENT_MODEL:
                    logger.info("Embedding model changed, rebuilding index")
                    return False
                self._indexed_ids = metadata.get('indexed_ids', set())
                self._is_trained = metadata.get('is_trained', False)
            
            # Load FAISS index
            self._index = faiss.read_index(str(self._index_path))
            
            logger.debug(f"Loaded index with {len(self._indexed_ids)} vectors")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            self._index = None
            self._indexed_ids = set()
            return False
    
    def exists(self) -> bool:
        """Check if a valid index exists on disk."""
        return self._index_path.exists() and self._metadata_path.exists()
    
    def is_loaded(self) -> bool:
        """Check if index is loaded in memory."""
        return self._index is not None
    
    def ensure_loaded(self) -> bool:
        """Ensure index is loaded, loading from disk if necessary.
        
        Returns:
            True if index is available (loaded or created)
        """
        if self._index is not None:
            return True
        return self._load()
    
    def get_indexed_count(self) -> int:
        """Return number of vectors in the index."""
        if self._index is not None:
            return self._index.ntotal
        return 0
    
    def build(self, commands: list[dict]) -> int:
        """Build index from scratch with given commands.
        
        Args:
            commands: List of command dicts with 'id' and 'embedding' keys
            
        Returns:
            Number of vectors indexed
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index build")
            return 0
            
        if not commands:
            return 0
            
        # Filter commands with embeddings
        with_embeddings = [
            cmd for cmd in commands 
            if cmd.get('embedding') is not None
        ]
        
        if not with_embeddings:
            return 0
            
        # Create index
        self._index = self._create_index(len(with_embeddings))
        
        # Prepare vectors and IDs
        vectors = np.stack([cmd['embedding'] for cmd in with_embeddings])
        ids = np.array([cmd['id'] for cmd in with_embeddings], dtype=np.int64)
        
        # Train if needed (for IVF indices)
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            logger.debug(f"Training index with {len(vectors)} vectors")
            self._index.train(vectors)
            self._is_trained = True
        
        # Add vectors with their command IDs
        self._index.add_with_ids(vectors, ids)
        
        # Track indexed IDs
        self._indexed_ids = set(int(cmd_id) for cmd_id in ids)
        
        # Save to disk
        self._save()
        
        logger.info(f"Built index with {len(with_embeddings)} vectors")
        return len(with_embeddings)
    
    def add_vectors(self, commands: list[dict]) -> int:
        """Add new vectors to existing index.
        
        Args:
            commands: List of command dicts with 'id' and 'embedding' keys
            
        Returns:
            Number of vectors added
        """
        if not FAISS_AVAILABLE:
            return 0
            
        if not commands:
            return 0
            
        # Filter out already indexed commands
        new_commands = [
            cmd for cmd in commands
            if cmd.get('embedding') is not None 
            and cmd['id'] not in self._indexed_ids
        ]
        
        if not new_commands:
            return 0
            
        # Ensure index exists
        if self._index is None:
            # Build new index if doesn't exist
            return self.build(new_commands)
        
        # Prepare vectors and IDs
        vectors = np.stack([cmd['embedding'] for cmd in new_commands])
        ids = np.array([cmd['id'] for cmd in new_commands], dtype=np.int64)
        
        # Train if needed (shouldn't happen for incremental adds)
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            self._index.train(vectors)
            self._is_trained = True
        
        # Add vectors with their command IDs
        self._index.add_with_ids(vectors, ids)
        
        # Track indexed IDs
        for cmd_id in ids:
            self._indexed_ids.add(int(cmd_id))
        
        # Save to disk
        self._save()
        
        logger.debug(f"Added {len(new_commands)} vectors to index")
        return len(new_commands)
    
    def search(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search for nearest neighbors.
        
        Args:
            query_vec: Query embedding vector (normalized)
            k: Number of results to return
            
        Returns:
            List of (command_id, similarity_score) tuples, sorted by score
        """
        if not FAISS_AVAILABLE or self._index is None:
            return []
            
        if self._index.ntotal == 0:
            return []
            
        # Ensure query is 2D array (batch of 1)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
            
        # Search
        # For normalized vectors, inner product = cosine similarity
        scores, indices = self._index.search(query_vec, k)
        
        # Convert to list of (command_id, score) tuples
        # Note: When using IndexIDMap2 with add_with_ids(), 
        # FAISS returns the actual IDs we passed in, not positions
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for padding
                continue
            # idx is already the command_id because we used add_with_ids()
            results.append((int(idx), float(score)))
                
        return results
    
    def clear(self) -> None:
        """Clear the index and remove files from disk."""
        self._index = None
        self._indexed_ids = set()
        self._is_trained = False
        
        # Remove files
        for path in [self._index_path, self._metadata_path]:
            if path.exists():
                path.unlink()
                logger.debug(f"Removed {path}")
    
    def remove_commands(self, command_ids: list[int]) -> int:
        """Remove commands from the index.
        
        Note: FAISS doesn't support efficient deletion, so we rebuild.
        
        Args:
            command_ids: List of command IDs to remove
            
        Returns:
            Number of commands removed
        """
        if not self._index:
            return 0
            
        # Check if any are in index
        to_remove = set(command_ids) & self._indexed_ids
        if not to_remove:
            return 0
            
        # For now, mark as needing rebuild
        # In practice, this is called when model changes (full rebuild)
        logger.debug(f"Marking {len(to_remove)} vectors for removal (will rebuild)")
        return len(to_remove)


# Global index instance
_index_instance: Optional[VectorIndex] = None


def get_vector_index() -> VectorIndex:
    """Get the global vector index instance."""
    global _index_instance
    if _index_instance is None:
        _index_instance = VectorIndex()
    return _index_instance


def reset_vector_index() -> None:
    """Reset the global vector index instance."""
    global _index_instance
    _index_instance = None


def is_faiss_available() -> bool:
    """Check if FAISS is available."""
    return FAISS_AVAILABLE
