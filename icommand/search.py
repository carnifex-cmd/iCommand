"""Search functionality for icommand.

Handles syncing command embeddings to the vector index and semantic search using
FAISS for Approximate Nearest Neighbor search with fallback to numpy brute-force.
"""

from dataclasses import dataclass

import numpy as np

from icommand.config import load_config
from icommand.db import (
    clear_stale_embeddings,
    get_all_embedded_commands,
    get_commands_by_ids,
    get_unembedded_commands,
    mark_embedded,
)
from icommand.embeddings import get_provider
from icommand.vector_index import get_vector_index, is_faiss_available


@dataclass
class SearchResult:
    """A single search result with command metadata and similarity score."""

    command: str
    directory: str
    timestamp: str
    similarity_score: float


# Model name stored alongside embeddings for migration detection.
_EMBEDDING_MODEL = "arctic-xs"


def _brute_force_search(
    query_vec: np.ndarray,
    commands: list[dict],
    max_results: int
) -> list[SearchResult]:
    """Fallback brute-force search using numpy.
    
    Args:
        query_vec: Normalized query embedding vector
        commands: List of commands with embeddings
        max_results: Maximum number of results
        
    Returns:
        List of SearchResult sorted by similarity
    """
    if not commands:
        return []
        
    # Build matrix: (N, embedding_dim)
    matrix = np.stack([cmd["embedding"] for cmd in commands])

    # Vectorized cosine similarity (query is already normalized)
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denominators = matrix_norms * query_norm
    denominators = np.where(denominators == 0, 1e-10, denominators)
    similarities = np.dot(matrix, query_vec) / denominators

    # Get top-N indices sorted by descending similarity
    top_indices = np.argsort(similarities)[::-1][:max_results]

    return [
        SearchResult(
            command=commands[i]["command"],
            directory=commands[i]["directory"] or "",
            timestamp=commands[i]["timestamp"],
            similarity_score=round(float(similarities[i]), 4),
        )
        for i in top_indices
    ]


def _ann_search(
    query_vec: np.ndarray,
    max_results: int
) -> list[SearchResult]:
    """ANN search using FAISS index.
    
    Args:
        query_vec: Normalized query embedding vector
        max_results: Maximum number of results
        
    Returns:
        List of SearchResult sorted by similarity
    """
    index = get_vector_index()
    
    if not index.ensure_loaded():
        return []
        
    # Search the index
    results = index.search(query_vec, k=max_results)
    
    if not results:
        return []
        
    # Get command IDs and scores
    command_ids = [cmd_id for cmd_id, _ in results]
    scores_by_id = {cmd_id: score for cmd_id, score in results}
    
    # Fetch command metadata from database
    commands = get_commands_by_ids(command_ids)
    
    # Build SearchResult objects preserving order
    search_results = []
    for cmd_id in command_ids:
        cmd = next((c for c in commands if c['id'] == cmd_id), None)
        if cmd:
            search_results.append(SearchResult(
                command=cmd['command'],
                directory=cmd.get('directory') or "",
                timestamp=cmd['timestamp'],
                similarity_score=round(scores_by_id[cmd_id], 4),
            ))
            
    return search_results


def sync() -> int:
    """Embed any unembedded commands and update the vector index.
    
    On model change, clears all old embeddings first so they get re-embedded.

    Returns:
        The number of newly synced commands.
    """
    config = load_config()

    # Clear stale embeddings if model has changed (e.g. MiniLM → Arctic XS)
    stale_count = clear_stale_embeddings(_EMBEDDING_MODEL)
    if stale_count > 0:
        # Rebuild index from scratch if model changed
        index = get_vector_index()
        index.clear()

    unembedded = get_unembedded_commands()

    if not unembedded:
        return 0

    provider = get_provider(config.provider)
    texts = [cmd["command"] for cmd in unembedded]
    embeddings = provider.embed_documents(texts)

    ids = [cmd["id"] for cmd in unembedded]
    embedding_arrays = [np.array(e, dtype=np.float32) for e in embeddings]

    mark_embedded(ids, embedding_arrays, _EMBEDDING_MODEL)
    
    # Update the vector index
    index = get_vector_index()
    
    # Prepare commands for index update
    commands_for_index = [
        {"id": cmd_id, "embedding": emb}
        for cmd_id, emb in zip(ids, embedding_arrays)
    ]
    
    if index.exists() and index.ensure_loaded():
        # Incremental update
        index.add_vectors(commands_for_index)
    else:
        # Build from scratch - need all embedded commands
        all_embedded = get_all_embedded_commands()
        index.build(all_embedded)

    return len(unembedded)


def search(query: str, max_results: int = 10) -> list[SearchResult]:
    """Semantically search command history using cosine similarity.

    Uses FAISS ANN search if available, falls back to brute-force numpy search.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult sorted by relevance (most similar first).
    """
    config = load_config()
    provider = get_provider(config.provider)

    # Embed the query
    query_vec = np.array(provider.embed_queries([query])[0], dtype=np.float32)

    # Try FAISS ANN search first
    if is_faiss_available():
        index = get_vector_index()
        if index.ensure_loaded() and index.get_indexed_count() > 0:
            return _ann_search(query_vec, max_results)

    # Fallback to brute-force search
    commands = get_all_embedded_commands()
    if not commands:
        return []
        
    return _brute_force_search(query_vec, commands, max_results)


def conversational_search(query: str, max_results: int = 10) -> SearchResult:
    """Conversational natural language search for commands.

    Args:
        query: Natural language question about a past command.
        max_results: Maximum candidate commands to consider.

    Returns:
        The single best matching SearchResult.
    """
    # TODO: Future conversational search flow:
    # 1. Embed the user query
    # 2. Vector search for top 20 candidate commands
    # 3. Pass candidates + original query to LLMProvider.complete()
    # 4. LLM reasons about which command the user meant and returns best match
    # 5. Return as SearchResult
    raise NotImplementedError("Conversational search coming soon")
