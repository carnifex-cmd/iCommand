"""Search functionality for icommand.

Handles syncing command embeddings to SQLite and semantic search using
numpy cosine similarity — no external vector database required.
"""

from dataclasses import dataclass

import numpy as np

from icommand.config import load_config
from icommand.db import get_all_embedded_commands, get_unembedded_commands, mark_embedded
from icommand.embeddings import get_provider


@dataclass
class SearchResult:
    """A single search result with command metadata and similarity score."""

    command: str
    directory: str
    timestamp: str
    similarity_score: float


def sync() -> int:
    """Embed any unembedded commands and store them in SQLite.

    Returns:
        The number of newly synced commands.
    """
    config = load_config()
    unembedded = get_unembedded_commands()

    if not unembedded:
        return 0

    provider = get_provider(config.provider)
    texts = [cmd["command"] for cmd in unembedded]
    embeddings = provider.embed(texts)

    ids = [cmd["id"] for cmd in unembedded]
    embedding_arrays = [np.array(e, dtype=np.float32) for e in embeddings]

    mark_embedded(ids, embedding_arrays)
    return len(unembedded)


def search(query: str, max_results: int = 10) -> list[SearchResult]:
    """Semantically search command history using cosine similarity.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of SearchResult sorted by relevance (most similar first).
    """
    config = load_config()
    provider = get_provider(config.provider)

    # Embed the query
    query_vec = np.array(provider.embed([query])[0], dtype=np.float32)

    # Load all stored embeddings
    commands = get_all_embedded_commands()
    if not commands:
        return []

    # Build matrix: (N, embedding_dim)
    matrix = np.stack([cmd["embedding"] for cmd in commands])

    # Vectorized cosine similarity
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
