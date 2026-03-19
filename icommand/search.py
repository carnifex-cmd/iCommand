"""Search and sync functionality for icommand."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from icommand.config import load_config
from icommand.db import (
    clear_stale_embeddings,
    get_command_count,
    get_commands_by_ids,
    get_embedded_command_count,
    get_max_command_id,
    get_unembedded_commands_for_hot_window,
    iter_embedded_commands,
    keyword_search,
    mark_embedded,
)
from icommand.embeddings import get_provider, is_provider_implemented
from icommand.maintenance import get_hot_floor_id, get_storage_usage_bytes, run_maintenance
from icommand.vector_index import CURRENT_MODEL, get_vector_index, is_faiss_available

EMBED_BATCH_SIZE = 256
REBUILD_BATCH_SIZE = 5_000
KEYWORD_RESULT_FETCH_LIMIT = 1000
SEMANTIC_FILL_MULTIPLIER = 3


class SearchCache:
    """LRU cache with TTL for search results."""

    def __init__(self, maxsize: int = 100, ttl_seconds: float = 60.0):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[list, float]] = {}
        self._access_order: list[str] = []

    def _make_key(self, query: str, max_results: int) -> str:
        return hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()

    def get(self, query: str, max_results: int) -> Optional[list]:
        key = self._make_key(query, max_results)
        if key not in self._cache:
            return None

        results, timestamp = self._cache[key]
        if time.time() - timestamp >= self.ttl:
            del self._cache[key]
            self._access_order.remove(key)
            return None

        self._access_order.remove(key)
        self._access_order.append(key)
        return results

    def set(self, query: str, max_results: int, results: list) -> None:
        key = self._make_key(query, max_results)
        if len(self._cache) >= self.maxsize and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = (results, time.time())
        if key not in self._access_order:
            self._access_order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()


_search_cache = SearchCache(maxsize=100, ttl_seconds=60.0)


def invalidate_search_cache() -> None:
    """Invalidate cached search results."""
    _search_cache.clear()


@dataclass
class SearchResult:
    """A single search result with command metadata and similarity score."""

    command: str
    directory: str
    timestamp: str
    similarity_score: float
    command_id: Optional[int] = None


@dataclass
class SyncResult:
    """Result of a sync/maintenance pass."""

    synced_commands: int
    retained_commands: int
    indexed_commands: int
    storage_usage_bytes: int
    embedding_paused: bool
    show_prune_notice: bool = False
    messages: list[str] = field(default_factory=list)


@dataclass
class SearchOutcome:
    """Search results plus any non-fatal notices shown to the user."""

    results: list[SearchResult]
    messages: list[str] = field(default_factory=list)


def _semantic_unavailable_message(provider_name: str, detail: Optional[str] = None) -> str:
    """Build a short user-facing semantic-search fallback message."""
    if detail:
        return (
            f"semantic search unavailable for provider '{provider_name}'; "
            f"using keyword matches only ({detail})"
        )
    return (
        f"semantic search unavailable for provider '{provider_name}'; "
        "using keyword matches only"
    )


def _rebuild_index(hot_floor_id: int) -> int:
    """Rebuild the FAISS index from the DB-backed hot window."""
    if not is_faiss_available():
        return 0

    index = get_vector_index()
    embedded_count = get_embedded_command_count(min_id=hot_floor_id)
    return index.build_from_batches(
        total_vectors=embedded_count,
        batches_factory=lambda: iter_embedded_commands(
            min_id=hot_floor_id,
            batch_size=REBUILD_BATCH_SIZE,
        ),
        hot_floor_id=hot_floor_id,
        model=CURRENT_MODEL,
    )


def _ensure_index_ready(hot_floor_id: int) -> int:
    """Ensure the FAISS index matches the current semantic hot window."""
    if not is_faiss_available():
        return 0

    index = get_vector_index()
    if index.needs_rebuild(
        expected_hot_floor_id=hot_floor_id,
        expected_model=CURRENT_MODEL,
    ):
        return _rebuild_index(hot_floor_id)

    if not index.ensure_loaded(
        expected_hot_floor_id=hot_floor_id,
        expected_model=CURRENT_MODEL,
    ):
        return _rebuild_index(hot_floor_id)

    return index.get_indexed_count()


def _ann_search(query_vec: np.ndarray, max_results: int, hot_floor_id: int) -> list[SearchResult]:
    """Search the current FAISS index."""
    index = get_vector_index()
    if not index.ensure_loaded(
        expected_hot_floor_id=hot_floor_id,
        expected_model=CURRENT_MODEL,
    ):
        return []

    raw_results = index.search(query_vec, k=max_results)
    if not raw_results:
        return []

    command_ids = [command_id for command_id, _ in raw_results]
    scores_by_id = {command_id: score for command_id, score in raw_results}
    commands = get_commands_by_ids(command_ids)
    commands_by_id = {command["id"]: command for command in commands}

    search_results = []
    for command_id in command_ids:
        command = commands_by_id.get(command_id)
        if not command:
            continue
        search_results.append(
            SearchResult(
                command=command["command"],
                directory=command.get("directory") or "",
                timestamp=command["timestamp"],
                similarity_score=round(scores_by_id[command_id], 4),
                command_id=command_id,
            )
        )
    return search_results


def _extract_search_terms(text: str) -> list[str]:
    """Tokenize text similarly to FTS-style term matching."""
    return list(dict.fromkeys(re.findall(r"[A-Za-z0-9_]+", text.lower())))


def _is_keyword_only_query(query: str) -> bool:
    """Treat very short queries as literal shell lookups instead of semantic search."""
    raw_terms = [term for term in query.strip().split() if term]
    if not raw_terms:
        return False
    if len(raw_terms) == 1:
        return len(raw_terms[0]) <= 3
    return all(len(term) <= 3 for term in raw_terms)


def _keyword_rank_key(command: dict, query_terms: list[str]) -> tuple[int, int, int, int]:
    """Rank commands so exact term matches beat prefix matches, then favor recency."""
    command_tokens = set(_extract_search_terms(command["command"]))
    exact_matches = sum(1 for term in query_terms if term in command_tokens)
    prefix_matches = sum(
        1
        for term in query_terms
        if term not in command_tokens
        and any(token.startswith(term) for token in command_tokens)
    )
    all_exact = int(exact_matches == len(query_terms) and len(query_terms) > 0)
    return (
        all_exact,
        exact_matches,
        prefix_matches,
        command["id"],
    )


def _build_keyword_results(query: str, max_results: int) -> list[SearchResult]:
    """Return exact/prefix keyword matches ranked ahead of semantic results."""
    query_terms = _extract_search_terms(query)
    if not query_terms:
        return []

    keyword_ids = keyword_search(
        query,
        limit=max(KEYWORD_RESULT_FETCH_LIMIT, max_results * 50),
    )
    if not keyword_ids:
        return []

    commands = get_commands_by_ids(keyword_ids)
    ranked_commands = sorted(
        commands,
        key=lambda command: _keyword_rank_key(command, query_terms),
        reverse=True,
    )
    return [
        SearchResult(
            command=command["command"],
            directory=command.get("directory") or "",
            timestamp=command["timestamp"],
            similarity_score=1.0,
            command_id=command["id"],
        )
        for command in ranked_commands[:max_results]
    ]


def _merge_results(
    primary_results: list[SearchResult],
    secondary_results: list[SearchResult],
    max_results: int,
) -> list[SearchResult]:
    """Append secondary results without duplicating commands already present."""
    merged = list(primary_results)
    seen_ids = {result.command_id for result in primary_results if result.command_id is not None}
    seen_fallback = {
        (result.command, result.directory, result.timestamp)
        for result in primary_results
        if result.command_id is None
    }

    for result in secondary_results:
        if len(merged) >= max_results:
            break
        if result.command_id is not None:
            if result.command_id in seen_ids:
                continue
            seen_ids.add(result.command_id)
        else:
            dedupe_key = (result.command, result.directory, result.timestamp)
            if dedupe_key in seen_fallback:
                continue
            seen_fallback.add(dedupe_key)
        merged.append(result)

    return merged


def sync() -> SyncResult:
    """Run maintenance, embed hot-window commands, and keep the index current."""
    config = load_config()
    maintenance = run_maintenance(config, model_name=CURRENT_MODEL)
    semantic_notice: Optional[str] = None

    stale_count = clear_stale_embeddings(CURRENT_MODEL)
    if stale_count > 0:
        get_vector_index().invalidate(
            hot_floor_id=maintenance.hot_floor_id,
            model=CURRENT_MODEL,
            delete_index=True,
        )
        maintenance.messages.append(
            "embedding model changed; rebuilding the semantic index for recent history"
        )

    synced_commands = 0
    index = get_vector_index()
    can_incrementally_update = (
        is_faiss_available()
        and stale_count == 0
        and not maintenance.rebuild_required
        and index.ensure_loaded(
            expected_hot_floor_id=maintenance.hot_floor_id,
            expected_model=CURRENT_MODEL,
        )
    )
    rebuild_needed = maintenance.rebuild_required or stale_count > 0 or not can_incrementally_update
    rebuild_marked = maintenance.rebuild_required or stale_count > 0
    provider = None

    if not is_provider_implemented(config.provider):
        semantic_notice = _semantic_unavailable_message(
            config.provider,
            "provider is not implemented",
        )

    while semantic_notice is None and not maintenance.embedding_paused:
        batch = get_unembedded_commands_for_hot_window(
            maintenance.hot_floor_id,
            EMBED_BATCH_SIZE,
        )
        if not batch:
            break

        try:
            if provider is None:
                provider = get_provider(config.provider)

            texts = [command["command"] for command in batch]
            embeddings = [
                np.array(embedding, dtype=np.float32)
                for embedding in provider.embed_documents(texts)
            ]
        except Exception as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            semantic_notice = _semantic_unavailable_message(config.provider, detail)
            break

        ids = [command["id"] for command in batch]

        if can_incrementally_update:
            index.prepare_update(
                hot_floor_id=maintenance.hot_floor_id,
                model=CURRENT_MODEL,
            )
        elif is_faiss_available() and not rebuild_marked:
            index.prepare_update(
                hot_floor_id=maintenance.hot_floor_id,
                model=CURRENT_MODEL,
            )
            rebuild_marked = True

        mark_embedded(ids, embeddings, CURRENT_MODEL)
        synced_commands += len(batch)

        if not can_incrementally_update:
            rebuild_needed = True
            continue

        commands_for_index = [
            {"id": command_id, "embedding": embedding}
            for command_id, embedding in zip(ids, embeddings)
        ]
        try:
            index.add_vectors(
                commands_for_index,
                hot_floor_id=maintenance.hot_floor_id,
                model=CURRENT_MODEL,
            )
        except Exception:
            can_incrementally_update = False
            rebuild_needed = True
            index.invalidate(
                hot_floor_id=maintenance.hot_floor_id,
                model=CURRENT_MODEL,
                delete_index=True,
            )

    indexed_commands = 0
    if semantic_notice is not None:
        maintenance.messages.append(semantic_notice)
    elif is_faiss_available():
        if rebuild_needed:
            indexed_commands = _rebuild_index(maintenance.hot_floor_id)
        else:
            indexed_commands = _ensure_index_ready(maintenance.hot_floor_id)

    if (
        synced_commands > 0
        or stale_count > 0
        or maintenance.pruned_rows > 0
        or maintenance.cold_embeddings_cleared > 0
        or semantic_notice is not None
    ):
        invalidate_search_cache()

    return SyncResult(
        synced_commands=synced_commands,
        retained_commands=get_command_count(),
        indexed_commands=indexed_commands,
        storage_usage_bytes=get_storage_usage_bytes(),
        embedding_paused=maintenance.embedding_paused,
        show_prune_notice=maintenance.show_prune_notice,
        messages=list(maintenance.messages),
    )


def search_with_messages(
    query: str,
    max_results: int = 10,
    use_cache: bool = True,
) -> SearchOutcome:
    """Search command history and return any non-fatal fallback notices."""
    normalized_query = query.strip()
    if not normalized_query:
        return SearchOutcome(results=[])

    config = load_config()
    semantic_enabled = is_provider_implemented(config.provider)

    if use_cache and semantic_enabled:
        cached = _search_cache.get(normalized_query, max_results)
        if cached is not None:
            return SearchOutcome(results=cached)

    keyword_results = _build_keyword_results(normalized_query, max_results)
    if _is_keyword_only_query(normalized_query):
        if use_cache and semantic_enabled:
            _search_cache.set(normalized_query, max_results, keyword_results)
        return SearchOutcome(results=keyword_results)
    if len(keyword_results) >= max_results:
        if use_cache and semantic_enabled:
            _search_cache.set(normalized_query, max_results, keyword_results)
        return SearchOutcome(results=keyword_results)

    if not semantic_enabled:
        return SearchOutcome(
            results=keyword_results,
            messages=[
                _semantic_unavailable_message(
                    config.provider,
                    "provider is not implemented",
                )
            ],
        )

    hot_floor_id = get_hot_floor_id(
        get_max_command_id(),
        config.semantic_command_limit,
    )

    semantic_results: list[SearchResult]
    messages: list[str] = []
    if is_faiss_available():
        indexed_count = _ensure_index_ready(hot_floor_id)
        if indexed_count > 0:
            try:
                provider = get_provider(config.provider)
                query_vec = np.array(
                    provider.embed_queries([normalized_query])[0],
                    dtype=np.float32,
                )
                semantic_results = _ann_search(
                    query_vec,
                    max_results * SEMANTIC_FILL_MULTIPLIER,
                    hot_floor_id,
                )
            except Exception as exc:
                detail = str(exc).strip() or exc.__class__.__name__
                messages.append(_semantic_unavailable_message(config.provider, detail))
                semantic_results = []
        else:
            semantic_results = []
    else:
        semantic_results = []

    if keyword_results:
        results = _merge_results(keyword_results, semantic_results, max_results)
    elif semantic_results:
        results = semantic_results[:max_results]
    else:
        results = []

    if use_cache and not messages:
        _search_cache.set(normalized_query, max_results, results)
    return SearchOutcome(results=results, messages=messages)


def search(query: str, max_results: int = 10, use_cache: bool = True) -> list[SearchResult]:
    """Search command history using exact-first hybrid ranking."""
    return search_with_messages(query, max_results=max_results, use_cache=use_cache).results


def conversational_search(query: str, max_results: int = 10) -> SearchResult:
    """Conversational natural language search for commands."""
    raise NotImplementedError("Conversational search coming soon")
