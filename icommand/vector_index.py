"""Vector index management using FAISS for semantic search."""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from icommand.config import get_icommand_dir

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Falling back to keyword search.")

EMBEDDING_DIM = 384
INDEX_FILE = "vectors.faiss"
METADATA_FILE = "vectors_metadata.pkl"
CURRENT_MODEL = "arctic-xs"
TRAINING_SAMPLE_SIZE = 50_000

logger = logging.getLogger(__name__)


def _default_metadata(
    *,
    model: str = CURRENT_MODEL,
    hot_floor_id: int = 1,
    rebuild_needed: bool = True,
    indexed_count: int = 0,
    is_trained: bool = False,
) -> dict:
    return {
        "model": model,
        "hot_floor_id": hot_floor_id,
        "rebuild_needed": rebuild_needed,
        "indexed_count": indexed_count,
        "is_trained": is_trained,
    }


class VectorIndex:
    """Manages the FAISS vector index for the semantic hot window."""

    def __init__(self) -> None:
        self._index: Optional[faiss.Index] = None
        self._icommand_dir = get_icommand_dir()
        self._index_path = self._icommand_dir / INDEX_FILE
        self._metadata_path = self._icommand_dir / METADATA_FILE
        self._metadata = _default_metadata()

    def _create_index(self, n_vectors: int) -> faiss.Index:
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available")

        if n_vectors < 10_000:
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
        else:
            nlist = min(4096, max(4, int(np.sqrt(n_vectors))))
            quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
            index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIM, nlist, 32, 8)
        return faiss.IndexIDMap2(index)

    def _read_metadata(self) -> dict:
        if not self._metadata_path.exists():
            return _default_metadata()

        try:
            with open(self._metadata_path, "rb") as handle:
                metadata = pickle.load(handle)
        except Exception as exc:
            logger.warning("Failed to read vector metadata: %s", exc)
            return _default_metadata()

        return {
            "model": metadata.get("model", CURRENT_MODEL),
            "hot_floor_id": int(metadata.get("hot_floor_id", 1)),
            "rebuild_needed": bool(metadata.get("rebuild_needed", False)),
            "indexed_count": int(metadata.get("indexed_count", 0)),
            "is_trained": bool(metadata.get("is_trained", False)),
        }

    def _write_metadata(self, metadata: dict) -> None:
        self._icommand_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self._icommand_dir,
            delete=False,
            suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            pickle.dump(metadata, handle)
        temp_path.replace(self._metadata_path)
        self._metadata = metadata

    def _save_index_and_metadata(self, metadata: dict) -> None:
        if self._index is None:
            self._write_metadata(metadata)
            return

        self._icommand_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self._icommand_dir,
            delete=False,
            suffix=".tmp",
        ) as handle:
            temp_path = Path(handle.name)
            faiss.write_index(self._index, str(temp_path))
        temp_path.replace(self._index_path)
        self._write_metadata(metadata)

    def load_metadata(self) -> dict:
        self._metadata = self._read_metadata()
        return dict(self._metadata)

    def exists(self) -> bool:
        return self._index_path.exists() and self._metadata_path.exists()

    def is_loaded(self) -> bool:
        return self._index is not None

    def get_indexed_count(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        return int(self.load_metadata().get("indexed_count", 0))

    def needs_rebuild(
        self,
        *,
        expected_hot_floor_id: int,
        expected_model: str = CURRENT_MODEL,
    ) -> bool:
        metadata = self.load_metadata()
        if metadata["model"] != expected_model:
            return True
        if metadata["hot_floor_id"] != expected_hot_floor_id:
            return True
        if metadata["rebuild_needed"]:
            return True
        if metadata["indexed_count"] == 0 and not self._index_path.exists():
            return False
        return not self._index_path.exists()

    def ensure_loaded(
        self,
        *,
        expected_hot_floor_id: int,
        expected_model: str = CURRENT_MODEL,
    ) -> bool:
        if not FAISS_AVAILABLE:
            return False

        if self._index is not None:
            metadata = self.load_metadata()
            if (
                metadata["model"] == expected_model
                and metadata["hot_floor_id"] == expected_hot_floor_id
                and not metadata["rebuild_needed"]
            ):
                return True
            self._index = None

        if self.needs_rebuild(
            expected_hot_floor_id=expected_hot_floor_id,
            expected_model=expected_model,
        ):
            return False

        try:
            self._index = faiss.read_index(str(self._index_path))
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            self._index = None
            return False

    def prepare_update(
        self,
        *,
        hot_floor_id: int,
        model: str = CURRENT_MODEL,
    ) -> None:
        metadata = self.load_metadata()
        metadata.update(
            {
                "model": model,
                "hot_floor_id": hot_floor_id,
                "rebuild_needed": True,
                "indexed_count": self.get_indexed_count(),
                "is_trained": metadata.get("is_trained", False),
            }
        )
        self._write_metadata(metadata)

    def invalidate(
        self,
        *,
        hot_floor_id: int,
        model: str = CURRENT_MODEL,
        delete_index: bool = True,
    ) -> None:
        self._index = None
        if delete_index and self._index_path.exists():
            self._index_path.unlink()
        metadata = _default_metadata(
            model=model,
            hot_floor_id=hot_floor_id,
            rebuild_needed=True,
            indexed_count=0,
            is_trained=False,
        )
        self._write_metadata(metadata)

    def build_from_batches(
        self,
        *,
        total_vectors: int,
        batches_factory: Callable[[], Iterator[list[dict]]],
        hot_floor_id: int,
        model: str = CURRENT_MODEL,
    ) -> int:
        if not FAISS_AVAILABLE:
            return 0

        if total_vectors <= 0:
            self.invalidate(hot_floor_id=hot_floor_id, model=model, delete_index=True)
            metadata = _default_metadata(
                model=model,
                hot_floor_id=hot_floor_id,
                rebuild_needed=False,
                indexed_count=0,
                is_trained=False,
            )
            self._save_index_and_metadata(metadata)
            return 0

        self._index = self._create_index(total_vectors)
        is_trained = True

        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            sample_vectors: list[np.ndarray] = []
            sample_count = 0
            for batch in batches_factory():
                for command in batch:
                    sample_vectors.append(command["embedding"])
                    sample_count += 1
                    if sample_count >= min(total_vectors, TRAINING_SAMPLE_SIZE):
                        break
                if sample_count >= min(total_vectors, TRAINING_SAMPLE_SIZE):
                    break

            if sample_vectors:
                training_matrix = np.stack(sample_vectors)
                self._index.train(training_matrix)
            is_trained = True

        indexed_count = 0
        for batch in batches_factory():
            vectors = np.stack([command["embedding"] for command in batch])
            ids = np.array([command["id"] for command in batch], dtype=np.int64)
            self._index.add_with_ids(vectors, ids)
            indexed_count += len(batch)

        metadata = _default_metadata(
            model=model,
            hot_floor_id=hot_floor_id,
            rebuild_needed=False,
            indexed_count=indexed_count,
            is_trained=is_trained,
        )
        self._save_index_and_metadata(metadata)
        logger.info("Built vector index with %s commands", indexed_count)
        return indexed_count

    def add_vectors(
        self,
        commands: list[dict],
        *,
        hot_floor_id: int,
        model: str = CURRENT_MODEL,
    ) -> int:
        if not FAISS_AVAILABLE or not commands:
            return 0

        if not self.ensure_loaded(
            expected_hot_floor_id=hot_floor_id,
            expected_model=model,
        ):
            raise RuntimeError("Cannot incrementally update an out-of-date FAISS index")

        vectors = np.stack([command["embedding"] for command in commands])
        ids = np.array([command["id"] for command in commands], dtype=np.int64)

        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            self._index.train(vectors)

        self._index.add_with_ids(vectors, ids)
        metadata = _default_metadata(
            model=model,
            hot_floor_id=hot_floor_id,
            rebuild_needed=False,
            indexed_count=int(self._index.ntotal),
            is_trained=True,
        )
        self._save_index_and_metadata(metadata)
        return len(commands)

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if not FAISS_AVAILABLE or self._index is None or self._index.ntotal == 0:
            return []

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def clear(self) -> None:
        self._index = None
        for path in (self._index_path, self._metadata_path):
            if path.exists():
                path.unlink()


_index_instance: Optional[VectorIndex] = None


def get_vector_index() -> VectorIndex:
    global _index_instance
    if _index_instance is None:
        _index_instance = VectorIndex()
    return _index_instance


def reset_vector_index() -> None:
    global _index_instance
    _index_instance = None


def is_faiss_available() -> bool:
    return FAISS_AVAILABLE
