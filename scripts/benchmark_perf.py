#!/usr/bin/env python3
"""Benchmark script to measure search performance improvements.

This script compares brute-force vs FAISS ANN search performance
at different dataset sizes.
"""

import time
import statistics
import tempfile
from pathlib import Path

import numpy as np

from icommand.vector_index import VectorIndex, EMBEDDING_DIM, is_faiss_available


def generate_synthetic_embeddings(n: int, dim: int = EMBEDDING_DIM) -> tuple[np.ndarray, list[int]]:
    """Generate random normalized embeddings for testing."""
    vectors = np.random.randn(n, dim).astype(np.float32)
    # Normalize
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ids = list(range(1, n + 1))
    return vectors, ids


def benchmark_brute_force(vectors: np.ndarray, query: np.ndarray, k: int = 10) -> float:
    """Benchmark brute-force cosine similarity search."""
    start = time.perf_counter()
    
    # Compute cosine similarities
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(vectors, axis=1)
    denominators = matrix_norms * query_norm
    denominators = np.where(denominators == 0, 1e-10, denominators)
    similarities = np.dot(vectors, query) / denominators
    
    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:k]
    
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_faiss(index: VectorIndex, query: np.ndarray, k: int = 10) -> float:
    """Benchmark FAISS ANN search."""
    start = time.perf_counter()
    results = index.search(query, k=k)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_at_scale(n_vectors: int, n_queries: int = 100) -> dict:
    """Benchmark both methods at a given scale."""
    print(f"\nBenchmarking with {n_vectors:,} vectors...")
    
    # Generate synthetic data
    vectors, ids = generate_synthetic_embeddings(n_vectors)
    commands = [{"id": i, "embedding": vectors[idx]} for idx, i in enumerate(ids)]
    
    # Build FAISS index
    with tempfile.TemporaryDirectory() as tmpdir:
        index = VectorIndex()
        index._icommand_dir = Path(tmpdir)
        index._index_path = index._icommand_dir / "vectors.faiss"
        index._idmap_path = index._icommand_dir / "vectors_idmap.pkl"
        index._metadata_path = index._icommand_dir / "vectors_metadata.pkl"
        
        build_start = time.perf_counter()
        index.build(commands)
        build_time = time.perf_counter() - build_start
        
        # Generate random queries
        queries = [np.random.randn(EMBEDDING_DIM).astype(np.float32) for _ in range(n_queries)]
        for q in queries:
            q /= np.linalg.norm(q)
        
        # Benchmark brute-force
        bf_times = []
        for query in queries:
            t = benchmark_brute_force(vectors, query)
            bf_times.append(t)
        
        # Benchmark FAISS
        faiss_times = []
        if is_faiss_available():
            for query in queries:
                t = benchmark_faiss(index, query)
                faiss_times.append(t)
        
        # Calculate memory usage (approximate)
        raw_memory_mb = vectors.nbytes / (1024 * 1024)
        index_memory_mb = index._index_path.stat().st_size / (1024 * 1024) if index._index_path.exists() else 0
        
        return {
            "n_vectors": n_vectors,
            "build_time": build_time,
            "brute_force_p50": statistics.median(bf_times),
            "brute_force_p99": np.percentile(bf_times, 99),
            "faiss_p50": statistics.median(faiss_times) if faiss_times else 0,
            "faiss_p99": np.percentile(faiss_times, 99) if faiss_times else 0,
            "raw_memory_mb": raw_memory_mb,
            "index_memory_mb": index_memory_mb,
            "speedup": statistics.median(bf_times) / statistics.median(faiss_times) if faiss_times else 0,
            "memory_reduction": raw_memory_mb / index_memory_mb if index_memory_mb > 0 else 0,
        }


def main():
    print("=" * 70)
    print("iCommand Search Performance Benchmark")
    print("=" * 70)
    print(f"FAISS available: {is_faiss_available()}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    
    # Test at different scales
    scales = [1_000, 10_000, 50_000]
    
    results = []
    for n in scales:
        result = benchmark_at_scale(n, n_queries=100)
        results.append(result)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Vectors':>12} {'Method':>15} {'p50 (ms)':>12} {'p99 (ms)':>12} {'Memory (MB)':>12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['n_vectors']:>12,} {'Brute-force':>15} {r['brute_force_p50']*1000:>12.2f} {r['brute_force_p99']*1000:>12.2f} {r['raw_memory_mb']:>12.1f}")
        if is_faiss_available():
            print(f"{'':>12} {'FAISS ANN':>15} {r['faiss_p50']*1000:>12.2f} {r['faiss_p99']*1000:>12.2f} {r['index_memory_mb']:>12.1f}")
            print(f"{'':>12} {'Improvement':>15} {r['speedup']:>11.1f}x {r['memory_reduction']:>11.1f}x")
        print()
    
    # Project to 1M vectors
    if results:
        last = results[-1]
        print("-" * 70)
        print("Projection to 1,000,000 vectors:")
        print(f"  Brute-force search: ~{last['brute_force_p50'] * 1000 * (1_000_000 / last['n_vectors']):.0f} ms (estimated)")
        print(f"  Brute-force memory: ~{last['raw_memory_mb'] * (1_000_000 / last['n_vectors']):.0f} MB")
        if is_faiss_available():
            print(f"  FAISS ANN search:   ~{last['faiss_p50'] * 1000:.2f} ms (sub-linear scaling)")
            print(f"  FAISS memory:       ~{last['index_memory_mb'] * (1_000_000 / last['n_vectors']):.0f} MB (linear scaling)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
