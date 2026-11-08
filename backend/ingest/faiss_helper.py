"""
faiss_helper.py — FAISS index management utilities.

Provides functions to load, update, and query the global FAISS index.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import faiss
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def load_faiss_index() -> tuple[Any | None, list[int]]:
    """Load the global FAISS index and ID mapping from disk."""
    index_file = config.FAISS_INDEX_PATH
    mapping_file = config.FAISS_MAPPING_PATH

    if not (index_file.exists() and mapping_file.exists()):
        logger.warning("FAISS index or mapping file not found")
        return None, []

    try:
        index = faiss.read_index(str(index_file))
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        logger.info("Loaded FAISS index with %d vectors", index.ntotal)
        return index, mapping
    except Exception as e:
        logger.error("Failed to load FAISS index: %s", e)
        return None, []


def search_faiss(
    index: Any,
    mapping: list[int],
    query_vector: list[float],
    top_k: int,
) -> list[int]:
    """
    Search the FAISS index and return chunk IDs.
    
    Args:
        index: FAISS index object
        mapping: List mapping FAISS positions to chunk IDs
        query_vector: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        List of chunk IDs ordered by similarity (most similar first)
    """
    if index is None or not mapping:
        return []

    query = np.asarray([query_vector], dtype="float32")
    faiss.normalize_L2(query)

    distances, indices = index.search(query, min(top_k, index.ntotal))

    chunk_ids = []
    for idx in indices[0]:
        if 0 <= idx < len(mapping):
            chunk_ids.append(mapping[idx])

    return chunk_ids
