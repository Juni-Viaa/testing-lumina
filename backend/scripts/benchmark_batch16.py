import os
import sys
import time
from pathlib import Path

# Setup Django environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumina.settings")
import django
django.setup()

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import config

def benchmark():
    doc_path = Path("/app/documents/Kurikulum-TRPL.pdf")
    docs = PyPDFLoader(str(doc_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    n_chunks = len(chunks)
    texts = [f"passage: {c.page_content}" for c in chunks]
    print(f"Loaded {len(docs)} pages, created {n_chunks} chunks.")

    cache_folder = str(config.EMBEDDING_MODEL_CACHE_DIR) if config.EMBEDDING_MODEL_CACHE_DIR else None

    # Load model once
    t0 = time.time()
    emb_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_folder=cache_folder,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    # Warmup query
    _ = emb_model.embed_query("query: test warmup")
    load_time = time.time() - t0
    print(f"Model load & warmup time: {load_time:.2f}s")

    # 1. Benchmark batch_size = 16
    print("\n--- Testing batch_size = 16 ---")
    emb_model.encode_kwargs["batch_size"] = 16
    t0 = time.time()
    _ = emb_model.embed_documents(texts)
    t_batch16 = time.time() - t0
    s_per_chunk_16 = t_batch16 / n_chunks
    print(f"Batch 16: total={t_batch16:.2f}s, per_chunk={s_per_chunk_16:.2f}s")

    # Summary
    print("\n=== BENCHMARK COMPLETE ===")
    print(f"n_chunks: {n_chunks}")
    print(f"load_s: {load_time:.2f}")
    print(f"batched_16_s_per_chunk: {s_per_chunk_16:.2f}")

if __name__ == "__main__":
    benchmark()
