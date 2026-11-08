import os
import sys
import time
from pathlib import Path

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

def main():
    doc_path = Path("/app/documents/Kurikulum-TRPL.pdf")
    if not doc_path.exists():
        print(f"File not found: {doc_path}")
        return

    print("=== MEASURING INGESTION ON Kurikulum-TRPL.pdf ===")
    print(f"File: {doc_path.name}")
    
    docs = PyPDFLoader(str(doc_path)).load()
    print(f"Loaded {len(docs)} pages")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    n_chunks = len(chunks)
    print(f"Created {n_chunks} chunks")

    texts = [f"passage: {c.page_content}" for c in chunks]
    
    cache_folder = str(config.EMBEDDING_MODEL_CACHE_DIR) if config.EMBEDDING_MODEL_CACHE_DIR else None

    # Test with batch_size=32
    print("\n=== Testing batch_size=32 ===")
    t0 = time.time()
    emb32 = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_folder=cache_folder,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    _ = emb32.embed_query("query: test warmup")
    load_time_32 = time.time() - t0
    print(f"Model load & warmup: {load_time_32:.2f}s")
    
    t0 = time.time()
    _ = emb32.embed_documents(texts)
    batch_time_32 = time.time() - t0
    s_per_chunk_32 = batch_time_32 / n_chunks
    print(f"Batch 32 total: {batch_time_32:.2f}s, per_chunk: {s_per_chunk_32:.2f}s")

    # Test with batch_size=16
    print("\n=== Testing batch_size=16 ===")
    t0 = time.time()
    emb16 = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_folder=cache_folder,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    _ = emb16.embed_query("query: test warmup")
    load_time_16 = time.time() - t0
    print(f"Model load & warmup: {load_time_16:.2f}s")
    
    t0 = time.time()
    _ = emb16.embed_documents(texts)
    batch_time_16 = time.time() - t0
    s_per_chunk_16 = batch_time_16 / n_chunks
    print(f"Batch 16 total: {batch_time_16:.2f}s, per_chunk: {s_per_chunk_16:.2f}s")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"n_chunks: {n_chunks}")
    print(f"batch_32: load_s={load_time_32:.2f}, batched_s_per_chunk={s_per_chunk_32:.2f}")
    print(f"batch_16: load_s={load_time_16:.2f}, batched_s_per_chunk={s_per_chunk_16:.2f}")

if __name__ == "__main__":
    main()
