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

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import config

def test_batch(batch_size):
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

    cache_folder = str(config.EMBEDDING_MODEL_CACHE_DIR) if config.EMBEDDING_MODEL_CACHE_DIR else None

    print(f"\n--- Starting benchmark for batch_size={batch_size} (chunks={n_chunks}) ---", flush=True)
    t0 = time.time()
    emb = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_folder=cache_folder,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
    _ = emb.embed_query("query: warmup")
    load_time = time.time() - t0
    print(f"Model load & warmup: {load_time:.2f}s", flush=True)

    # Measure one-by-one (batch 1) on sample 2 chunks to estimate one-by-one s per chunk
    t0 = time.time()
    for i in range(2):
        _ = emb.embed_documents([texts[i]])
    one_by_one_s = (time.time() - t0) / 2
    print(f"One-by-one estimation (sample 2 chunks): {one_by_one_s:.2f}s/chunk", flush=True)

    t0 = time.time()
    print("Embedding documents in batch...", flush=True)
    _ = emb.embed_documents(texts)
    batch_total = time.time() - t0
    batched_s_per_chunk = batch_total / n_chunks
    print(f"Batch {batch_size} total: {batch_total:.2f}s ({batched_s_per_chunk:.2f}s/chunk)", flush=True)

    return load_time, one_by_one_s, batched_s_per_chunk, n_chunks

if __name__ == "__main__":
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    test_batch(bs)
