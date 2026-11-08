import os
import sys
import time
from pathlib import Path

sys.path.insert(0, '/app')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumina.settings")
import django
django.setup()

import torch
torch.set_num_threads(4)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import config

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
print(f"Loaded {len(docs)} pages, created {n_chunks} chunks")

cache_folder = str(config.EMBEDDING_MODEL_CACHE_DIR) if config.EMBEDDING_MODEL_CACHE_DIR else None

# Test batch_size=32
print("\n--- Testing batch_size=32 ---", flush=True)
t0 = time.time()
emb32 = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    cache_folder=cache_folder,
    model_kwargs={"device": config.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)
_ = emb32.embed_query("query: warmup")
load32 = time.time() - t0
print(f"Load & warmup: {load32:.2f}s", flush=True)

t0 = time.time()
_ = emb32.embed_documents(texts[:2])
one32 = (time.time() - t0) / 2
print(f"One-by-one (2 chunks): {one32:.2f}s/chunk", flush=True)

t0 = time.time()
_ = emb32.embed_documents(texts)
batch32 = (time.time() - t0) / n_chunks
print(f"Batch 32 total: {batch32*n_chunks:.2f}s, per_chunk: {batch32:.2f}s", flush=True)

# Test batch_size=16
print("\n--- Testing batch_size=16 ---", flush=True)
t0 = time.time()
emb16 = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    cache_folder=cache_folder,
    model_kwargs={"device": config.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
)
_ = emb16.embed_query("query: warmup")
load16 = time.time() - t0
print(f"Load & warmup: {load16:.2f}s", flush=True)

t0 = time.time()
_ = emb16.embed_documents(texts[:2])
one16 = (time.time() - t0) / 2
print(f"One-by-one (2 chunks): {one16:.2f}s/chunk", flush=True)

t0 = time.time()
_ = emb16.embed_documents(texts)
batch16 = (time.time() - t0) / n_chunks
print(f"Batch 16 total: {batch16*n_chunks:.2f}s, per_chunk: {batch16:.2f}s", flush=True)

print("\n=== SUMMARY ===", flush=True)
print(f"n_chunks: {n_chunks}", flush=True)
print(f"load_s: {load32:.2f}", flush=True)
print(f"one_by_one_s_per_chunk: {one32:.2f}", flush=True)
print(f"batched_s_per_chunk: {batch32:.2f}", flush=True)
print(f"---", flush=True)
print(f"batch_size=16", flush=True)
print(f"load_s: {load16:.2f}", flush=True)
print(f"one_by_one_s_per_chunk: {one16:.2f}", flush=True)
print(f"batched_s_per_chunk: {batch16:.2f}", flush=True)
