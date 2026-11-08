import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumina.settings")
print("Setting up Django...")
import django
django.setup()
print("Django setup complete.")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import config
from ingest.services import get_embeddings

def main():
    doc_path = Path("/app/documents/Kurikulum-TRPL.pdf")
    print("Loading document...")
    docs = PyPDFLoader(str(doc_path)).load()
    print(f"Loaded {len(docs)} pages.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    n_chunks = len(chunks)
    print(f"Created {n_chunks} chunks.")
    texts = [f"passage: {c.page_content}" for c in chunks]

    print("Getting embeddings...")
    embeddings = get_embeddings()
    print("Embeddings loaded.")

    # Measure 1 chunk one-by-one
    print("Measuring 1 chunk one-by-one:")
    t0 = time.time()
    _ = embeddings.embed_documents([texts[0]])
    one_chunk_time = time.time() - t0
    print(f"1 chunk time: {one_chunk_time:.2f}s")

    # Measure full batch
    print("Measuring full batch:")
    t0 = time.time()
    vectors = embeddings.embed_documents(texts)
    batch_time = time.time() - t0
    print(f"Full batch ({n_chunks} chunks): {batch_time:.2f}s ({batch_time/n_chunks:.2f}s/chunk)")

    print("Finished measure.")

if __name__ == "__main__":
    main()
