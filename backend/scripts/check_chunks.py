import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumina.settings")
import django
django.setup()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import config

doc_path = Path("/app/Kurikulum-TRPL.pdf")
docs = PyPDFLoader(str(doc_path)).load()
print(f"Pages: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    add_start_index=True,
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"Chunk {i+1} len: {len(c.page_content)}")
