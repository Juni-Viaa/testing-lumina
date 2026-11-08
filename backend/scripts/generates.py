"""
Tugas 4: Generate Answers Batch (retrieval LOKAL)

Arsitektur:
  - Dense retrieval  : FAISS index di folder /vectorstore (dibaca lokal)
  - Sparse retrieval : PostgreSQL full-text search (container Docker, diakses via localhost)
  - Database Django  : PostgreSQL di Docker (Query / Conversation / User)
  - LLM              : Google Gemini (input/output token dicatat dari usage_metadata)
  - Output           : results/answers.csv, ditulis SETIAP SELESAI 1 pertanyaan

Konfigurasi lewat environment variable (semua opsional kecuali API key):
  GEMINI_API_KEY / GOOGLE_API_KEY   API key Gemini
  GEMINI_MODEL                      default: gemini-2.5-flash
  LOCAL_DB_HOST                     default: localhost   (host Postgres Docker dari sisi lokal)
  LOCAL_DB_PORT                     default: (ikut settings) -> isi mis. 5432 / 5433 sesuai port mapping
  VECTORSTORE_DIR                   default: <root proyek>/vectorstore
  FAISS_INDEX_NAME                  default: index  (-> index.faiss + index.pkl)
  EMBEDDING_MODEL                   default: intfloat/multilingual-e5-large
  FTS_TABLE                         default: ingest_chunk
  FTS_CONTENT_COL / FTS_SOURCE_COL / FTS_PAGE_COL   default: content / source / page
  FTS_VECTOR_COL                    (opsional) kolom tsvector yang sudah ada, mis. search_vector
  FTS_CONFIG                        default: simple  (ganti 'indonesian' jika tersedia)
  TOP_K_DENSE / TOP_K_SPARSE / TOP_K_FINAL   default: 10 / 10 / 5
  PAUSE_EVERY / PAUSE_SECONDS       default: 10 / 180 (jeda rate limit; set PAUSE_EVERY=0 untuk mematikan)
  LOG_TO_DB                         default: 1 (catat Query ke DB Django); 0 untuk mematikan
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
backend_dir = BASE_DIR.parent
sys.path.insert(0, str(backend_dir))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumina.settings")
print(f"DEBUG: sys.path[0] = {sys.path[0]}")

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
LOCAL_DB_HOST = os.getenv("LOCAL_DB_HOST", "localhost")
LOCAL_DB_PORT = os.getenv("LOCAL_DB_PORT")

VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", str(BASE_DIR.parent / "vectorstore")))
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

FTS_TABLE = os.getenv("FTS_TABLE", "chunks")
FTS_CONTENT_COL = os.getenv("FTS_CONTENT_COL", "chunk_text")
FTS_SOURCE_COL = os.getenv("FTS_SOURCE_COL", "document_id")
FTS_PAGE_COL = os.getenv("FTS_PAGE_COL", "page")
FTS_VECTOR_COL = os.getenv("FTS_VECTOR_COL", "search_vector")
FTS_CONFIG = os.getenv("FTS_CONFIG", "simple")

TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "10"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "10"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))
RRF_K = 60

PAUSE_EVERY = int(os.getenv("PAUSE_EVERY", "10"))
PAUSE_SECONDS = int(os.getenv("PAUSE_SECONDS", "180"))
LOG_TO_DB = os.getenv("LOG_TO_DB", "1") == "1"
MAX_RETRIES = 3

CSV_FIELDS = [
    "qid", "answer", "cited_sources", "gen_time_s", "total_time_s",
    "input_tokens", "output_tokens", "run_date",
]

# ----------------------------------------------------------------------------
# Django setup (DB tetap di Docker, diakses lewat localhost)
# ----------------------------------------------------------------------------
print("Setting up Django...")
import sys
import os
sys.path.insert(0, str(BASE_DIR.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumina.settings")

# Patch: Hapus semua setting storage yang conflict dari global_settings
from django.conf import global_settings
storage_attrs = ['DEFAULT_FILE_STORAGE', 'STATICFILES_STORAGE']
for attr in storage_attrs:
    if hasattr(global_settings, attr):
        delattr(global_settings, attr)

import django  # noqa: E402
django.setup()
print("Django setup complete.")

from django.conf import settings  # noqa: E402
from django.contrib.auth import get_user_model  # noqa: E402
from django.db import connection, connections  # noqa: E402

# Arahkan koneksi DB ke container Docker dari sisi lokal
# (settings biasanya memakai nama service Docker seperti "db" / "postgres").
_db_cfg = connections.databases["default"]
_db_cfg["HOST"] = LOCAL_DB_HOST
if LOCAL_DB_PORT:
    _db_cfg["PORT"] = LOCAL_DB_PORT
connections["default"].close()

from core.models import Query, Conversation  # noqa: E402

User = get_user_model()

# ----------------------------------------------------------------------------
# Retriever lokal: FAISS (dense) + PostgreSQL FTS (sparse) + RRF
# ----------------------------------------------------------------------------
_embeddings = None
_faiss_store = None


def load_faiss():
    """Load FAISS index & embedding model sekali saja."""
    global _embeddings, _faiss_store
    if _faiss_store is not None:
        return _faiss_store

    import faiss
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    import pickle

    index_file = VECTORSTORE_DIR / f"{FAISS_INDEX_NAME}.faiss"
    if not index_file.exists():
        index_file = VECTORSTORE_DIR / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index tidak ditemukan: {index_file}. "
                f"Set VECTORSTORE_DIR / FAISS_INDEX_NAME dengan benar."
            )

    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print(f"Loading FAISS index from {index_file} ...")
    faiss_index = faiss.read_index(str(index_file))
    
    pkl_file = VECTORSTORE_DIR / "faiss_mapping.json"
    if not pkl_file.exists():
        pkl_file = VECTORSTORE_DIR / "faiss_mapping.json.bak"
    
    print(f"Loading metadata from {pkl_file} ...")
    with open(pkl_file, "r", encoding="utf-8") as f:
        import json
        metadata_list = json.load(f)
    
    documents = []
    for meta in metadata_list:
        if isinstance(meta, dict):
            content = meta.get("page_content", "")
            metadata = meta.get("metadata", {})
        else:
            content = str(meta)
            metadata = {}
        documents.append(Document(page_content=content, metadata=metadata))
    
    from langchain_community.vectorstores import FAISS
    _faiss_store = FAISS.from_documents(documents, _embeddings)
    _faiss_store.index = faiss_index
    
    return _faiss_store


def _doc_key(source, page, content):
    norm = re.sub(r"\s+", " ", (content or "")).strip()[:120]
    return f"{source}|{page}|{norm}"


def dense_search(question, k):
    store = load_faiss()
    # multilingual-e5 wajib memakai prefix "query: "
    results = store.similarity_search_with_score(f"query: {question}", k=k)
    out = []
    for doc, _score in results:
        meta = doc.metadata or {}
        out.append({
            "source": meta.get("source") or meta.get("file_name") or "",
            "page": meta.get("page"),
            "content": doc.page_content,
        })
    return out


def sparse_search(question, k):
    """Full-text search PostgreSQL (di Docker)."""
    from core.models import Chunk, Document
    
    tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    if not tokens:
        return []
    tsquery_text = " | ".join(dict.fromkeys(tokens))

    # Gunakan ORM untuk query dengan join ke Document
    # karena FTS_SOURCE_COL adalah foreign key (document_id)
    qs = Chunk.objects.all()
    
    if FTS_VECTOR_COL:
        # Jika sudah ada search_vector column
        from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
        qs = qs.annotate(search=SearchVector(FTS_VECTOR_COL)).filter(search=tsquery_text)
    else:
        # Gunakan chunk_text
        from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
        qs = qs.annotate(search=SearchVector(FTS_CONTENT_COL)).filter(search=tsquery_text)
    
    results = qs.order_by('-id')[:k]
    
    return [{
        "source": r.document.document_name if hasattr(r, 'document') and r.document else "",
        "page": r.page,
        "content": getattr(r, FTS_CONTENT_COL, "")
    } for r in results]


def hybrid_retrieve(question):
    """Gabungkan dense + sparse dengan Reciprocal Rank Fusion."""
    dense = dense_search(question, TOP_K_DENSE)
    sparse = sparse_search(question, TOP_K_SPARSE)

    fused = {}
    for ranked in (dense, sparse):
        for rank, item in enumerate(ranked, start=1):
            key = _doc_key(item["source"], item["page"], item["content"])
            entry = fused.setdefault(key, {"item": item, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank)

    ordered = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    return [e["item"] for e in ordered[:TOP_K_FINAL]]


# ----------------------------------------------------------------------------
# Generasi jawaban dengan Gemini + pencatatan token
# ----------------------------------------------------------------------------
_gemini_client = None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or getattr(settings, "GEMINI_API_KEY", None)
            or getattr(settings, "GOOGLE_API_KEY", None)
        )
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY belum di-set.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


SYSTEM_INSTRUCTION = (
    "Kamu adalah asisten akademik. Jawab pertanyaan HANYA berdasarkan konteks dokumen yang diberikan. "
    "Jika informasi tidak ada di konteks, katakan bahwa informasi tidak ditemukan. "
    "Setiap klaim wajib diberi sitasi dengan format persis: **nama_dokumen, hal. N**. "
    "Jawab dalam Bahasa Indonesia yang jelas dan ringkas."
)


def build_prompt(question, chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        page = c["page"]
        try:
            page = int(page)
        except (TypeError, ValueError):
            pass
        blocks.append(f"[Sumber {i}] {c['source']}, hal. {page}\n{c['content']}")
    context = "\n\n".join(blocks) if blocks else "(tidak ada konteks ditemukan)"
    return f"KONTEKS:\n{context}\n\nPERTANYAAN:\n{question}\n\nJAWABAN:"


def generate_with_gemini(prompt):
    """Return (answer_text, input_tokens, output_tokens)."""
    from google.genai import types

    client = get_gemini_client()
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                ),
            )
            usage = getattr(resp, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            # output = token jawaban + token "thinking" (ikut ditagih sebagai output)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
                getattr(usage, "thoughts_token_count", 0) or 0
            )
            return (resp.text or "").strip(), input_tokens, output_tokens
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 15 * attempt
            print(f"  Gemini error (attempt {attempt}/{MAX_RETRIES}): {e} -> retry {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Gemini gagal setelah {MAX_RETRIES} percobaan: {last_err}")


# ----------------------------------------------------------------------------
# Sitasi
# ----------------------------------------------------------------------------
def extract_cited_sources(answer_text):
    """Parse answer text to extract cited document:page pairs."""
    citations = []
    patterns = [
        r"\*\*([^,*]+),\s*hal\.\s*(\d+)\*\*",
        r"\*([^,*]+),\s*hal\.\s*(\d+)\*",
        r"\[([^,\]]+),\s*hal\.\s*(\d+)\]",
        r"\(([^,)]+),\s*hal\.\s*(\d+)\)",
    ]
    for pattern in patterns:
        for doc, page in re.findall(pattern, answer_text):
            citations.append(f"{doc.strip()}:{page}")
    return ";".join(dict.fromkeys(citations)) if citations else ""


def sources_to_cited(chunks):
    pairs = []
    for c in chunks:
        try:
            page = int(c["page"])
        except (TypeError, ValueError):
            continue
        if c["source"]:
            pairs.append(f"{c['source']}:{page}")
    return ";".join(dict.fromkeys(pairs))


# ----------------------------------------------------------------------------
# Penyimpanan langsung ke CSV & JSONL
# ----------------------------------------------------------------------------
def load_done_qids(output_path):
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8", newline="") as f:
        return {row["qid"] for row in csv.DictReader(f) if row.get("qid")}


def append_answer_row(output_path, row):
    """Tulis 1 baris ke answers.csv langsung (flush + fsync)."""
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def append_metrics(metrics_file, metrics):
    with metrics_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        f.flush()


def log_query_to_db(qid, question_text, input_tokens, output_tokens):
    """Catat Query ke DB Django (Postgres Docker). Best-effort, tidak boleh menggagalkan batch."""
    try:
        user, _ = User.objects.get_or_create(
            email="benchmark@polibatam.ac.id",
            defaults={"username": "benchmark", "role": "mahasiswa"},
        )
        conversation, _ = Conversation.objects.get_or_create(
            user=user,
            title=f"Benchmark {qid}",
            defaults={"title": f"Benchmark {qid}"},
        )
        query = Query.objects.create(
            user=user,
            conversation=conversation,
            query_text=question_text,
            query_title=f"{qid}: {question_text[:100]}",
            status=Query.Status.PENDING,
        )
        updates = []
        if hasattr(query, "llm_input_tokens"):
            query.llm_input_tokens = input_tokens
            updates.append("llm_input_tokens")
        if hasattr(query, "llm_output_tokens"):
            query.llm_output_tokens = output_tokens
            updates.append("llm_output_tokens")
        if updates:
            query.save(update_fields=updates)
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] Gagal mencatat ke DB: {e}")


# ----------------------------------------------------------------------------
# Proses utama
# ----------------------------------------------------------------------------
def process_questions(questions_file, output_file, fresh=False):
    questions_path = Path(questions_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fresh and output_path.exists():
        output_path.unlink()

    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f, delimiter=";"))
    print(f"Loaded {len(questions)} questions from {questions_file}")

    done = load_done_qids(output_path)
    if done:
        print(f"Resume: {len(done)} qid sudah ada di {output_path.name}, akan dilewati.")

    metrics_dir = output_path.parent / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    metrics_file = metrics_dir / "metrics.jsonl"

    # Fail fast: pastikan koneksi DB & FAISS siap sebelum memulai batch
    connection.ensure_connection()
    print(f"DB OK -> {connection.settings_dict['HOST']}:{connection.settings_dict['PORT']}")
    load_faiss()
    print(f"Gemini model: {GEMINI_MODEL}")

    processed = 0
    MAX_QUESTIONS = 10
    START_FROM_QID = "Q074"
    END_AT_QID = "Q075"
    start_processing = False
    
    for idx, row in enumerate(questions):
        qid = row["qid"]
        question_text = row["question"]

        if qid == START_FROM_QID:
            start_processing = True
        
        if not start_processing:
            continue
            
        if qid in done:
            continue

        if processed >= MAX_QUESTIONS or qid > END_AT_QID:
            print(f"\n[STOP] Sudah menjawab {processed} pertanyaan atau mencapai {END_AT_QID}, hentikan.")
            break

        print(f"[{qid}] Processing...")
        start_total = time.time()

        answer_text = ""
        cited_sources = ""
        gen_time_s = 0.0
        input_tokens = 0
        output_tokens = 0
        chunks = []
        error_msg = ""

        try:
            chunks = hybrid_retrieve(question_text)
            print(f"  Retrieved {len(chunks)} chunks")

            prompt = build_prompt(question_text, chunks)

            start_gen = time.time()
            answer_text, input_tokens, output_tokens = generate_with_gemini(prompt)
            gen_time_s = round(time.time() - start_gen, 2)

            cited_sources = extract_cited_sources(answer_text) or sources_to_cited(chunks)
            print(f"  Answer generated ({gen_time_s}s)")
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            print(f"  ERROR: {error_msg}")

        total_time_s = round(time.time() - start_total, 2)

        if LOG_TO_DB:
            log_query_to_db(qid, question_text, input_tokens, output_tokens)

        # Simpan langsung ke answers.csv (juga untuk baris error, agar tidak hilang)
        append_answer_row(output_path, {
            "qid": qid,
            "answer": answer_text if not error_msg else f"[ERROR] {error_msg}",
            "cited_sources": cited_sources,
            "gen_time_s": gen_time_s,
            "total_time_s": total_time_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "run_date": date.today().isoformat(),
        })

        append_metrics(metrics_file, {
            "qid": qid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_time_s": total_time_s,
            "gen_time_s": gen_time_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "sources_count": len(chunks),
            "cited_sources": cited_sources,
            "model": GEMINI_MODEL,
            "status": "error" if error_msg else "success",
            "error": error_msg,
        })

        print(f"  [{qid}] Saved to {output_path.name} "
              f"({total_time_s:.2f}s, in={input_tokens} / out={output_tokens} tokens)")

        processed += 1
        is_last = idx == len(questions) - 1
        if PAUSE_EVERY and processed % PAUSE_EVERY == 0 and not is_last:
            print(f"\n[PAUSE] Rate limit: jeda {PAUSE_SECONDS} detik...\n")
            time.sleep(PAUSE_SECONDS)

    print(f"\n[FINAL] Semua hasil tersimpan di {output_file}")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Generate answers batch (retrieval lokal)")
    parser.add_argument("--questions", default=str(BASE_DIR / "questions.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "results" / "answers.csv"))
    parser.add_argument("--fresh", action="store_true",
                        help="Hapus answers.csv lama dan mulai dari awal (default: resume)")
    args = parser.parse_args()

    print("=" * 60)
    print("Tugas 4: Generate Answers Batch (local retrieval)")
    print("=" * 60)
    print(f"Input      : {args.questions}")
    print(f"Output     : {args.output}")
    print(f"Vectorstore: {VECTORSTORE_DIR}")
    print(f"DB host    : {LOCAL_DB_HOST}:{LOCAL_DB_PORT or '(settings)'}")
    print(f"Rate limit : jeda {PAUSE_SECONDS}s setiap {PAUSE_EVERY} pertanyaan"
          if PAUSE_EVERY else "Rate limit : nonaktif")
    print("=" * 60)

    n = process_questions(args.questions, args.output, fresh=args.fresh)
    print(f"\nCompleted: {n} questions processed")


if __name__ == "__main__":
    main()