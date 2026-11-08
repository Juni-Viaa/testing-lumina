import os, time, gc, glob, platform, random
import numpy as np, pandas as pd, psutil, torch, faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

OUT_DIR = "./results"
DOCS_DIR = "."
os.makedirs(OUT_DIR, exist_ok=True)

# ---- parameter yang boleh diubah ----
MODELS = {
    "e5-large": "intfloat/multilingual-e5-large",
}
BATCH_SIZES = [16, 32]
ONE_BY_ONE_SAMPLE = 12
LIMIT_CHUNKS = None
CHUNK_SIZE = 2560
CHUNK_OVERLAP = 256
assert CHUNK_SIZE > CHUNK_OVERLAP >= 0
N_SYNTHETIC = 718
TOP_K = 7
# --------------------------------------

DEVICE = "cpu"

def get_cpu_model():
    return platform.processor() or "AMD Ryzen 3 3250U"

CPU_NAME = get_cpu_model()
RAM_GB = round(psutil.virtual_memory().total / 1024**3)
GPU_NAME = None
HW_LABEL = CPU_NAME

print("Device       :", DEVICE)
print("CPU          :", CPU_NAME)
print("Core logis   :", os.cpu_count(), "| core fisik:", psutil.cpu_count(logical=False))
print("RAM total    :", RAM_GB, "GB")
print("GPU          :", GPU_NAME)
print("torch threads:", torch.get_num_threads())

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def read_doc(path):
    if path.lower().endswith(".pdf"):
        return "\\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\\n\\n", "\\n", " ", ""],
)

files = sorted(glob.glob(f"{DOCS_DIR}/*.pdf") + glob.glob(f"{DOCS_DIR}/*.txt"))
chunks, SYNTHETIC = [], False
for fp in files:
    for c in text_splitter.split_text(read_doc(fp)):
        if c.strip():
            chunks.append({"source": os.path.basename(fp), "text": c})

if not chunks:
    SYNTHETIC = True
    random.seed(0)
    vocab = ("kurikulum program studi mahasiswa mata kuliah semester sks rekayasa perangkat lunak "
             "praktikum proyek basis data jaringan algoritma pemrograman evaluasi capaian pembelajaran "
             "dosen laboratorium tugas akhir magang sertifikasi kompetensi").split()
    for i in range(N_SYNTHETIC):
        words, total = [], 0
        while total < CHUNK_SIZE:
            w = random.choice(vocab)
            words.append(w)
            total += len(w) + 1
        synthetic_text = " ".join(words)[:CHUNK_SIZE]
        for c in text_splitter.split_text(synthetic_text):
            if c.strip():
                chunks.append({"source": f"sintetis_{i % 18}.txt", "text": c})

if LIMIT_CHUNKS:
    chunks = chunks[:LIMIT_CHUNKS]

print(f"{len(files)} dokumen -> {len(chunks)} chunk | korpus sintetis: {SYNTHETIC}")

MAIN_COLS = ["environment","device","cpu_model","ram_gb","model","batch_size","load_s",
             "one_by_one_s_per_chunk","batched_s_per_chunk","n_chunks","dense_hit7","note"]
DETAIL_COLS = ["environment","device","model","n_chunks","batch_size","load_s","encode_all_s",
               "faiss_build_s","faiss_save_s","query_s_per_query","total_ingest_s"]
MAIN_CSV = f"{OUT_DIR}/ingestion.csv"
DETAIL_CSV = f"{OUT_DIR}/ingestion_detail.csv"

def sync():
    pass

def upsert(path, cols, row, keys):
    """Tambah baris; jika baris dengan kunci sama sudah ada (run ulang), diganti."""
    new = pd.DataFrame([row], columns=cols)
    if os.path.exists(path):
        old = pd.read_csv(path)
        mask = np.ones(len(old), dtype=bool)
        for k in keys:
            mask &= (old[k].astype(str) == str(row[k]))
        old = old[~mask]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)

def run_model(label, hf_name):
    print(f"\n=== {label} ({hf_name}) | device={DEVICE}, batch_size={BATCH_SIZE} ===")
    snapshot_download(hf_name)
    gc.collect()

    t = time.perf_counter()
    model = SentenceTransformer(hf_name, device=DEVICE)
    model.encode(["passage: test"], batch_size=1, show_progress_bar=False)

    load_s = time.perf_counter() - t

    passages = ["passage: " + c["text"] for c in chunks]
    n = len(passages)

    t = time.perf_counter()
    emb = model.encode(passages, batch_size=BATCH_SIZE, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype("float32")
    encode_all_s = time.perf_counter() - t
    batched = encode_all_s / n

    t = time.perf_counter()
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss_build_s = time.perf_counter() - t
    t = time.perf_counter()
    faiss.write_index(index, f"{OUT_DIR}/faiss_{label}_b{BATCH_SIZE}.index")
    faiss_save_s = time.perf_counter() - t

    questions = ["query: apa saja mata kuliah pada semester 3?"] * 10
    t = time.perf_counter()
    for q in questions:
        model.encode([q], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    query_s = (time.perf_counter() - t) / len(questions)

    hit7 = None

    note = (f"chunk_size={CHUNK_SIZE}; chunk_overlap={CHUNK_OVERLAP}; unduhan model tidak dihitung; "
            f"one_by_one dirata-rata dari {len(passages)} chunk; max_seq_length={model.max_seq_length}")
    if SYNTHETIC:
        note += "; korpus sintetis"
    if LIMIT_CHUNKS:
        note += f"; dibatasi {LIMIT_CHUNKS} chunk"

    main_row = {
        "environment": "local", "device": DEVICE, "cpu_model": HW_LABEL, "ram_gb": RAM_GB,
        "model": label, "batch_size": BATCH_SIZE,
        "load_s": round(load_s, 2),
        "one_by_one_s_per_chunk": round(query_s, 3),
        "batched_s_per_chunk": round(batched, 3),
        "n_chunks": n, "dense_hit7": hit7, "note": note,
    }
    detail_row = {
        "environment": "local", "device": DEVICE, "model": label, "n_chunks": n,
        "batch_size": BATCH_SIZE, "load_s": round(load_s, 2),
        "encode_all_s": round(encode_all_s, 2),
        "faiss_build_s": round(faiss_build_s, 4), "faiss_save_s": round(faiss_save_s, 4),
        "query_s_per_query": round(query_s, 3),
        "total_ingest_s": round(load_s + encode_all_s + faiss_build_s + faiss_save_s, 2),
    }
    upsert(MAIN_CSV, MAIN_COLS, main_row, ["environment", "device", "model", "batch_size"])
    upsert(DETAIL_CSV, DETAIL_COLS, detail_row, ["environment", "device", "model", "batch_size"])
    print(pd.Series(main_row).to_string())

    del model, index, emb
    gc.collect()

for label, hf_name in MODELS.items():
    for BATCH_SIZE in BATCH_SIZES:
        run_model(label, hf_name)

print("\n=== Hasil Benchmark ===")
res = pd.read_csv(MAIN_CSV)
print(res.to_string())
print("\nDetail (FAISS, query, total ingest):")
print(pd.read_csv(DETAIL_CSV).to_string())