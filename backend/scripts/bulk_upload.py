"""
Bulk upload script for Lumina RAG system.

Uploads all supported files from D:\Project\Dataset to Django backend,
waits for FAISS indexing completion, and logs results.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Configuration
DATASET_DIR = r"D:\Project\Dataset"
BACKEND_URL = "http://localhost:8001"
LOGIN_ENDPOINT = "/api/auth/login/"
UPLOAD_ENDPOINT = "/api/ingest/upload/"
STATUS_ENDPOINT = "/api/ingest/status/{document_id}/"

LOGIN_EMAIL = "lumina@lumina.gmail.com"
LOGIN_PASSWORD = "IF4md08rag"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
POLL_INTERVAL = 2
POLL_TIMEOUT = 60

SCRIPTS_DIR = Path(__file__).parent
RESULTS_FILE = SCRIPTS_DIR / "upload_results.json"

# ── Logging ────────────────────────────────────────────────────────────────────
def log_step(num: int, total: int, filename: str, status: str, message: str = ""):
    """Print progress step to console."""
    icons = {
        "upload": "[",
        "indexed": "[OK]",
        "failed": "[FAIL]",
        "timeout": "[TIMEOUT]",
        "error": "[ERROR]",
    }
    icon = icons.get(status, "[WAIT]")
    prefix = f"[{num}/{total}] {icon} {filename}"
    if message:
        print(f"{prefix} -> {message}")
    else:
        print(prefix)


def log_error(message: str):
    """Print error message."""
    print(f"\n❌ ERROR: {message}\n")


# ── API Calls ──────────────────────────────────────────────────────────────────
def login() -> Optional[str]:
    """Login to Django API and return auth token."""
    url = f"{BACKEND_URL}{LOGIN_ENDPOINT}"
    payload = {"email": LOGIN_EMAIL, "password": LOGIN_PASSWORD}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("token")
    except requests.exceptions.RequestException as e:
        log_error(f"Login failed: {e}")
        return None


def upload_file(token: str, filepath: Path) -> Optional[dict]:
    """Upload single file to backend."""
    url = f"{BACKEND_URL}{UPLOAD_ENDPOINT}"
    headers = {"Authorization": f"Token {token}"}

    try:
        with open(filepath, "rb") as f:
            files = {"document": (filepath.name, f, "application/octet-stream")}
            response = requests.post(url, headers=headers, files=files, timeout=300)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        log_error(f"Upload failed for {filepath.name}: {e}")
        return None


def poll_status(token: str, document_id: int, session_id: str, timeout: int = POLL_TIMEOUT) -> dict:
    """Poll ingestion status until completion or timeout."""
    url = f"{BACKEND_URL}{STATUS_ENDPOINT.format(document_id=document_id)}"
    headers = {"Authorization": f"Token {token}"}
    params = {"session": session_id}

    start_time = time.time()
    last_log_id = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            return {
                "status": "timeout",
                "logs": [],
                "duration_seconds": round(elapsed, 1),
                "error_message": f"Polling timeout after {timeout} seconds",
            }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            logs = data.get("logs", [])
            current_status = data.get("status", "processing")

            # Process new logs
            for log in logs:
                if log["id"] > last_log_id:
                    log_step(0, 0, "", "", f"Step: {log['step']} - {log['message']}")
                    last_log_id = log["id"]

            if current_status == "indexed":
                # Get chunk count from logs
                chunks = extract_chunks_from_logs(logs)
                return {
                    "status": "indexed",
                    "logs": logs,
                    "duration_seconds": round(elapsed, 1),
                    "chunks": chunks,
                    "error_message": None,
                }
            elif current_status == "failed":
                error_msg = extract_error_from_logs(logs)
                return {
                    "status": "failed",
                    "logs": logs,
                    "duration_seconds": round(elapsed, 1),
                    "chunks": 0,
                    "error_message": error_msg,
                }

            time.sleep(POLL_INTERVAL)

        except requests.exceptions.RequestException as e:
            log_error(f"Status poll failed for document {document_id}: {e}")
            time.sleep(POLL_INTERVAL)


def extract_chunks_from_logs(logs: list) -> int:
    """Extract chunk count from ingestion logs."""
    for log in reversed(logs):
        msg = log.get("message", "")
        if "chunks" in msg.lower():
            # Try to extract number from message like "Created 45 chunks"
            parts = msg.split()
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0 and "chunk" in parts[i - 1].lower():
                    return int(part)
    return 0


def extract_error_from_logs(logs: list) -> str:
    """Extract error message from failed ingestion logs."""
    for log in reversed(logs):
        if log.get("status") == "failed":
            msg = log.get("message", "")
            err = log.get("error_message")
            return err or msg or "Unknown error"
    return "Ingestion failed without specific error message"


def count_chunks_in_logs(logs: list) -> int:
    """Count total chunks from log messages."""
    for log in logs:
        msg = log.get("message", "")
        if "Created" in msg and "chunks" in msg:
            try:
                parts = msg.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i > 0 and "chunk" in parts[i - 1].lower():
                        return int(part)
            except Exception:
                pass
    return 0


# ── File Scanning ──────────────────────────────────────────────────────────────
def scan_dataset(directory: str) -> list:
    """Scan directory for supported file types."""
    dir_path = Path(directory)
    files = []

    if not dir_path.exists():
        log_error(f"Dataset directory not found: {directory}")
        return files

    for item in dir_path.iterdir():
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)

    return sorted(files, key=lambda x: x.name)


# ── Results Logging ────────────────────────────────────────────────────────────
def save_results(results: list, filepath: Path):
    """Save upload results to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filepath}")


def print_summary(results: list):
    """Print summary of upload results."""
    total = len(results)
    indexed = sum(1 for r in results if r["status"] == "indexed")
    failed = sum(1 for r in results if r["status"] == "failed")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    total_chunks = sum(r.get("chunks", 0) for r in results)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total files:     {total}")
    print(f"Indexed:         {indexed}")
    print(f"Failed:          {failed}")
    print(f"Timeout:         {timeout}")
    print(f"Total chunks:    {total_chunks}")
    print("=" * 50)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Main entry point."""
    print("Lumina Bulk Upload Script")
    print("=" * 50)

    # Step 1: Login
    print("\nStep 1: Logging in...")
    token = login()
    if not token:
        log_error("Failed to login. Exiting.")
        sys.exit(1)
    print("[OK] Login successful")

    # Step 2: Scan dataset
    print(f"\nStep 2: Scanning dataset: {DATASET_DIR}")
    files = scan_dataset(DATASET_DIR)
    if not files:
        log_error("No supported files found in dataset directory.")
        sys.exit(0)
    print(f"Found {len(files)} supported files")

    # Step 3: Upload files
    print("\nStep 3: Uploading files...")
    results = []

    for i, filepath in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Uploading: {filepath.name}", flush=True)

        # Upload
        upload_resp = upload_file(token, filepath)
        if not upload_resp:
            result = {
                "filename": filepath.name,
                "document_id": None,
                "session_id": None,
                "status": "error",
                "chunks": 0,
                "duration_seconds": 0,
                "error_message": "Upload request failed",
            }
            results.append(result)
            log_step(i, len(files), filepath.name, "error", "Upload request failed")
            continue

        doc_id = upload_resp.get("document_id")
        session_id = upload_resp.get("session_id")
        print(f"  Upload response: doc_id={doc_id}, session_id={session_id}", flush=True)

        # Poll status until indexed or failed (or 60s timeout)
        status_resp = poll_status(token, doc_id, session_id, timeout=POLL_TIMEOUT)
        status = status_resp["status"]

        if status == "indexed":
            chunks = status_resp.get("chunks", 0)
            result = {
                "filename": filepath.name,
                "document_id": doc_id,
                "session_id": session_id,
                "status": status,
                "chunks": chunks,
                "duration_seconds": status_resp["duration_seconds"],
                "error_message": None,
            }
            results.append(result)
            log_step(i, len(files), filepath.name, "indexed", f"{chunks} chunks in {status_resp['duration_seconds']}s")

        elif status == "failed":
            result = {
                "filename": filepath.name,
                "document_id": doc_id,
                "session_id": session_id,
                "status": status,
                "chunks": 0,
                "duration_seconds": status_resp["duration_seconds"],
                "error_message": status_resp.get("error_message"),
            }
            results.append(result)
            log_step(i, len(files), filepath.name, "failed", status_resp.get("error_message"))

        else:
            result = {
                "filename": filepath.name,
                "document_id": doc_id,
                "session_id": session_id,
                "status": status,
                "chunks": 0,
                "duration_seconds": status_resp["duration_seconds"],
                "error_message": status_resp.get("error_message"),
            }
            results.append(result)
            log_step(i, len(files), filepath.name, "timeout", status_resp.get("error_message"))

    # Step 4: Save results
    print("\nStep 4: Saving results...", flush=True)
    save_results(results, RESULTS_FILE)

    # Step 5: Print summary
    print_summary(results)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
