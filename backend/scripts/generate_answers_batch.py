import os
import sys
import time
import csv
import json
import re
from pathlib import Path
from datetime import date, datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumina.settings")

print("Setting up Django...")
import django
django.setup()
print("Django setup complete.")

from django.contrib.auth import get_user_model
from core.models import Query, Answer, Conversation
from ingest.retrieval import run_rag_query, RagPipelineError


User = get_user_model()


def extract_cited_sources(answer_text):
    """Parse answer text to extract cited document:page pairs."""
    citations = []
    
    patterns = [
        r'\*\*([^,]+),\s*hal\.\s*(\d+)\*\*',
        r'\*([^,]+),\s*hal\.\s*(\d+)\*',
        r'\[([^,]+),\s*hal\.\s*(\d+)\]',
        r'\(([^,]+),\s*hal\.\s*(\d+)\)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, answer_text)
        for doc, page in matches:
            citations.append(f"{doc}:{page}")
    
    return ";".join(dict.fromkeys(citations)) if citations else ""


def process_questions(questions_file, output_file):
    """Process all questions and generate answers.csv."""
    
    questions_path = Path(questions_file)
    output_path = Path(output_file)

    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        questions = list(reader)
    
    print(f"Loaded {len(questions)} questions from {questions_file}")
    
    results = []
    
    for idx, row in enumerate(questions):
        qid = row["qid"]
        question_text = row["question"]
        
        print(f"[{qid}] Processing...")
        
        start_total = time.time()
        
        answer_text = ""
        cited_sources = ""
        gen_time_s = 0.0
        total_time_s = 0.0
        input_tokens = 0
        output_tokens = 0
        error_msg = ""
        
        try:
            user, created = User.objects.get_or_create(
                email="benchmark@polibatam.ac.id",
                defaults={"username": "benchmark", "role": "mahasiswa"}
            )
            
            conversation, _ = Conversation.objects.get_or_create(
                user=user,
                title=f"Benchmark {qid}",
                defaults={"title": f"Benchmark {qid}"}
            )
            
            query = Query.objects.create(
                user=user,
                conversation=conversation,
                query_text=question_text,
                query_title=f"{qid}: {question_text[:100]}",
                status=Query.Status.PENDING,
            )
            
            query_id = query.id
            
            result = run_rag_query(query_id, question_text)
            
            answer_text = result.get("answer", "")
            sources = result.get("sources", [])
            response_time_ms = result.get("response_time_ms", 0)
            
            cited_sources = extract_cited_sources(answer_text)
            
            if not cited_sources and sources:
                doc_pages = []
                for source in sources:
                    doc_name = source.get("source", "")
                    page = source.get("page")
                    if doc_name and page:
                        doc_pages.append(f"{doc_name}:{int(page)}")
                if doc_pages:
                    cited_sources = ";".join(dict.fromkeys(doc_pages))
            
            gen_time_s = round(response_time_ms / 1000, 2)
            total_time_s = round(time.time() - start_total, 2)
            
            query.refresh_from_db()
            if hasattr(query, 'llm_input_tokens') and hasattr(query, 'llm_output_tokens'):
                input_tokens = query.llm_input_tokens
                output_tokens = query.llm_output_tokens
            
            print(f"  Answer generated ({gen_time_s}s)")
            
            # Save answer to results list
            results.append({
                "qid": qid,
                "answer": answer_text,
                "cited_sources": cited_sources,
                "gen_time_s": gen_time_s,
                "total_time_s": total_time_s,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "run_date": date.today().isoformat(),
            })
            
        except RagPipelineError as e:
            print(f"  RAG error: {e.user_message}")
            error_msg = e.user_message
            cited_sources = ""
        except Exception as e:
            print(f"  Unexpected error: {e}")
            error_msg = str(e)
            cited_sources = ""
            gen_time_s = round(time.time() - start_total, 2)
            total_time_s = gen_time_s
            if 'sources' not in locals():
                sources = []
        
        # Save metrics to JSONL after each answer
        sources_count = len(sources) if 'sources' in locals() else 0
        metrics = {
            "qid": qid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_time_s": total_time_s,
            "gen_time_s": gen_time_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "sources_count": sources_count,
            "cited_sources": cited_sources,
            "status": "success" if not error_msg else "error",
        }
        metrics_dir = output_path.parent / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        metrics_file = metrics_dir / "metrics.jsonl"
        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        
        print(f"  [{qid}] Answer saved ({total_time_s:.2f}s, {input_tokens}/{output_tokens} tokens)")
        
        if (idx + 1) % 2 == 0:
            # Save answers.csv after every 2 questions (checkpoint)
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "qid", "answer", "cited_sources", "gen_time_s", "total_time_s",
                    "input_tokens", "output_tokens", "run_date"
                ])
                writer.writeheader()
                writer.writerows(results)
            
            print(f"  [CHECKPOINT] Saved {idx + 1} answers to {output_file}")
            
            if (idx + 1) < len(questions):
                print(f"\n⏱️ Waiting 10 seconds before continuing...\n")
                time.sleep(10)
    
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "qid", "answer", "cited_sources", "gen_time_s", "total_time_s",
            "input_tokens", "output_tokens", "run_date"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[FINAL] All results saved to {output_file}")
    return results


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    questions_file = str(script_dir / "questions.csv")
    output_dir = script_dir / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = str(output_dir / "answers.csv")
    
    print("=" * 60)
    print("Tugas 4: Generate Answers Batch")
    print("=" * 60)
    print(f"Input: {questions_file}")
    print(f"Output: {output_file}")
    print(f"Rate limit: Pause 3 minutes every 10 questions")
    print("=" * 60)
    
    results = process_questions(questions_file, output_file)
    print(f"\nCompleted: {len(results)} questions processed")
