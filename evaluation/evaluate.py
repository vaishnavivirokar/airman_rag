"""
Evaluation script for the AIRMAN RAG system.
Computes: retrieval hit-rate, faithfulness, hallucination rate, qualitative analysis.
Run with API server active: uvicorn app.main:app --reload
"""

import json
import re
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000/ask"
REFUSAL = "This information is not available in the provided document(s)."
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "evaluation_results.json"


def normalize(text):
    """Normalize text for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize(text):
    """Simple word tokenization."""
    return set(re.findall(r"\b\w+\b", normalize(text)))


def overlap_score(answer_tokens, chunk_tokens):
    """Jaccard-like overlap: |answer ∩ chunks| / |answer|."""
    if not answer_tokens:
        return 1.0
    overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)
    return overlap


def retrieval_hit(answer, chunks):
    """Did the retrieved chunks contain information to answer the question?"""
    if not chunks or not chunks[0].get("text"):
        return False
    if answer.strip() == REFUSAL.strip():
        return False
    ans_tok = tokenize(answer)
    chunk_tok = set()
    for c in chunks:
        chunk_tok |= tokenize(c.get("text", ""))
    return overlap_score(ans_tok, chunk_tok) >= 0.15


def faithfulness(answer, chunks):
    """Is the answer fully grounded in retrieved text? Returns 0-1."""
    if answer.strip() == REFUSAL.strip():
        return 1.0
    if not chunks:
        return 0.0
    ans_tok = tokenize(answer)
    chunk_tok = set()
    for c in chunks:
        chunk_tok |= tokenize(c.get("text", ""))
    return overlap_score(ans_tok, chunk_tok)


def is_hallucination(answer, chunks):
    """Unsupported claims = hallucination."""
    if answer.strip() == REFUSAL.strip():
        return False
    return faithfulness(answer, chunks) < 0.25


def run_evaluation():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    results = []

    for i, q in enumerate(questions):
        qtext = q["question"]
        qtype = q.get("type", "factual")
        try:
            r = requests.post(URL, json={"question": qtext, "debug": True}, timeout=60)
            data = r.json()
        except Exception as e:
            results.append({
                "question": qtext,
                "type": qtype,
                "answer": "",
                "citations": [],
                "chunks": [],
                "error": str(e),
                "retrieval_hit": False,
                "faithfulness": 0.0,
                "hallucination": True,
            })
            continue

        answer = data.get("answer", "")
        chunks = data.get("chunks", [])
        citations = data.get("citations", [])

        hit = retrieval_hit(answer, chunks)
        faith = faithfulness(answer, chunks)
        hall = is_hallucination(answer, chunks)

        results.append({
            "question": qtext,
            "type": qtype,
            "answer": answer,
            "citations": citations,
            "chunks": chunks,
            "retrieval_hit": hit,
            "faithfulness": faith,
            "hallucination": hall,
        })
        print(f"  [{i+1}/{len(questions)}] {qtype}: {qtext[:50]}... -> hit={hit}, faith={faith:.2f}, hall={hall}")

    # Save results
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compute metrics
    n = len(results)
    hits = sum(1 for r in results if r.get("retrieval_hit"))
    faithful = sum(1 for r in results if r.get("faithfulness", 0) >= 0.25)
    halls = sum(1 for r in results if r.get("hallucination"))
    answered = sum(1 for r in results if r.get("answer", "").strip() != REFUSAL.strip())

    retrieval_hit_rate = hits / n if n else 0
    faithfulness_rate = faithful / n if n else 0
    hallucination_rate = halls / n if n else 0

    # Qualitative: 5 best, 5 worst
    scored = []
    for r in results:
        s = 0.0
        if r.get("retrieval_hit"):
            s += 1.0
        s += r.get("faithfulness", 0)
        if r.get("hallucination"):
            s -= 2.0
        if r.get("answer", "").strip() == REFUSAL.strip() and not r.get("retrieval_hit"):
            s += 0.5  # Correct refusal
        scored.append((s, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_5 = [r for _, r in scored[:5]]
    worst_5 = [r for _, r in scored[-5:]][::-1]

    report = {
        "summary": {
            "total_questions": n,
            "answered": answered,
            "refused": n - answered,
            "retrieval_hit_rate": round(retrieval_hit_rate, 4),
            "faithfulness_rate": round(faithfulness_rate, 4),
            "hallucination_rate": round(hallucination_rate, 4),
        },
        "best_5": best_5,
        "worst_5": worst_5,
    }

    report_path = Path(__file__).parent.parent / "report.md"
    write_report(report, report_path)
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions:     {n}")
    print(f"Answered:            {answered}")
    print(f"Refused:             {n - answered}")
    print(f"Retrieval hit-rate:  {retrieval_hit_rate:.2%}")
    print(f"Faithfulness rate:   {faithfulness_rate:.2%}")
    print(f"Hallucination rate:  {hallucination_rate:.2%}")
    print(f"\nReport written to: {report_path}")
    return report


def write_report(report, path):
    s = report["summary"]
    md = f"""# AIRMAN RAG — Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Total questions | {s['total_questions']} |
| Answered | {s['answered']} |
| Refused | {s['refused']} |
| **Retrieval hit-rate** | {s['retrieval_hit_rate']:.2%} |
| **Faithfulness rate** | {s['faithfulness_rate']:.2%} |
| **Hallucination rate** | {s['hallucination_rate']:.2%} |

## Qualitative Analysis

### 5 Best Answers

"""
    for i, r in enumerate(report["best_5"], 1):
        md += f"""#### {i}. {r['question'][:80]}...
- **Type:** {r.get('type', 'N/A')}
- **Answer:** {r.get('answer', '')[:300]}...
- **Retrieval hit:** {r.get('retrieval_hit')} | **Faithfulness:** {r.get('faithfulness', 0):.2f} | **Hallucination:** {r.get('hallucination')}
- **Citations:** {r.get('citations', [])}

"""
    md += """
### 5 Worst Answers

"""
    for i, r in enumerate(report["worst_5"], 1):
        md += f"""#### {i}. {r['question'][:80]}...
- **Type:** {r.get('type', 'N/A')}
- **Answer:** {r.get('answer', '')[:300]}...
- **Retrieval hit:** {r.get('retrieval_hit')} | **Faithfulness:** {r.get('faithfulness', 0):.2f} | **Hallucination:** {r.get('hallucination')}
- **Citations:** {r.get('citations', [])}

"""
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    run_evaluation()
