"""Evaluate the RAG system on a set of test questions.

Metrics:
- Retrieval hit rate: did the right document show up in the retrieved chunks?
- Answer accuracy: does the answer contain the expected keywords?
- Faithfulness: does an LLM judge agree every claim is supported by the context? (catches made-up answers)
- Average latency

Run with: python eval.py
"""
import json
import os
import time

from rag import answer, generate

EVAL_FILE = "eval_set.json"
RESULTS_FILE = "eval_results.json"
DELAY = float(os.getenv("EVAL_DELAY", "7"))  # pause between calls to stay inside free-tier rate limits

JUDGE_PROMPT = """You are grading an AI answer for faithfulness.

CONTEXT:
{context}

ANSWER:
{answer}

Is every factual claim in the ANSWER supported by the CONTEXT?
If the answer says it couldn't find the information, count it as faithful.
Reply with exactly one word on the first line: FAITHFUL or UNFAITHFUL.
Then one short sentence explaining why."""


def keyword_score(text: str, keywords):
    if not keywords:
        return None
    text = text.lower()
    return sum(kw.lower() in text for kw in keywords) / len(keywords)


def main():
    if not os.path.exists(EVAL_FILE):
        print(f"Create {EVAL_FILE} first (copy eval_set.example.json and edit it).")
        return
    with open(EVAL_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        print(f"[{i}/{len(cases)}] {q}")

        start = time.time()
        res = answer(q)
        latency = time.time() - start
        time.sleep(DELAY)

        verdict = generate(JUDGE_PROMPT.format(context=res["context"], answer=res["answer"]))
        faithful = verdict.strip().upper().startswith("FAITHFUL")
        time.sleep(DELAY)

        expected_source = case.get("expected_source")
        retrieved = [s["source"] for s in res["sources"]]
        hit = (expected_source in retrieved) if expected_source else None

        results.append({
            "question": q,
            "answer": res["answer"],
            "retrieved_sources": retrieved,
            "retrieval_hit": hit,
            "keyword_score": keyword_score(res["answer"], case.get("expected_keywords", [])),
            "faithful": faithful,
            "judge_reason": verdict,
            "latency_s": round(latency, 2),
        })
        print(f"   hit={hit}  keywords={results[-1]['keyword_score']}  faithful={faithful}  {latency:.1f}s")

    def avg(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "questions": len(results),
        "retrieval_hit_rate": avg([r["retrieval_hit"] for r in results]),
        "answer_accuracy": avg([r["keyword_score"] for r in results]),
        "faithfulness": avg([r["faithful"] for r in results]),
        "avg_latency_s": avg([r["latency_s"] for r in results]),
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        if isinstance(v, float) and k != "avg_latency_s":
            print(f"{k:22} {v:.0%}")
        elif isinstance(v, float):
            print(f"{k:22} {v:.2f}s")
        else:
            print(f"{k:22} {v}")
    print(f"\nFull details saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
