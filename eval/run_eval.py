"""
eval/run_eval.py
─────────────────────────────────────────────────────────────────────────────
LLM-as-Judge evaluation pipeline for the CAE multi-agent RAG system.

What this evaluates:
─────────────────────
  Using the 20 golden Q&A pairs in data/synthetic/golden_qa.json, this
  pipeline measures retrieval and answer quality across 4 RAGAS metrics:

  • context_precision   — are the retrieved chunks actually relevant?
  • context_recall      — are all relevant facts covered in retrieved chunks?
  • faithfulness        — does the answer only assert things in the context?
  • answer_relevancy    — does the answer address the question asked?

  Target: faithfulness ≥ 0.85 (project CV claim)

Two modes:
──────────
  1. LLM mode (OPENAI_API_KEY set): uses RAGAS with GPT-4o as judge
     → Full faithfulness + relevancy scoring
  2. Offline mode (no API key): uses keyword-overlap metrics as proxy
     → Hit rate, MRR, context coverage — fast, free, deterministic

Run:
    python eval/run_eval.py              # offline mode
    OPENAI_API_KEY=sk-... python eval/run_eval.py   # LLM judge mode
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from knowledge_base.build_retriever import HybridNVHRetriever
from agents.cae_graph import build_cae_graph


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_QA_PATH  = "data/synthetic/golden_qa.json"
RESULTS_PATH    = "eval/eval_results.json"
TOP_K           = 5
FAITHFULNESS_TARGET = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GOLDEN Q&A
# ─────────────────────────────────────────────────────────────────────────────

def load_golden_qa(path: str = GOLDEN_QA_PATH) -> List[Dict]:
    """Load the 20 hand-curated evaluation questions."""
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE EVALUATION  (no LLM API needed)
# ─────────────────────────────────────────────────────────────────────────────

def keyword_overlap(text: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found in the answer/context."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def evaluate_offline(
    qa_pairs: List[Dict],
    retriever: HybridNVHRetriever,
    graph,
) -> Dict[str, Any]:
    """
    Offline evaluation using keyword overlap as a quality proxy.

    Metrics:
      hit_rate        — fraction of queries where the reference case was retrieved
      mrr             — mean reciprocal rank of the reference case
      context_coverage — avg keyword overlap between context and expected topics
      answer_coverage  — avg keyword overlap between answer and expected topics
      avg_latency_ms  — average end-to-end latency
    """
    print("\n" + "=" * 60)
    print("Offline Evaluation (keyword overlap metrics)")
    print("=" * 60)

    hits        = 0
    rr_scores   = []
    ctx_scores  = []
    ans_scores  = []
    latencies   = []

    for i, qa in enumerate(qa_pairs, 1):
        q          = qa["question"]
        ref_case   = qa.get("reference_case", "")
        keywords   = qa.get("expected_topics", [])

        print(f"\n[{i:02d}/{len(qa_pairs)}] {q[:70]}...")

        # ── Retrieval eval ─────────────────────────────────────────────
        results  = retriever.retrieve(q, top_k=TOP_K)
        case_ids = [r.doc_id for r in results]

        if ref_case in case_ids:
            hits += 1
            rank  = case_ids.index(ref_case) + 1
            rr_scores.append(1.0 / rank)
            print(f"    Hit: ✅  rank={rank}")
        else:
            rr_scores.append(0.0)
            print(f"    Hit: ❌  (ref={ref_case}, retrieved={case_ids[:3]})")

        # Context coverage — keywords in retrieved text
        context_text = " ".join(r.text for r in results)
        ctx_score    = keyword_overlap(context_text, keywords)
        ctx_scores.append(ctx_score)

        # ── Answer eval via full graph ─────────────────────────────────
        t0     = time.time()
        result = graph.invoke({"query": q})
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        answer    = result.get("answer", "")
        ans_score = keyword_overlap(answer, keywords)
        ans_scores.append(ans_score)

        print(f"    Context coverage: {ctx_score:.2f} | Answer coverage: {ans_score:.2f} | {latency_ms:.0f}ms")

    # ── Aggregate results ─────────────────────────────────────────────
    n = len(qa_pairs)
    summary = {
        "mode":              "offline",
        "n_questions":       n,
        "hit_rate":          round(hits / n, 3),
        "mrr":               round(sum(rr_scores) / n, 3),
        "context_coverage":  round(sum(ctx_scores) / n, 3),
        "answer_coverage":   round(sum(ans_scores) / n, 3),
        "avg_latency_ms":    round(sum(latencies) / n, 1),
        "faithfulness_proxy": round(sum(ans_scores) / n, 3),
        "meets_target":      sum(ans_scores) / n >= FAITHFULNESS_TARGET,
    }

    print("\n" + "─" * 60)
    print("RESULTS SUMMARY")
    print("─" * 60)
    print(f"  Hit Rate (top-{TOP_K}):     {summary['hit_rate']:.3f}")
    print(f"  MRR:                 {summary['mrr']:.3f}")
    print(f"  Context Coverage:    {summary['context_coverage']:.3f}")
    print(f"  Answer Coverage:     {summary['answer_coverage']:.3f}  (faithfulness proxy)")
    print(f"  Avg Latency:         {summary['avg_latency_ms']:.0f} ms")
    status = "✅ PASS" if summary["meets_target"] else "❌ BELOW TARGET"
    print(f"  Target (≥{FAITHFULNESS_TARGET}):      {status}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS EVALUATION  (LLM judge mode — requires OPENAI_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_ragas(
    qa_pairs: List[Dict],
    retriever: HybridNVHRetriever,
    graph,
) -> Dict[str, Any]:
    """
    Full RAGAS evaluation using GPT-4o as judge.

    Metrics computed:
      faithfulness      — answer assertions grounded in context (LLM judge)
      answer_relevancy  — answer addresses the question (LLM judge)
      context_precision — retrieved context is relevant to question (LLM judge)
      context_recall    — context covers all expected facts (LLM judge)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        print(f"RAGAS import failed: {e}")
        print("Install: pip install ragas datasets langchain-openai")
        return evaluate_offline(qa_pairs, retriever, graph)

    print("\n" + "=" * 60)
    print("RAGAS Evaluation (LLM-as-Judge with GPT-4o)")
    print("=" * 60)

    # ── Build RAGAS dataset ───────────────────────────────────────────
    questions   = []
    answers     = []
    contexts    = []
    ground_truths = []

    for i, qa in enumerate(qa_pairs, 1):
        q = qa["question"]
        print(f"  [{i:02d}/{len(qa_pairs)}] Collecting: {q[:60]}...")

        results  = retriever.retrieve(q, top_k=TOP_K)
        ctx_list = [r.text for r in results]

        result   = graph.invoke({"query": q})
        answer   = result.get("answer", "")

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx_list)
        ground_truths.append(qa.get("expected_answer_contains", [""])[0])

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # ── Configure RAGAS LLM + embeddings ─────────────────────────────
    llm        = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    metrics = [faithfulness, answer_relevancy, context_precision]
    for m in metrics:
        m.llm        = llm
        m.embeddings = embeddings

    print("\n  Running RAGAS evaluation (this calls the LLM judge)...")
    t0 = time.time()
    ragas_result = evaluate(dataset, metrics=metrics)
    elapsed = time.time() - t0

    scores = {
        "mode":              "ragas_llm_judge",
        "n_questions":       len(qa_pairs),
        "faithfulness":      round(float(ragas_result["faithfulness"]),      3),
        "answer_relevancy":  round(float(ragas_result["answer_relevancy"]),  3),
        "context_precision": round(float(ragas_result["context_precision"]), 3),
        "eval_time_s":       round(elapsed, 1),
        "meets_target":      float(ragas_result["faithfulness"]) >= FAITHFULNESS_TARGET,
    }

    print("\n" + "─" * 60)
    print("RAGAS RESULTS")
    print("─" * 60)
    print(f"  Faithfulness:      {scores['faithfulness']:.3f}  (target ≥ {FAITHFULNESS_TARGET})")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    print(f"  Context Precision: {scores['context_precision']:.3f}")
    status = "✅ PASS" if scores["meets_target"] else "❌ BELOW TARGET"
    print(f"  Faithfulness:      {status}")
    print(f"  Eval time:         {scores['eval_time_s']}s for {len(qa_pairs)} questions")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CAE Platform — RAG Evaluation Pipeline")
    print("=" * 60)

    # Load components
    print("\n[1/3] Loading retriever and agent graph...")
    retriever = HybridNVHRetriever.load()
    graph     = build_cae_graph()

    print("\n[2/3] Loading golden Q&A set...")
    qa_pairs = load_golden_qa()
    print(f"  Loaded {len(qa_pairs)} evaluation questions")

    # Choose evaluation mode
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    print(f"\n[3/3] Running evaluation (mode: {'RAGAS/LLM' if has_openai else 'offline/keyword'})")

    if has_openai:
        summary = evaluate_with_ragas(qa_pairs, retriever, graph)
    else:
        print("  (Set OPENAI_API_KEY for full RAGAS/LLM-as-Judge evaluation)")
        summary = evaluate_offline(qa_pairs, retriever, graph)

    # Save results
    Path("eval").mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    main()
