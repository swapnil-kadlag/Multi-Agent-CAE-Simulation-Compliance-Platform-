"""
build_all.py
─────────────────────────────────────────────────────────────────────────────
Rebuild all ML artifacts (retriever index + surrogate model) from scratch.

IMPORTANT: Always run THIS file, not the individual module scripts directly.
Running build_retriever.py or surrogate_model.py as __main__ causes pickle
to store class names as '__main__.NVHDocument' which breaks loading from
other modules. This wrapper imports them as proper package modules so pickle
stores 'knowledge_base.build_retriever.NVHDocument' — the correct path.

Usage:
    python build_all.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("CAE NVH Platform — Build All ML Artifacts")
print("=" * 60)

# ── Step 1: Build retriever (FAISS + BM25 index) ─────────────────────────────
print("\n[1/2] Building hybrid retriever index (FAISS + BM25)...")
from knowledge_base.build_retriever import (
    load_knowledge_base, build_faiss_index, build_bm25_index, HybridNVHRetriever
)

parents, children = load_knowledge_base("data/synthetic/nvh_knowledge_base.json")
faiss_index, vectorizer, tfidf_matrix = build_faiss_index(children)
bm25_index = build_bm25_index(children)

retriever = HybridNVHRetriever(
    parents      = parents,
    children     = children,
    faiss_index  = faiss_index,
    vectorizer   = vectorizer,
    tfidf_matrix = tfidf_matrix,
    bm25_index   = bm25_index,
)
retriever.save("data/retriever")

# Quick smoke test
test_results = retriever.retrieve("electric motor electromagnetic whine 500 Hz", top_k=3)
print(f"  Smoke test: retrieved {len(test_results)} cases")
for r in test_results:
    print(f"    [{r.metadata['case_id']}] {r.metadata['title'][:55]}")
print("  Retriever: OK")

# ── Step 2: Train surrogate model (GradientBoosting) ─────────────────────────
print("\n[2/2] Training NVH surrogate model...")
from tools.surrogate_model import load_training_data, train_surrogate, save_model, predict_nvh_level

X, y = load_training_data()
train_result = train_surrogate(X, y)
save_model(train_result)

# Quick smoke test
pred = predict_nvh_level(rpm=3000, load_nm=80)
print(f"  Smoke test: 3000 rpm / 80 Nm → {pred['predicted_nvh_db']} dB (severity {pred['severity']})")
print("  Surrogate model: OK")

print("\n" + "=" * 60)
print("All artifacts built successfully!")
print("=" * 60)
print()
print("Next steps:")
print("  Start server:  uvicorn api.main:app --reload --port 8000")
print("  Run tests:     pytest tests/ -v")
print("  Run eval:      python eval/run_eval.py")
