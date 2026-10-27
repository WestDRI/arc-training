"""nDCG@10 of the retrieval methods restricted to the 50-query LLM subset,
so era 4 is compared apples to apples."""

import json
import re
from pathlib import Path

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent.parent  # scripts live in src/, data one level up
DATA = HERE / "data/scifact/scifact"
CACHE = HERE / ".emb_cache"
K, RRF_K = 10, 60
QUERY_PREFIX = {
    "mxbai-embed-large": "Represent this sentence for searching relevant passages: "
}


def ndcg_at_k(ranked, qrels, k=K):
    dcg = sum(qrels.get(d, 0) / np.log2(r + 2) for r, d in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


corpus = {}
with open(DATA / "corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = {"title": doc.get("title") or "", "text": doc["text"]}
docs = list(corpus)
queries, qrels = {}, {}
with open(DATA / "queries.jsonl", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q["text"]
with open(DATA / "qrels" / "test.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        qid, did, score = line.split("\t")
        qrels.setdefault(qid, {})[did.strip()] = int(score)

eval_qids = [q for q in qrels if q in queries and any(d in corpus for d in qrels[q])]
llm_qids = sorted(eval_qids)[:50]  # same subset as experiment.py era 4

# era-4 settings from the full run: every embedder + the reranker scores
rows = json.loads((HERE / "results" / "results.json").read_text())
emb_models = [r["method"].split()[1] for r in rows
              if "rerank" not in r["method"] and "BM25" not in r["method"]
              and "hybrid" not in r["method"]]

full_text = {d: f"{corpus[d]['title']}. {corpus[d]['text']}" for d in docs}
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bm25 = BM25Okapi([tok(full_text[d]) for d in docs])


def mean_ndcg(rankings):
    return float(np.mean([ndcg_at_k(rankings[q], qrels[q]) for q in llm_qids]))


bm25_r = {
    qid: [docs[i] for i in np.argsort(-bm25.get_scores(tok(queries[qid])))]
    for qid in llm_qids
}
dense_r = {}
for model in emb_models:
    E = np.load(CACHE / f"{model}-{len(docs)}.npy")  # doc order == docs
    dense_r[model] = {}
    for qid in llm_qids:
        q = np.asarray(
            ollama.embed(
                model=model,
                input=[QUERY_PREFIX.get(model, "") + queries[qid]],
            )["embeddings"][0],
            dtype=np.float32,
        )
        q /= np.linalg.norm(q)
        dense_r[model][qid] = [docs[i] for i in np.argsort(-(E @ q))]

# hybrid partner: the era-3 winner of the full run (not of this subset)
full_ndcg = {r["method"].split()[1]: r["ndcg10"] for r in rows
             if r["method"].split()[1] in emb_models}
best = max(emb_models, key=lambda m: full_ndcg[m])

rrf_r = {}
for qid in llm_qids:
    s = {}
    for ranking in (bm25_r[qid], dense_r[best][qid]):
        for rank, d in enumerate(ranking):
            s[d] = s.get(d, 0.0) + 1.0 / (RRF_K + rank)
    rrf_r[qid] = sorted(s, key=s.get, reverse=True)

print(f"on the 50-query LLM subset (nDCG@10):")
print(f"  {'BM25':<18} {mean_ndcg(bm25_r):.3f}")
for model in emb_models:
    print(f"  {model:<18} {mean_ndcg(dense_r[model]):.3f}")
print(f"  {'RRF hybrid':<18} {mean_ndcg(rrf_r):.3f}")
for r in rows:  # rerankers were scored on this very subset
    if "rerank" in r["method"]:
        print(f"  {r['method'].split(' ', 1)[1]:<42} {r['ndcg10']:.3f}"
              "  (from experiment.py)")
