"""nDCG@10 of the retrieval methods restricted to the 50-query LLM subset,
so era 3 is compared apples to apples."""

import json
import re
from pathlib import Path

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent / "data/scifact/scifact"
CACHE = Path(__file__).parent / ".emb_cache"
K, RRF_K = 10, 60


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
llm_qids = sorted(eval_qids)[:50]  # same subset as experiment.py era 3

full_text = {d: f"{corpus[d]['title']}. {corpus[d]['text']}" for d in docs}
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bm25 = BM25Okapi([tok(full_text[d]) for d in docs])

E = np.load(CACHE / "nomic-embed-text-5183.npy")  # doc order == docs


def mean_ndcg(rankings):
    return float(np.mean([ndcg_at_k(rankings[q], qrels[q]) for q in llm_qids]))


bm25_r, dense_r = {}, {}
for qid in llm_qids:
    bm25_r[qid] = [docs[i] for i in np.argsort(-bm25.get_scores(tok(queries[qid])))]
    q = np.asarray(
        ollama.embed(model="nomic-embed-text", input=[queries[qid]])["embeddings"][0],
        dtype=np.float32,
    )
    q /= np.linalg.norm(q)
    dense_r[qid] = [docs[i] for i in np.argsort(-(E @ q))]

rrf_r = {}
for qid in llm_qids:
    s = {}
    for ranking in (bm25_r[qid], dense_r[qid]):
        for rank, d in enumerate(ranking):
            s[d] = s.get(d, 0.0) + 1.0 / (RRF_K + rank)
    rrf_r[qid] = sorted(s, key=s.get, reverse=True)

print("on the 50-query LLM subset (nDCG@10):")
for name, r in [("BM25", bm25_r), ("nomic-embed-text", dense_r), ("RRF hybrid", rrf_r)]:
    print(f"  {name:<18} {mean_ndcg(r):.3f}")
print("  llama3.2 rerank   0.615  (from experiment.py)")
