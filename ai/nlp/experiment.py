"""Semantic search across three eras of NLP: lexical, dense, generative.

Experiment backing the webinar "From pipelines to prompts: NLP in the LLM era"
(see wb_llm_era.qmd). One task - semantic search - on one benchmark:

    BEIR/SciFact: 5,183 scientific abstracts, ~300 test queries
    (scientific claims) with gold relevance judgements.

The eras:

1. Lexical search   BM25 (no model, exact-term matching)
2. Dense retrieval  embedding similarity, three model sizes
3. Generative       small LLM (llama3.2 via ollama) reranking a shortlist

Plus the standard hybrid trick (reciprocal rank fusion of eras 1 and 2).

Everything is scored with nDCG@10 on the same qrels. No method is given an
unfair edge: the numbers land where they land and the webinar conclusion
follows the results, not the other way around.

Data:   curl -L -o scifact.zip
        https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
Models: ollama pull all-minilm nomic-embed-text mxbai-embed-large llama3.2

Run with `uv run python -I experiment.py` (SMOKE=1 for a quick validation
run on a corpus subsample).
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import ollama
import polars as pl
from rank_bm25 import BM25Okapi

# --- Config -------------------------------------------------------------------

SMOKE = os.environ.get("SMOKE") == "1"  # tiny corpus, few queries, fast checks
DATA = Path(__file__).parent / "data/scifact/scifact"
K = 10  # nDCG depth
LLM_QUERIES = 5 if SMOKE else 50  # LLM reranking is slow: subset only
RRF_K = 60  # reciprocal rank fusion constant

# Embedding models, oldest/smallest first: does model quality flip the
# BM25 vs dense ranking? (SciFact is full of rare exact terms - a BM25-friendly
# domain - so this is a fair fight, not a strawman.)
# "mxbai-embed-large" (335M) is a stronger third candidate if disk allows.
EMB_MODELS = [
    "all-minilm",  # 23M params, 2020-era miniature
    "nomic-embed-text",  # 137M params, modern
]
LLM_MODEL = "llama3.2"  # 3B params

RERANK_FORMAT = {  # structured output for the reranker
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
}

RESULTS = []


def timed(fn, *args, **kwargs):
    """Return (result, wall-clock seconds) for a call."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def tokenize(text: str) -> list[str]:
    """Crude but sufficient for BM25: lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int = K) -> float:
    """nDCG@k, graded gains (SciFact qrels happen to be binary)."""
    dcg = sum(qrels.get(d, 0) / np.log2(r + 2) for r, d in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


# --- Corpus, queries, relevance judgements -------------------------------------

corpus = {}  # doc id -> {"title": ..., "text": ...}
with open(DATA / "corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = {
            "title": doc.get("title") or "",
            "text": doc["text"],
        }

if SMOKE:  # shrink the corpus, keep only queries still answerable from it
    corpus = dict(list(corpus.items())[:500])
docs = list(corpus)

queries = {}  # query id -> text
with open(DATA / "queries.jsonl", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q["text"]

qrels = {}  # query id -> {doc id: relevance}
with open(DATA / "qrels" / "test.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        qid, did, score = line.split("\t")
        qrels.setdefault(qid, {})[did.strip()] = int(score)

eval_qids = [
    qid for qid in qrels if qid in queries and any(d in corpus for d in qrels[qid])
]
if SMOKE:
    eval_qids = eval_qids[:30]
print(
    f"corpus: {len(corpus)} docs | eval queries: {len(eval_qids)}"
    f" | LLM subset: {LLM_QUERIES}\n"
)

full_text = {d: f"{corpus[d]['title']}. {corpus[d]['text']}" for d in docs}


# --- Era 1: lexical search (BM25) ----------------------------------------------

# No model at all: score docs by exact (unigram) term overlap with the query.
# Fails on vocabulary mismatch - and on SciFact it is still a serious
# baseline: scientific queries are full of rare exact terms.

t0 = time.perf_counter()
bm25 = BM25Okapi([tokenize(full_text[d]) for d in docs])
bm25_index_time = time.perf_counter() - t0

t0 = time.perf_counter()
bm25_rankings = {  # query id -> doc ids, best first
    qid: [docs[i] for i in np.argsort(-bm25.get_scores(tokenize(queries[qid])))]
    for qid in eval_qids
}
bm25_search_time = time.perf_counter() - t0

bm25_ndcg = float(np.mean([ndcg_at_k(bm25_rankings[q], qrels[q]) for q in eval_qids]))
RESULTS.append(
    dict(
        method="1a BM25 (lexical)",
        ndcg10=bm25_ndcg,
        index_s=bm25_index_time,
        search_ms=1000 * bm25_search_time / len(eval_qids),
    )
)
print(
    f"era 1  BM25             nDCG@10 {bm25_ndcg:.3f}"
    f"  ({bm25_search_time * 1000 / len(eval_qids):.1f} ms/query)"
)


# --- Era 2: dense retrieval (embeddings) ---------------------------------------

CACHE = Path(__file__).parent / ".emb_cache"


def embed(texts: list[str], model: str, batch: int = 64) -> np.ndarray:
    """Embed texts with an ollama model, L2-normalized (disk-cached:
    the doc list is deterministic, so length is a sufficient cache key)."""
    cache = CACHE / f"{model}-{len(texts)}.npy"
    if cache.exists():
        return np.load(cache)
    CACHE.mkdir(exist_ok=True)
    out = []
    for i in range(0, len(texts), batch):
        resp = ollama.embed(model=model, input=texts[i : i + batch])
        out.extend(resp["embeddings"])
        if (i // batch) % 10 == 0:
            print(f"  {model}: {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    emb = np.asarray(out, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(cache, emb)
    return emb


E = {}  # model -> (doc embeddings, per-query rankings, timings)
best_dense = None  # (model, rankings) of the era 2 winner
for model in EMB_MODELS:
    t0 = time.perf_counter()
    doc_emb = embed([full_text[d] for d in docs], model)
    index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rankings = {}
    for qid in eval_qids:
        q = ollama.embed(model=model, input=[queries[qid]])["embeddings"][0]
        q = np.asarray(q, dtype=np.float32)
        q /= np.linalg.norm(q)
        rankings[qid] = [docs[i] for i in np.argsort(-(doc_emb @ q))]
    search_time = time.perf_counter() - t0

    ndcg = float(np.mean([ndcg_at_k(rankings[q], qrels[q]) for q in eval_qids]))
    RESULTS.append(
        dict(
            method=f"2 {model}",
            ndcg10=ndcg,
            index_s=index_time,
            search_ms=1000 * search_time / len(eval_qids),
        )
    )
    print(
        f"era 2  {model:<17} nDCG@10 {ndcg:.3f}"
        f"  ({index_time:.0f}s index, "
        f"{1000 * search_time / len(eval_qids):.1f} ms/query)"
    )
    if best_dense is None or ndcg > best_dense[0]:
        best_dense = (ndcg, model, rankings)

_, dense_model, dense_rankings = best_dense


# --- Hybrid: reciprocal rank fusion of eras 1 and 2 -----------------------------


def rrf(rankings: dict[str, list[str]]) -> dict[str, list[str]]:
    """Fuse rankings: score = sum over methods of 1 / (RRF_K + rank)."""
    fused = {}
    for qid in eval_qids:
        scores = {}
        for ranking in rankings.values():
            for rank, d in enumerate(ranking[qid]):
                scores[d] = scores.get(d, 0.0) + 1.0 / (RRF_K + rank)
        fused[qid] = sorted(scores, key=scores.get, reverse=True)
    return fused


rrf_rankings = rrf({"bm25": bm25_rankings, "dense": dense_rankings})
rrf_ndcg = float(np.mean([ndcg_at_k(rrf_rankings[q], qrels[q]) for q in eval_qids]))
RESULTS.append(
    dict(
        method=f"2+1 hybrid RRF ({dense_model} + BM25)",
        ndcg10=rrf_ndcg,
        index_s=np.nan,
        search_ms=np.nan,
    )
)
print(f"hyb   RRF fusion       nDCG@10 {rrf_ndcg:.3f}")


# --- Era 3: LLM reranker ---------------------------------------------------------

# A generative model cannot scan the corpus; it can only reorder a shortlist.
# Standard pattern: retrieve top-k with era 1/2 methods, let the LLM rerank.
# This is the only place compute explodes: one LLM call per query.


def rerank(query: str, shortlist: list[str]) -> tuple[list[str], bool]:
    listing = "\n".join(
        f"{n + 1}. {full_text[d][:350]}" for n, d in enumerate(shortlist)
    )
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a search reranker."},
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments:\n{listing}\n\n"
                "The documents are numbered starting from 1. "
                "Rank them by relevance to the query. "
                "Reply with the document numbers, most relevant "
                "first.",
            },
        ],
        format=RERANK_FORMAT,
        options={"temperature": 0, "num_predict": 120},
    )
    ok = True
    try:
        nums = list(
            dict.fromkeys(  # dedupe, order-preserving
                json.loads(resp["message"]["content"])["ranking"]
            )
        )[:K]
        ranked = [shortlist[n - 1] for n in nums if 1 <= n <= len(shortlist)]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        ok = False
        ranked = []
        with open(
            Path(__file__).parent / ".rerank_failures.log", "a", encoding="utf-8"
        ) as f:
            f.write(
                f"query: {query!r}\nerror: {e}\n"
                f"response: {resp['message']['content']!r}\n\n"
            )
    # pad with the retriever order (docs the LLM did not list)
    return ranked + [d for d in shortlist if d not in ranked], ok


# Unload the embedding models from the ollama server so the LLM has
# room to load (this box has 7.8 GiB RAM total).
for m in EMB_MODELS:
    ollama.embed(model=m, input=["."], keep_alive=0)

llm_qids = sorted(eval_qids)[:LLM_QUERIES]
t0 = time.perf_counter()
n_fail = 0
llm_rankings = {}
for qid in llm_qids:
    shortlist = list(dict.fromkeys(bm25_rankings[qid][:K] + dense_rankings[qid][:K]))
    for attempt in range(3):
        try:
            llm_rankings[qid], ok = rerank(queries[qid], shortlist)
            break
        except ollama.ResponseError:  # runner hiccup: give it a moment
            print(f"  rerank retry {attempt + 1} for query {qid}", flush=True)
            time.sleep(10)
    else:  # all retries failed: fall back to the retriever order
        llm_rankings[qid], ok = shortlist[:K], False
    n_fail += not ok
llm_time = time.perf_counter() - t0

llm_ndcg = float(np.mean([ndcg_at_k(llm_rankings[q], qrels[q]) for q in llm_qids]))
RESULTS.append(
    dict(
        method=f"3 LLM rerank on {dense_model}+BM25 top-{K}",
        ndcg10=llm_ndcg,
        index_s=np.nan,
        search_ms=1000 * llm_time / len(llm_qids),
    )
)
print(
    f"era 3 llama3.2 rerank nDCG@10 {llm_ndcg:.3f}"
    f"  ({llm_time / len(llm_qids):.1f} s/query,"
    f" {len(llm_qids)} queries, {n_fail} unparseable)"
)


# --- Summary ---------------------------------------------------------------------

df = pl.DataFrame(RESULTS)
print("\n== BEIR/SciFact - nDCG@10 ==")
print(
    df.to_string(
        index=False,
        formatters={
            "ndcg10": "{:.3f}".format,
            "index_s": "{:.0f}".format,
            "search_ms": "{:.1f}".format,
        },
    )
)
df.to_json(Path(__file__).parent / "results.json", orient="records", indent=2)
