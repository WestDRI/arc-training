# start ollama first
# normal workflow: sudo systemctl start ollama.service
# in bbox, no systemd, start with: ollama serve

"""Semantic search across the data-driven eras of NLP: lexical, dense, generative.

Experiment backing the webinar "From pipelines to prompts: NLP in the LLM era"
(see wb_llm_era.qmd). One task - semantic search - on one benchmark:

    BEIR/SciFact: 5,183 scientific abstracts, ~300 test queries
    (scientific claims) with gold relevance judgements.

The eras (numbered as in the webinar slides, where era 1 is the symbolic
era, not tested here as it involves no data-driven method):

2. Lexical search   BM25 (no model, exact-term matching)
3. Dense retrieval  embedding similarity, three model sizes
4. Generative       LLMs (llama3.2 3B, llama3.1 8B via ollama) reranking
                    a shortlist, two ways: the brute-force listwise prompt
                    and pointwise P(yes) scoring read from the model
                    logprobs

Plus the standard hybrid trick (reciprocal rank fusion of eras 2 and 3).

Everything is scored with nDCG@10 on the same qrels. No method is given an
unfair edge: the numbers land where they land and the webinar conclusion
follows the results, not the other way around.

Data:   curl -L -o scifact.zip
        https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
Models: ollama pull all-minilm nomic-embed-text mxbai-embed-large llama3.2 llama3.1:8b

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

# --- Config ------------------------------------------------------------------

SMOKE = os.environ.get("SMOKE") == "1"  # tiny corpus, few queries, fast checks
DATA = Path(__file__).parent / "data/scifact/scifact"
K = 10  # nDCG depth
LLM_QUERIES = 5 if SMOKE else 50  # LLM reranking is slow: subset only
RRF_K = 60  # reciprocal rank fusion constant

# Embedding models, oldest/smallest first: does model quality flip the
# BM25 vs dense ranking? (SciFact is full of rare exact terms - a BM25-friendly
# domain - so this is a fair fight, not a strawman.)
EMB_MODELS = [
    "all-minilm",  # 23M params, 2020-era miniature
    "nomic-embed-text",  # 137M params, modern
    "mxbai-embed-large",  # 335M params, modern
]
# mxbai wants a task instruction prefixed to queries (per its model card);
# nomic's ollama template already adds its own prefixes.
QUERY_PREFIX = {
    "mxbai-embed-large": "Represent this sentence for searching relevant passages: "
}
LLM_MODELS = ["llama3.2", "llama3.1:8b"]  # 3B then 8B: does scaling fix era 4?
# Reranking full abstracts (median ~360 tokens, p99 ~790) means ~8-16k token
# prompts, far past ollama's 4k default context. Weights + 16k of KV cache only
# fit the 6 GiB GPU for the 8B model with flash attention and a compressed cache.
LLM_OPTS = {
    "llama3.2": {"num_ctx": 16384},
    "llama3.1:8b": {
        "num_ctx": 16384,
        "flash_attention": True,
        "kv_cache_type": "q8_0",
    },
}

RERANK_FORMAT = {  # structured output for the reranker
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
}

RESULTS = []


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


# --- Era 2: lexical search (BM25) ----------------------------------------------

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
    {
        "method": "2 BM25 (lexical)",
        "ndcg10": bm25_ndcg,
        "index_s": bm25_index_time,
        "search_ms": 1000 * bm25_search_time / len(eval_qids),
    }
)
print(
    f"era 2  BM25             nDCG@10 {bm25_ndcg:.3f}"
    f"  ({bm25_search_time * 1000 / len(eval_qids):.1f} ms/query)"
)


# --- Era 3: dense retrieval (embeddings) ---------------------------------------

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


best_dense = None  # (ndcg, model, rankings) of the era 3 winner
for model in EMB_MODELS:
    t0 = time.perf_counter()
    doc_emb = embed([full_text[d] for d in docs], model)
    index_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rankings = {}
    for qid in eval_qids:
        q = ollama.embed(
            model=model, input=[QUERY_PREFIX.get(model, "") + queries[qid]]
        )["embeddings"][0]
        q = np.asarray(q, dtype=np.float32)
        q /= np.linalg.norm(q)
        rankings[qid] = [docs[i] for i in np.argsort(-(doc_emb @ q))]
    search_time = time.perf_counter() - t0

    ndcg = float(np.mean([ndcg_at_k(rankings[q], qrels[q]) for q in eval_qids]))
    RESULTS.append(
        {
            "method": f"3 {model}",
            "ndcg10": ndcg,
            "index_s": index_time,
            "search_ms": 1000 * search_time / len(eval_qids),
        }
    )
    print(
        f"era 3  {model:<17} nDCG@10 {ndcg:.3f}"
        f"  ({index_time:.0f}s index, "
        f"{1000 * search_time / len(eval_qids):.1f} ms/query)"
    )
    if best_dense is None or ndcg > best_dense[0]:
        best_dense = (ndcg, model, rankings)

_, dense_model, dense_rankings = best_dense


# --- Hybrid: reciprocal rank fusion of eras 2 and 3 -----------------------------


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
    {
        "method": f"3+2 hybrid RRF ({dense_model} + BM25)",
        "ndcg10": rrf_ndcg,
        "index_s": np.nan,
        "search_ms": np.nan,
    }
)
print(f"hyb   RRF fusion       nDCG@10 {rrf_ndcg:.3f}")


# --- Era 4: LLM rerankers ---------------------------------------------------------

# A generative model cannot scan the corpus; it can only reorder a shortlist.
# Standard pattern: retrieve top-k with era 2/3 methods, let the LLM rerank.
# This is the only place compute explodes: one LLM call per query.


def rerank(model: str, query: str, shortlist: list[str]) -> tuple[list[str], bool]:
    listing = "\n".join(f"{n + 1}. {full_text[d]}" for n, d in enumerate(shortlist))
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a search reranker."},
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments:\n{listing}\n\n"
                f"There are {len(shortlist)} documents, numbered 1 to {len(shortlist)}. "
                f"Reply with the {K} most relevant document numbers, "
                "most relevant first.",
            },
        ],
        format=RERANK_FORMAT,
        options={"temperature": 0, "num_predict": 200, **LLM_OPTS[model]},
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


# Unload the embedding models from the ollama server so the LLMs have
# room to load (they compete for the 6 GiB of GPU memory).
for m in EMB_MODELS:
    ollama.embed(model=m, input=["."], keep_alive=0)

llm_qids = sorted(eval_qids)[:LLM_QUERIES]
shortlists = {  # same shortlists for every LLM: only the reranker changes
    qid: list(dict.fromkeys(bm25_rankings[qid][:K] + dense_rankings[qid][:K]))
    for qid in llm_qids
}
for llm in LLM_MODELS:
    t0 = time.perf_counter()
    n_fail = 0
    llm_rankings = {}
    for qid in llm_qids:
        for attempt in range(3):
            try:
                llm_rankings[qid], ok = rerank(llm, queries[qid], shortlists[qid])
                break
            except ollama.ResponseError:  # runner hiccup: give it a moment
                print(f"  rerank retry {attempt + 1} for query {qid}", flush=True)
                time.sleep(10)
        else:  # all retries failed: fall back to the retriever order
            llm_rankings[qid], ok = shortlists[qid][:K], False
        n_fail += not ok
    llm_time = time.perf_counter() - t0
    ollama.generate(model=llm, keep_alive=0)  # free memory for the next model

    llm_ndcg = float(
        np.mean([ndcg_at_k(llm_rankings[q], qrels[q]) for q in llm_qids])
    )
    RESULTS.append(
        {
            "method": f"4 {llm} listwise rerank on {dense_model}+BM25 top-{K}",
            "ndcg10": llm_ndcg,
            "index_s": np.nan,
            "search_ms": 1000 * llm_time / len(llm_qids),
        }
    )
    print(
        f"era 4 {llm:<15} listwise rerank nDCG@10 {llm_ndcg:.3f}"
        f"  ({llm_time / len(llm_qids):.1f} s/query,"
        f" {len(llm_qids)} queries, {n_fail} unparseable)"
    )


# --- Era 4, take 2: pointwise scoring ---------------------------------------------

# The listwise prompt asks the model to do arithmetic over list positions -
# exactly what small transformers fumble (counting loops, countdown replies).
# Alternative: judge one document at a time and read the answer from the
# model's own distribution: score = P(yes) / (P(yes) + P(no)) on the first
# generated token. No list to count, no JSON to truncate, no fallback. And
# one abstract per prompt fits a small context window (no 16k KV cache).

YES = ("yes", "Yes", " yes", " Yes")
NO = ("no", "No", " no", " No")


def p_yes(resp) -> float:
    """P(yes) / (P(yes) + P(no)) from the first generated token's logprobs."""
    lp = {e["token"]: e["logprob"] for e in resp["logprobs"][0]["top_logprobs"]}
    yes = max((lp[t] for t in YES if t in lp), default=None)
    no = max((lp[t] for t in NO if t in lp), default=None)
    if yes is None and no is None:
        return 0.5  # the model dodged the question: no signal
    if no is None:
        return 1.0
    if yes is None:
        return 0.0
    ey, en = np.exp(yes), np.exp(no)
    return float(ey / (ey + en))


for llm in LLM_MODELS:
    t0 = time.perf_counter()
    llm_rankings = {}
    for qid in llm_qids:
        for attempt in range(3):
            try:
                scores = []
                for d in shortlists[qid]:
                    resp = ollama.chat(
                        model=llm,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a search reranker.",
                            },
                            {
                                "role": "user",
                                "content": f"Claim: {queries[qid]}\n\n"
                                f"Abstract: {full_text[d]}\n\n"
                                "Is this abstract relevant to the claim? "
                                "Answer with exactly one word: yes or no.",
                            },
                        ],
                        logprobs=True,
                        top_logprobs=10,
                        options={"temperature": 0, "num_predict": 1,
                                 "num_ctx": 2048},
                    )
                    scores.append(p_yes(resp))
                break
            except ollama.ResponseError:  # runner hiccup: give it a moment
                print(f"  pointwise retry {attempt + 1} for query {qid}",
                      flush=True)
                time.sleep(10)
        else:  # all retries failed: neutral scores keep the retriever order
            scores = [0.5] * len(shortlists[qid])
        # stable sort: ties keep the retriever order
        order = np.argsort(-np.asarray(scores), kind="stable")
        llm_rankings[qid] = [shortlists[qid][i] for i in order]
    llm_time = time.perf_counter() - t0
    ollama.generate(model=llm, keep_alive=0)  # free memory for the next model

    llm_ndcg = float(
        np.mean([ndcg_at_k(llm_rankings[q], qrels[q]) for q in llm_qids])
    )
    RESULTS.append(
        {
            "method": f"4 {llm} pointwise rerank on {dense_model}+BM25 top-{K}",
            "ndcg10": llm_ndcg,
            "index_s": np.nan,
            "search_ms": 1000 * llm_time / len(llm_qids),
        }
    )
    print(
        f"era 4 {llm:<15} pointwise rerank nDCG@10 {llm_ndcg:.3f}"
        f"  ({llm_time / len(llm_qids):.1f} s/query,"
        f" {len(llm_qids)} queries)"
    )


# --- Summary ---------------------------------------------------------------------

df = pl.DataFrame(RESULTS)
print("\n== BEIR/SciFact - nDCG@10 ==")
print(
    str(
        df.with_columns(
            pl.col("ndcg10").map_elements(lambda x: f"{x:.3f}", return_dtype=pl.String),
            pl.col("index_s").map_elements(
                lambda x: f"{x:.0f}", return_dtype=pl.String
            ),
            pl.col("search_ms").map_elements(
                lambda x: f"{x:.1f}", return_dtype=pl.String
            ),
        )
    )
)

df.write_json(Path(__file__).parent / "results.json")
