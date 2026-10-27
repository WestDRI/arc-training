"""Pointwise LLM reranking: score each shortlisted document with the model's
own P(yes) instead of asking it to generate a ranking.

Companion to experiment.py era 4 (listwise reranking): same 50 queries, same
shortlists (BM25 + dense top-10 union), same qrels. Instead of one giant
"rank these documents" prompt (16k-token context, JSON contract), each
document gets its own small prompt, and the score is read from the first
generated token's logprobs:

    score = P(yes) / (P(yes) + P(no))

Nothing to count, no JSON to truncate, no fallback needed.

Run with `uv run python -I src/pointwise_rerank.py` (ollama serve first).
POINTWISE_QUERIES=3 for a quick sanity check.
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

HERE = Path(__file__).resolve().parent.parent  # scripts live in src/, data one level up
DATA = HERE / "data/scifact/scifact"
SRC = HERE / "src"  # scratch artifacts: .emb_cache/, lora_reranker/
CACHE = SRC / ".emb_cache"
K = 10
N_QUERIES = int(os.environ.get("POINTWISE_QUERIES", "50"))
QUERY_PREFIX = {
    "mxbai-embed-large": "Represent this sentence for searching relevant passages: "
}
LLM_MODELS = ["llama3.2", "llama3.1:8b"]
# one abstract per prompt: a small context window is enough (and cheap)
LLM_OPTS = {"temperature": 0, "num_predict": 1, "num_ctx": 2048}
YES = ("yes", "Yes", " yes", " Yes")
NO = ("no", "No", " no", " No")


def ndcg_at_k(ranked, qrels, k=K):
    dcg = sum(qrels.get(d, 0) / np.log2(r + 2) for r, d in enumerate(ranked[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def p_yes(resp) -> float:
    """P(yes)/(P(yes)+P(no)) from the first generated token's top logprobs."""
    lp = {e["token"]: e["logprob"]
          for e in resp["logprobs"][0]["top_logprobs"]}
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


# --- Same data, same shortlists as experiment.py ---------------------------------

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
llm_qids = sorted(eval_qids)[:N_QUERIES]
print(f"corpus: {len(corpus)} docs | pointwise queries: {len(llm_qids)}\n", flush=True)

full_text = {d: f"{corpus[d]['title']}. {corpus[d]['text']}" for d in docs}
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bm25 = BM25Okapi([tok(full_text[d]) for d in docs])

dense_model = max(
    (r for r in json.loads((HERE / "results" / "results.json").read_text())
     if "rerank" not in r["method"] and "BM25" not in r["method"]
     and "hybrid" not in r["method"]),
    key=lambda r: r["ndcg10"],
)["method"].split()[1]
E = np.load(CACHE / f"{dense_model}-{len(docs)}.npy")  # doc order == docs

shortlists = {}
for qid in llm_qids:
    bm25_top = [docs[i] for i in np.argsort(-bm25.get_scores(tok(queries[qid])))][:K]
    q = np.asarray(
        ollama.embed(model=dense_model,
                     input=[QUERY_PREFIX.get(dense_model, "") + queries[qid]])
        ["embeddings"][0], dtype=np.float32)
    q /= np.linalg.norm(q)
    dense_top = [docs[i] for i in np.argsort(-(E @ q))][:K]
    shortlists[qid] = list(dict.fromkeys(bm25_top + dense_top))

for m in (dense_model,):  # free the GPU for the LLMs
    ollama.embed(model=m, input=["."], keep_alive=0)


# --- Era 4, take 2: score, don't generate ---------------------------------------

def pointwise(model: str, query: str, shortlist: list[str]) -> list[str]:
    """Rank the shortlist by the model's P(yes), ties keep retriever order."""
    scores = []
    for d in shortlist:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a search reranker."},
                {
                    "role": "user",
                    "content": f"Claim: {query}\n\nAbstract: {full_text[d]}\n\n"
                    "Is this abstract relevant to the claim? "
                    "Answer with exactly one word: yes or no.",
                },
            ],
            logprobs=True,
            top_logprobs=10,
            options=LLM_OPTS,
        )
        scores.append(p_yes(resp))
    order = np.argsort(-np.asarray(scores), kind="stable")
    return [shortlist[i] for i in order], scores


RESULTS = []
for llm in LLM_MODELS:
    t0 = time.perf_counter()
    ndcg, dodged = [], 0
    for n, qid in enumerate(llm_qids):
        for attempt in range(3):
            try:
                ranked, scores = pointwise(llm, queries[qid], shortlists[qid])
                break
            except ollama.ResponseError:  # runner hiccup: give it a moment
                print(f"  retry {attempt + 1} for query {qid}", flush=True)
                time.sleep(10)
        else:  # all retries failed: fall back to the retriever order
            ranked = shortlists[qid]
        dodged += sum(s == 0.5 for s in scores)
        ndcg.append(ndcg_at_k(ranked, qrels[qid]))
        if n < 3:  # exhibit material: gold vs distractors for the first queries
            gold = [d for d in qrels[qid] if d in shortlists[qid]]
            print(f"  query {n + 1}: {queries[qid][:60]!r}", flush=True)
            for d, s in sorted(zip(shortlists[qid], scores), key=lambda x: -x[1])[:3]:
                mark = " <- gold" if d in gold else ""
                print(f"    P(yes) {s:.2f}  {full_text[d][:60]!r}{mark}", flush=True)
    llm_time = time.perf_counter() - t0
    ollama.generate(model=llm, keep_alive=0)  # free memory for the next model

    mean_ndcg = float(np.mean(ndcg))
    RESULTS.append(
        {
            "method": f"4 {llm} pointwise P(yes) on {dense_model}+BM25 top-{K}",
            "ndcg10": mean_ndcg,
            "search_ms": 1000 * llm_time / len(llm_qids),
            "dodged": dodged,
        }
    )
    print(
        f"era 4 {llm:<15} pointwise nDCG@10 {mean_ndcg:.3f}"
        f"  ({llm_time / len(llm_qids):.1f} s/query,"
        f" {dodged} dodged answers)\n",
        flush=True,
    )

Path(HERE / "results" / "run05_results_pointwise.json").write_text(json.dumps(RESULTS, indent=2))
