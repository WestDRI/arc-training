"""One-shot probe: what does llama3.2 actually return for the rerank prompt?"""

import json
import re

import numpy as np
import ollama
from rank_bm25 import BM25Okapi

DATA = "data/scifact/scifact"

corpus = {}
with open(f"{DATA}/corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = {"title": doc.get("title") or "", "text": doc["text"]}
corpus = dict(list(corpus.items())[:500])
docs = list(corpus)

qrels = {}
with open(f"{DATA}/qrels/test.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        qid, did, score = line.split("\t")
        qrels.setdefault(qid, {})[did.strip()] = int(score)

queries = {}
with open(f"{DATA}/queries.jsonl", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q["text"]

eval_qids = [q for q in qrels if q in queries and any(d in corpus for d in qrels[q])]
qid = min(eval_qids)
query = queries[qid]

full_text = {d: f"{corpus[d]['title']}. {corpus[d]['text']}" for d in docs}
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bm25 = BM25Okapi([tok(full_text[d]) for d in docs])
shortlist = [docs[i] for i in np.argsort(-bm25.get_scores(tok(query)))[:10]]

print(f"qid {qid}: {query!r}")
print(f"relevant in corpus: {[(d, qrels[qid][d]) for d in qrels[qid] if d in corpus]}")
print(f"relevant in shortlist: {[d for d in qrels[qid] if d in shortlist]}\n")

listing = "\n".join(f"{n + 1}. {full_text[d][:350]}" for n, d in enumerate(shortlist))
print(listing[:600], "...\n")

resp = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a search reranker."},
        {
            "role": "user",
            "content": f"Query: {query}\n\nDocuments:\n{listing}\n\n"
            "Rank the documents by relevance to the query. "
            "Reply with the document numbers, most relevant first.",
        },
    ],
    format={
        "type": "object",
        "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
        "required": ["ranking"],
    },
    options={"temperature": 0, "num_predict": 120},
)
print("RAW RESPONSE:", resp["message"]["content"])
