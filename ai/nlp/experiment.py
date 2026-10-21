"""Three eras of NLP on one corpus: classic pipeline, embeddings, small LLM.

Experiment backing the webinar "From pipelines to prompts: NLP in the LLM era"
(see wb_llm_era.qmd). Same corpus (AG News headlines), two tasks:

- Task A: topic classification (World / Sports / Business / Sci-Tech)
- Task B: semantic search

For each era we record the labels required, the size of the pretrained model,
and the wall-clock time, so the three approaches can be compared side by side.

Dependencies (add to the uv project in this directory):

    uv add scikit-learn pandas datasets sentence-transformers transformers accelerate

Quick test: shrink N_SEARCH / N_TRAIN / N_TEST below (era 3 is by far the
slowest step; everything else runs in seconds).
"""

import random
import re
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import pipeline as hf_pipeline

# --- Config -----------------------------------------------------------------

SEED = 42
N_SEARCH = 1_000   # retrieval corpus for task B
N_TRAIN = 500      # labeled docs to train the era 1 / era 2 classifiers
N_TEST = 100       # docs to classify at inference (keep small: era 3 is slow)
K = 3              # results shown per query

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # 494M params, small enough for CPU

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
# How each class is described for the zero-label methods of eras 2 and 3:
# wording matters, this is prompt engineering in miniature.
CLASS_TEXTS = [
    "world news",
    "sports news",
    "business news",
    "science and technology news",
]

QUERIES = [
    "spacecraft heading to Mars",
    "automobile manufacturer recalls vehicles over safety flaws",
    "athlete wins gold medal at the Olympics",
    "stock prices surge on Wall Street",
    "fighting escalates in the Middle East",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def timed(fn, *args, **kwargs):
    """Return (result, wall-clock seconds) for a call."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def top_k(sims, k=K):
    """Indices of the k largest similarities, best first."""
    best = np.argpartition(-sims, k - 1)[:k]
    return best[np.argsort(-sims[best])]


def overlap(a, b) -> float:
    """Fraction of top-k agreement between two result lists."""
    return len({int(i) for i in a} & {int(i) for i in b}) / K


# --- Corpus -----------------------------------------------------------------

ag = load_dataset("ag_news")

corpus_texts = ag["train"]["text"][:N_SEARCH]
corpus_labels = [LABELS[i] for i in ag["train"]["label"][:N_SEARCH]]

# Labeled samples for task A, class balance preserved.
pool = ag["train"].select(range(N_SEARCH, N_SEARCH + 20_000))
train_texts, _, train_y, _ = train_test_split(
    pool["text"], pool["label"],
    train_size=N_TRAIN, stratify=pool["label"], random_state=SEED,
)
test_pool = ag["test"].select(range(2_000))
test_texts, _, test_y, _ = train_test_split(
    test_pool["text"], test_pool["label"],
    train_size=N_TEST, stratify=test_pool["label"], random_state=SEED,
)
train_labels = [LABELS[i] for i in train_y]
test_labels = [LABELS[i] for i in test_y]
print(f"corpus: {N_SEARCH} docs for search, {N_TRAIN} labeled, {N_TEST} test\n")


# --- Task A: topic classification -------------------------------------------

# Era 1 - classic pipeline: TF-IDF features + task-specific classifier.
# Trained from scratch: needs labeled data and a new model for every task.
classic = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])
_, t_fit_classic = timed(classic.fit, train_texts, train_labels)
pred_classic, t_pred_classic = timed(classic.predict, test_texts)
acc_classic = accuracy_score(test_labels, pred_classic)
print(f"era 1  tfidf+logreg       accuracy {acc_classic:.1%}"
      f"  (fit {t_fit_classic:.2f}s, predict {t_pred_classic:.2f}s)")

# Era 2 - embeddings: one encoder for every task.
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(EMB_MODEL)
E_corpus, _ = timed(
    embedder.encode, corpus_texts,
    normalize_embeddings=True, show_progress_bar=False,
)
E_train, t_enc_train = timed(
    embedder.encode, train_texts,
    normalize_embeddings=True, show_progress_bar=False,
)
# Both era 2 variants below share this one encoding of the test set.
E_test, t_enc_test = timed(
    embedder.encode, test_texts,
    normalize_embeddings=True, show_progress_bar=False,
)

# (a) zero-shot: no labels at all - classify by similarity to the class names.
class_vecs = embedder.encode(CLASS_TEXTS, normalize_embeddings=True)
pred_zero = [LABELS[np.argmax(class_vecs @ e)] for e in E_test]
acc_zero = accuracy_score(test_labels, pred_zero)
print(f"era 2  embeddings 0-shot  accuracy {acc_zero:.1%}"
      f"  (no labels, encode {t_enc_test:.2f}s)")

# (b) few-shot: the very same classifier as era 1, but on embeddings.
head = LogisticRegression(max_iter=1000)
_, t_fit_head = timed(head.fit, E_train, train_labels)
pred_emb, _ = timed(head.predict, E_test)
acc_emb = accuracy_score(test_labels, pred_emb)
print(f"era 2  embeddings+logreg accuracy {acc_emb:.1%}"
      f"  (encode {t_enc_train:.2f}s + fit {t_fit_head:.2f}s,"
      f" predict {t_enc_test:.2f}s)")

# Era 3 - small LLM: zero labels, prompt only.
llm = hf_pipeline("text-generation", model=LLM_MODEL, device_map="auto")
llm.tokenizer.pad_token_id = llm.tokenizer.eos_token_id
llm.tokenizer.padding_side = "left"  # safe batching for decoder-only models


def parse_label(out: str) -> str | None:
    out = out.lower()
    for label in LABELS:
        if label.lower() in out:
            return label
    return None


def classify_llm(texts: list[str]) -> list[str | None]:
    messages = [
        [
            {"role": "system",
             "content": "You are a classifier. Reply with the label only."},
            {"role": "user",
             "content": "Classify this news headline as one of: "
                        f"{', '.join(LABELS)}.\n\nHeadline: {text}"},
        ]
        for text in texts
    ]
    outs = llm(messages, max_new_tokens=4, do_sample=False, batch_size=16)
    return [parse_label(o[0]["generated_text"]) for o in outs]


pred_llm, t_pred_llm = timed(classify_llm, test_texts)
n_unparsed = sum(p is None for p in pred_llm)
acc_llm = accuracy_score(test_labels, [p or "" for p in pred_llm])
print(f"era 3  LLM 0-shot        accuracy {acc_llm:.1%}"
      f"  (predict {t_pred_llm:.0f}s, {n_unparsed} unparseable)\n")


# --- Task B: semantic search -------------------------------------------------

# Era 1 - lexical search: TF-IDF cosine similarity.
tfidf = TfidfVectorizer(ngram_range=(1, 2)).fit(corpus_texts)
D = tfidf.transform(corpus_texts)


def search_tfidf(query: str, k: int = K) -> np.ndarray:
    # Query words absent from the corpus vocabulary are silently dropped:
    # vocabulary mismatch in action.
    q = tfidf.transform([query])
    return top_k((q @ D.T).toarray().ravel(), k)


# Era 2 - dense retrieval: embedding cosine similarity.
def search_embed(query: str, k: int = K) -> np.ndarray:
    q = embedder.encode([query], normalize_embeddings=True)[0]
    return top_k(E_corpus @ q, k)


# Era 3 - LLM as reranker: cannot scan a corpus, so it reorders a shortlist.
def search_llm(query: str, shortlist: list[int]) -> list[int]:
    listing = "\n".join(
        f"{n}. {corpus_texts[i]}" for n, i in enumerate(shortlist))
    messages = [[
        {"role": "system",
         "content": "You are a search engine. Answer with numbers only."},
        {"role": "user",
         "content": f"Query: {query}\n\nDocuments:\n{listing}\n\n"
                    f"Which {K} documents are most relevant? "
                    "Reply with their numbers, most relevant first."},
    ]]
    out = llm(messages, max_new_tokens=16, do_sample=False)[0][0][
        "generated_text"]
    nums = [int(n) for n in re.findall(r"\d+", out)][:K]
    return [shortlist[n] for n in nums if n < len(shortlist)]


def show(method: str, ids) -> None:
    print(f"  {method}")
    for i in ids:
        print(f"    [{corpus_labels[int(i)]:>8}] {corpus_texts[int(i)][:70]}")


t_tfidf = t_embed = t_llm = 0.0
overlaps_tfidf, overlaps_llm = [], []
for query in QUERIES:
    hits_tfidf, dt = timed(search_tfidf, query)
    t_tfidf += dt
    hits_embed, dt = timed(search_embed, query)
    t_embed += dt

    # Era 3 cannot scan the corpus, only rerank the top hits of a retriever
    # (here the union of both) - the standard reranking pattern.
    shortlist = sorted({*search_tfidf(query, k=10).tolist(),
                        *search_embed(query, k=10).tolist()})
    hits_llm, dt = timed(search_llm, query, shortlist)
    t_llm += dt

    overlaps_tfidf.append(overlap(hits_tfidf, hits_embed))
    overlaps_llm.append(overlap(hits_llm, hits_embed))

    print(f"\nquery: {query!r}")
    show("era 1 tfidf", hits_tfidf)
    show("era 2 embeddings", hits_embed)
    show("era 3 llm rerank", hits_llm)


# --- Summary -----------------------------------------------------------------

cls_rows = pd.DataFrame([
    dict(era="1 classic pipeline", method="tfidf + logistic reg",
         labels=N_TRAIN, params="0", fit_s=t_fit_classic,
         infer_s=t_pred_classic, accuracy=acc_classic),
    dict(era="2 embeddings", method="zero-shot (class names)",
         labels=0, params="22M", fit_s=0.0, infer_s=t_enc_test,
         accuracy=acc_zero),
    dict(era="2 embeddings", method="embeddings + logistic reg",
         labels=N_TRAIN, params="22M", fit_s=t_enc_train + t_fit_head,
         infer_s=t_enc_test, accuracy=acc_emb),
    dict(era="3 small LLM", method="zero-shot prompt",
         labels=0, params="494M", fit_s=0.0, infer_s=t_pred_llm,
         accuracy=acc_llm),
])
print("\nTask A - classification")
print(cls_rows.to_string(index=False))

search_rows = pd.DataFrame([
    dict(method="tfidf", total_s=t_tfidf,
         overlap_with_embeddings=np.mean(overlaps_tfidf)),
    dict(method="embeddings", total_s=t_embed, overlap_with_embeddings=1.0),
    dict(method="llm reranker", total_s=t_llm,
         overlap_with_embeddings=np.mean(overlaps_llm)),
])
print(f"\nTask B - search ({len(QUERIES)} queries)")
print(search_rows.to_string(index=False))

cls_rows.to_json("results.json", orient="records", indent=2)
