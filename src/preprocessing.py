"""Text preprocessing for the arXiv-AI classification project (same spirit as the course labs).

Two levels, built from the title and the abstract of each paper:

1. LIGHT cleaning (`clean_light`, `build_text`) - for EVERY model, transformers included:
   HTML entities, URLs, LaTeX formulas (replaced by a placeholder), repeated whitespace.
   Case, punctuation and numbers are NOT touched: a transformer needs them, and its own
   tokenizer (WordPiece / BPE) does the rest.

2. SPARSE normalization (`preprocess_sparse`, `lemma_tokens`) - only for TF-IDF, Word2Vec/Doc2Vec
   and the topic models LDA/NMF: lowercase, tokenization, standalone numbers, one-character
   tokens, stop words, then optionally stemming or lemmatization.

Choices made on purpose (all of them are compared in notebook 03):
  * hyphens split the word ("self-supervised" -> "self", "supervised"), they are not glued;
  * numbers INSIDE words are kept ("3d", "gpt4", "l1"): only standalone numbers are dropped;
  * negations ("not", "no", "nor", "...n't") are removed from the standard stop word list;
  * everything that is learned from the data (stop word lists by frequency, vocabulary, IDF)
    must be computed on the TRAIN set only.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

MATH_TOKEN = "mathexpr"      # placeholder for LaTeX formulas ($...$)
NEGATIONS = {"not", "no", "nor"}
# Academic boilerplate (inflected and base forms: lemmas and stems are filtered AFTER this list).
# Starting point only: check it against `document_frequency` and the ablation of notebook 03.
DOMAIN_STOPWORDS_SEED = (
    "paper", "propose", "proposed", "proposes", "present", "presented", "presents",
    "show", "shown", "shows", "result", "results", "approach", "novel", "study",
    "also", "however", "may", "can", "et", "al",
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MATH_RE = re.compile(r"\$\$[^$]{1,500}\$\$|\$[^$]{1,300}\$")
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?")          # letters and digits, keeps "don't"
_POSSESSIVE_RE = re.compile(r"(?<=\w)'s\b")
_HYPHEN_RE = re.compile(r"(?<=\w)-(?=\w)")


# ------------------------------------------------------------------ 1. light cleaning
def clean_light(text: str, math_token: str = MATH_TOKEN) -> str:
    """Common cleaning for all models: HTML entities, URLs, LaTeX formulas, whitespace."""
    text = html.unescape(str(text))
    text = _URL_RE.sub(" ", text)                       # links before anything that touches punctuation
    text = _MATH_RE.sub(f" {math_token} ", text)
    return " ".join(text.split())


def build_text(title: str, abstract: str, math_token: str = MATH_TOKEN) -> str:
    """'Title. Abstract' after light cleaning. Model-specific formats (e.g. [SEP]) are added later."""
    t, a = clean_light(title, math_token), clean_light(abstract, math_token)
    if t and t[-1] not in ".?!:;":
        t += "."
    return f"{t} {a}".strip()


def build_texts(df: pd.DataFrame, math_token: str = MATH_TOKEN) -> list[str]:
    return [build_text(t, a, math_token) for t, a in zip(df["title"], df["abstract"])]


# ------------------------------------------------------------------ 2. sparse normalization
def load_nltk_stopwords() -> set[str]:
    """NLTK English stop words (as in the labs); downloads the list the first time."""
    from nltk.corpus import stopwords
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def build_stopwords(base=None, keep_negations: bool = True, extra=()) -> set[str]:
    """Standard stop words (NLTK unless `base` is given), without negations, plus `extra`."""
    words = set(load_nltk_stopwords() if base is None else base)
    if keep_negations:
        words = {w for w in words if w not in NEGATIONS and not w.endswith("n't")}
    return words | set(extra)


def tokenize(text: str, glue_hyphens: bool = False) -> list[str]:
    """Lowercase tokens made of letters and digits. Hyphens split the word unless `glue_hyphens`."""
    text = text.lower().replace("’", "'")
    if glue_hyphens:
        text = _HYPHEN_RE.sub("", text)
    return _TOKEN_RE.findall(_POSSESSIVE_RE.sub("", text))


def filter_tokens(tokens, stopwords=frozenset(), drop_numbers: bool = True, min_len: int = 2) -> list[str]:
    """Drop standalone numbers ("2024"; "3d" stays), tokens shorter than `min_len`, stop words."""
    return [t for t in tokens
            if not (drop_numbers and t.isdigit()) and len(t) >= min_len and t not in stopwords]


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    from nltk.stem.porter import PorterStemmer
    return PorterStemmer().stem(word)


def stem_tokens(tokens) -> list[str]:
    return [_stem(t) for t in tokens]


def preprocess_sparse(text: str, stopwords=frozenset(), drop_numbers: bool = True, min_len: int = 2,
                      stem: bool = False, glue_hyphens: bool = False) -> list[str]:
    """tokenize -> filter (numbers, short tokens, stop words) -> optional Porter stemming."""
    toks = filter_tokens(tokenize(text, glue_hyphens), stopwords, drop_numbers, min_len)
    return stem_tokens(toks) if stem else toks


# ------------------------------------------------------------------ lemmatization (spaCy)
def load_spacy(model: str = "en_core_web_sm"):
    """spaCy pipeline with only what the lemmatizer needs (no parser, no NER)."""
    import spacy
    try:
        return spacy.load(model, disable=["parser", "ner"])
    except OSError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
        return spacy.load(model, disable=["parser", "ner"])


def lemma_tokens(texts, nlp, stopwords=frozenset(), drop_numbers: bool = True, min_len: int = 2,
                 n_process: int = 1, batch_size: int = 512, progress: bool = True) -> list[list[str]]:
    """Lemmatize the light-cleaned texts (case and punctuation are kept: the tagger needs them),
    then apply the same filters as `preprocess_sparse`."""
    texts = list(texts)
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    if progress:
        from tqdm.auto import tqdm
        docs = tqdm(docs, total=len(texts), desc="lemmatizing")
    out = []
    for doc in docs:
        toks = []
        for t in doc:
            if not (t.is_space or t.is_punct):
                toks += _TOKEN_RE.findall(t.lemma_.lower().replace("’", "'"))
        out.append(filter_tokens(toks, stopwords, drop_numbers, min_len))
    return out


# ------------------------------------------------------------------ analysis helpers
def document_frequency(token_lists, top: int = 100) -> pd.DataFrame:
    """Most frequent tokens by number of documents that contain them (to choose domain stop words)."""
    counter = Counter()
    for toks in token_lists:
        counter.update(set(toks))
    n = max(len(token_lists), 1)
    df = pd.DataFrame(counter.most_common(top), columns=["token", "docs"])
    df["% docs"] = (100 * df["docs"] / n).round(1)
    return df


def vocabulary_steps(texts, stopwords, domain_stopwords=(), lemmas=None) -> pd.DataFrame:
    """Vocabulary size after each cumulative step (Exercise 1.2 of Lab 1.1, on our corpus).

    `lemmas`: optional output of `lemma_tokens` for the same texts (already filtered with the
    standard + domain stop words), to add the lemmatization row. Two side rows (hyphens glued,
    stemming) branch from the last common step and are compared with it.
    """
    texts = list(texts)
    n_docs = max(len(texts), 1)
    std = frozenset(stopwords)
    dom = std | frozenset(domain_stopwords)
    rows = []

    def measure(name, token_lists, ref=None):
        vocab, total = set(), 0
        for toks in token_lists:
            vocab.update(toks)
            total += len(toks)
        base = ref if ref is not None else (rows[-1]["vocabulary"] if rows else None)
        change = None if base is None else round(100 * (len(vocab) / base - 1), 1)
        rows.append({"step": name, "vocabulary": len(vocab), "change %": change,
                     "tokens / doc": round(total / n_docs, 1)})

    measure("light-cleaned text, split on spaces", (t.split() for t in texts))
    measure("+ lowercase", (t.lower().split() for t in texts))
    measure("+ tokenization (letters and digits; hyphens and punctuation split)", (tokenize(t) for t in texts))
    measure("+ drop standalone numbers", (filter_tokens(tokenize(t), min_len=1) for t in texts))
    measure("+ drop one-character tokens", (filter_tokens(tokenize(t)) for t in texts))
    measure("+ standard stop words (negations kept)", (filter_tokens(tokenize(t), std) for t in texts))
    measure("+ domain stop words", (filter_tokens(tokenize(t), dom) for t in texts))
    last = rows[-1]["vocabulary"]
    measure("  side: Porter stemming (vs. previous row)",
            (stem_tokens(filter_tokens(tokenize(t), dom)) for t in texts), ref=last)
    if lemmas is not None:
        measure("  side: spaCy lemmatization (vs. domain stop words)", lemmas, ref=last)
    measure("  side: hyphens glued instead of split (not used)",
            (filter_tokens(tokenize(t, glue_hyphens=True), dom) for t in texts), ref=last)
    return pd.DataFrame(rows)


def token_length_stats(lengths, limits=(256, 512)) -> pd.Series:
    """Summary of token lengths: mean, median, 95th/99th percentile, max and % above each limit."""
    x = np.asarray(lengths)
    out = {"mean": x.mean(), "median": np.median(x), "p95": np.percentile(x, 95),
           "p99": np.percentile(x, 99), "max": x.max()}
    for lim in limits:
        out[f"% > {lim}"] = 100 * (x > lim).mean()
    return pd.Series(out).round(2)
