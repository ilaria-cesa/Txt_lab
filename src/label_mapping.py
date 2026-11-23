"""Label construction for the arXiv-AI project.

Two independent label sets are derived from the arXiv `categories` field
(space-separated codes, primary category first):

1. TECHNIQUE label  -> target of the CLASSIFICATION task (ONE label per paper).
   Which AI sub-field / modality does the paper belong to?
   (Computer Vision, NLP, Robotics, Speech & Audio, ...).
   Priority rule (`technique_label`):
     a) if the PRIMARY category is a specific technique, that is the label;
     b) otherwise the first specific technique among the other categories, in the
        order listed by arXiv;
     c) otherwise GENERAL (only generic categories: cs.AI, cs.LG, stat.ML, cs.NE).
   `technique_labels` keeps ALL the techniques of the paper ('|'-joined, ~7% of the
   papers have more than one): used to analyse the ambiguous papers, not as target.

2. DOMAIN labels -> NOT a classification target.
   Which field is AI applied to (physics, life sciences, economics, ...)?
   Used only as an *external reference* to interpret / evaluate the topic model
   (e.g. NMI between topics and domains), so the abstract text never has to
   "predict" them and no label leakage can occur.

Rules are deterministic, depend only on the category codes (never on the text)
and every code of the arXiv taxonomy must be covered (see `check_coverage`).
"""
from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

# ---------------------------------------------------------------- technique
GENERAL = "General ML/AI"

# category code -> technique label
TECHNIQUE_MAP = {
    "cs.CV": "Computer Vision",
    "eess.IV": "Computer Vision",          # image/video processing (incl. medical imaging)
    "cs.CL": "Natural Language Processing",
    "cs.RO": "Robotics",
    "cs.SD": "Speech & Audio",
    "eess.AS": "Speech & Audio",
    "cs.IR": "Information Retrieval & RecSys",
    "cs.MA": "Multi-Agent Systems",
}
# generic catch-all categories: carry no technique information.
# cs.NE is here because its meaning drifted (deep learning in 2015, evolutionary computing today).
GENERIC_AI = {"cs.AI", "cs.LG", "stat.ML", "cs.NE"}
TECHNIQUE_CLASSES = [GENERAL] + sorted(set(TECHNIQUE_MAP.values()))

# ------------------------------------------------------------------- domain
NO_DOMAIN = "No application domain"

_DOMAIN_EXPLICIT = {
    # computer science (non-technique)
    "cs.CR": "Security & Privacy",
    "cs.SE": "Software & Systems", "cs.PL": "Software & Systems", "cs.DC": "Software & Systems",
    "cs.AR": "Software & Systems", "cs.OS": "Software & Systems", "cs.PF": "Software & Systems",
    "cs.NI": "Software & Systems", "cs.DB": "Software & Systems", "cs.ET": "Software & Systems",
    "cs.MS": "Software & Systems", "cs.OH": "Software & Systems", "cs.GL": "Software & Systems",
    "cs.SI": "Information & Social Networks", "cs.DL": "Information & Social Networks",
    "cs.HC": "Humans & Society", "cs.CY": "Humans & Society", "physics.soc-ph": "Humans & Society",
    "cs.GT": "Logic & Games", "cs.LO": "Logic & Games", "cs.FL": "Logic & Games", "cs.SC": "Logic & Games",
    "cs.CC": "Math, Statistics & Theory", "cs.DS": "Math, Statistics & Theory",
    "cs.DM": "Math, Statistics & Theory", "cs.CG": "Math, Statistics & Theory",
    "cs.NA": "Math, Statistics & Theory", "cs.IT": "Math, Statistics & Theory",
    "cs.CE": "Math, Statistics & Theory",
    "cs.GR": "Signals, Graphics & Multimedia", "cs.MM": "Signals, Graphics & Multimedia",
    "eess.SP": "Signals, Graphics & Multimedia",
    "cs.SY": "Systems & Control", "eess.SY": "Systems & Control", "math.OC": "Systems & Control",
    # earth / medicine carved out of physics
    "physics.ao-ph": "Earth & Climate", "physics.geo-ph": "Earth & Climate",
    "physics.med-ph": "Life Sciences & Medicine",
}
# prefix -> domain (checked after the explicit table)
_DOMAIN_PREFIX = [
    ("math-ph", "Physics & Astronomy"),
    ("math.", "Math, Statistics & Theory"),
    ("stat.", "Math, Statistics & Theory"),
    ("q-bio.", "Life Sciences & Medicine"),
    ("q-fin.", "Economics & Finance"),
    ("econ.", "Economics & Finance"),
    ("physics.", "Physics & Astronomy"),
    ("astro-ph", "Physics & Astronomy"),
    ("cond-mat", "Physics & Astronomy"),
    ("hep-", "Physics & Astronomy"),
    ("nucl-", "Physics & Astronomy"),
    ("gr-qc", "Physics & Astronomy"),
    ("quant-ph", "Physics & Astronomy"),
    ("nlin.", "Physics & Astronomy"),
]


def domain_of(code: str) -> str | None:
    """Domain of one category code; None if the code is a technique/generic category."""
    if code in TECHNIQUE_MAP or code in GENERIC_AI:
        return None
    if code in _DOMAIN_EXPLICIT:
        return _DOMAIN_EXPLICIT[code]
    for prefix, dom in _DOMAIN_PREFIX:
        if code.startswith(prefix):
            return dom
    raise KeyError(f"unmapped arXiv category: {code}")


DOMAIN_CLASSES = sorted({d for d in _DOMAIN_EXPLICIT.values()}
                        | {d for _, d in _DOMAIN_PREFIX}) + [NO_DOMAIN]


# ------------------------------------------------------------------- labels
def technique_labels(categories: str) -> list[str]:
    cats = categories.split()
    labs = sorted({TECHNIQUE_MAP[c] for c in cats if c in TECHNIQUE_MAP})
    return labs or [GENERAL]


def technique_label(categories: str, primary: str | None = None) -> str:
    """Single technique label with the priority rule (primary first, then listed order)."""
    cats = categories.split()
    primary = primary or (cats[0] if cats else None)
    if primary in TECHNIQUE_MAP:
        return TECHNIQUE_MAP[primary]
    for c in cats:
        if c in TECHNIQUE_MAP:
            return TECHNIQUE_MAP[c]
    return GENERAL


def domain_labels(categories: str) -> list[str]:
    cats = categories.split()
    labs = sorted({d for d in (domain_of(c) for c in cats) if d})
    return labs or [NO_DOMAIN]


def add_labels(df: pd.DataFrame, sep: str = "|") -> pd.DataFrame:
    """Add `technique_label` (single, target), `technique_labels` (all, joined by `sep`) and
    `domain_labels` to a copy of df. Uses `primary_category` if the column is present."""
    out = df.copy()
    prim = out["primary_category"] if "primary_category" in out else [None] * len(out)
    out["technique_label"] = [technique_label(c, p) for c, p in zip(out["categories"], prim)]
    out["technique_labels"] = out["categories"].map(lambda s: sep.join(technique_labels(s)))
    out["domain_labels"] = out["categories"].map(lambda s: sep.join(domain_labels(s)))
    return out


def to_multihot(series: pd.Series, classes: list[str], sep: str = "|") -> pd.DataFrame:
    """Multi-hot indicator matrix (one column per class) from a `sep`-joined label column."""
    sets = series.str.split(sep, regex=False).map(set)
    return pd.DataFrame({c: sets.map(lambda s, c=c: int(c in s)) for c in classes}, index=series.index)


def check_coverage() -> None:
    """Every code of the arXiv taxonomy must be mapped (raises KeyError otherwise)."""
    from arxiv_taxonomy import CATEGORY_INFO
    for code in CATEGORY_INFO:
        domain_of(code)


# -------------------------------------------------------------------- stats
def label_report(df: pd.DataFrame) -> str:
    lines = []
    n = len(df)
    cols = [("technique_label", "TECHNIQUE label (single, classification target)"),
            ("technique_labels", "TECHNIQUE labels (all, for analysis)"),
            ("domain_labels", "DOMAIN labels (reference for topic modeling)")]
    for col, title in cols:
        if col not in df:
            continue
        sets = df[col].str.split("|", regex=False)
        cnt = Counter(l for s in sets for l in s)
        card = sets.map(len)
        lines.append(f"\n{title} labels (papers={n}, mean labels/paper={card.mean():.2f}, "
                     f"multi-label share={(card > 1).mean():.1%})")
        for lab, c in cnt.most_common():
            lines.append(f"  {lab:<36}{c:>9}  {c / n:6.1%}")
        if "year" in df:
            lines.append("  -- share of papers per year (%):")
            years = sorted(df["year"].unique())
            lines.append("  " + " " * 36 + "".join(f"{y:>7}" for y in years))
            for lab, _ in cnt.most_common():
                row = [df.loc[sets.map(lambda s, lab=lab: lab in s) & (df["year"] == y)].shape[0]
                       / max((df["year"] == y).sum(), 1) * 100 for y in years]
                lines.append(f"  {lab:<36}" + "".join(f"{v:>7.1f}" for v in row))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build technique/domain labels and print their distribution.")
    ap.add_argument("csv", help="dataset CSV produced by arxiv_ai_dataset.py")
    ap.add_argument("--out", help="optional path to save the CSV with the label columns")
    args = ap.parse_args(argv)
    check_coverage()
    df = add_labels(pd.read_csv(args.csv, dtype={"arxiv_id": str}, usecols=lambda c: c in {"arxiv_id", "year", "categories", "primary_category"}))
    print(label_report(df))
    if args.out:
        df[["arxiv_id", "technique_labels", "domain_labels"]].to_csv(args.out, index=False)
        print(f"\nsaved labels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
