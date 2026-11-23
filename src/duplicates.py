"""Duplicate handling for the arXiv-AI dataset.

Workflow (run on the full table, BEFORE the train/validation/test split):

1. Exact duplicate abstracts  -> `report_exact_abstract_duplicates` writes a CSV to look at,
   `drop_exact_abstract_duplicates` removes them (keeps the earliest paper of each group).
2. Same title                 -> `build_title_review` writes a CSV to review by hand
   (columns `status` and `drop`), `apply_title_decisions` removes the rows marked `drop = 1`.

Nothing is decided silently: every function returns statistics and the review files list
every row involved. Both steps only use text equality / similarity, never the labels.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import pandas as pd

REVIEW_COLS = ["arxiv_id", "published", "year", "categories", "title", "abstract"]
SIM_SAME = 0.95      # abstract similarity >= this  -> almost certainly the same paper
SIM_DIFFERENT = 0.50  # abstract similarity <  this  -> almost certainly different papers


def _order(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest paper first (ties broken by id) so that 'first' is well defined."""
    return df.sort_values(["published", "arxiv_id"], kind="mergesort")


def _cat_key(s: str) -> frozenset:
    return frozenset(str(s).split())


# --------------------------------------------------------------------------- 1. abstracts
def exact_abstract_groups(df: pd.DataFrame) -> pd.DataFrame:
    """All rows whose abstract appears more than once, with a group id and a `keep` flag."""
    key = df["abstract"].str.strip()
    dup = df[key.duplicated(keep=False)].copy()
    dup["_key"] = key[dup.index]
    dup = _order(dup)
    dup["group_id"] = dup.groupby("_key", sort=False).ngroup() + 1
    dup["group_size"] = dup.groupby("group_id")["arxiv_id"].transform("size")
    first_cats = dup.groupby("group_id")["categories"].transform("first")
    dup["same_categories_as_first"] = [_cat_key(a) == _cat_key(b) for a, b in zip(dup["categories"], first_cats)]
    dup["keep"] = (~dup.duplicated("group_id")).astype(int)        # 1 = earliest copy
    dup["n_words"] = dup["abstract"].str.split().str.len()
    return dup.drop(columns="_key").sort_values(["group_id", "published", "arxiv_id"])


def report_exact_abstract_duplicates(df: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    g = exact_abstract_groups(df)
    cols = ["group_id", "group_size", "keep", "same_categories_as_first", "n_words"] + REVIEW_COLS
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    g[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    n_groups = g["group_id"].nunique()
    extra = int((g["keep"] == 0).sum())
    diff = g.groupby("group_id")["same_categories_as_first"].all().eq(False).sum()
    print(f"exact duplicate abstracts: {n_groups} groups, {extra} rows would be removed")
    print(f"groups whose copies have DIFFERENT categories: {diff}")
    print(f"groups with more than 2 copies: {(g.groupby('group_id').size() > 2).sum()}"
          f" | groups with abstract < 60 words: {(g.groupby('group_id')['n_words'].first() < 60).sum()}")
    print("years of the removable rows:", g.loc[g['keep'] == 0, "year"].value_counts().sort_index().to_dict())
    print("saved ->", out_csv)
    return g


def drop_exact_abstract_duplicates(df: pd.DataFrame, merge_categories: bool = False):
    """Keep the earliest paper of each group of identical abstracts.

    merge_categories=True also gives the kept paper the union of the categories of all its
    copies (use it if copies carry different labels). Returns (clean_df, n_removed).
    """
    g = exact_abstract_groups(df)
    out = df.copy()
    if merge_categories and len(g):
        union = g.groupby("group_id")["categories"].agg(lambda s: " ".join(dict.fromkeys(" ".join(s).split())))
        for gid, idx in zip(g.loc[g["keep"] == 1, "group_id"], g.index[g["keep"] == 1]):
            out.at[idx, "categories"] = union[gid]
    remove = g.index[g["keep"] == 0]
    out = out.drop(index=remove)
    return out, len(remove)


# ------------------------------------------------------------------------------ 2. titles
def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def build_title_review(df: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    """Rows sharing an exact title (after exact-abstract duplicates were removed).

    `abstract_similarity` compares each row's abstract to the earliest one of its group.
    `status`: REVIEW (to look at: 0.5-0.95), auto (>=0.95, prefilled drop=1 for later copies),
    different (<0.5: probably different papers with the same title, kept).
    Fill the `drop` column with 1 for the rows to remove, then call `apply_title_decisions`.
    """
    key = df["title"].str.strip()
    dup = _order(df[key.duplicated(keep=False)].copy())
    dup["_key"] = key[dup.index]
    dup["group_id"] = dup.groupby("_key", sort=False).ngroup() + 1
    dup["group_size"] = dup.groupby("group_id")["arxiv_id"].transform("size")
    first_abs = dup.groupby("group_id")["abstract"].transform("first")
    is_first = ~dup.duplicated("group_id")
    dup["abstract_similarity"] = [1.0 if f else round(_sim(a, b), 3)
                                  for f, a, b in zip(is_first, dup["abstract"], first_abs)]
    first_cats = dup.groupby("group_id")["categories"].transform("first")
    dup["same_categories_as_first"] = [_cat_key(a) == _cat_key(b) for a, b in zip(dup["categories"], first_cats)]
    worst = dup[~is_first].groupby("group_id")["abstract_similarity"]
    gmax = worst.max().reindex(dup["group_id"].unique()).fillna(0)
    gmin = worst.min().reindex(dup["group_id"].unique()).fillna(0)

    def status(gid):
        hi, lo = gmax[gid], gmin[gid]
        if lo >= SIM_SAME:
            return "auto"
        if hi < SIM_DIFFERENT:
            return "different"
        return "REVIEW"

    dup["status"] = dup["group_id"].map(status)
    dup["drop"] = ((dup["status"] == "auto") & ~is_first).astype(int).replace(0, "")
    order = {"REVIEW": 0, "auto": 1, "different": 2}
    dup = dup.sort_values(["status", "group_id", "published", "arxiv_id"],
                          key=lambda s: s.map(order) if s.name == "status" else s)
    cols = ["status", "group_id", "group_size", "drop", "abstract_similarity", "same_categories_as_first"] + REVIEW_COLS
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dup[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    per_status = dup.groupby("status")["group_id"].nunique().to_dict()
    print(f"title duplicates: {dup['group_id'].nunique()} groups, {len(dup)} rows | groups by status: {per_status}")
    print("saved ->", out_csv, "  (fill the `drop` column with 1, then call apply_title_decisions)")
    return dup[cols]


def apply_title_decisions(df: pd.DataFrame, review_csv: str | Path):
    """Remove the rows marked drop = 1 in the (edited) review CSV. Returns (clean_df, n_removed)."""
    rev = pd.read_csv(review_csv, dtype={"arxiv_id": str}, encoding="utf-8-sig")
    drop_ids = set(rev.loc[pd.to_numeric(rev["drop"], errors="coerce") == 1, "arxiv_id"])
    out = df[~df["arxiv_id"].isin(drop_ids)]
    return out, len(df) - len(out)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Write the duplicate review files for the dataset CSV.")
    ap.add_argument("csv")
    ap.add_argument("--out-dir", default="duplicates")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.csv, dtype={"arxiv_id": str}, usecols=REVIEW_COLS)
    out = Path(args.out_dir)
    report_exact_abstract_duplicates(df, out / "exact_abstract_duplicates.csv")
    clean, n = drop_exact_abstract_duplicates(df)
    print(f"after removing exact abstract duplicates: {len(clean)} rows ({n} removed)")
    build_title_review(clean, out / "title_duplicates_review.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
