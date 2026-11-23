"""Temporal train / validation / test split and train-only rebalancing.

* `split_by_year`: train = years <= TRAIN_END, validation = VAL_YEAR, test = TEST_YEAR.
  The cut is on whole calendar years, so no paper of the test period can appear in training
  and seasonality (conference deadlines) does not distort a split made in the middle of a year.
* `cap_general`: rebalancing that touches the TRAIN set only. It reduces the papers whose
  technique label is `General ML/AI`, sampling them uniformly within each month so that
  the temporal composition of the training set is preserved. Validation and test keep their
  natural distribution, otherwise the metrics would describe a world that does not exist.
"""
from __future__ import annotations

import pandas as pd

import label_mapping as lm


def split_by_year(df: pd.DataFrame, train_end: int = 2024, val_year: int = 2025, test_year: int = 2026):
    """Return {'train','val','test'} DataFrames (copies). Raises if years do not fit the scheme."""
    assert train_end < val_year < test_year, "years must be increasing"
    years = set(df["year"].unique())
    unexpected = {y for y in years if y > test_year}
    if unexpected:
        raise ValueError(f"papers published after the test year {test_year}: {sorted(unexpected)}")
    out = {
        "train": df[df["year"] <= train_end].copy(),
        "val": df[df["year"] == val_year].copy(),
        "test": df[df["year"] == test_year].copy(),
    }
    ids = [set(out[k]["arxiv_id"]) for k in out]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "papers shared between splits"
    assert sum(len(v) for v in out.values()) == len(df), "some papers fall in no split"
    return out


def general_only_mask(df: pd.DataFrame) -> pd.Series:
    """Papers whose (single) technique label is `General ML/AI`."""
    return df["technique_label"] == lm.GENERAL


def cap_general(train: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """Keep at most ~`cap` papers whose only label is General ML/AI, uniformly within each month.

    All other papers are kept. The result is approximately `cap` (each month is rounded).
    """
    mask = general_only_mask(train)
    gen, rest = train[mask], train[~mask]
    if cap >= len(gen):
        return train.copy()
    month = gen["published"].str[:7]
    frac = cap / len(gen)
    kept = gen.groupby(month, group_keys=False).sample(frac=frac, random_state=seed)
    return pd.concat([rest, kept]).sort_values(["published", "arxiv_id"], kind="mergesort")


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Size and % of papers per technique label in every split (one row per split).

    Columns: n, share of the data, % per class (they sum to 100) and `% ambiguous`
    (papers that had more than one technique category before the priority rule)."""
    total = max(sum(len(v) for v in splits.values()), 1)
    rows = []
    for name, d in splits.items():
        n = max(len(d), 1)
        row = {"split": name, "n": len(d), "% of data": round(100 * len(d) / total, 1)}
        for c in lm.TECHNIQUE_CLASSES:
            row[c] = round(100 * (d["technique_label"] == c).sum() / n, 1)
        row["% ambiguous"] = round(100 * d["technique_labels"].str.contains("|", regex=False).sum() / n, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")
