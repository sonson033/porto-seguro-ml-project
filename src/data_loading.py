"""
Column typing for the Porto Seguro feature set.
=================================================

Derives the categorical-vs-quantity split from column-name suffixes, exactly
as established in the data sanity pass (issue #6, notebooks/01_data_sanity_pass.ipynb):
`_cat`/`_bin` suffix -> categorical (codes/labels), everything else -> quantity.

Kept here as reusable code so every ticket that needs to tell these two
groups apart (encoding, scaling, ...) uses the same lists instead of
re-deriving them ad hoc.
"""


def feature_groups(columns):
    """
    Split feature column names into (categorical_cols, quantity_cols).

    `columns` is any iterable of column names, typically a DataFrame's
    `.columns`. `id` and `target` are excluded from both groups.
    """
    feature_cols = [c for c in columns if c not in ("id", "target")]
    categorical_cols = [c for c in feature_cols if c.endswith("_cat") or c.endswith("_bin")]
    quantity_cols = [c for c in feature_cols if c not in categorical_cols]
    return categorical_cols, quantity_cols
