"""
Feature engineering -- row missingness and high-cardinality encoding
=====================================================================

Issue #22. Two engineered features, each scored separately on the logistic
regression and on the 37-column (ps_calc_* dropped) forest from #13/#17, so
each of the four results can be attributed to exactly one change:

    1. missing-value-count feature -> logistic regression
    2. missing-value-count feature -> 37-column forest
    3. ps_car_11_cat frequency encoding -> logistic regression
    4. ps_car_11_cat frequency encoding -> 37-column forest

Both features build on the 37-column base (all ps_calc_* columns dropped),
not the original 57-column set, per the ticket's technical considerations.

Run this file directly to reproduce every number printed below:
    uv run python src/train_feature_engineering.py
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loading import feature_groups
from evaluation import SEED, cross_validate_model, load_train
from train_baseline_models import REALISTIC_GINI_CEILING, _build_preprocessor, _check_realistic

CAR11_COL = "ps_car_11_cat"


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Maps each category to its frequency (share of rows) *within whatever it
    was fit on*. Used inside a Pipeline exactly like OneHotEncoder, so
    cross_validate_model()'s per-fold clone-and-fit fits this on only that
    fold's training rows.

    This is the leakage risk the ticket calls out explicitly: fitting on the
    whole 80% split before cross-validation would let each fold's validation
    rows influence the frequencies it's scored against. Unseen categories
    (present in a validation fold but not that fold's training rows) map to
    0.0 -- the same "never seen it" signal OneHotEncoder's handle_unknown
    ="ignore" gives for a fold's unseen categories.
    """

    def fit(self, X, y=None):
        values = np.asarray(X).reshape(-1)
        uniques, counts = np.unique(values, return_counts=True)
        self.freq_map_ = dict(zip(uniques, counts / counts.sum()))
        return self

    def transform(self, X):
        values = np.asarray(X).reshape(-1)
        return np.array([self.freq_map_.get(v, 0.0) for v in values],
                         dtype=np.float64).reshape(-1, 1)


def _drop_calc(categorical_cols, quantity_cols):
    """The 37-column base: every ps_calc_* column removed, per #13/#17."""
    cat_37 = [c for c in categorical_cols if not c.startswith("ps_calc_")]
    qty_37 = [c for c in quantity_cols if not c.startswith("ps_calc_")]
    return cat_37, qty_37


def _build_freq_preprocessor(n_categorical_other, n_quantity, scale):
    """
    Same preprocessing as _build_preprocessor(), except ps_car_11_cat is
    pulled out of the one-hot group and routed through FrequencyEncoder
    instead. Column order expected: [other categorical cols] + [car_11_cat]
    + [quantity cols].
    """
    quantity_steps = [SimpleImputer(missing_values=-1, strategy="median")]
    if scale:
        quantity_steps.append(StandardScaler())

    other_cat_idx = list(range(n_categorical_other))
    car11_idx = [n_categorical_other]
    quantity_idx = list(range(n_categorical_other + 1, n_categorical_other + 1 + n_quantity))

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                               sparse_output=False, dtype=np.float32),
         other_cat_idx),
        ("car11_freq", FrequencyEncoder(), car11_idx),
        ("qty", make_pipeline(*quantity_steps), quantity_idx),
    ])


def _run(label, pipeline, X, y, baseline_mean, baseline_label):
    t0 = time.perf_counter()
    mean, std, scores = cross_validate_model(pipeline, X, y)
    runtime = time.perf_counter() - t0
    delta = mean - baseline_mean
    beats_std = delta > std
    print(f"  {label}")
    print(f"    Gini      {mean:+.5f} +/- {std:.5f}   ({runtime:.1f}s)")
    print(f"    Delta vs {baseline_label} ({baseline_mean:+.5f}): {delta:+.5f}   "
          f"beats std? {'yes' if beats_std else 'no'}")
    _check_realistic(label, mean)
    return mean, std, delta, beats_std, runtime


def main():
    train = load_train()  # the 80% split only; load_final_test() is never called in this file
    categorical_cols, quantity_cols = feature_groups(train.columns)
    cat_37, qty_37 = _drop_calc(categorical_cols, quantity_cols)

    # ---- reference point: an isolated 37-column LR with NEITHER new
    # feature, for transparency only (not one of the 4 recorded rows). The
    # only recorded LR baseline (#9, 0.2572 +/- 0.0036) is on 57 columns, so
    # a raw delta against it would conflate "dropped calc" with "added this
    # feature". This isolates the feature's own effect for the Notes.
    print("=" * 70)
    print("Reference only: 37-column LR, no new feature (not a results.md row)")
    print("=" * 70)
    X_37 = train[cat_37 + qty_37]
    y = train["target"]
    lr_37_pipeline = make_pipeline(
        _build_preprocessor(len(cat_37), len(qty_37), scale=True),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    lr_37_mean, lr_37_std, _ = cross_validate_model(lr_37_pipeline, X_37, y, verbose=False)
    print(f"  Gini {lr_37_mean:+.5f} +/- {lr_37_std:.5f}")

    LR_BASELINE = 0.2572   # #9, recorded in results.md, 57 columns
    RF_BASELINE = 0.27244  # #13, recorded in results.md, 37 columns (calc dropped)

    results = {}

    # ---- Feature 1: missing-value count -----------------------------------
    missing_count = (train[cat_37 + qty_37] == -1).sum(axis=1).astype(np.float64)
    X_missing = train[cat_37 + qty_37].copy()
    X_missing["missing_count"] = missing_count
    qty_37_plus_count = qty_37 + ["missing_count"]

    print()
    print("=" * 70)
    print("FEATURE 1: missing-value count (37-col base + 1 numeric column)")
    print("=" * 70)
    lr_missing_pipeline = make_pipeline(
        _build_preprocessor(len(cat_37), len(qty_37_plus_count), scale=True),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    results["missing_lr"] = _run(
        "Logistic regression + missing-count", lr_missing_pipeline,
        X_missing[cat_37 + qty_37_plus_count], y, LR_BASELINE, "#9 LR baseline")

    rf_missing_pipeline = make_pipeline(
        _build_preprocessor(len(cat_37), len(qty_37_plus_count), scale=False),
        RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=SEED),
    )
    results["missing_rf"] = _run(
        "Random forest + missing-count", rf_missing_pipeline,
        X_missing[cat_37 + qty_37_plus_count], y, RF_BASELINE, "#13 37-col RF")

    # ---- Feature 2: ps_car_11_cat frequency encoding -----------------------
    cat_37_other = [c for c in cat_37 if c != CAR11_COL]
    freq_cols = cat_37_other + [CAR11_COL] + qty_37
    X_freq = train[freq_cols]

    print()
    print("=" * 70)
    print("FEATURE 2: ps_car_11_cat frequency encoding (fit inside each fold)")
    print("=" * 70)
    lr_freq_pipeline = make_pipeline(
        _build_freq_preprocessor(len(cat_37_other), len(qty_37), scale=True),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    results["freq_lr"] = _run(
        "Logistic regression + car_11_cat freq-encoded", lr_freq_pipeline,
        X_freq, y, LR_BASELINE, "#9 LR baseline")

    rf_freq_pipeline = make_pipeline(
        _build_freq_preprocessor(len(cat_37_other), len(qty_37), scale=False),
        RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=SEED),
    )
    results["freq_rf"] = _run(
        "Random forest + car_11_cat freq-encoded", rf_freq_pipeline,
        X_freq, y, RF_BASELINE, "#13 37-col RF")

    print()
    print("=" * 70)
    print("Isolated feature effect (vs. the 37-column LR reference above, not a row)")
    print("=" * 70)
    print(f"  missing-count on LR:  {results['missing_lr'][0] - lr_37_mean:+.5f}")
    print(f"  freq-encoding on LR:  {results['freq_lr'][0] - lr_37_mean:+.5f}")


if __name__ == "__main__":
    main()
