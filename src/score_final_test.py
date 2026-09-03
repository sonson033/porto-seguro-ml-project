"""
Score the final test set -- Porto Seguro claim risk ranking (issue #24)
=========================================================================

The frozen winning configuration, retrained on the WHOLE 80% training split
and scored ONCE against the 20% holdout no decision has ever touched.

Frozen configuration (unchanged since #13; results.md; commit 6c794f2):
    RandomForestClassifier(n_estimators=200, min_samples_leaf=50, n_jobs=-1,
    random_state=42) -- 37 columns, all `ps_calc_*` dropped -- median-impute
    the -1 sentinel (quantity cols), one-hot encode (categorical cols,
    drop="if_binary"), no scaling.
No experiment is in flight: #17's extra column drop and #22's two engineered
features were both REJECT, so this is still the model on top.

`evaluation.load_final_test()` is called exactly once in this file, and this
is the only file in the project that calls it. Everything before that call
runs on `load_train()`'s 80% only, so that if the pipeline needs debugging,
it gets debugged there.

Run this file directly:  uv run python src/score_final_test.py
"""

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from data_loading import feature_groups
from evaluation import (
    SEED,
    cross_validate_model,
    gini_normalized,
    load_final_test,
    load_train,
)
from train_baseline_models import REALISTIC_GINI_CEILING, _build_preprocessor, _check_realistic


def _winning_columns(df):
    """The 37-column set: every `ps_calc_*` column dropped, per #13."""
    categorical_cols, quantity_cols = feature_groups(df.columns)
    cat_nocalc = [c for c in categorical_cols if "_calc_" not in c]
    qty_nocalc = [c for c in quantity_cols if "_calc_" not in c]
    return cat_nocalc, qty_nocalc


def _build_model(n_cat, n_qty):
    return make_pipeline(
        _build_preprocessor(n_cat, n_qty, scale=False),
        RandomForestClassifier(
            n_estimators=200, min_samples_leaf=50, n_jobs=-1, random_state=SEED,
        ),
    )


def _capture_rate(y_true, y_score, top_pct=0.10):
    """Share of actual claims falling in the riskiest `top_pct` of rows."""
    n_top = int(len(y_true) * top_pct)
    top_idx = np.argsort(y_score)[::-1][:n_top]
    return y_true[top_idx].sum() / y_true.sum(), n_top


def _dry_run(cat_nocalc, qty_nocalc):
    """
    Prove the pipeline mechanics work end to end using ONLY the 80% training
    split, fitting on half of it and scoring on the other half.
    `load_final_test` plays no part in this function. If anything about the
    pipeline needs debugging, this is where to do it -- not against the
    holdout.
    """
    print("=" * 70)
    print("DRY RUN -- pipeline mechanics, training split only")
    print("=" * 70)

    train = load_train()
    X = train[cat_nocalc + qty_nocalc]
    y = train["target"]

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train))
    fit_idx, check_idx = idx[: len(idx) // 2], idx[len(idx) // 2:]

    model = _build_model(len(cat_nocalc), len(qty_nocalc))
    model.fit(X.values[fit_idx], y.values[fit_idx])
    y_score = model.predict_proba(X.values[check_idx])[:, 1]
    dry_gini = gini_normalized(y.values[check_idx], y_score)

    print(f"  half-split sanity Gini: {dry_gini:+.5f} "
          f"(a smaller, noisier version of the CV number is expected here)")
    assert 0.0 < dry_gini < REALISTIC_GINI_CEILING, (
        "dry run Gini is outside the plausible range -- debug the pipeline "
        "here before it ever touches the holdout"
    )
    print("  PASS -- pipeline works end to end\n")


def main():
    train = load_train()
    cat_nocalc, qty_nocalc = _winning_columns(train)
    print(f"columns: {len(cat_nocalc)} categorical + {len(qty_nocalc)} quantity "
          f"= {len(cat_nocalc) + len(qty_nocalc)}\n")

    _dry_run(cat_nocalc, qty_nocalc)

    print("=" * 70)
    print("CROSS-VALIDATION -- reference number, 80% split, 5 folds")
    print("=" * 70)
    X_train = train[cat_nocalc + qty_nocalc]
    y_train = train["target"]
    t0 = time.perf_counter()
    cv_mean, cv_std, _ = cross_validate_model(
        _build_model(len(cat_nocalc), len(qty_nocalc)), X_train, y_train,
    )
    cv_runtime = time.perf_counter() - t0
    print(f"  Gini {cv_mean:+.5f} +/- {cv_std:.5f}  ({cv_runtime:.1f}s)\n")

    print("=" * 70)
    print("FINAL FIT -- the whole 80% training split, one model")
    print("=" * 70)
    final_model = _build_model(len(cat_nocalc), len(qty_nocalc))
    t0 = time.perf_counter()
    final_model.fit(X_train.values, y_train.values)
    fit_runtime = time.perf_counter() - t0
    print(f"  fitted on {len(train):,} rows in {fit_runtime:.1f}s\n")

    print("=" * 70)
    print("FINAL TEST -- load_final_test(), called once, scored once")
    print("=" * 70)
    final = load_final_test(confirm="final evaluation")
    X_final = final[cat_nocalc + qty_nocalc]
    y_final = final["target"].values

    y_score = final_model.predict_proba(X_final.values)[:, 1]
    final_gini = gini_normalized(y_final, y_score)
    capture, n_top = _capture_rate(y_final, y_score)
    oof_capture = 0.2109  # #18's out-of-fold figure, for comparison only

    delta = final_gini - cv_mean
    print(f"  final test rows: {len(final):,}  (claims: {int(y_final.sum()):,})")
    print(f"  final test Gini: {final_gini:+.5f}")
    print(f"  cross-validation Gini: {cv_mean:+.5f} +/- {cv_std:.5f}")
    print(f"  delta (final - CV): {delta:+.5f}  ({delta / cv_std:+.2f}x the CV fold std)")
    _check_realistic("Final test", final_gini)

    print()
    print(f"  riskiest 10% of final test rows: {n_top:,}")
    print(f"  share of actual claims captured: {capture * 100:.2f}%")
    print(f"  #18's out-of-fold figure for comparison: {oof_capture * 100:.2f}%")
    print(f"  delta: {(capture - oof_capture) * 100:+.2f}pp")


if __name__ == "__main__":
    main()
