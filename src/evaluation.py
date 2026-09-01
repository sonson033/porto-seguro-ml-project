"""
Evaluation harness — Porto Seguro claim risk ranking
====================================================

One shared scoring function and one shared split, so that when two of us
compare models we are comparing the same thing.

Run this file directly to verify it:  uv run python src/evaluation.py
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5
FINAL_TEST_PCT = 20
_HASH_BUCKETS = 100
DATA_PATH = "data/raw/train.csv"


# --------------------------------------------------------------- the metric

def gini_normalized(y_true, y_score):
    """
    Normalized Gini coefficient — the competition metric.

    Algebraically identical to 2 * AUC - 1. Both measure the same thing: how
    often the model ranks a claimer above a non-claimer. AUC puts that on a
    0.5-to-1 scale where 0.5 is random; Gini rescales to 0-to-1.

        0.0  no better than random
        1.0  perfect ranking
       -1.0  perfectly reversed

    Only the ORDER of y_score matters, never its absolute values.
    """
    return 2 * roc_auc_score(y_true, y_score) - 1


# ---------------------------------------------------------------- the split

def _hash_bucket(row_id):
    """
    Map a row id to one of 100 stable buckets.

    hashlib rather than Python's built-in hash(): the built-in is randomised
    per process for strings, so it gives different answers in different runs.
    md5 is a fixed mathematical function — same input, same output, on any
    machine, in any library version, forever. That is what AC-2 needs.

    (md5 is used here for bucketing, not for security.)
    """
    digest = hashlib.md5(str(int(row_id)).encode()).hexdigest()
    return int(digest, 16) % _HASH_BUCKETS


def is_final_test(ids):
    """
    True for rows belonging to the final test set.

    Membership depends ONLY on the row's own id — never on its position in the
    file, never on library behaviour. So reordering the rows, rerunning, or
    running on a different machine cannot move a row across the boundary.
    """
    return np.array([_hash_bucket(i) < FINAL_TEST_PCT for i in np.asarray(ids)])


# --------------------------------------------------------------- the loaders

def _read_raw():
    """The only place in the project that touches the data file."""
    df = pd.read_csv(DATA_PATH)
    assert df["id"].is_unique, "id is not unique — the split assumes it is"
    return df


def load_train():
    """
    The everyday loader: 80% of the rows. Use this for everything —
    exploration, cross-validation, model choice, tuning.

    Missing values come back as the raw -1 sentinel. Handling them is a
    modelling decision and belongs inside each model's pipeline, so that
    alternatives can still be compared fairly.
    """
    df = _read_raw()
    return df.loc[~is_final_test(df["id"].values)].reset_index(drop=True)


def load_final_test(confirm=None):
    """
    The final 20%. Scored ONCE, at the end of the project.

    Deliberately awkward to call, because it must take part in no decision:
    not model choice, not feature selection, not tuning.
    """
    if confirm != "final evaluation":
        raise ValueError(
            "The final test set is scored ONCE, at the end of the project.\n"
            "It takes part in no decision: not model choice, not feature\n"
            "selection, not tuning. If you are certain, call:\n"
            '    load_final_test(confirm="final evaluation")'
        )
    df = _read_raw()
    return df.loc[is_final_test(df["id"].values)].reset_index(drop=True)


# ---------------------------------------------------------- verification

def _verify_metric():
    """AC-1: random scores score ~0, perfect scores score ~1."""
    print("AC-1  metric")

    y = np.array([0, 0, 0, 1, 1])
    perfect = gini_normalized(y, y)
    reverse = gini_normalized(y, 1 - y)
    flat = gini_normalized(y, np.full(len(y), 0.5))

    rng = np.random.default_rng(SEED)
    y_big = (rng.random(200_000) < 0.036448).astype(int)
    random_scores = gini_normalized(y_big, rng.random(200_000))

    p = rng.random(200_000)
    invariant = np.isclose(gini_normalized(y_big, p),
                           gini_normalized(y_big, p * 7 + 3))

    print(f"      perfect ranking   {perfect:+.6f}   expect +1.0")
    print(f"      reversed ranking  {reverse:+.6f}   expect -1.0")
    print(f"      constant score    {flat:+.6f}   expect  0.0")
    print(f"      random scores     {random_scores:+.6f}   expect ~0.0")
    print(f"      scale-invariant   {invariant}       expect True")

    assert np.isclose(perfect, 1.0), "perfect ranking must score 1.0"
    assert np.isclose(reverse, -1.0), "reversed ranking must score -1.0"
    assert np.isclose(flat, 0.0), "a constant score must be 0.0"
    assert abs(random_scores) < 0.01, "random scores must be ~0"
    assert invariant, "Gini must depend only on rank order"
    print("      PASS\n")


def _verify_split():
    """AC-2 and AC-3."""
    print("AC-2  split is stable and depends only on the row id")

    raw = _read_raw()
    n = len(raw)
    ids = raw["id"].values
    mask = is_final_test(ids)

    reordered = raw.sample(frac=1, random_state=99).reset_index(drop=True)
    same_after_reorder = (set(raw.loc[mask, "id"])
                          == set(reordered.loc[is_final_test(reordered["id"].values), "id"]))
    same_on_rerun = set(raw.loc[mask, "id"]) == set(raw.loc[is_final_test(ids), "id"])

    print(f"      same rows after reordering the file: {same_after_reorder}")
    print(f"      same rows on a repeat run:           {same_on_rerun}")
    assert same_after_reorder, "split moved when rows were reordered"
    assert same_on_rerun, "split is not deterministic"
    print("      PASS\n")

    print("AC-3  80/20, and the positive rate matches")
    train, final = raw.loc[~mask], raw.loc[mask]
    print(f"      {'':12}{'rows':>9}{'share':>9}{'positive':>11}")
    for name, part in [("train", train), ("final test", final), ("whole file", raw)]:
        print(f"      {name:12}{len(part):>9,}{len(part)/n*100:>8.3f}%{part.target.mean()*100:>10.4f}%")

    assert len(train) + len(final) == n, "rows were lost or duplicated"
    assert len(set(train["id"]) & set(final["id"])) == 0, "the two parts overlap"
    assert 18 <= len(final) / n * 100 <= 22, "final test set is not near 20%"

    # The two positive rates will never match exactly — assigning by hash is
    # random assignment, so they differ by sampling error. Compare the gap to
    # that expected error rather than to an arbitrary tolerance.
    p = train["target"].mean()
    gap = abs(p - final["target"].mean())
    se = np.sqrt(p * (1 - p) * (1 / len(train) + 1 / len(final)))
    print(f"\n      positive-rate gap {gap*100:.4f} pp")
    print(f"      expected sampling error {se*100:.4f} pp  ->  gap is {gap/se:.2f} SE")
    assert gap < 3 * se, "positive rates differ by more than sampling error explains"
    print("      PASS\n")

# ------------------------------------------------- the cross-validation runner

def cross_validate_model(model, X, y, n_folds=N_FOLDS, seed=SEED, verbose=True):
    """
    Score a model on 5 stratified, shuffled folds and report mean AND spread.

    Stratified means every fold holds the same share of claims, so the spread
    between folds reflects the MODEL rather than lucky or unlucky folds.

    IMPORTANT — leakage: pass a scikit-learn Pipeline containing any imputer
    or scaler, not a bare model with the data pre-processed beforehand. This
    function calls fit() inside each fold, so a Pipeline gets fitted on the
    training rows only. Scaling everything up front instead lets the
    validation rows influence training and inflates every score afterwards.

    Returns (mean, std, per_fold_scores).
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []

    for i, (train_idx, val_idx) in enumerate(splitter.split(X_arr, y_arr), 1):
        fold_model = clone(model)          # a fresh, unfitted copy every fold
        fold_model.fit(X_arr[train_idx], y_arr[train_idx])

        if hasattr(fold_model, "predict_proba"):
            y_score = fold_model.predict_proba(X_arr[val_idx])[:, 1]
        else:
            y_score = fold_model.decision_function(X_arr[val_idx])

        score = gini_normalized(y_arr[val_idx], y_score)
        scores.append(score)
        if verbose:
            print(f"      fold {i}  gini {score:+.5f}  "
                  f"(claims in fold: {int(y_arr[val_idx].sum()):,})")

    scores = np.array(scores)
    mean, std = scores.mean(), scores.std()
    if verbose:
        print(f"      mean {mean:+.5f}  std {std:.5f}")
    return mean, std, scores


def _verify_cv():
    """AC-5: two runs on the same model return identical numbers."""
    print("AC-5  cross-validation runner")

    train = load_train()
    features = [c for c in train.columns if c not in ("id", "target")]
    X, y = train[features], train["target"]

    # A Pipeline, so imputing and scaling happen inside each fold.
    # missing_values=-1 handles this dataset's sentinel directly.
    model = make_pipeline(
        SimpleImputer(missing_values=-1, strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )

    mean_a, std_a, scores_a = cross_validate_model(model, X, y)
    print()
    mean_b, std_b, scores_b = cross_validate_model(model, X, y, verbose=False)

    identical = np.allclose(scores_a, scores_b)
    print(f"      repeat run identical: {identical}")
    assert identical, "cross-validation is not reproducible"
    assert len(scores_a) == N_FOLDS, f"expected {N_FOLDS} folds"
    print("      PASS\n")
    
if __name__ == "__main__":
    _verify_metric()
    _verify_split()
    _verify_cv()
