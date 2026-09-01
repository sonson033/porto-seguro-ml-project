"""
Baseline models -- Porto Seguro claim risk ranking
===================================================

Three reference models scored through the shared harness (src/evaluation.py):
a constant-prediction floor, a logistic regression baseline to beat, and a
random forest as the strong upper end without tuning or feature engineering.

Run this file directly to reproduce every number printed below:
    uv run python src/train_baseline_models.py
"""

import time

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loading import feature_groups
from evaluation import N_FOLDS, SEED, cross_validate_model, gini_normalized, load_train

# A score at or above this is treated as a leak, not a win, and blocks
# recording the number. Issue #9 says to check scores against "the realistic
# range recorded in the README (from #2)" -- that range does not actually
# exist in README.md (issue #2's own spec never asked for one, and none was
# added later; see the project log). Substituting the well-documented public
# benchmark for this exact Kaggle competition instead: tuned, heavily
# feature-engineered, stacked solutions topped out around Gini 0.297 on the
# private leaderboard. This ticket does no tuning, no feature engineering and
# no stacking, so a single model approaching -- let alone exceeding -- that
# ceiling is a leak, not real signal.
REALISTIC_GINI_CEILING = 0.30


def _build_preprocessor(n_categorical, n_quantity, scale):
    """
    Shared preprocessing for the logistic regression and the random forest:
    median-impute the -1 sentinel in quantity columns (per issue #6), one-hot
    encode categorical columns. `drop="if_binary"` keeps a true _bin column
    as a single 0/1 column instead of doubling it.

    `scale` adds a StandardScaler after imputation for quantity columns.
    Only logistic regression needs it -- trees are invariant to monotonic
    per-feature rescaling, so skipping it for the forest saves memory and
    time without changing what it can learn.

    ps_car_11_cat has 104 distinct values (checked directly against the
    data). It is one-hot encoded the same as every other categorical column
    rather than given special treatment -- any alternative (frequency or
    target encoding) would be feature engineering, which this ticket
    explicitly excludes.

    Selects columns by *position*, not name: evaluation.cross_validate_model()
    converts X to a bare numpy array internally before fitting each fold, so
    a name-based ColumnTransformer would fail on the array it actually
    receives. `main()` builds X as categorical columns followed by quantity
    columns, so positions 0..n_categorical-1 are categorical and the rest
    are quantity.
    """
    quantity_steps = [SimpleImputer(missing_values=-1, strategy="median")]
    if scale:
        quantity_steps.append(StandardScaler())

    categorical_idx = list(range(n_categorical))
    quantity_idx = list(range(n_categorical, n_categorical + n_quantity))

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                               sparse_output=False, dtype=np.float32),
         categorical_idx),
        ("qty", make_pipeline(*quantity_steps), quantity_idx),
    ])


def _cv_accuracy_and_gini(model, X, y, n_folds=N_FOLDS, seed=SEED):
    """
    Same fold structure as evaluation.cross_validate_model(), but records
    accuracy alongside Gini. Used only for the constant-prediction floor:
    AC1 needs both metrics for that row, and cross_validate_model() only
    returns Gini.
    """
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    gini_scores, acc_scores = [], []
    for train_idx, val_idx in splitter.split(X_arr, y_arr):
        fold_model = clone(model)
        fold_model.fit(X_arr[train_idx], y_arr[train_idx])
        y_score = fold_model.predict_proba(X_arr[val_idx])[:, 1]
        y_pred = fold_model.predict(X_arr[val_idx])
        gini_scores.append(gini_normalized(y_arr[val_idx], y_score))
        acc_scores.append((y_pred == y_arr[val_idx]).mean())

    gini_scores, acc_scores = np.array(gini_scores), np.array(acc_scores)
    return gini_scores.mean(), gini_scores.std(), acc_scores.mean(), acc_scores.std()


def _check_realistic(name, gini_mean):
    assert gini_mean < REALISTIC_GINI_CEILING, (
        f"{name} scored {gini_mean:.4f}, at or above the "
        f"{REALISTIC_GINI_CEILING} realistic ceiling for this dataset "
        "without tuning or feature engineering -- check for a leak before "
        "recording this number."
    )


def main():
    train = load_train()  # the 80% split only; load_final_test() is never called in this file
    categorical_cols, quantity_cols = feature_groups(train.columns)
    X = train[categorical_cols + quantity_cols]
    y = train["target"]

    print("=" * 70)
    print("FLOOR -- constant prediction (DummyClassifier, most_frequent)")
    print("=" * 70)
    t0 = time.perf_counter()
    constant_model = DummyClassifier(strategy="most_frequent")
    const_gini_mean, const_gini_std, const_acc_mean, const_acc_std = \
        _cv_accuracy_and_gini(constant_model, X, y)
    const_runtime = time.perf_counter() - t0
    print(f"  Gini      {const_gini_mean:+.5f} +/- {const_gini_std:.5f}")
    print(f"  Accuracy  {const_acc_mean:.5%} +/- {const_acc_std:.5%}")
    print(f"  Runtime   {const_runtime:.1f}s")
    _check_realistic("Constant baseline", const_gini_mean)

    print()
    print("=" * 70)
    print("BASELINE TO BEAT -- logistic regression")
    print("=" * 70)
    t0 = time.perf_counter()
    lr_pipeline = make_pipeline(
        _build_preprocessor(len(categorical_cols), len(quantity_cols), scale=True),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    lr_mean, lr_std, lr_scores = cross_validate_model(lr_pipeline, X, y)
    lr_runtime = time.perf_counter() - t0
    print(f"  Gini      {lr_mean:+.5f} +/- {lr_std:.5f}")
    print(f"  Runtime   {lr_runtime:.1f}s")
    _check_realistic("Logistic regression", lr_mean)

    print()
    print("=" * 70)
    print("STRONG -- random forest (n_jobs=-1, min_samples_leaf=50)")
    print("=" * 70)
    t0 = time.perf_counter()
    rf_pipeline = make_pipeline(
        _build_preprocessor(len(categorical_cols), len(quantity_cols), scale=False),
        RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=SEED,
        ),
    )
    rf_mean, rf_std, rf_scores = cross_validate_model(rf_pipeline, X, y)
    rf_runtime = time.perf_counter() - t0
    print(f"  Gini      {rf_mean:+.5f} +/- {rf_std:.5f}")
    print(f"  Runtime   {rf_runtime:.1f}s")
    _check_realistic("Random forest", rf_mean)

    delta = rf_mean - lr_mean
    print()
    print(f"  Delta over logistic regression: {delta:+.5f}")
    print(f"  (logistic regression fold std: {lr_std:.5f}, "
          f"random forest fold std: {rf_std:.5f})")
    # Not abs(delta): results.md defines this as the signed delta exceeding
    # the row's own std, so a model that's worse than LR by more than its
    # std never reads as "yes" just because the gap is large.
    print(f"  Beats std? {'yes' if delta > rf_std else 'no'}")

    print()
    print("Reproducibility check: rerunning the strongest model (random forest)")
    _, _, rf_scores_b = cross_validate_model(rf_pipeline, X, y, verbose=False)
    print(f"  repeat run identical: {np.allclose(rf_scores, rf_scores_b)}")


if __name__ == "__main__":
    main()
