"""Cumulative gains / decile capture for the 37-column forest.

Mirrors notebooks/04_error_analysis_and_reflection.ipynb exactly up to the
out-of-fold predictions, then extends the single 10% point (AC-3) to every
review cutoff, which is what the presentation needs.

Run from anywhere inside the repo:  uv run python scripts/gains_curve.py
"""
import os
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from data_loading import feature_groups
from evaluation import SEED, N_FOLDS, gini_normalized, load_train
from train_baseline_models import _build_preprocessor

# --- identical to notebook 04 ------------------------------------------------
train = load_train()
categorical_cols, quantity_cols = feature_groups(train.columns)
cat_nocalc = [c for c in categorical_cols if "_calc_" not in c]
qty_nocalc = [c for c in quantity_cols if "_calc_" not in c]
X = train[cat_nocalc + qty_nocalc]
y = train["target"]
print(f"rows {len(train):,} | {len(cat_nocalc) + len(qty_nocalc)} columns")

model = make_pipeline(
    _build_preprocessor(len(cat_nocalc), len(qty_nocalc), scale=False),
    RandomForestClassifier(n_estimators=200, min_samples_leaf=50,
                           n_jobs=-2, random_state=SEED),
)
splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("fitting 5 folds - expect roughly 6 to 10 minutes, no output until done...")
oof_risk = cross_val_predict(model, X, y, cv=splitter, method="predict_proba")[:, 1]
train = train.assign(oof_risk=oof_risk)

# --- gate: did we reproduce his run? ----------------------------------------
pooled_gini = gini_normalized(y, oof_risk)
drift = abs(pooled_gini - 0.27243)
print(f"\npooled OOF gini : {pooled_gini:.5f}")
print(f"his notebook    : 0.27243")
print(f"difference      : {drift:.5f}   -> "
      f"{'PASS - same run reproduced' if drift < 0.0005 else 'DRIFT - do not use these numbers, tell Claude'}")

# --- new: capture at every cutoff, his method extended ----------------------
total_claims = int(train["target"].sum())
n = len(train)


def capture_at(pct):
    """His AC-3 calculation at an arbitrary cutoff (nlargest, as he used)."""
    k = int(n * pct / 100)
    claims = int(train.nlargest(k, "oof_risk")["target"].sum())
    return k, claims, claims / total_claims


print(f"\ntotal claims: {total_claims:,} of {n:,} drivers "
      f"({total_claims / n * 100:.4f}%)")

print("\ncumulative gains - reviewing the riskiest X% by predicted risk")
print(f"{'reviewed':>9}  {'drivers':>9}  {'claims found':>12}  {'% of claims':>11}  {'lift':>5}")
for pct in range(10, 101, 10):
    k, claims, cap = capture_at(pct)
    print(f"{pct:>8}%  {k:>9,}  {claims:>12,}  {cap * 100:>10.2f}%  {cap / (pct / 100):>4.2f}x")

print("\n--- the two numbers for slide 8 ---")
for pct in (10, 20):
    k, claims, cap = capture_at(pct)
    print(f"riskiest {pct}% ({k:,} drivers) contains {claims:,} of {total_claims:,} claims "
          f"= {cap * 100:.2f}%  ({cap / (pct / 100):.2f}x random)")

# --- save, so nobody has to refit for the next question ---------------------
out = ROOT / "data" / "processed" / "oof_risk_37col.npy"
out.parent.mkdir(parents=True, exist_ok=True)
np.save(out, oof_risk)
print(f"\npredictions saved to {out.relative_to(ROOT)} "
      f"(row order of load_train(); {len(oof_risk):,} values)")