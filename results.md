# Results

One row per scored model. A number that isn't in this table doesn't exist.

| Column | What goes in it |
| --- | --- |
| Date | When the run happened |
| Owner | Who ran it |
| Model | Model type and any non-default settings |
| Preprocessing | Missing-value handling, scaling, category encoding |
| Gini mean | Mean Normalized Gini across the CV folds |
| Gini std | Standard deviation across the same folds |
| Delta vs LR | Gini mean minus the logistic regression baseline's Gini mean |
| Beats std? | `yes` if the delta is larger than this row's Gini std. `n/a` for the baseline itself |
| Runtime | How long the run took |
| Commit | Commit SHA the run was made from |
| Notes | Anything that would change how someone reads the number |

| Date | Owner | Model | Preprocessing | Gini mean | Gini std | Delta vs LR | Beats std? | Runtime | Commit | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-01 | Jassin | DummyClassifier(strategy=most_frequent) | None — always predicts the majority class | 0.0000 | 0.0000 | -0.2572 | no | 0.5s | c11d1ba | Accuracy 96.331% ± 0.0004%. High accuracy, zero ranking power — evidence for the "not optimizing accuracy" line in the README (#2). |
| 2026-09-01 | Jassin | LogisticRegression(max_iter=1000) | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); standardize (quantity cols) | 0.2572 | 0.0036 | n/a | n/a | 25.1s | c11d1ba | Supersedes #14's self-check number (0.2389 ± 0.0052), which fed raw `_cat` codes into the model as numbers instead of one-hot encoding — see project log entries 13 and 15. `ps_car_11_cat` (104 levels) one-hot encoded as-is, no special handling, per #9's no-feature-engineering rule. |
| 2026-09-01 | Jassin | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-1 | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling (tree-based, scale-invariant) | 0.2687 | 0.0037 | +0.0114 | yes | 3m 53s | c11d1ba | LR fold std for comparison: ±0.0036. |
| 2026-09-02 | Burcak | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-1 — 37 columns, all 20 `ps_calc_*` dropped | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling | 0.2724 | 0.0031 | +0.0152 | yes | ~5m (10.2m for both arms) | 6c794f2 | Ablation for #13. Same folds and settings as the 57-column RF above; only the calc columns differ. Delta vs that RF: **+0.0038**, 1.24× this row's fold std. 4 of 5 folds improved, none got worse (fifth −0.00003). Fold spread fell 23% (0.00397 → 0.00305). First result to clear the 0.27 target. |
| 2026-09-02 | Burcak | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-2 — 34 columns, `ps_calc_*` + `ps_reg_*` dropped | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling | 0.25866 | 0.00517 | +0.0015 | no | 4.9m | 6c794f2 | Diagnostic for #13, not a proposed model. Group ablation from the 37-column calc-dropped model (0.27244): delta **-0.01378** across 3 columns = **-0.00459 per column**, the densest group in the dataset. Fold std rose 0.00305 → 0.00517 (+70%). |
| 2026-09-02 | Burcak | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-2 — 21 columns, `ps_calc_*` + `ps_car_*` dropped | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling | 0.23577 | 0.00698 | -0.0214 | no | 2.0m | 6c794f2 | Diagnostic for #13, not a proposed model. Group ablation from the 37-column model: delta **-0.03667** across 16 columns = -0.00229 per column. Fold std rose 0.00305 → 0.00698 (+129%). |
| 2026-09-02 | Burcak | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-2 — 19 columns, `ps_calc_*` + `ps_ind_*` dropped | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling | 0.21493 | 0.00517 | -0.0423 | no | 4.8m | 6c794f2 | Diagnostic for #13, not a proposed model. Group ablation from the 37-column model: delta **-0.05751** across 18 columns = -0.00320 per column — the largest total cost, despite `ind` also containing the three columns the forest never used. Fold std rose 0.00305 → 0.00517 (+70%). |
| 2026-09-02 | Fritz | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-1 — 26 columns, `ps_calc_*` dropped + 11 individually-dead columns (permutation drop ≤ 0) | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling | 0.27558 | 0.00667 | +0.0184 | yes | ~2.3m (1.0m permutation importance + 1.3m cross-validation) | e677f2b | Experiment for #17, from #13's shortlist item 3. Re-derived the dead-column list fresh (per the ticket's design) rather than reusing #13's: permutation importance on the 37-column model found 11 of 37 columns with drop ≤ 0, one fewer than #13's 12 — consistent with the noise-floor discrepancy #13 already flagged, not a real disagreement. Reference point is the 37-column calc-dropped model (0.27244 ± 0.00305): delta **+0.00314**, which does **not** exceed this row's fold std (0.00667). Fold spread rose 119% (0.00305 → 0.00667) instead of falling further, the opposite of #13's expectation for this lever. **REJECT** — the gain doesn't clear its own noise. |
| 2026-09-03 | Jassin | LogisticRegression(max_iter=1000) + missing-value-count feature — 37 columns (`ps_calc_*` dropped) + 1 numeric column | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); standardize (quantity cols incl. new feature) | 0.25981 | 0.00400 | +0.0026 | no | 20.9s | 5c8d274 | Experiment for #22, item 1. Previous best for this model is #9's LR (0.2572 ± 0.0036): delta **+0.00261**, does **not** clear this row's fold std (0.00400). Isolated feature effect (vs. an internal, unrecorded 37-column-LR-only reference of 0.25886 ± 0.00441): +0.00095 — indistinguishable from noise either way. **REJECT**. |
| 2026-09-03 | Jassin | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-1 + missing-value-count feature — 37 columns (`ps_calc_*` dropped) + 1 numeric column | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling | 0.27213 | 0.00509 | +0.0149 | yes | 3m 27s | 5c8d274 | Experiment for #22, item 1. Previous best for this model is #13's 37-column RF (0.27244 ± 0.00305): delta **-0.00031** (slightly negative), does **not** clear this row's fold std (0.00509). **REJECT**. |
| 2026-09-03 | Jassin | LogisticRegression(max_iter=1000) + `ps_car_11_cat` frequency encoding (fit inside each fold) — 37 columns (`ps_calc_*` dropped), one-hot replaced by 1 dense column | -1 → median impute (quantity cols); one-hot encode (other categorical cols, drop=if_binary); frequency-encode `ps_car_11_cat`; standardize (quantity cols) | 0.25845 | 0.00445 | +0.0013 | no | 11.9s | 5c8d274 | Experiment for #22, item 2. Previous best for this model is #9's LR (0.2572 ± 0.0036): delta **+0.00125**, does **not** clear this row's fold std (0.00445). Isolated feature effect (vs. the same 37-column-LR-only reference, 0.25886 ± 0.00441): **-0.00041**, i.e. slightly worse. **REJECT**. |
| 2026-09-03 | Jassin | RandomForest, n_estimators=200, min_samples_leaf=50, n_jobs=-1 + `ps_car_11_cat` frequency encoding (fit inside each fold) — 37 columns (`ps_calc_*` dropped), one-hot replaced by 1 dense column | -1 → median impute (quantity cols); one-hot encode (other categorical cols, drop=if_binary); frequency-encode `ps_car_11_cat`; no scaling | 0.27116 | 0.00504 | +0.0140 | yes | 2m 35s | 5c8d274 | Experiment for #22, item 2. Previous best for this model is #13's 37-column RF (0.27244 ± 0.00305): delta **-0.00128** (slightly negative), does **not** clear this row's fold std (0.00504). **REJECT**. |
| 2026-09-03 | Fritz | **FINAL TEST SET (scored once) — RandomForest**, n_estimators=200, min_samples_leaf=50, n_jobs=-1 — 37 columns, `ps_calc_*` dropped (frozen config, #13) | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); no scaling | **0.27308** | n/a — single score, not a CV mean | +0.0159 | n/a | 23.9s (final refit only) | f688e03 | **Issue #24 — the only call to `load_final_test` in the project.** Refit on the whole 80% training split (475,967 rows), scored once against the untouched 20% holdout (119,245 rows, 4,233 claims). Freshly-recomputed CV reference on this same commit: 0.27248 ± 0.00360 (5 folds: 0.26674, 0.27559, 0.27303, 0.27038, 0.27667) — final-vs-CV delta **+0.00060**, 0.17× the CV fold std, comfortably inside the ±0.003 gap the ticket called normal. Riskiest 10% of final-test predictions capture **21.45%** of actual claims vs #18's out-of-fold **21.09%** (+0.36pp) — read against #18. No modelling decision follows this number (AC-7). Script: `src/score_final_test.py`. |
| 2026-09-17 | Fritz | **Neural network (PyTorch MLP)** — 1 hidden layer (128 units), ReLU, dropout 0.3, Adam lr=1e-3, batch 1024, 9 epochs, weight decay 1e-4, output bias initialised to `logit(base rate)` — 37 columns, `ps_calc_*` dropped. Headline is the mean of 3 seeds (42/43/44) | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); standardize (quantity cols) | 0.27801 | 0.00678 | +0.0208 | yes | 1m 46s (3 × 5-fold CV; 35s per run) | cd7cb20 | Notebook: `notebooks/05_neural_network_comparison.ipynb`. Same 5 folds as every row above; `src/evaluation.py` imported unmodified. Per-seed CV means 0.27900 / 0.27780 / 0.27723 — seed spread **0.00074**, so the CV mean is stable even though a single fold at a fixed epoch is not. **Against the 37-column RF, re-run on this commit (0.27248 ± 0.00360, reproducing #13's 0.27244 to +0.00004): delta +0.00552, which does NOT clear this row's fold std (0.00678) — inconclusive by the standing rule that rejected #17 and #22.** The folds are shared, so the paired test also applies: the NN is higher on **5 of 5 folds**, paired mean +0.00552 ± 0.00372, t = 3.32 (4 df). Both readings are in the notebook; the unpaired rule is the conservative one, the paired one the more sensitive. Epoch budget fixed a priori (as `n_estimators=200` was) from inner 85/15 splits inside fold-training rows only — never an outer fold — chosen on a 3-epoch rolling mean over 4 curves and converted through optimizer steps (10 epochs × 317 steps on 324k rows → 9 epochs on 381k), so the NN trains on the same rows the forest did. Tuning grid lr {1e-3, 3e-4} × batch {1024, 4096} × {1 layer, 2 layers}: all 8 configs fell within seed noise of each other, so the winner was picked on **curve flatness** (tail-spread 0.0057 vs 0.0100–0.0250), flatness being what makes a fixed epoch count safe rather than a bet. Determinism: CPU, `set_num_threads(1)`, `use_deterministic_algorithms(True)`; rerunning seed 42 is bitwise identical. MPS rejected — it gives a different answer from CPU. Capture at riskiest 10%: **21.49%** (lift 2.15×) vs the forest's out-of-fold 21.09% (#18), **+0.40pp**. |
| 2026-09-17 | Fritz | Neural network (PyTorch MLP), config as above **+ `pos_weight=26.26`** in `BCEWithLogitsLoss` — seed 42 only | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); standardize (quantity cols) | 0.27269 | 0.00561 | +0.0155 | yes | ≈35s (one 5-fold CV; not separately timed) | cd7cb20 | Diagnostic for the class-imbalance question, not a proposed model. Reference is the seed-42 run above (0.27900): delta **-0.00631**, i.e. reweighting the loss made the ranking *worse* by more than this row's fold std. Expected: under weighted BCE the optimum is `w·p(x) / (w·p(x) + 1 − p(x))`, a strictly increasing function of `p(x)`, and Gini is invariant to strictly increasing transforms — so reweighting cannot change AUC in the well-specified limit. Whatever it does in finite samples has no guaranteed sign, and here the sign is negative. **Do not reweight.** |
| 2026-09-17 | Fritz | Neural network (PyTorch MLP), config as row 1 but **output bias left at 0** instead of `logit(base rate)` — seed 42 only | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); standardize (quantity cols) | 0.27043 | 0.00532 | +0.0132 | yes | ≈35s (one 5-fold CV; not separately timed) | cd7cb20 | Diagnostic, not a proposed model. Reference is the seed-42 run above (0.27900): delta **-0.00857**, larger than this row's fold std. At a 3.67% base rate the optimizer otherwise spends its first few hundred steps dragging a single scalar from 0 down to −3.27 while the weight gradients are tiny. One line of code, and it is worth more than every hyperparameter in the tuning grid combined. |
| 2026-09-17 | Fritz | LogisticRegression(max_iter=1000, **class_weight="balanced"**) — 37 columns, `ps_calc_*` dropped | -1 → median impute (quantity cols); one-hot encode (categorical cols, drop=if_binary); standardize (quantity cols) | 0.25821 | 0.00405 | +0.0010 | no | ≈50s (one 5-fold CV; not separately timed) | cd7cb20 | Diagnostic for the same class-imbalance question, on a model nobody can accuse of being mistuned. Matched 37-column plain LR, same folds and same commit: **0.25886 ± 0.00441**. Delta from rebalancing: **-0.00065**, an order of magnitude inside the fold std — indistinguishable from zero. Model-agnostic in-project evidence that reweighting does not move a rank metric, which is why the network above trains on unweighted BCE. |

---

## Head to head — neural network vs. random forest

Both models scored on the **same five folds** through `src/evaluation.py`, and both timed in the same
session on the same machine (Apple M2 Pro, 10 cores), so the cost rows are like-for-like rather than
quoted across runs. Working: `notebooks/05_neural_network_comparison.ipynb`, commit `cd7cb20`.

| Dimension | Neural network | Random forest | Winner |
| --- | --- | --- | --- |
| **Primary metric (Gini)** | **0.27801** ± 0.00678 | 0.27248 ± 0.00360 | NN by +0.00552 (+2.03%) — 5/5 folds, but inside its own fold std |
| **Capture @ riskiest 10%** | **21.49%** (2.15× lift) | 21.09% | NN by +0.40pp |
| **Training time** (5-fold CV, own defaults) | **35s** — 1 thread | 115s — `n_jobs=-1`, 10 cores | NN |
| **Training time** (one fold, single core) | **5.2s** | 98.5s | NN by 19× |
| **Inference** (batch 8192, per row) | **0.15 µs** (6.6M rows/s) | 4.15 µs (241k rows/s) | NN by 27× |
| **Inference** (single row) | **0.045 ms** | 16.51 ms | NN by 367× |
| **Data volume used** | 475,967 rows / 17,461 claims / 206 features | identical | tie — by construction |
| **Interpretability (1–5)** | **1** | **3** | forest, decisively |
| **Engineering effort** | ~8 hours | ~2 hours | forest |
| **Would you ship it?** | **No** | **Yes** | forest |

### Interpretability — 1 vs 3, justified

The forest is not a glass box, but it answers questions. `feature_importances_` ranks columns,
permutation importance gives a defensible per-column contribution — that is what made the ablations
in #13 and #17 possible at all — and individual trees can be printed and read. It loses points
because 200 trees cannot be reasoned about as a whole, and it offers no per-applicant explanation
without extra tooling.

The network scores 1 because none of that exists. No native importance, a learned representation of
128 anonymous units over 206 already-anonymized inputs, and any explanation requires bolting on SHAP
or integrated gradients — more code, more compute, and an approximation rather than the model's own
answer. For a model that would feed insurance pricing, where the README already flags proxy
discrimination as a deployment blocker, that is not a cosmetic loss.

### Engineering effort — where the ~8 hours went

Almost none of it to modelling; the network itself is about fifteen lines. The time went to adding a
121MB dependency to a shared lockfile; writing an estimator that survives sklearn's `clone()`
contract, where most mistakes fail *silently* (building the module in `__init__` draws its initial
weights off the global RNG at clone time, so folds start from different weights for invisible
reasons); discovering that `BCEWithLogitsLoss` is log-sum-exp stable, so a catastrophically diverged
run returns a large *finite* loss and needs an explicit threshold rather than an `isfinite` check;
proving bitwise reproducibility across processes; and designing a tuning protocol whose epoch budget
is not fitted to noise.

This is a **one-time** cost. A second network on this codebase would be far cheaper — which matters
if the answer here is "not yet" rather than "never".

### Would you ship it? No — and not because of the metric

The network is better on the metric and dramatically cheaper to run. It is still not the one to
ship, because +0.40pp of extra claims captured in the top decile does not pay for losing the audit
surface on a model that feeds pricing. See the verdict section in `README.md` for what we would tell
stakeholders and what would flip the answer.
