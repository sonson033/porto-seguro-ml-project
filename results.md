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
