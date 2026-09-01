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
