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

> The first row is an example, not a result. The numbers are illustrative. Delete it
> once the first real row lands.

| Date | Owner | Model | Preprocessing | Gini mean | Gini std | Delta vs LR | Beats std? | Runtime | Commit | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-02 | EXAMPLE | RandomForest, n_estimators=200, min_samples_leaf=50 | -1 → NaN, median impute + missing flag, cat columns as codes | 0.2701 | 0.0041 | +0.0189 | yes | 8m 12s | a1b2c3d | _Example row, not a real result_ |
