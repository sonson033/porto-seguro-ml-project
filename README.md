## Objective

We rank Porto Seguro policyholders by how likely they are to file an auto insurance claim in the next year, so that underwriting can match premium to risk instead of cross-subsidising risky drivers with safe ones. We measure success by Normalized Gini on held-out data.

### Primary KPI — Normalized Gini (= 2 × AUC − 1)

The product is a ranking, not a decision. Gini measures how well that ranking orders policyholders by risk, and it needs no probability threshold — which matters, because we never set one.

### Metrics we are explicitly not reporting

| Metric | Why not |
|---|---|
| **Accuracy** | Only ~3.6% of policyholders file a claim, so predicting "no claim" for everyone scores ~96% while ranking nobody. It is the most flattering and least useful number available to us. |
| **Precision / recall / F1** | These require choosing a cutoff. Our output is an ordering; there is no cutoff to choose. |
| **Log-loss** | Measures how well-calibrated the predicted probabilities are. We use only their order, not their values. |

### Scope limit — what this dataset cannot answer

The data contains **no premium, no claim cost, and no policy exposure**. We can produce the risk ordering that pricing consumes; we cannot compute a price, a loss ratio, or money saved. Any figure in euros would be invented.

The features are also anonymized, so the model cannot be audited for proxy discrimination. For a system that would feed real insurance pricing that is a genuine deployment blocker, and it is out of scope for this project.

---

*Figures marked ~ are to be confirmed in the data sanity pass.*

---

## Setup

You need [uv](https://docs.astral.sh/uv/) and git. Nothing else — uv installs Python 3.12 itself.

```bash
git clone git@github.com:sonson033/porto-seguro-ml-project.git
cd porto-seguro-ml-project
uv sync
```

`uv sync` reads `uv.lock` and installs the exact same package versions for everyone. No Kaggle account needed.

### Get the data

`data/train.zip` is committed. Extract it once:

```bash
uv run python -c "import zipfile; zipfile.ZipFile('data/train.zip').extract('train.csv', 'data/raw')"
```

That puts `train.csv` into `data/raw/`, which is git-ignored. Sanity check: 595,212 rows x 59 columns, claim rate 3.645%.

### Running things

Run through the environment rather than activating it:

```bash
uv run jupyter lab
uv run python scripts/<name>.py
```

### Adding a package

```bash
uv add <package>
```

This updates `pyproject.toml` and `uv.lock` together, so everyone stays in sync. **Never use `pip install`** — it installs only on your machine and your environment silently drifts from the rest of the team's.

## Layout

| Path | Contents |
|---|---|
| `data/train.zip` | Kaggle training data, committed |
| `data/raw/` | extracted CSVs — git-ignored |
| `data/processed/` | derived datasets — git-ignored |
| `notebooks/` | exploratory work, one per person |
| `src/` | shared code, imported by notebooks |
| `scripts/` | runnable scripts |
| `docs/` | GitHub Pages source — the published model-comparison page |

## Notes

We train on `train.csv` only. `test.csv` and `sample_submission.csv` are not in the repo — their only use is submitting to the Kaggle leaderboard. Whoever makes that submission downloads `test.csv` from Kaggle at the time.

---

## Evaluation

All models are scored with **normalized Gini** (`2 × AUC − 1`), from `src/evaluation.py`.

### The split

Every row belongs to one of two parts, decided by a hash of its `id`:

- **80% — training.** Everything happens here: exploration, cross-validation, model choice, tuning. Load with `load_train()`.
- **20% — final test.** Scored once, at the very end. It takes part in no decision — not model choice, not feature selection, not tuning. Load with `load_final_test(confirm="final evaluation")`, which is deliberately awkward to call by accident.

Membership depends **only** on the row's own `id`, never on its position in the file. So reordering the rows, rerunning, or running on a different machine cannot move a row across the boundary. `train_test_split(random_state=…)` does not give that guarantee — it assigns by position, and we confirmed it puts different rows in the test set after a reorder.

We considered stratifying the hash by class to make the positive rates match more exactly, and rejected it: that would make a row's membership depend on which other rows are present, which is a weaker guarantee than the one above.

### Cross-validation

**5 stratified, shuffled folds, `random_state=42`**, on the 80% only. Stratified so every fold holds the same share of claims — which is why the spread between folds reflects the model rather than fold composition.

### Two rules

**Always report mean and standard deviation**, never the mean alone. It is how we tell a real improvement from noise.

**Pass a Pipeline**, not a pre-processed dataset. `cross_validate_model()` calls `fit()` inside each fold, so an imputer or scaler inside a Pipeline is fitted on that fold's training rows only. Scaling everything up front lets validation rows influence training and inflates every score afterwards.

### Verify it

```bash
uv run python src/evaluation.py
```

Checks the metric, the split's stability, the 80/20 proportions and the positive-rate match, and that cross-validation is reproducible.

### Measured

|  | rows | share | positive |
|---|---|---|---|
| training | 475,967 | 79.966% | 3.6685% |
| final test | 119,245 | 20.034% | 3.5498% |
| whole file | 595,212 | 100.000% | 3.6448% |

The positive rates differ by 0.1187 percentage points — 1.95 standard errors of random sampling, which is within what random assignment produces.

Logistic regression baseline (median imputation of `-1`, standardised): **Gini 0.2389 ± 0.0052** across the 5 folds.

---

## Verdict — neural network vs. random forest

> **[Read this as a page →](https://sonson033.github.io/porto-seguro-ml-project/)**
> The same comparison with both architectures drawn out and the results interactive —
> built for presenting. Source: `docs/index.html`.
>
> Preview it locally with `python3 -m http.server 8000 --directory docs`.

We built a feedforward neural network (PyTorch; one hidden layer of 128 units, ReLU, dropout) and
scored it on the **identical five folds** as the random forest, through the same harness. Full
working: `notebooks/05_neural_network_comparison.ipynb`. The 20% holdout took no part in it.

### Did the network beat the forest? Marginally, and consistently.

| | Neural network | Random forest |
|---|---|---|
| **Gini** | **0.27801** ± 0.00678 | 0.27248 ± 0.00360 |
| Claims caught in the riskiest 10% | **21.49%** | 21.09% |

**+0.00552 Gini, a 2.03% relative improvement.** The network scored higher on **all five folds**
(paired t = 3.32 on 4 df), so the gain is consistent rather than one lucky fold.

But it does **not** clear our standing bar. Our rule — a delta counts only if it exceeds its own
row's fold standard deviation — is what rejected the experiments in #17 and #22, and by it +0.00552
against a fold std of 0.00678 is inconclusive. The two tests disagree because the paired one is
more sensitive; we report both rather than picking the flattering one.

In business terms the whole difference is **+0.40 percentage points** of claims captured in the
top-risk decile.

### Was it worth the cost? The cost turned out not to be where we expected.

The network is **cheaper to run than the forest, not dearer** — the opposite of the usual story:

| | Neural network | Random forest | |
|---|---|---|---|
| Training, 5-fold CV | **35s** | 115s | 3× faster |
| Training, one fold, single core | **5.2s** | 98.5s | 19× faster |
| Inference throughput | **0.15 µs/row** | 4.15 µs/row | 27× faster |
| Single-row latency | **0.045 ms** | 16.5 ms | 367× faster |

So "too expensive" is not the objection. The real costs are elsewhere:

- **Interpretability: 1/5 versus 3/5.** The forest gives feature importances and permutation
  importance — that is what made the column ablations in #13 and #17 possible at all. The network
  gives none of it. Any explanation needs SHAP or integrated gradients bolted on: more code, and an
  approximation rather than the model's own answer.
- **Engineering effort: ~8 hours versus ~2.** Almost none of it modelling. It went to a 121MB
  dependency in the shared lockfile, an estimator that survives scikit-learn's `clone()` contract
  (where most mistakes fail silently), proving bitwise reproducibility, and designing a tuning
  protocol that does not fit noise. The network itself is about fifteen lines.
- **Stability: roughly twice the fold-to-fold variance** (0.00678 vs 0.00360). The better average
  comes with a wider spread.

### Which model ships: the random forest.

**What we would tell stakeholders.** We tested a neural network properly rather than assuming trees
were better. It ranks about 2% more accurately and it scores far faster. We are not shipping it,
because it cannot explain why it ranked any particular policyholder where it did — and this model
feeds insurance pricing. This README already flags that anonymized features prevent us auditing for
proxy discrimination; that is a known deployment blocker. Giving up the little explanation tooling
we have left, to gain 0.4 percentage points of claim capture, makes a blocked system harder to
unblock. The forest keeps the audit surface at no measurable cost to the business outcome.

That is a judgement about **this** application, not about neural networks in general. The experiment
was worth running: we now know the ceiling is close, and we know where the network's real advantages
sit — which is not where we would have guessed.

### What would flip the answer

- **A latency requirement.** Today scoring is batch, so a 367× single-row latency advantage buys
  nothing. Real-time per-applicant quoting at meaningful volume would make it decisive.
- **More claims.** 17,461 positives is thin for a network. A substantially larger book, or
  multi-year claim history, is the single most likely thing to turn +0.4pp into something that pays
  for the interpretability trade.
- **A different problem shape.** This is the real one. If the data gained structure a network can
  exploit — telematics sequences, free-text claim descriptions, damage photographs — the comparison
  stops being close, because a forest has no way to use any of it. Today's data is flat, anonymized
  and tabular, which is precisely where trees are strongest.
- **A change in the explanation requirement.** If SHAP-style post-hoc attribution were accepted for
  pricing review, interpretability stops being the binding constraint and the decision comes down to
  the metric — where the network is already ahead.
- **Wanting an ensemble rather than a single model.** The network's errors come from a different
  inductive bias than the forest's, so blending the two would likely beat either alone. That is how
  neural networks actually appeared in the top solutions to this competition — as components, not as
  winners. We did not test it; it is the obvious next experiment.
