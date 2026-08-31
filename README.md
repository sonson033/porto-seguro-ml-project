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
| `docs/` | charter, experiment log, retrospective |

## Notes

We train on `train.csv` only. `test.csv` and `sample_submission.csv` are not in the repo — their only use is submitting to the Kaggle leaderboard. Whoever makes that submission downloads `test.csv` from Kaggle at the time.
