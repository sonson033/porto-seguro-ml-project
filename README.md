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
