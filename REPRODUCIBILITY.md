# Reproducibility Protocol

## Experimental unit

One experimental unit is a complete WSN simulation for one routing protocol and one random seed. The same seed initializes the same node topology and environmental stochastic process across competing protocols, enabling paired statistical comparison.

## Default design

The manuscript-matched configuration uses 200 nodes in a 1000 × 1000 m² area, 2 J initial energy, 100 m nominal communication radius, 4000-bit packets, 1000 rounds, a 30-particle PSO swarm, and 100 PSO iterations.

## Recommended evaluation

Use at least 10 independent topology seeds for the main table. For a more defensible revision, 20–30 seeds are preferable when runtime permits.

Evaluate:

- cumulative energy consumption
- packet-delivery ratio
- mean end-to-end latency
- routing success
- mean hop count
- energy-balance fairness
- first-node-death round
- half-node-death round
- final surviving nodes
- delivered bits per joule

## ML validation

The ML component predicts next-step node energy drain from node telemetry. Node IDs are split by groups so that validation nodes are not seen during training. The repository reports MAE, RMSE, R², train size, validation size, and feature importance.

The model is not trained on final test-route outcomes. It is trained on a reproducible telemetry generator using the same physical energy model and independent node groups. If real hardware traces become available, replace the telemetry source while preserving the validation interface.

## Statistical validation

For each metric, the proposed method is paired against each baseline by identical seed. The analysis reports:

- paired Wilcoxon signed-rank test
- rank-biserial correlation as paired effect size
- Benjamini–Hochberg adjusted p-value
- bootstrap 95% confidence interval for the mean

Do not report “significant improvement” unless the statistical output supports the statement.

## Reproduction commands

Quick:

```bash
python reproduce_all.py --quick
```

Full:

```bash
python reproduce_all.py
```

Tests:

```bash
pytest -q
```
