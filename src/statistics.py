from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HIGHER_IS_BETTER = {
    "fnd_round": True,
    "hnd_round": True,
    "final_alive_nodes": True,
    "total_energy_consumed_j": False,
    "packet_delivery_ratio": True,
    "mean_latency_ms": False,
    "mean_hops": False,
    "mean_routing_success": True,
    "mean_energy_fairness": True,
    "delivered_bits_per_joule": True,
}


def bootstrap_ci(values, samples=2000, confidence=0.95, seed=12345):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = np.mean(rng.choice(x, size=len(x), replace=True))
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def rank_biserial_paired(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    denom = pos + neg
    return float((pos - neg) / denom) if denom else 0.0


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    prev = 1.0
    for k in range(n - 1, -1, -1):
        idx = order[k]
        rank = k + 1
        val = min(prev, p[idx] * n / rank)
        adjusted[idx] = val
        prev = val
    return np.clip(adjusted, 0.0, 1.0)


def create_summary_and_pairwise_stats(cfg, output_dir: str | Path):
    out = Path(output_dir)
    runs = pd.read_csv(out / "runs.csv")

    bs = int(cfg["statistics"]["bootstrap_samples"])
    conf = float(cfg["statistics"]["confidence"])

    summary_rows = []
    for protocol, g in runs.groupby("protocol"):
        for metric in HIGHER_IS_BETTER:
            vals = g[metric].astype(float).to_numpy()
            lo, hi = bootstrap_ci(vals, bs, conf, seed=2026)
            summary_rows.append({
                "protocol": protocol,
                "metric": metric,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            })
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)

    if "proposed" not in set(runs["protocol"]):
        pd.DataFrame().to_csv(out / "pairwise_stats.csv", index=False)
        return

    rows = []
    baselines = [p for p in sorted(runs["protocol"].unique()) if p != "proposed"]

    for baseline in baselines:
        for metric, higher_better in HIGHER_IS_BETTER.items():
            a = runs[runs["protocol"] == "proposed"][["seed", metric]]
            b = runs[runs["protocol"] == baseline][["seed", metric]]
            m = a.merge(b, on="seed", suffixes=("_proposed", "_baseline"))
            if len(m) < 2:
                continue

            x = m[f"{metric}_proposed"].astype(float).to_numpy()
            y = m[f"{metric}_baseline"].astype(float).to_numpy()
            try:
                stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                p = float(p)
                stat = float(stat)
            except ValueError:
                stat, p = 0.0, 1.0

            diff = x - y
            if not higher_better:
                diff = -diff

            rows.append({
                "baseline": baseline,
                "metric": metric,
                "n_pairs": len(m),
                "proposed_mean": float(np.mean(x)),
                "baseline_mean": float(np.mean(y)),
                "improvement_directional_mean": float(np.mean(diff)),
                "wilcoxon_stat": stat,
                "p_value": p,
                "rank_biserial_effect": rank_biserial_paired(x if higher_better else -x, y if higher_better else -y),
            })

    stats = pd.DataFrame(rows)
    if len(stats):
        stats["p_fdr_bh"] = bh_adjust(stats["p_value"].to_numpy())
    stats.to_csv(out / "pairwise_stats.csv", index=False)
