# Energy-Aware 6G WSN Reproducibility Framework

This repository provides a reproducible implementation of the computational framework described in the manuscript **“Energy-Aware 6G Wireless Sensor Networks Using Machine Learning-Based Routing Protocols.”** It is designed to address the major-revision requests concerning novelty, an explicit machine-learning workflow, mathematical clarity, algorithmic detail, broader comparison, statistical validation, 6G-oriented evaluation, and reproducibility.

The repository deliberately contains only two source folders (`src/` and `tests/`). Simulation outputs are generated at run time and are not committed.

## What is implemented

The code separates the current manuscript logic from the reviewer-strengthening extensions:

1. **First-order radio-energy model** for transmission, reception, aggregation, and sensing.
2. **Markov state model** for sleep/idle/RX/TX state transitions and one-step expected energy drain.
3. **Explicit supervised ML energy predictor** trained and validated on node-level telemetry. This resolves the concern that PSO alone is not a machine-learning model.
4. **Energy-aware clustering and scheduling** that considers residual energy, predicted drain, node centrality, link quality, and traffic.
5. **PSO-assisted multi-objective routing** in which PSO learns routing-policy weights rather than being incorrectly described as ML.
6. **6G-oriented QoS terms** (configurable latency, reliability, dense-device load, and edge-processing delay) used as simulation constraints/proxies rather than claimed as a formal 6G standard.
7. **Baselines and ablations**: LEACH, LEACH-C, greedy energy-aware routing, PSO-only, Markov+PSO, ML+PSO, and the full Markov+ML+PSO method.
8. **Repeated-seed statistical validation** with bootstrap confidence intervals, paired Wilcoxon tests, effect size, and false-discovery-rate correction.
9. **Reproducible artifacts**: per-run CSV files, aggregate summaries, pairwise statistics, ML validation metrics, convergence traces, and publication-ready figures.

## Important scientific alignment

The uploaded manuscript states a Markov-chain energy predictor, clustering-based scheduling, and PSO-assisted routing. The reviewer correctly notes that PSO is an optimization method rather than a machine-learning model. Therefore:

- `paper_core`-type components reproduce the manuscript’s Markov + clustering + PSO logic.
- `ml_pso` and `proposed` contain an **actual trained ML model**.
- The ML results must be reported in the revised manuscript only after the repository has been run and the generated outputs have been checked.
- External literature methods are not falsely presented as exact reproductions unless their original source code and parameterization are available. The in-repository baselines are transparent reference implementations.

## Repository layout

```text
6G-WSN-ML-PSO-Reproducibility/
├── README.md
├── CODE_AVAILABILITY.md
├── METHODS_AND_EQUATIONS.md
├── REPRODUCIBILITY.md
├── REVIEWER_RESPONSE_MAP.md
├── config.yaml
├── requirements.txt
├── run_experiments.py
├── reproduce_all.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── energy.py
│   ├── markov.py
│   ├── ml.py
│   ├── clustering.py
│   ├── pso.py
│   ├── routing.py
│   ├── protocols.py
│   ├── simulator.py
│   ├── metrics.py
│   ├── statistics.py
│   ├── plotting.py
│   └── experiment.py
└── tests/
    ├── test_energy.py
    ├── test_markov.py
    ├── test_pso.py
    ├── test_protocols.py
    └── test_reproducibility.py
```

## Manuscript-matched default parameters

The default `config.yaml` uses:

- 200 sensor nodes
- 1000 × 1000 m² network area
- 2 J initial node energy
- 100 m communication radius
- 4000-bit packets
- 1000 simulation rounds
- 30 PSO particles
- 100 maximum PSO iterations
- first-order radio-energy model

These are the values reported in the manuscript. Any departure is saved with the run metadata.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Quick validation

Run the unit tests:

```bash
pytest -q
```

Run a short smoke experiment:

```bash
python run_experiments.py --protocols leach leach_c proposed --seeds 7 11 --rounds 100
```

## Full reproducibility run

```bash
python reproduce_all.py
```

This executes all configured protocols over repeated seeds, writes results under a generated `results/` directory, performs statistical testing, and creates figures.

For a faster development check, `--quick` runs a representative LEACH/LEACH-C/proposed smoke workflow with a reduced node count, rounds, ML estimator count, and PSO budget:

```bash
python reproduce_all.py --quick
```

## Single protocol

```bash
python run_experiments.py --protocols proposed --seeds 7 11 19 23 29
```

Supported names:

```text
leach
leach_c
greedy
pso_only
markov_pso
ml_pso
proposed
```

## Primary outputs

After execution, the generated `results/` directory contains:

- `runs.csv` — one row per protocol/seed
- `round_metrics.csv` — longitudinal metrics
- `summary.csv` — mean, standard deviation, bootstrap 95% CI
- `pairwise_stats.csv` — proposed-vs-baseline paired tests and effect sizes
- `ml_validation.csv` — supervised-predictor validation metrics
- `pso_convergence.csv` — optimization convergence information
- PNG/PDF figures for network lifetime, residual energy, PDR, energy balance, and PSO convergence
- `run_manifest.json` — configuration, seeds, versions, and reproducibility metadata

## Reproducibility controls

- Explicit random seeds are propagated through NumPy and scikit-learn.
- Identical topology seeds are reused across protocols for paired comparison.
- Input configuration is saved with each run.
- Group-wise ML validation prevents the same sensor node from appearing in both training and validation sets.
- Statistical tests operate on paired seeds.
- Deterministic run manifests record package versions and platform information.

## Scope and limitations

This is a simulation-based framework. It does not claim hardware-level 6G validation. The QoS variables are configurable simulation proxies for latency, reliability, dense connectivity, and edge-processing effects. Real-world channel traces, device power measurements, mobility, hardware faults, and standardized 6G testbed validation should be added when such data are available.

No software license is included in this repository.
