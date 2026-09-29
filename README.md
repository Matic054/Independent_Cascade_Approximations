# Independent Cascade Approximation Benchmarks

This repository contains implementations and benchmark code for analytical approximations of node activation probabilities under the **Independent Cascade (IC)** model.

The project focuses on comparing approximation families under a common experimental protocol, including cumulative, step-wise, cavity/message-passing, linear/path-based, and higher-order corrections.

## Main components

| File | Purpose |
|---|---|
| `GraphIC.py` | Compact IC graph representation, Monte Carlo reference simulation, and approximation methods |
| `ic_calibration.py` | Edge-probability models and calibration to target secondary activation |
| `synthetic_benchmark.py` | Final calibration/test benchmark on synthetic graph families |
| `real_world_benchmark.py` | Final calibration/test benchmark on real-world networks |
| `utilities.py` | Dataset loading and evaluation metrics |
| `edge_probabilities.py` | Auxiliary edge-probability utilities used by `utilities.py` |
| `benchmark_experiments.ipynb` | Reproducible entry point for running the final benchmarks |

The old exploratory experiment scripts are not required by the final benchmark pipeline.

## Implemented approximation methods

`GraphIC.py` includes, among others:

- **NM** (`Naive`)
- **DMP** (`dmp_est`)
- **second-order / triangle-aware DMP** (`dmp_est_r2`)
- **steady-state DMP** (`dmp_inf`)
- **SWE** (`swe`)
- **cavity SWE** (`swe_cavity`)
- **SWE-NO** (`swe_no`)
- **cavity SWE-NO** (`swe_no_cavity`)
- **additive SWE** and its cavity extension
- **SWE-HIB-C** (`swe_hib_cavity`)
- **ALE** (`ALE_heuristic`)
- **ALE2** (`ALE2`)
- **IPL** (`modified_ALE`)
- **IPL2** (`modified_ALE2`)
- **cavity ALE** (`cavity_ALE`)
- **cavity IPL** (`modified_ALE_cavity`)
- **SPM**
- **SP1M**
- **SSS**
- **SSS-Noself**
- **MIA**
- PageRank baseline

The public names in the implementation preserve some historical code names for compatibility with the experiment scripts. In particular:

- `Naive` = NM
- `modified_ALE` = IPL
- `modified_ALE2` = IPL2
- `dmp_inf` = steady-state DMP / cavity SSS

## Installation

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate ic-diffusion-benchmarks
```

### Option 2: pip

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository can also be installed in editable mode:

```bash
pip install -e ".[notebook]"
```

Python 3.10 or newer is required. The supplied Conda environment uses Python 3.12.

`GraphIC.py` can run without Numba, but Numba is included in the environments because it substantially improves some graph-preprocessing operations.

## Data

The raw real-world datasets are not redistributed with this repository.
Download them from their original/public sources using the links in
[`data/README.md`](data/README.md), then place the files in the `data/`
directory before running the real-world benchmark.

The current loader expects the filenames listed in `data/README.md`.

## Running the experiments

The easiest entry point is:

```bash
jupyter lab benchmark_experiments_clean.ipynb
```

The notebook contains two switches:

```python
RUN_SYNTHETIC = True
RUN_REAL_WORLD = True
```

Set either one to `False` when only one benchmark is needed.

### Synthetic benchmark

The default synthetic benchmark uses:

- Erdős-Rényi
- Barabási-Albert
- Watts-Strogatz
- Stochastic Block Model
- Holme-Kim

with:

- `n = 5000`
- target mean degrees `4, 10, 20`
- seed fractions `0.01, 0.05, 0.10, 0.20`
- target secondary activation `0.10, 0.35, 0.60`
- edge models:
  - constant
  - weighted cascade
  - heterogeneous Dirichlet
- 3 calibration instances per setting
- 5 held-out test instances per setting

Calibration and held-out evaluation use **independent graph realizations**.

### Real-world benchmark

The default real-world benchmark includes:

- NetHEPT
- WikiVote
- Enron
- Epinions
- Slashdot
- CondMat
- cit-HepTh

The graph topology is fixed for a dataset. Calibration and held-out instances vary the seed set and, for the heterogeneous Dirichlet condition, the stochastic edge-weight realization.

The Monte Carlo budgets are dataset-dependent and are defined in `real_world_benchmark.py`.

## Hyperparameter selection

For methods with a finite propagation horizon, the default search is

```text
T = 1, ..., 20
```

For SWE-NO and cavity SWE-NO, the continuation depth is also calibrated over

```text
L = 0, ..., 5
```

For each experimental setting:

1. every candidate parameter combination is evaluated on the calibration instances;
2. parameters are selected by mean calibration RMSE;
3. the selected parameters are frozen;
4. the method is evaluated unchanged on held-out instances.

This avoids selecting method parameters directly on the held-out benchmark data.

## Diffusion-strength calibration

Instead of fixing a single global edge-probability scale, experiments are calibrated to a target **secondary activation fraction**.

For a selected seed prior and an edge-probability shape, `ic_calibration.py` searches for a scalar multiplier

```text
p_e(alpha) = min(1, alpha * w_e)
```

using a small Monte Carlo pilot run.

The final Monte Carlo simulation used as the benchmark reference is separate from this pilot calibration.

## Output files

Both benchmark drivers write incremental CSV outputs.

Typical output directory:

```text
results/
├── synthetic_benchmark/
│   ├── graphs.csv
│   ├── instances.csv
│   ├── calibration_curves.csv
│   ├── selected_hyperparameters.csv
│   └── benchmark_results.csv
└── real_world_benchmark/
    ├── graphs.csv
    ├── instances.csv
    ├── calibration_curves.csv
    ├── selected_hyperparameters.csv
    └── benchmark_results.csv
```

The files contain:

- graph metadata;
- calibrated diffusion-instance metadata;
- Monte Carlo reference summaries;
- complete calibration curves;
- selected method hyperparameters;
- held-out RMSE, MAE, Pearson correlation, Spearman correlation, AUC, and runtime.

The benchmark drivers use incremental writes and support:

```python
resume=True
```

so interrupted long-running experiments can continue without recomputing completed cells.

## Runtime measurements

Per-method benchmark runtimes exclude:

- dataset loading;
- Monte Carlo reference generation;
- hyperparameter calibration;
- metric computation.

For second-order DMP, triangle-state construction is performed before the timed method execution so that the reported runtime measures repeated inference on a prepared graph rather than one-time topology preprocessing.

## Reproducibility

Randomness is controlled through deterministic seeds derived from a single `base_seed`.

The benchmark design keeps comparisons paired where appropriate:

- seed realizations are reused across edge models within the same graph replicate;
- edge-shape realizations are reused across seed fractions and target diffusion strengths where intended;
- calibration and test instances remain separate.

Monte Carlo estimates are used as high-precision **reference marginals**, not as exact ground truth.

## Minimal programmatic example

```python
import numpy as np

from GraphIC import ICGraph, dmp_est, dmp_est_r2, swe
from ic_calibration import make_base_probabilities

graph = ICGraph.from_edges(
    n=3,
    src=np.array([0, 1], dtype=np.int32),
    dst=np.array([1, 2], dtype=np.int32),
    prob=np.array([0.2, 0.3], dtype=np.float32),
)

p0 = np.array([1.0, 0.0, 0.0])

print(dmp_est(graph, p0, T=3))
print(dmp_est_r2(graph, p0, T=3))
print(swe(graph, p0, T=3))
```

## Repository scope

The final benchmark code is intentionally separated from earlier exploratory scripts. The repository is intended to contain the implementation required to reproduce the reported experiments without requiring the exploratory development history.

## Paper analysis

The final paper-facing summaries and figures can be reproduced with:

```bash
jupyter lab analysis/paper_analysis.ipynb
```

The notebook uses `analysis/paper_analysis.py` and writes its outputs to
`analysis/figures/` and `analysis/tables/`. It includes the overall benchmark
summaries, DMP/DMP2 paired comparisons, ALE/IPL/ALE2/IPL2 analyses, SWE and
cavity comparisons, diffusion-regime breakdowns, runtime-accuracy plots, and
selected-hyperparameter summaries. The large calibration-curve CSVs are not
needed for these paper-level analyses.

