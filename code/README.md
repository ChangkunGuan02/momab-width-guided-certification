This folder contains the computational code for the width-guided certification study.

Dependencies:

- Python 3
- `numpy`
- `matplotlib`

Tested with Python `3.12.2`, NumPy `1.26.4`, and Matplotlib `3.10.0`.

Install the Python dependencies in your environment before running the scripts, for example:

```bash
pip install numpy matplotlib
```

File layout:

- `momab_synthetic_core.py`
  - shared definitions for the synthetic family, Pareto UCB1, the width-guided policy, and experiment metadata
- `run_synthetic_experiments.py`
  - simulation CLI
  - runs the Monte Carlo experiments
  - writes raw run-level caches plus a manifest under `../computational_results/momab_synthetic_experiments/` by default
- `plot_synthetic_experiments.py`
  - plotting/reporting CLI
  - reads the cached simulation outputs, validates that they match the current simulation code, and regenerates:
    - `derived_manifest.json`
    - `summary.json`
    - `trajectory_summary.json`
    - `results_table.tex`
    - PDF figures in `../plots/`

The synthetic instance family matches the paper notation: two Pareto-optimal arms fix the certification gap, while a variable number of dominated crowd arms have small Pareto gaps without entering the champion objective's top-two race.

This codebase is focused on the experiment family studied in the paper rather than a general benchmark harness. The instance families and representative trajectory settings are fixed in the code, and the warm start pulls each of the 20 arms once, so `--T` must be at least `20`.

CLI 1: run simulations

From the `code/` directory:

```bash
python3 run_synthetic_experiments.py
```

From the project root:

```bash
python3 code/run_synthetic_experiments.py
```

Arguments for the simulation CLI:

- `--T`: time horizon for each run; default `10000`
- `--n-runs`: number of Monte Carlo runs for the final-regret study; default `10`
- `--trajectory-runs`: number of Monte Carlo runs for the trajectory study; default `20`
- `--seed`: base random seed; default `7`
- `--outdir`: cache/output directory; default `../computational_results/momab_synthetic_experiments/`

Files written by the simulation CLI:

- `manifest.json`
- `run_cache.npz`
- `trajectory_cache.npz`

CLI 2: regenerate tables and figures from cached simulations

From the `code/` directory:

```bash
python3 plot_synthetic_experiments.py
```

From the project root:

```bash
python3 code/plot_synthetic_experiments.py
```

Arguments for the plotting CLI:

- `--outdir`: directory containing `manifest.json`, `run_cache.npz`, and `trajectory_cache.npz`
- `--plots-dir`: destination for PDF figures; default `../plots/`

Files written by the plotting/reporting CLI:

- `derived_manifest.json`
- `summary.json`
- `trajectory_summary.json`
- `results_table.tex`
- `comparison_plots.pdf` and `trajectory_plots.pdf` in `--plots-dir`

Workflow notes:

- The intended workflow is `simulation CLI -> plotting/reporting CLI`.
- The raw cache files are the primary cached outputs; the derived summaries and figures should be regenerated from them.
- The plotting CLI refuses to use stale caches if the simulation code has changed. If that happens, rerun `run_synthetic_experiments.py` first and then rerun `plot_synthetic_experiments.py`.
- `derived_manifest.json` records the plotting-side provenance for the regenerated outputs, including plotting code hashes and the Matplotlib version used to render the figures.
- If `--plots-dir` is omitted, the plotting CLI writes to `../plots/` only for the canonical cache directory. For any other `--outdir`, it defaults to `<outdir>/plots/` so exploratory runs do not overwrite the main figures.
