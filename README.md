# MOMAB Width-Guided Certification

This repository contains the computational code and cached outputs for the width-guided certification study in stochastic multi-objective bandits. It includes only:

- runnable experiment code in `code/`
- cached simulation outputs in `computational_results/`
- paper figures in `plots/`

It does not include the paper writeup or manuscript LaTeX sources. The only `.tex` file in the repository is a generated table snippet in `computational_results/`.

## Dependencies

- Python 3
- `numpy`
- `matplotlib`

Tested with:

- Python `3.12.2`
- NumPy `1.26.4`
- Matplotlib `3.10.0`

Install the Python dependencies in your environment before running the scripts, for example:

```bash
pip install numpy matplotlib
```

## Quick Start

From the repository root:

```bash
python3 code/run_synthetic_experiments.py
python3 code/plot_synthetic_experiments.py
```

The first command regenerates the raw simulation caches under `computational_results/momab_synthetic_experiments/`. The second command regenerates the derived summaries and paper figures from those caches.

## Repository Layout

- `code/`
  - experiment code and CLI documentation
- `computational_results/`
  - raw caches, manifests, JSON summaries, and the LaTeX table
- `plots/`
  - PDF figures

## Notes

- The raw cache files in `computational_results/momab_synthetic_experiments/` are the primary cached outputs.
- The plotting CLI validates the simulation-side code hashes before regenerating summaries or figures.
- The detailed CLI arguments and workflow notes are documented in `code/README.md`.
