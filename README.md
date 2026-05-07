# MOMAB Computational Study Artifact

This directory contains the supplementary computational artifact for the study.
The bundled outputs reproduce the manuscript tables and figures from the
prepared benchmark instances included under `data/`.

## Environment

The final rerun reported in the manuscript used:

- Python 3.9.21
- NumPy 2.0.2
- Matplotlib 3.9.4
- CPU-only Slurm jobs, with no GPU request

Install the Python dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The command-line entry points are script-oriented and should be run from this
directory as `python3 src/<script>.py ...`.
Prepared real-data `.npz` instances use NumPy object arrays for variable-length
empirical reward samples, so replacement instance files should be treated as
trusted inputs.

## Files

- `src/real_benchmark.py`: real-data Spotify and KuaiRec benchmark runner.
- `src/synthetic_benchmark.py`: sharded synthetic benchmark runner.
- `src/synthetic_core.py`: shared policy, instance, and metric utilities.
- `src/synthetic_report.py`: synthetic table and figure generator.
- `src/prepare_spotify_instance.py`: raw Spotify-to-instance preparation script.
- `src/prepare_kuairec_cohort_instance.py`: raw KuaiRec-to-instance preparation script.
- `data/`: prepared benchmark instances used by the manuscript experiments.
- `data/prepared_instances_manifest.json`: checksums and array schemas for the prepared instances.
- `data/LICENSES.md`: attribution and license notes for the prepared data files.
- `frozen_subsets/`: held-out subset definitions used for the manuscript rows.
- `results/`: bundled final outputs used in the manuscript.
- `scripts/`: Slurm run, aggregation, and zip-building helpers.
- `slurm/`: Slurm array task entry points.
- `LICENSE`: code license and third-party data notice.
- `CITATION.cff`: citation metadata.

## Inspect Bundled Outputs

The main bundled real-data outputs are:

- `results/real/spotify_main/tables/table_6_1_main.tex`
- `results/real/spotify_easy/tables/table_6_1_main.tex`
- `results/real/kuairec/tables/table_6_1_main.tex`

The main bundled synthetic outputs are:

- `results/synthetic/main/results_table.tex`
- `results/synthetic/main/summary.json`
- `results/synthetic/main/figures/comparison_plots.pdf`
- `results/synthetic/main/figures/trajectory_plots.pdf`

The final synthetic PDFs and per-job trajectory outputs are included. The large
aggregated trajectory cache (`results/synthetic/main/trajectory_cache.npz`) is
omitted from GitHub; it can be included in the supplementary zip when present
locally or regenerated from the bundled job outputs.

## Local Smoke Test

The following small synthetic run checks the main code path without Slurm:

```bash
python3 src/synthetic_benchmark.py plan --T 20 --n-runs 1 --trajectory-runs 1 \
  --outdir /tmp/momab_synth_smoke --force-clean
python3 src/synthetic_benchmark.py run-shard --outdir /tmp/momab_synth_smoke \
  --shard-index 0 --n-shards 1 --workers 1
python3 src/synthetic_benchmark.py aggregate --outdir /tmp/momab_synth_smoke
python3 src/synthetic_report.py --outdir /tmp/momab_synth_smoke \
  --plots-dir /tmp/momab_synth_smoke/figures
```

For a real-data smoke test, use a temporary output directory so the bundled
result files are not touched:

```bash
python3 src/real_benchmark.py plan \
  --instance data/spotify_genre_instance_d6.npz \
  --dataset-name spotify_d6 \
  --outdir /tmp/momab_real_smoke \
  --T 20 \
  --n-runs 1 \
  --seed 20260704 \
  --k-values 10 \
  --subset-types easy_separated \
  --selection-mode geometry \
  --policies width_guided_c0.02,pareto_ucb1 \
  --frozen-subsets frozen_subsets/spotify_easy_selected_subsets.json \
  --force-clean
python3 src/real_benchmark.py run-local --outdir /tmp/momab_real_smoke \
  --workers 1 --limit 2
python3 src/real_benchmark.py aggregate --outdir /tmp/momab_real_smoke \
  --allow-missing
```

The `--allow-missing` flag is only for partial smoke diagnostics. Final tables
are generated only after all planned jobs have finished.

## Full Slurm Rerun

Full reproduction of the manuscript-facing runs requires a Slurm cluster with
`sbatch`. Each final run uses the fixed per-shard profile below:

- 20 Slurm array shards
- 20 CPU cores per shard
- 40 GB RAM per shard
- 12 hour wall-clock limit per shard
- CPU only

The submit scripts refuse to overwrite bundled result outputs by default. To
replace the bundled outputs in a working copy, set `FORCE_CLEAN=1` explicitly:

```bash
FORCE_CLEAN=1 bash scripts/submit_real_fixed.sh spotify-main
FORCE_CLEAN=1 bash scripts/submit_real_fixed.sh spotify-easy
FORCE_CLEAN=1 bash scripts/submit_real_fixed.sh kuairec
FORCE_CLEAN=1 bash scripts/submit_synthetic_fixed.sh
```

After all jobs finish:

```bash
bash scripts/aggregate_all.sh
```

The account, partition, and QoS can be set through `ACCOUNT`, `PARTITION`, and
`QOS`. Without `FORCE_CLEAN=1`, the planner stops if generated outputs already
exist.

## Build Supplementary Zip

To create the supplementary zip from this directory:

```bash
bash scripts/make_supplement_zip.sh
```

If the output zip already exists, use `FORCE=1 bash scripts/make_supplement_zip.sh`
to replace it. The zip helper excludes scheduler logs, Python bytecode,
Matplotlib caches, and local runtime caches that should not be uploaded. It
keeps the scientific result caches needed to regenerate the manuscript figures.
