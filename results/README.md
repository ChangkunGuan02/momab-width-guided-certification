# Published Result Outputs

This folder contains the fixed-resource rerun outputs used for the manuscript
computational study.

## Fixed Resource Profile

Each final real-data and synthetic evaluation was rerun as a CPU-only Slurm
array job with:

- 20 array shards
- 20 CPU cores per shard
- 40 GB RAM per shard
- conservative 12 hour wall-clock limit per shard
- no GPU request

On the execution cluster, all completed final-evaluation shards finished within
6 minutes of wall-clock time, excluding scheduler queueing. Scheduler logs are
not bundled, because they can contain cluster-specific usernames and paths.

## Main Outputs

Real-data tables:

- `real/spotify_main/tables/table_6_1_main.tex`
- `real/spotify_easy/tables/table_6_1_main.tex`
- `real/kuairec/tables/table_6_1_main.tex`

Synthetic outputs:

- `synthetic/main/results_table.tex`
- `synthetic/main/summary.json`
- `synthetic/main/figures/comparison_plots.pdf`
- `synthetic/main/figures/trajectory_plots.pdf`

The real-data metadata uses paths relative to the artifact root, including
`data/` for prepared instances and `frozen_subsets/` for frozen subset files.
The synthetic manifest records the raw runner hash from the final evaluation
run. The report generator validates the numerical simulation core
(`synthetic_core.py`) before regenerating tables and figures, so later
packaging-only edits to the sharded runner do not force a full synthetic rerun.
