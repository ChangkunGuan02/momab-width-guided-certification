#!/bin/bash
#SBATCH --job-name=momab-synth
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PUBLISH_DIR="${MOMAB_PUBLISH_DIR:?MOMAB_PUBLISH_DIR must be set by the submit script}"
OUTDIR="${MOMAB_OUTDIR:?MOMAB_OUTDIR must be set by the submit script}"
N_SHARDS="${MOMAB_N_SHARDS:-20}"
ASSIGNMENT="${MOMAB_ASSIGNMENT:-balanced}"
ASSIGNMENT_SEED="${MOMAB_ASSIGNMENT_SEED:-20260504}"
WORKERS="${SLURM_CPUS_PER_TASK:-20}"

cd "$PUBLISH_DIR"
python3 src/synthetic_benchmark.py run-shard \
  --outdir "$OUTDIR" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --n-shards "$N_SHARDS" \
  --workers "$WORKERS" \
  --assignment "$ASSIGNMENT" \
  --assignment-seed "$ASSIGNMENT_SEED"
