#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PUBLISH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARTITION="${PARTITION:-cpu-small}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"

T=1000000
N_RUNS=20
TRAJECTORY_RUNS=20
SEED=7
N_SHARDS=20
CPUS_PER_TASK=20
MEM=40G
TIME_LIMIT=12:00:00
SHARD_ASSIGNMENT=balanced
SHARD_ASSIGNMENT_SEED=20260504
OUTDIR="$PUBLISH_DIR/results/synthetic/main"

PLAN_ARGS=()
if [[ "$FORCE_CLEAN" == "1" ]]; then
  PLAN_ARGS+=(--force-clean)
fi

python3 "$PUBLISH_DIR/src/synthetic_benchmark.py" plan \
  --T "$T" \
  --n-runs "$N_RUNS" \
  --trajectory-runs "$TRAJECTORY_RUNS" \
  --seed "$SEED" \
  --outdir "$OUTDIR" \
  "${PLAN_ARGS[@]}"

N_JOBS="$(wc -l < "$OUTDIR/jobs.jsonl")"
if [[ "$N_JOBS" -le 0 ]]; then
  echo "No jobs were planned." >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Planned synthetic benchmark: $N_JOBS jobs in $OUTDIR."
  echo "DRY_RUN=1, so no Slurm job was submitted."
  exit 0
fi

mkdir -p "$OUTDIR/slurm_logs"

LAST_SHARD="$((N_SHARDS - 1))"
SBATCH_ARGS=(
  --partition="$PARTITION"
  --array="0-${LAST_SHARD}"
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$TIME_LIMIT"
  --output="$OUTDIR/slurm_logs/%x_%A_%a.out"
  --export=NONE,MOMAB_PUBLISH_DIR="$PUBLISH_DIR",MOMAB_OUTDIR="$OUTDIR",MOMAB_N_SHARDS="$N_SHARDS",MOMAB_ASSIGNMENT="$SHARD_ASSIGNMENT",MOMAB_ASSIGNMENT_SEED="$SHARD_ASSIGNMENT_SEED"
)
if [[ -n "$ACCOUNT" ]]; then
  SBATCH_ARGS=(--account="$ACCOUNT" "${SBATCH_ARGS[@]}")
fi
if [[ -n "$QOS" ]]; then
  SBATCH_ARGS=(--qos="$QOS" "${SBATCH_ARGS[@]}")
fi
sbatch "${SBATCH_ARGS[@]}" "$PUBLISH_DIR/slurm/synthetic_array.sbatch"

echo "Submitted synthetic benchmark: $N_JOBS jobs as $N_SHARDS shards."
echo "Fixed resources per shard: $CPUS_PER_TASK CPU cores, $MEM RAM, $TIME_LIMIT wall time."
echo "Aggregate after completion with: python3 src/synthetic_benchmark.py aggregate --outdir \"$OUTDIR\""
echo "Then report plots with: python3 src/synthetic_report.py --outdir \"$OUTDIR\" --plots-dir \"$OUTDIR/figures\""
