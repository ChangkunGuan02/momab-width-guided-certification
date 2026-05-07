#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PUBLISH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PUBLISH_DIR"

python3 src/real_benchmark.py aggregate --outdir results/real/spotify_main
python3 src/real_benchmark.py aggregate --outdir results/real/spotify_easy
python3 src/real_benchmark.py aggregate --outdir results/real/kuairec

python3 src/synthetic_benchmark.py aggregate --outdir results/synthetic/main
python3 src/synthetic_report.py --outdir results/synthetic/main --plots-dir results/synthetic/main/figures

echo "Aggregated all fixed-resource publish outputs under $PUBLISH_DIR/results."
