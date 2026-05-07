#!/bin/bash
set -euo pipefail

PUBLISH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${FORCE_CLEAN:-0}" != "1" ]]; then
  echo "This helper replaces bundled result outputs. Re-run with FORCE_CLEAN=1 to continue." >&2
  exit 2
fi

bash "$PUBLISH_DIR/scripts/submit_real_fixed.sh" spotify-main
bash "$PUBLISH_DIR/scripts/submit_real_fixed.sh" spotify-easy
bash "$PUBLISH_DIR/scripts/submit_real_fixed.sh" kuairec
bash "$PUBLISH_DIR/scripts/submit_synthetic_fixed.sh"

echo "Submitted all fixed-resource jobs. Run scripts/aggregate_all.sh after Slurm jobs complete."
