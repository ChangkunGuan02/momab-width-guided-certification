#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ZIP="${OUT_ZIP:-${ROOT_DIR}/../momab_computational_artifact.zip}"
FORCE="${FORCE:-0}"

python3 - "$ROOT_DIR" "$OUT_ZIP" "$FORCE" <<'PY'
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out_zip = Path(sys.argv[2]).resolve()
force = sys.argv[3] == "1"

excluded_names = {
    ".DS_Store",
}
excluded_suffixes = {
    ".pyc",
}
excluded_parts = {
    "__pycache__",
    "slurm_logs",
    ".mplconfig",
}

def include(path: Path) -> bool:
    rel = path.relative_to(root)
    if path == out_zip:
        return False
    if path.name in excluded_names:
        return False
    if path.suffix in excluded_suffixes:
        return False
    return not any(part in excluded_parts for part in rel.parts)

out_zip.parent.mkdir(parents=True, exist_ok=True)
if out_zip.exists() and not force:
    raise SystemExit(f"{out_zip} already exists. Set FORCE=1 to overwrite it.")

with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for path in sorted(root.rglob("*")):
        if path.is_file() and include(path):
            zf.write(path, Path("code_publish") / path.relative_to(root))

print(out_zip)
PY
