"""Prepare a KuaiRec cohort-objective MO-MAB instance.

Each video is treated as an arm. Objectives are user cohorts derived from
KuaiRec user features. For a selected video, the j-th reward coordinate is a
normalized watch-ratio sample from users in cohort j.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


SOURCE_URL = "https://github.com/chongminggao/KuaiRec"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a KuaiRec cohort-objective MO-MAB instance.")
    parser.add_argument("--input-dir", required=True, help="KuaiRec root or data directory.")
    parser.add_argument("--outdir", required=True, help="Directory for the prepared instance.")
    parser.add_argument("--matrix", default="small_matrix.csv", help="Interaction matrix CSV name.")
    parser.add_argument("--user-features", default="user_features.csv", help="User feature CSV name.")
    parser.add_argument(
        "--cohort-cols",
        default="user_active_degree,gender,age_range",
        help="Comma-separated user feature columns used to define cohorts.",
    )
    parser.add_argument(
        "--cohort-mode",
        choices=["joint", "single"],
        default="joint",
        help="joint uses feature-value combinations; single uses each feature=value group as an objective.",
    )
    parser.add_argument("--max-objectives", type=int, default=30, help="Keep the largest cohorts up to this count.")
    parser.add_argument("--min-users-per-objective", type=int, default=20, help="Minimum users per retained cohort.")
    parser.add_argument("--top-k", type=int, default=300, help="Number of video arms to retain.")
    parser.add_argument(
        "--min-samples-per-objective",
        type=int,
        default=10,
        help="Minimum video interactions required for every retained video and objective.",
    )
    parser.add_argument("--watch-cap", type=float, default=5.0, help="Clip watch_ratio before normalizing.")
    parser.add_argument(
        "--rank-by",
        choices=["mean_sum", "interaction_count"],
        default="mean_sum",
        help="How to rank videos after filtering.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional interaction-row cap for smoke tests.")
    return parser.parse_args()


def _resolve_file(input_dir: Path, name: str) -> Path:
    candidates = [
        input_dir / name,
        input_dir / "data" / name,
        input_dir / "KuaiRec 2.0" / "data" / name,
        input_dir / "KuaiRec" / "data" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name!r} under {input_dir}. Tried: {candidates}")


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _clean_value(raw: Optional[str]) -> str:
    if raw is None or raw == "":
        return "unknown"
    return str(raw).strip().replace(" ", "_").replace("/", "_")


def _read_user_cohort_labels(
    path: Path,
    *,
    cohort_cols: List[str],
    cohort_mode: str,
    max_objectives: int,
    min_users_per_objective: int,
    allowed_user_ids: Optional[set] = None,
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, int]]:
    user_to_labels_all: Dict[str, List[str]] = {}
    counts: Counter = Counter()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "user_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a user_id column.")
        missing = [col for col in cohort_cols if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing cohort columns: {missing}. Available: {reader.fieldnames}")
        for row in reader:
            user_id = str(row["user_id"])
            if allowed_user_ids is not None and user_id not in allowed_user_ids:
                continue
            if cohort_mode == "joint":
                pieces = [f"{col}={_clean_value(row.get(col))}" for col in cohort_cols]
                labels = ["|".join(pieces)]
            else:
                labels = [f"{col}={_clean_value(row.get(col))}" for col in cohort_cols]
            user_to_labels_all[user_id] = labels
            counts.update(labels)

    candidates = [
        label
        for label, count in sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
        if count >= min_users_per_objective
    ]
    objective_names = candidates[: int(max_objectives)]
    if not objective_names:
        raise ValueError("No cohorts passed --min-users-per-objective.")
    objective_set = set(objective_names)
    user_to_labels = {
        user_id: [label for label in labels if label in objective_set]
        for user_id, labels in user_to_labels_all.items()
    }
    user_to_labels = {user_id: labels for user_id, labels in user_to_labels.items() if labels}
    return user_to_labels, objective_names, {label: int(counts[label]) for label in objective_names}


def _read_matrix_user_ids(path: Path, max_rows: Optional[int]) -> set:
    user_ids = set()
    rows_seen = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "user_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a user_id column.")
        for row in reader:
            if max_rows is not None and rows_seen >= max_rows:
                break
            rows_seen += 1
            user_ids.add(str(row["user_id"]))
    if not user_ids:
        raise ValueError(f"No user IDs found in {path}.")
    return user_ids


def _reward_from_watch_ratio(raw: str, watch_cap: float) -> float:
    try:
        val = float(raw)
        if not np.isfinite(val):
            val = 0.0
    except (TypeError, ValueError):
        val = 0.0
    return float(max(0.0, min(watch_cap, val)) / watch_cap)


def _aggregate_matrix(
    path: Path,
    *,
    user_to_labels: Dict[str, List[str]],
    objective_names: List[str],
    watch_cap: float,
    max_rows: Optional[int],
) -> Tuple[Dict[str, Dict[str, List[float]]], int, int]:
    objective_set = set(objective_names)
    values_by_video: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    rows_seen = 0
    rows_used = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"user_id", "video_id", "watch_ratio"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}.")
        for row in reader:
            if max_rows is not None and rows_seen >= max_rows:
                break
            rows_seen += 1
            user_id = str(row["user_id"])
            labels = user_to_labels.get(user_id, [])
            if not labels:
                continue
            video_id = str(row["video_id"])
            reward = _reward_from_watch_ratio(row.get("watch_ratio", "0"), watch_cap)
            for label in labels:
                if label in objective_set:
                    values_by_video[video_id][label].append(reward)
                    rows_used += 1
    return values_by_video, rows_seen, rows_used


def _select_videos(
    values_by_video: Dict[str, Dict[str, List[float]]],
    *,
    objective_names: List[str],
    top_k: int,
    min_samples_per_objective: int,
    rank_by: str,
) -> List[str]:
    candidates = []
    for video_id, by_objective in values_by_video.items():
        if all(len(by_objective.get(objective, [])) >= min_samples_per_objective for objective in objective_names):
            candidates.append(video_id)
    if not candidates:
        raise ValueError("No video passed --min-samples-per-objective.")

    def score(video_id: str) -> Tuple[float, int]:
        by_objective = values_by_video[video_id]
        total_count = sum(len(by_objective[objective]) for objective in objective_names)
        if rank_by == "interaction_count":
            return float(total_count), total_count
        mean_sum = sum(float(np.mean(by_objective[objective])) for objective in objective_names)
        return mean_sum, total_count

    return sorted(candidates, key=score, reverse=True)[:top_k]


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_cols = _split_csv(args.cohort_cols)
    if not cohort_cols:
        raise ValueError("--cohort-cols must contain at least one column.")
    if args.watch_cap <= 0:
        raise ValueError("--watch-cap must be positive.")

    matrix_path = _resolve_file(input_dir, args.matrix)
    features_path = _resolve_file(input_dir, args.user_features)
    matrix_user_ids = _read_matrix_user_ids(matrix_path, args.max_rows)
    user_to_labels, objective_names, cohort_user_counts = _read_user_cohort_labels(
        features_path,
        cohort_cols=cohort_cols,
        cohort_mode=args.cohort_mode,
        max_objectives=int(args.max_objectives),
        min_users_per_objective=int(args.min_users_per_objective),
        allowed_user_ids=matrix_user_ids,
    )
    values_by_video, rows_seen, rows_used = _aggregate_matrix(
        matrix_path,
        user_to_labels=user_to_labels,
        objective_names=objective_names,
        watch_cap=float(args.watch_cap),
        max_rows=args.max_rows,
    )
    selected = _select_videos(
        values_by_video,
        objective_names=objective_names,
        top_k=int(args.top_k),
        min_samples_per_objective=int(args.min_samples_per_objective),
        rank_by=args.rank_by,
    )

    objective_values = np.empty(len(selected), dtype=object)
    mu = np.zeros((len(selected), len(objective_names)), dtype=float)
    sample_counts = np.zeros((len(selected), len(objective_names)), dtype=np.int64)
    for arm_idx, video_id in enumerate(selected):
        per_objective = np.empty(len(objective_names), dtype=object)
        by_objective = values_by_video[video_id]
        for obj_idx, objective in enumerate(objective_names):
            values = np.asarray(by_objective[objective], dtype=float)
            per_objective[obj_idx] = values
            mu[arm_idx, obj_idx] = float(np.mean(values))
            sample_counts[arm_idx, obj_idx] = int(values.shape[0])
        objective_values[arm_idx] = per_objective

    np.savez(
        out_dir / "kuairec_cohort_instance.npz",
        mu=mu,
        objective_values=objective_values,
        arm_ids=np.array(selected, dtype=str),
        arm_names=np.array(selected, dtype=str),
        sample_counts=sample_counts,
        objective_names=np.array(objective_names, dtype=str),
    )

    metadata = {
        "source": "KuaiRec",
        "source_url": SOURCE_URL,
        "input_dir_name": input_dir.name,
        "matrix_file": matrix_path.name,
        "user_features_file": features_path.name,
        "cohort_cols": cohort_cols,
        "cohort_mode": args.cohort_mode,
        "objective_names": objective_names,
        "n_objectives": int(len(objective_names)),
        "cohort_user_counts": cohort_user_counts,
        "rows_seen": int(rows_seen),
        "matrix_users_seen": int(len(matrix_user_ids)),
        "rows_used_for_selected_objectives": int(rows_used),
        "n_videos_seen": int(len(values_by_video)),
        "n_videos_retained": int(len(selected)),
        "top_k": int(args.top_k),
        "min_samples_per_objective": int(args.min_samples_per_objective),
        "watch_cap": float(args.watch_cap),
        "rank_by": args.rank_by,
        "retained_video_ids": selected,
        "sample_counts_by_objective": sample_counts.tolist(),
        "mean_vectors": mu.tolist(),
    }
    with open(out_dir / "kuairec_cohort_instance_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Prepared {len(selected)} video arms with {len(objective_names)} cohort objectives in {out_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
