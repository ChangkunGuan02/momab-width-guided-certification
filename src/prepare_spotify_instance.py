"""Prepare a real-data MO-MAB instance from Spotify track audio features.

Each genre is treated as an arm. Each track in that genre is treated as one
stochastic reward vector. The default objectives are popularity and normalized
audio features, giving a moderate-dimensional real-data benchmark without
adding synthetic objectives.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


SOURCE_URL = "https://huggingface.co/datasets/sfiore/spotify-tracks-dataset"
DEFAULT_OBJECTIVES = [
    "popularity",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Spotify genre-level MO-MAB instance.")
    parser.add_argument("--input", required=True, help="Path to the Spotify tracks CSV.")
    parser.add_argument("--outdir", required=True, help="Directory for the prepared instance.")
    parser.add_argument("--top-k", type=int, default=125, help="Maximum number of genre arms to retain.")
    parser.add_argument("--min-tracks", type=int, default=200, help="Minimum tracks per retained genre.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument(
        "--rank-by",
        choices=["mean_sum", "track_count"],
        default="track_count",
        help="How to rank genres after filtering.",
    )
    parser.add_argument(
        "--objectives",
        default=",".join(DEFAULT_OBJECTIVES),
        help=(
            "Comma-separated objectives among popularity, danceability, energy, loudness, "
            "speechiness, acousticness, instrumentalness, liveness, valence, tempo."
        ),
    )
    parser.add_argument(
        "--genres",
        default=None,
        help="Optional comma-separated genre names to retain instead of ranking by --rank-by.",
    )
    parser.add_argument(
        "--keep-duplicate-track-ids",
        action="store_true",
        help="Keep duplicate track_id entries within a genre. By default duplicates are removed within genre.",
    )
    return parser.parse_args()


def _float_value(raw: Optional[str]) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if np.isfinite(val) else None


def _clip_unit(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _scale_popularity(raw: Optional[str]) -> Optional[float]:
    val = _float_value(raw)
    if val is None:
        return None
    return _clip_unit(val / 100.0)


def _scale_unit(raw: Optional[str]) -> Optional[float]:
    val = _float_value(raw)
    if val is None:
        return None
    return _clip_unit(val)


def _scale_loudness(raw: Optional[str]) -> Optional[float]:
    val = _float_value(raw)
    if val is None:
        return None
    clipped = max(-60.0, min(0.0, val))
    return _clip_unit((clipped + 60.0) / 60.0)


def _scale_tempo(raw: Optional[str]) -> Optional[float]:
    val = _float_value(raw)
    if val is None:
        return None
    return _clip_unit(val / 250.0)


def _normalizers() -> Dict[str, Callable[[Optional[str]], Optional[float]]]:
    return {
        "popularity": _scale_popularity,
        "danceability": _scale_unit,
        "energy": _scale_unit,
        "loudness": _scale_loudness,
        "speechiness": _scale_unit,
        "acousticness": _scale_unit,
        "instrumentalness": _scale_unit,
        "liveness": _scale_unit,
        "valence": _scale_unit,
        "tempo": _scale_tempo,
    }


def _parse_objectives(raw: str) -> List[str]:
    objectives = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not objectives:
        raise ValueError("At least one objective is required.")
    known = set(_normalizers())
    unknown = [x for x in objectives if x not in known]
    if unknown:
        raise ValueError(f"Unknown objectives {unknown}. Known objectives: {sorted(known)}.")
    return objectives


def _reward_from_row(row: Dict[str, str], objectives: Iterable[str]) -> Optional[List[float]]:
    normalizers = _normalizers()
    reward: List[float] = []
    for objective in objectives:
        val = normalizers[objective](row.get(objective))
        if val is None:
            return None
        reward.append(val)
    return reward


def _aggregate_tracks_by_genre(
    path: Path,
    *,
    objectives: List[str],
    max_rows: Optional[int],
    deduplicate_within_genre: bool,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, int], int, int]:
    values_by_genre: Dict[str, List[List[float]]] = defaultdict(list)
    seen_track_ids: Dict[str, set] = defaultdict(set)
    duplicate_rows = 0
    rows_seen = 0
    rows_used = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = set(objectives) | {"track_genre"}
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Input CSV is missing required columns: {missing_columns}.")

        for row in reader:
            if max_rows is not None and rows_seen >= max_rows:
                break
            rows_seen += 1
            genre = (row.get("track_genre") or "").strip()
            if not genre:
                continue
            if deduplicate_within_genre:
                track_id = (row.get("track_id") or "").strip()
                if track_id:
                    if track_id in seen_track_ids[genre]:
                        duplicate_rows += 1
                        continue
                    seen_track_ids[genre].add(track_id)
            reward = _reward_from_row(row, objectives)
            if reward is None:
                continue
            values_by_genre[genre].append(reward)
            rows_used += 1
    track_counts = {genre: len(values) for genre, values in values_by_genre.items()}
    return values_by_genre, track_counts, rows_seen, duplicate_rows


def _select_arms(
    values_by_genre: Dict[str, List[List[float]]],
    *,
    top_k: int,
    min_tracks: int,
    rank_by: str,
) -> List[str]:
    candidates = [genre for genre, vals in values_by_genre.items() if len(vals) >= min_tracks]
    if not candidates:
        raise ValueError("No genre passed --min-tracks. Lower the threshold or use more rows.")

    def score(genre: str) -> Tuple[float, int, str]:
        arr = np.asarray(values_by_genre[genre], dtype=float)
        if rank_by == "track_count":
            return float(arr.shape[0]), arr.shape[0], genre
        return float(np.sum(np.mean(arr, axis=0))), arr.shape[0], genre

    return sorted(candidates, key=score, reverse=True)[:top_k]


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    objectives = _parse_objectives(args.objectives)
    if args.top_k <= 1:
        raise ValueError("--top-k must be at least 2.")
    if args.min_tracks <= 0:
        raise ValueError("--min-tracks must be positive.")

    values_by_genre, track_counts_by_genre, rows_seen, duplicate_rows = _aggregate_tracks_by_genre(
        input_path,
        objectives=objectives,
        max_rows=args.max_rows,
        deduplicate_within_genre=not args.keep_duplicate_track_ids,
    )
    if args.genres:
        selected = [x.strip() for x in args.genres.split(",") if x.strip()]
        missing = [x for x in selected if x not in values_by_genre]
        too_small = [x for x in selected if x in values_by_genre and len(values_by_genre[x]) < args.min_tracks]
        if missing:
            raise ValueError(f"Requested genres were not found: {missing}.")
        if too_small:
            raise ValueError(f"Requested genres below --min-tracks: {too_small}.")
    else:
        selected = _select_arms(
            values_by_genre,
            top_k=int(args.top_k),
            min_tracks=int(args.min_tracks),
            rank_by=args.rank_by,
        )

    reward_values = np.empty(len(selected), dtype=object)
    mu = np.zeros((len(selected), len(objectives)), dtype=float)
    track_counts = np.zeros(len(selected), dtype=np.int64)
    for idx, genre in enumerate(selected):
        arr = np.asarray(values_by_genre[genre], dtype=float)
        reward_values[idx] = arr
        mu[idx] = np.mean(arr, axis=0)
        track_counts[idx] = arr.shape[0]

    np.savez(
        out_dir / "spotify_genre_instance.npz",
        mu=mu,
        reward_values=reward_values,
        arm_ids=np.array(selected, dtype=str),
        arm_names=np.array(selected, dtype=str),
        track_counts=track_counts,
        objective_names=np.array(objectives, dtype=str),
    )

    metadata = {
        "source": "Spotify tracks dataset",
        "source_url": SOURCE_URL,
        "input_file": input_path.name,
        "rows_seen": int(rows_seen),
        "duplicate_rows_removed_within_genre": int(duplicate_rows),
        "n_genres_seen": int(len(values_by_genre)),
        "n_genres_retained": int(len(selected)),
        "top_k": int(args.top_k),
        "min_tracks": int(args.min_tracks),
        "rank_by": args.rank_by,
        "genres_argument": args.genres,
        "objectives": objectives,
        "normalization": {
            "popularity": "clip(value / 100, 0, 1)",
            "unit_interval_features": "clip(value, 0, 1)",
            "loudness": "clip((clip(value, -60, 0) + 60) / 60, 0, 1)",
            "tempo": "clip(value / 250, 0, 1)",
        },
        "retained_genres": selected,
        "retained_track_counts": [int(x) for x in track_counts.tolist()],
        "all_genre_track_counts": {genre: int(count) for genre, count in sorted(track_counts_by_genre.items())},
        "mean_vectors": mu.tolist(),
    }
    with open(out_dir / "spotify_genre_instance_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Prepared {len(selected)} genre arms with {len(objectives)} objectives in {out_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
