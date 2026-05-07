"""Sharded runner for the synthetic MOMAB family.

This script produces the same cache layout consumed by
``synthetic_report.py``, but it distributes individual Monte Carlo runs across
Slurm array shards.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from synthetic_core import (
    DEFAULT_OUTDIR,
    DERIVED_MANIFEST_NAME,
    EXPERIMENT_FAMILY_NAME,
    MANIFEST_NAME,
    RESULTS_TABLE_NAME,
    RUN_CACHE_NAME,
    SCHEMA_VERSION,
    SUMMARY_NAME,
    TRAJECTORY_CACHE_NAME,
    TRAJECTORY_SUMMARY_NAME,
    build_final_regret_settings,
    build_synthetic_instance,
    build_trajectory_settings,
    provenance_metadata,
    run_cache_key,
    run_pareto_ucb1,
    run_width_guided_policy,
    summarize_instance_metadata,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan, run, and aggregate sharded synthetic experiments.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--T", type=int, default=1_000_000)
    plan.add_argument("--n-runs", type=int, default=20)
    plan.add_argument("--trajectory-runs", type=int, default=20)
    plan.add_argument("--seed", type=int, default=7)
    plan.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    plan.add_argument(
        "--force-clean",
        action="store_true",
        help="Remove existing planned synthetic outputs in --outdir before writing a new plan.",
    )

    run = sub.add_parser("run-shard")
    run.add_argument("--outdir", type=str, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--n-shards", type=int, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--assignment", choices=["modulo", "balanced"], default="balanced")
    run.add_argument("--assignment-seed", type=int, default=20260504)

    agg = sub.add_parser("aggregate")
    agg.add_argument("--outdir", type=str, required=True)

    return parser.parse_args()


def _draw_run_seeds(base_seed: int, run_idx: int) -> Dict[str, int]:
    rng = np.random.default_rng(int(base_seed))
    seeds: Dict[str, int] = {}
    for _ in range(run_idx + 1):
        seeds = {
            "reward": int(rng.integers(1, 10**9)),
            "pareto": int(rng.integers(1, 10**9)),
            "width": int(rng.integers(1, 10**9)),
        }
    return seeds


def _first_certification_round(run_result) -> float:
    if run_result.debug is None:
        return float("nan")
    certified_idx = np.flatnonzero(run_result.debug["objective_certified"])
    if certified_idx.size == 0:
        return float("nan")
    return float(certified_idx[0] + 1)


def _setting_payloads(t_horizon: int) -> Dict[str, List[Dict[str, object]]]:
    final_settings: List[Dict[str, object]] = []
    for idx, exp in enumerate(build_final_regret_settings()):
        mu = build_synthetic_instance(delta=float(exp["delta"]), crowd_size=int(exp["crowd_size"]))
        final_settings.append({"index": idx, **exp, **summarize_instance_metadata(mu, t_horizon)})

    trajectory_settings: List[Dict[str, object]] = []
    for idx, exp in enumerate(build_trajectory_settings()):
        mu = build_synthetic_instance(delta=float(exp["delta"]), crowd_size=int(exp["crowd_size"]))
        trajectory_settings.append({"index": idx, **exp, **summarize_instance_metadata(mu, t_horizon)})

    return {"final_settings": final_settings, "trajectory_settings": trajectory_settings}


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _clean_synthetic_outputs(out_dir: Path) -> None:
    """Remove generated files that would otherwise make a new plan reuse stale runs."""
    for name in [
        "jobs.jsonl",
        MANIFEST_NAME,
        RUN_CACHE_NAME,
        TRAJECTORY_CACHE_NAME,
        SUMMARY_NAME,
        TRAJECTORY_SUMMARY_NAME,
        RESULTS_TABLE_NAME,
        DERIVED_MANIFEST_NAME,
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()
    for directory_name in ["job_outputs", "figures"]:
        directory = out_dir / directory_name
        if directory.exists():
            for path in directory.glob("*"):
                if path.is_file():
                    path.unlink()


def _synthetic_outputs_exist(out_dir: Path) -> List[str]:
    """Return generated files that can make a new synthetic plan ambiguous."""
    generated = [
        "jobs.jsonl",
        MANIFEST_NAME,
        RUN_CACHE_NAME,
        TRAJECTORY_CACHE_NAME,
        SUMMARY_NAME,
        TRAJECTORY_SUMMARY_NAME,
        RESULTS_TABLE_NAME,
        DERIVED_MANIFEST_NAME,
    ]
    existing = [name for name in generated if (out_dir / name).exists()]
    job_outputs = out_dir / "job_outputs"
    if job_outputs.exists() and any(job_outputs.iterdir()):
        existing.append("job_outputs/*")
    figures = out_dir / "figures"
    if figures.exists() and any(figures.iterdir()):
        existing.append("figures/*")
    return existing


def _plan_synthetic_benchmark(args: argparse.Namespace) -> None:
    if args.T < 20:
        raise ValueError("The synthetic family uses 20 arms, so --T must be at least 20.")
    if args.n_runs <= 0 or args.trajectory_runs <= 0:
        raise ValueError("Run counts must be positive.")

    out_dir = Path(args.outdir)
    existing = _synthetic_outputs_exist(out_dir)
    if existing and not args.force_clean:
        raise RuntimeError(
            f"Refusing to overwrite existing synthetic outputs in {out_dir}: {existing[:10]}. "
            "Use --force-clean to remove generated outputs before planning a new run."
        )
    if existing and args.force_clean:
        _clean_synthetic_outputs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job_outputs").mkdir(exist_ok=True)

    settings = _setting_payloads(int(args.T))
    jobs: List[Dict[str, object]] = []
    for setting in settings["final_settings"]:
        idx = int(setting["index"])
        for run_idx in range(int(args.n_runs)):
            jobs.append(
                {
                    "job_index": len(jobs),
                    "job_type": "final",
                    "setting_index": idx,
                    "run_idx": run_idx,
                    "delta": float(setting["delta"]),
                    "crowd_size": int(setting["crowd_size"]),
                    "base_seed": int(args.seed) + 100 * idx,
                }
            )
    for setting in settings["trajectory_settings"]:
        idx = int(setting["index"])
        for run_idx in range(int(args.trajectory_runs)):
            jobs.append(
                {
                    "job_index": len(jobs),
                    "job_type": "trajectory",
                    "setting_index": idx,
                    "run_idx": run_idx,
                    "delta": float(setting["delta"]),
                    "crowd_size": int(setting["crowd_size"]),
                    "base_seed": int(args.seed) + 1000 + 100 * idx,
                }
            )

    with open(out_dir / "jobs.jsonl", "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": EXPERIMENT_FAMILY_NAME,
        "config": {
            "T": int(args.T),
            "n_runs": int(args.n_runs),
            "trajectory_runs": int(args.trajectory_runs),
            "seed": int(args.seed),
            "runner": "synthetic_benchmark.py",
        },
        "files": {
            "run_cache": RUN_CACHE_NAME,
            "trajectory_cache": TRAJECTORY_CACHE_NAME,
        },
        "provenance": provenance_metadata(),
        **settings,
        "n_jobs": len(jobs),
    }
    _write_json(out_dir / MANIFEST_NAME, manifest)
    print(f"Planned {len(jobs)} synthetic jobs in {out_dir}.", flush=True)


def _load_jobs(out_dir: Path) -> List[Dict[str, object]]:
    with open(out_dir / "jobs.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _select_jobs_for_shard(
    jobs: Sequence[Dict[str, object]],
    *,
    shard_index: int,
    n_shards: int,
    assignment: str,
    assignment_seed: int,
) -> List[Dict[str, object]]:
    if assignment == "modulo":
        return [job for pos, job in enumerate(jobs) if pos % n_shards == shard_index]
    rng = np.random.default_rng(int(assignment_seed))
    order = np.arange(len(jobs))
    rng.shuffle(order)
    buckets: List[List[int]] = [[] for _ in range(n_shards)]
    for pos, job_idx in enumerate(order):
        buckets[pos % n_shards].append(int(job_idx))
    return [jobs[i] for i in sorted(buckets[shard_index])]


def _run_synthetic_job(job: Dict[str, object], out_dir: str, t_horizon: int) -> Dict[str, object]:
    out_path = Path(out_dir) / "job_outputs" / f"job_{int(job['job_index']):05d}.json"
    npz_path = Path(out_dir) / "job_outputs" / f"job_{int(job['job_index']):05d}.npz"
    if out_path.exists():
        return {"job_index": int(job["job_index"]), "status": "skipped"}

    mu = build_synthetic_instance(delta=float(job["delta"]), crowd_size=int(job["crowd_size"]))
    seeds = _draw_run_seeds(int(job["base_seed"]), int(job["run_idx"]))
    reward_rng = np.random.default_rng(seeds["reward"])
    reward_table = reward_rng.binomial(1, mu[:, None, :], size=(mu.shape[0], t_horizon, mu.shape[1])).astype(float)

    env_p = PrecomputedBernoulliBanditForShard(mu, reward_table)
    env_w = PrecomputedBernoulliBanditForShard(mu, reward_table)
    res_p = run_pareto_ucb1(env_p, t_horizon, seed=seeds["pareto"])
    res_w = run_width_guided_policy(env_w, t_horizon, seed=seeds["width"], return_debug=True)
    cert_round = _first_certification_round(res_w)

    payload = {
        **job,
        "T": int(t_horizon),
        "reward_seed": seeds["reward"],
        "pareto_seed": seeds["pareto"],
        "width_seed": seeds["width"],
        "pareto_regret_final": float(res_p.cum_regret[-1]),
        "width_regret_final": float(res_w.cum_regret[-1]),
        "certified_flag": float(np.isfinite(cert_round)),
        "certified_round": None if not np.isfinite(cert_round) else float(cert_round),
    }
    if str(job["job_type"]) == "trajectory":
        tmp_npz = npz_path.with_suffix(".npz.tmp")
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(
                f,
                pareto_regret_path=res_p.cum_regret,
                width_regret_path=res_w.cum_regret,
            )
        os.replace(tmp_npz, npz_path)
        payload["trajectory_cache"] = npz_path.name

    _write_json(out_path, payload)
    return {"job_index": int(job["job_index"]), "status": "done"}


class PrecomputedBernoulliBanditForShard:
    """Minimal local environment to avoid importing private runner helpers."""

    def __init__(self, mu: np.ndarray, reward_table: np.ndarray) -> None:
        self.mu = np.asarray(mu, dtype=float)
        self.reward_table = np.asarray(reward_table, dtype=float)
        self.pull_counts = np.zeros(self.mu.shape[0], dtype=int)

    @property
    def k(self) -> int:
        return int(self.mu.shape[0])

    @property
    def d(self) -> int:
        return int(self.mu.shape[1])

    def pull(self, arm: int) -> np.ndarray:
        arm = int(arm)
        idx = int(self.pull_counts[arm])
        reward = self.reward_table[arm, idx].copy()
        self.pull_counts[arm] += 1
        return reward


def _run_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    with open(out_dir / MANIFEST_NAME, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    jobs = _load_jobs(out_dir)
    assigned = _select_jobs_for_shard(
        jobs,
        shard_index=int(args.shard_index),
        n_shards=int(args.n_shards),
        assignment=str(args.assignment),
        assignment_seed=int(args.assignment_seed),
    )
    print(f"Shard {args.shard_index}/{args.n_shards}: {len(assigned)} jobs, workers={args.workers}.", flush=True)
    if not assigned:
        return

    t_horizon = int(manifest["config"]["T"])
    if int(args.workers) <= 1:
        for job in assigned:
            result = _run_synthetic_job(job, str(out_dir), t_horizon)
            print(result, flush=True)
        return

    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = [pool.submit(_run_synthetic_job, job, str(out_dir), t_horizon) for job in assigned]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)


def _aggregate_synthetic_results(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    with open(out_dir / MANIFEST_NAME, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    jobs = _load_jobs(out_dir)
    outputs: Dict[int, Dict[str, object]] = {}
    for path in sorted((out_dir / "job_outputs").glob("job_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        outputs[int(item["job_index"])] = item

    missing = [int(job["job_index"]) for job in jobs if int(job["job_index"]) not in outputs]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} synthetic jobs; first missing: {missing[:10]}")

    run_arrays: Dict[str, np.ndarray] = {}
    for setting in manifest["final_settings"]:
        idx = int(setting["index"])
        rows = sorted(
            [item for item in outputs.values() if item["job_type"] == "final" and int(item["setting_index"]) == idx],
            key=lambda item: int(item["run_idx"]),
        )
        run_arrays[run_cache_key(idx, "pareto_regret_final")] = np.array(
            [float(item["pareto_regret_final"]) for item in rows], dtype=float
        )
        run_arrays[run_cache_key(idx, "width_regret_final")] = np.array(
            [float(item["width_regret_final"]) for item in rows], dtype=float
        )
        run_arrays[run_cache_key(idx, "certified_flag")] = np.array(
            [float(item["certified_flag"]) for item in rows], dtype=float
        )
        run_arrays[run_cache_key(idx, "certified_round")] = np.array(
            [
                float("nan") if item["certified_round"] is None else float(item["certified_round"])
                for item in rows
            ],
            dtype=float,
        )
        for field in ["reward_seed", "pareto_seed", "width_seed"]:
            run_arrays[run_cache_key(idx, field)] = np.array([int(item[field]) for item in rows], dtype=np.int64)

    trajectory_arrays: Dict[str, np.ndarray] = {}
    for setting in manifest["trajectory_settings"]:
        idx = int(setting["index"])
        rows = sorted(
            [item for item in outputs.values() if item["job_type"] == "trajectory" and int(item["setting_index"]) == idx],
            key=lambda item: int(item["run_idx"]),
        )
        p_paths = []
        w_paths = []
        for item in rows:
            with np.load(out_dir / "job_outputs" / str(item["trajectory_cache"])) as npz:
                p_paths.append(np.asarray(npz["pareto_regret_path"], dtype=float))
                w_paths.append(np.asarray(npz["width_regret_path"], dtype=float))
        trajectory_arrays[run_cache_key(idx, "pareto_regret_paths")] = np.stack(p_paths, axis=0)
        trajectory_arrays[run_cache_key(idx, "width_regret_paths")] = np.stack(w_paths, axis=0)
        trajectory_arrays[run_cache_key(idx, "certified_round")] = np.array(
            [
                float("nan") if item["certified_round"] is None else float(item["certified_round"])
                for item in rows
            ],
            dtype=float,
        )
        for field in ["reward_seed", "pareto_seed", "width_seed"]:
            trajectory_arrays[run_cache_key(idx, field)] = np.array([int(item[field]) for item in rows], dtype=np.int64)

    np.savez(out_dir / RUN_CACHE_NAME, **run_arrays)
    np.savez(out_dir / TRAJECTORY_CACHE_NAME, **trajectory_arrays)
    print(f"Aggregated {len(outputs)} / {len(jobs)} jobs into {out_dir}.", flush=True)


def main() -> None:
    args = _parse_args()
    if args.cmd == "plan":
        _plan_synthetic_benchmark(args)
    elif args.cmd == "run-shard":
        _run_shard(args)
    elif args.cmd == "aggregate":
        _aggregate_synthetic_results(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
