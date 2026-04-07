"""Simulation CLI for the synthetic MOMAB experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from momab_synthetic_core import (
    DEFAULT_OUTDIR,
    EXPERIMENT_FAMILY_NAME,
    MANIFEST_NAME,
    RUN_CACHE_NAME,
    SCHEMA_VERSION,
    TRAJECTORY_CACHE_NAME,
    PrecomputedBernoulliBandit,
    build_final_regret_settings,
    build_trajectory_settings,
    build_synthetic_instance,
    provenance_metadata,
    run_cache_key,
    run_width_guided_policy,
    run_pareto_ucb1,
    summarize_instance_metadata,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the simulation CLI."""
    parser = argparse.ArgumentParser(
        description="Run self-contained Pareto UCB1 vs width-guided certification simulations."
    )
    parser.add_argument("--T", type=int, default=10000, help="Time horizon for each run.")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of Monte Carlo runs for the final-regret study.")
    parser.add_argument(
        "--trajectory-runs",
        type=int,
        default=20,
        help="Number of Monte Carlo runs for the trajectory study.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Directory in which raw simulation caches and the manifest are stored.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate basic simulation arguments before any work starts."""
    if args.T < 20:
        raise ValueError("The synthetic family uses 20 arms, so --T must be at least 20.")
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive.")
    if args.trajectory_runs <= 0:
        raise ValueError("--trajectory-runs must be positive.")


def _draw_run_seeds(base_rng: np.random.Generator) -> Dict[str, int]:
    """Draw the coupled seeds used for one Monte Carlo run."""
    return {
        "reward": int(base_rng.integers(1, 10**9)),
        "pareto": int(base_rng.integers(1, 10**9)),
        "width": int(base_rng.integers(1, 10**9)),
    }


def _first_certification_round(run_result) -> float:
    """Return the first certification round, or NaN if certification never occurs."""
    if run_result.debug is None:
        return float("nan")

    certified_idx = np.flatnonzero(run_result.debug["objective_certified"])
    if certified_idx.size == 0:
        return float("nan")
    return float(certified_idx[0] + 1)


def _run_coupled_policies(
    mu: np.ndarray,
    *,
    t_horizon: int,
    run_seeds: Dict[str, int],
):
    """Run both policies on the same pre-sampled reward table."""
    n_arms, n_obj = mu.shape
    reward_rng = np.random.default_rng(run_seeds["reward"])
    reward_table = reward_rng.binomial(1, mu[:, None, :], size=(n_arms, t_horizon, n_obj)).astype(float)

    env_p = PrecomputedBernoulliBandit(mu, reward_table)
    env_w = PrecomputedBernoulliBandit(mu, reward_table)

    res_p = run_pareto_ucb1(env_p, t_horizon, seed=run_seeds["pareto"])
    res_w = run_width_guided_policy(
        env_w,
        t_horizon,
        seed=run_seeds["width"],
        return_debug=True,
    )
    return res_p, res_w, _first_certification_round(res_w)


def _simulate_final_runs(mu: np.ndarray, *, t_horizon: int, n_runs: int, seed: int) -> Dict[str, np.ndarray]:
    """Run the final-regret study for one setting and return raw per-run outputs."""
    base_rng = np.random.default_rng(seed)

    pareto_regret_final = np.zeros(n_runs, dtype=float)
    width_regret_final = np.zeros(n_runs, dtype=float)
    certified_flag = np.zeros(n_runs, dtype=float)
    certified_round = np.full(n_runs, np.nan, dtype=float)
    reward_seed = np.zeros(n_runs, dtype=np.int64)
    pareto_seed = np.zeros(n_runs, dtype=np.int64)
    width_seed = np.zeros(n_runs, dtype=np.int64)

    for run_idx in range(n_runs):
        # Each run shares one reward table across both policies so the comparison
        # reflects policy differences rather than different reward draws.
        run_seeds = _draw_run_seeds(base_rng)
        reward_seed[run_idx] = run_seeds["reward"]
        pareto_seed[run_idx] = run_seeds["pareto"]
        width_seed[run_idx] = run_seeds["width"]
        res_p, res_w, first_cert_round = _run_coupled_policies(
            mu,
            t_horizon=t_horizon,
            run_seeds=run_seeds,
        )

        pareto_regret_final[run_idx] = float(res_p.cum_regret[-1])
        width_regret_final[run_idx] = float(res_w.cum_regret[-1])
        if np.isfinite(first_cert_round):
            certified_flag[run_idx] = 1.0
            certified_round[run_idx] = first_cert_round

    return {
        "pareto_regret_final": pareto_regret_final,
        "width_regret_final": width_regret_final,
        "certified_flag": certified_flag,
        "certified_round": certified_round,
        "reward_seed": reward_seed,
        "pareto_seed": pareto_seed,
        "width_seed": width_seed,
    }


def _simulate_trajectory_runs(mu: np.ndarray, *, t_horizon: int, n_runs: int, seed: int) -> Dict[str, np.ndarray]:
    """Run the trajectory study for one setting and return raw path data."""
    base_rng = np.random.default_rng(seed)

    pareto_regret_paths = np.zeros((n_runs, t_horizon), dtype=float)
    width_regret_paths = np.zeros((n_runs, t_horizon), dtype=float)
    certified_round = np.full(n_runs, np.nan, dtype=float)
    reward_seed = np.zeros(n_runs, dtype=np.int64)
    pareto_seed = np.zeros(n_runs, dtype=np.int64)
    width_seed = np.zeros(n_runs, dtype=np.int64)

    for run_idx in range(n_runs):
        # The trajectory study uses the same coupled-reward design as the
        # final-regret study, but keeps the full regret paths.
        run_seeds = _draw_run_seeds(base_rng)
        reward_seed[run_idx] = run_seeds["reward"]
        pareto_seed[run_idx] = run_seeds["pareto"]
        width_seed[run_idx] = run_seeds["width"]
        res_p, res_w, first_cert_round = _run_coupled_policies(
            mu,
            t_horizon=t_horizon,
            run_seeds=run_seeds,
        )

        pareto_regret_paths[run_idx] = res_p.cum_regret
        width_regret_paths[run_idx] = res_w.cum_regret
        certified_round[run_idx] = first_cert_round

    return {
        "pareto_regret_paths": pareto_regret_paths,
        "width_regret_paths": width_regret_paths,
        "certified_round": certified_round,
        "reward_seed": reward_seed,
        "pareto_seed": pareto_seed,
        "width_seed": width_seed,
    }


def _save_manifest(out_dir: Path, manifest: Dict[str, object]) -> None:
    """Write the raw-simulation manifest to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    """Run all synthetic experiments and save the raw caches."""
    args = _parse_args()
    _validate_args(args)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_settings: List[Dict[str, object]] = []
    run_cache_arrays: Dict[str, np.ndarray] = {}
    for idx, exp in enumerate(build_final_regret_settings()):
        # Final-regret settings cover both the delta-variation and crowd-size
        # variation discussed in the paper.
        mu = build_synthetic_instance(delta=float(exp["delta"]), crowd_size=int(exp["crowd_size"]))
        setting_meta = {
            "index": idx,
            **exp,
            **summarize_instance_metadata(mu, args.T),
        }
        final_settings.append(setting_meta)

        raw = _simulate_final_runs(
            mu,
            t_horizon=args.T,
            n_runs=args.n_runs,
            seed=args.seed + 100 * idx,
        )
        for field, values in raw.items():
            run_cache_arrays[run_cache_key(idx, field)] = values

        print(
            setting_meta["label"],
            "c_pucb=",
            f"{setting_meta['c_pucb_exact']:.2f}",
            "g_dagger=",
            f"{setting_meta['g_dagger']:.2f}",
            flush=True,
        )

    trajectory_settings: List[Dict[str, object]] = []
    trajectory_cache_arrays: Dict[str, np.ndarray] = {}
    for idx, exp in enumerate(build_trajectory_settings()):
        # Trajectory settings are a small representative subset used only for
        # the dynamic mechanism figure.
        mu = build_synthetic_instance(delta=float(exp["delta"]), crowd_size=int(exp["crowd_size"]))
        setting_meta = {
            "index": idx,
            **exp,
            **summarize_instance_metadata(mu, args.T),
        }
        trajectory_settings.append(setting_meta)

        raw = _simulate_trajectory_runs(
            mu,
            t_horizon=args.T,
            n_runs=args.trajectory_runs,
            seed=args.seed + 1000 + 100 * idx,
        )
        for field, values in raw.items():
            trajectory_cache_arrays[run_cache_key(idx, field)] = values

    np.savez(out_dir / RUN_CACHE_NAME, **run_cache_arrays)
    np.savez(out_dir / TRAJECTORY_CACHE_NAME, **trajectory_cache_arrays)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_family": EXPERIMENT_FAMILY_NAME,
        "config": {
            "T": int(args.T),
            "n_runs": int(args.n_runs),
            "trajectory_runs": int(args.trajectory_runs),
            "seed": int(args.seed),
        },
        "files": {
            "run_cache": RUN_CACHE_NAME,
            "trajectory_cache": TRAJECTORY_CACHE_NAME,
        },
        "provenance": provenance_metadata(),
        "final_settings": final_settings,
        "trajectory_settings": trajectory_settings,
    }
    _save_manifest(out_dir, manifest)

    print(
        f"Saved raw simulation caches to {out_dir}. "
        f"Run plot_synthetic_experiments.py to regenerate tables and figures.",
        flush=True,
    )


if __name__ == "__main__":
    main()
