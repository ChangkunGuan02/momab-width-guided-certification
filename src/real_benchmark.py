"""Real-data outcome benchmark for the supplementary artifact.

This runner selects held-out real-data subsets by empirical mean geometry,
runs the paper-facing methods, and reports the outcome metrics used in the
computational study.
The simulation path streams rewards and keeps compact counters so long-horizon
benchmarks do not materialize K x T x d reward tensors.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from synthetic_core import (
    compute_exact_pucb_coefficient,
    objective_winner_gaps,
    pareto_arm_regrets,
    pareto_nondominated_indices,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES = [
    "width_guided",
    "pareto_ucb1",
    "annealing_pareto",
    "scalarized_ucb_multi",
]
MAIN_METRICS = [
    "final_regret",
    "terminal_zero_regret",
    "terminal_pareto_optimal",
    "fairness_regret",
]
CERTIFICATE_DIAGNOSTIC_METRICS = [
    "certified",
    "certification_round",
    "certified_objective",
    "certified_arm",
    "theory_radius_same_certificate_at_empirical_time",
    "theory_radius_any_certificate_at_empirical_time",
]
FRONT_DIAGNOSTIC_METRICS = [
    "front_precision",
    "front_recall",
    "front_coverage_entropy",
]
ALL_METRICS = MAIN_METRICS + CERTIFICATE_DIAGNOSTIC_METRICS + FRONT_DIAGNOSTIC_METRICS
SUBSET_TYPE_LABELS = {
    "misleading_near_front": "Misleading near-front",
    "friendly_near_front": "Friendly near-front",
    "easy_separated": "Easy separated",
    "random_subsets": "Random subsets",
}
POLICY_LABELS = {
    "width_guided": "Width-guided",
    "width_guided_b0.25": "Width-guided",
    "width_guided_b0.5": "Width-guided",
    "width_guided_b0.75": "Width-guided",
    "pareto_ucb1": "Pareto UCB1",
    "annealing_pareto": "Annealing-Pareto",
    "annealing_pareto_random": "Annealing-Pareto random decay",
    "empirical_front_annealing": "Empirical-front annealing",
    "scalarized_ucb": "Equal-weight Scalarized UCB",
    "scalarized_ucb_equal": "Equal-weight Scalarized UCB",
    "scalarized_ucb_multi": "Scalarized UCB",
    "empirical_commit": "Uncertified commit",
}


def _policy_label(policy: str, *, show_width_coefficient: bool = False) -> str:
    if policy.startswith("width_guided_c"):
        return f"Width-guided (c={_format_width_coefficient(policy)})" if show_width_coefficient else "Width-guided"
    if policy.startswith("width_guided_b"):
        return f"Width-guided (c={_format_width_coefficient(policy)})" if show_width_coefficient else "Width-guided"
    if policy == "width_guided" and show_width_coefficient:
        return f"Width-guided (c={_format_width_coefficient(policy)})"
    if policy.startswith("annealing_pareto_decay"):
        return f"Annealing-Pareto (decay={policy[len('annealing_pareto_decay'):]})"
    return POLICY_LABELS.get(policy, policy)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real-data benchmark from the computational study.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--outdir", required=True, help="Benchmark output directory.")

    plan = subparsers.add_parser("plan", parents=[common], help="Select subsets and write job specs.")
    plan.add_argument("--instance", required=True, help="Prepared real-data NPZ instance.")
    plan.add_argument("--dataset-name", required=True, help="Dataset label for reporting.")
    plan.add_argument("--T", type=int, default=50000, help="Simulation horizon.")
    plan.add_argument("--n-runs", type=int, default=20, help="Monte Carlo runs per selected subset.")
    plan.add_argument("--seed", type=int, default=20260504, help="Base seed.")
    plan.add_argument("--k-values", default="10", help="Comma-separated subset sizes.")
    plan.add_argument("--n-subsets", type=int, default=3, help="Selected subsets per geometry type and K.")
    plan.add_argument("--n-candidates", type=int, default=2000, help="Random candidates if enumeration is disabled.")
    plan.add_argument(
        "--enumerate-limit",
        type=int,
        default=300000,
        help="Enumerate all K-subsets when n choose k is at most this value.",
    )
    plan.add_argument(
        "--subset-types",
        default="misleading_near_front,friendly_near_front,easy_separated",
        help="Comma-separated geometry groups.",
    )
    plan.add_argument("--policies", default=",".join(DEFAULT_POLICIES), help="Comma-separated policy list.")
    plan.add_argument(
        "--policies-by-type",
        default="",
        help=(
            "Optional semicolon-separated mapping from subset type to comma-separated policies, "
            "e.g. misleading_near_front:width_guided_c0.05,pareto_ucb1;friendly_near_front:..."
        ),
    )
    plan.add_argument(
        "--frozen-subsets",
        default="",
        help=(
            "Optional path to a selected_subsets.json file. When provided, reuse those "
            "subsets instead of re-running subset selection."
        ),
    )
    plan.add_argument(
        "--near-gap",
        type=float,
        default=0.02,
        help="Positive Pareto gaps at or below this value are treated as near-front.",
    )
    plan.add_argument(
        "--tiny-gap",
        type=float,
        default=0.002,
        help="Positive Pareto gaps at or below this value are treated as cheap near-front mistakes.",
    )
    plan.add_argument(
        "--gap-tol",
        type=float,
        default=1e-12,
        help="Tolerance for identifying zero Pareto gap arms.",
    )
    plan.add_argument(
        "--selection-mode",
        choices=["geometry", "contamination", "contamination_balanced"],
        default="geometry",
        help="Subset selector. contamination targets positive-gap arms on bootstrapped empirical fronts.",
    )
    plan.add_argument(
        "--contamination-samples-per-arm",
        type=int,
        default=50,
        help="Samples per arm used for bootstrapped empirical-front contamination scoring.",
    )
    plan.add_argument(
        "--contamination-bootstraps",
        type=int,
        default=64,
        help="Bootstrap repetitions used for empirical-front contamination scoring.",
    )
    plan.add_argument(
        "--contamination-prefilter",
        type=int,
        default=500,
        help="Number of geometry-misleading candidates rescored by contamination before selection.",
    )
    plan.add_argument(
        "--force-clean",
        action="store_true",
        help="Remove existing generated outputs in --outdir before writing a new plan.",
    )

    aggregate = subparsers.add_parser("aggregate", parents=[common], help="Aggregate finished job outputs.")
    aggregate.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write partial summaries even if some planned job outputs are missing.",
    )

    run_job = subparsers.add_parser("run-job", parents=[common], help="Run one job from jobs.jsonl.")
    run_job.add_argument("--job-index", type=int, required=True, help="Zero-based job index.")

    run_local = subparsers.add_parser("run-local", parents=[common], help="Run all planned jobs locally.")
    run_local.add_argument("--workers", type=int, default=1, help="Local worker processes.")
    run_local.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of planned jobs to run, for smoke tests.",
    )

    run_shard = subparsers.add_parser("run-shard", parents=[common], help="Run one modulo shard of jobs.")
    run_shard.add_argument("--shard-index", type=int, required=True, help="Zero-based shard index.")
    run_shard.add_argument("--n-shards", type=int, required=True, help="Total number of modulo shards.")
    run_shard.add_argument("--workers", type=int, default=1, help="Worker processes inside this Slurm task.")
    run_shard.add_argument(
        "--assignment",
        choices=["modulo", "balanced"],
        default="modulo",
        help="Shard assignment rule. balanced gives each shard a mix of policies.",
    )
    run_shard.add_argument(
        "--assignment-seed",
        type=int,
        default=20260504,
        help="Seed used to shuffle jobs within policy groups for balanced assignment.",
    )
    run_shard.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of shard jobs to run, for smoke tests.",
    )

    return parser.parse_args()


def _split_csv(raw: str, cast=str) -> List:
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_policies_by_type(raw: str, subset_types: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    valid = set(subset_types)
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid policies-by-type entry without ':': {item}")
        subset_type, policy_text = item.split(":", 1)
        subset_type = subset_type.strip()
        if subset_type not in valid:
            raise ValueError(f"Unknown subset type in policies-by-type: {subset_type}")
        policies = _split_csv(policy_text, str)
        if not policies:
            raise ValueError(f"No policies provided for subset type {subset_type}")
        out[subset_type] = policies
    missing = [subset_type for subset_type in subset_types if subset_type not in out]
    if missing:
        raise ValueError(f"policies-by-type is missing subset types: {', '.join(missing)}")
    return out


def _unique_policy_order(policy_groups: Iterable[Sequence[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for group in policy_groups:
        for policy in group:
            if policy not in seen:
                seen.add(policy)
                out.append(policy)
    return out


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1))


def _safe_median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _metric_summary(rows: Sequence[Dict[str, float]], metrics: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if metric in row and row[metric] is not None]
        out[f"{metric}_mean"] = _safe_mean(values)
        out[f"{metric}_std"] = _safe_std(values)
        out[f"{metric}_median"] = _safe_median(values)
    return out


def _instance_metadata(mu: np.ndarray, t_horizon: int, *, near_gap: float, tiny_gap: float, gap_tol: float) -> Dict[str, object]:
    opt_idx = pareto_nondominated_indices(mu)
    delta = pareto_arm_regrets(mu, opt_idx)
    gaps = objective_winner_gaps(mu)
    positive = delta > gap_tol
    subopt = delta[positive]
    near_positive = positive & (delta <= near_gap)
    tiny_positive = positive & (delta <= tiny_gap)
    misleading_positive = (delta > tiny_gap) & (delta <= near_gap)
    zero_gap = delta <= gap_tol
    separated_positive = positive & (delta > near_gap)
    return {
        "k": int(mu.shape[0]),
        "d": int(mu.shape[1]),
        "pareto_size": int(len(opt_idx)),
        "pareto_indices": [int(x) for x in opt_idx],
        "g_dagger": float(np.max(gaps)),
        "objective_winner_gaps": [float(x) for x in gaps.tolist()],
        "delta_min_p": float(np.min(subopt)) if subopt.size else 0.0,
        "delta_median_positive_p": float(np.median(subopt)) if subopt.size else 0.0,
        "sum_inv_delta_p": float(np.sum(1.0 / subopt)) if subopt.size else 0.0,
        "positive_gap_arms": int(np.sum(positive)),
        "zero_gap_arms": int(np.sum(zero_gap)),
        "near_positive_gap_arms": int(np.sum(near_positive)),
        "tiny_positive_gap_arms": int(np.sum(tiny_positive)),
        "misleading_near_positive_gap_arms": int(np.sum(misleading_positive)),
        "separated_positive_gap_arms": int(np.sum(separated_positive)),
        "misleading_near_burden": float(np.sum(1.0 / delta[misleading_positive])) if np.any(misleading_positive) else 0.0,
        "tiny_near_burden": float(np.sum(1.0 / delta[tiny_positive])) if np.any(tiny_positive) else 0.0,
        "c_pucb_exact": float(compute_exact_pucb_coefficient(mu, t_horizon)),
    }


def _subset_score(meta: Dict[str, object]) -> Dict[str, float]:
    misleading_count = float(meta["misleading_near_positive_gap_arms"])
    tiny_count = float(meta["tiny_positive_gap_arms"])
    near_count = float(meta["near_positive_gap_arms"])
    zero_count = float(meta["zero_gap_arms"])
    burden = float(meta["misleading_near_burden"])
    g_dagger = float(meta["g_dagger"])
    sum_inv = float(meta["sum_inv_delta_p"])
    positive_count = float(meta["positive_gap_arms"])
    separated_count = float(meta["separated_positive_gap_arms"])
    return {
        "misleading_score": 1000.0 * misleading_count + math.log1p(burden),
        "friendly_score": 1000.0 * (zero_count + tiny_count) - 100.0 * misleading_count - math.log1p(burden),
        "easy_score": 1000.0 * separated_count + 100.0 * g_dagger - 500.0 * near_count - math.log1p(sum_inv),
    }


def _reward_source_from_data(data: np.lib.npyio.NpzFile) -> Dict[str, object]:
    if "joint_probs" in data and "combo_rewards" in data:
        return {
            "kind": "joint",
            "joint_probs": np.asarray(data["joint_probs"], dtype=float),
            "combo_rewards": np.asarray(data["combo_rewards"], dtype=float),
        }
    if "reward_values" in data:
        return {
            "kind": "empirical",
            "reward_values": [np.asarray(item, dtype=float) for item in np.asarray(data["reward_values"], dtype=object)],
        }
    if "objective_values" in data:
        return {
            "kind": "objective_empirical",
            "objective_values": [
                [np.asarray(component, dtype=float) for component in item]
                for item in np.asarray(data["objective_values"], dtype=object)
            ],
        }
    raise ValueError("Instance must contain joint_probs/combo_rewards, reward_values, or objective_values.")


def _draw_empirical_subset_means(
    source: Dict[str, object],
    indices: Sequence[int],
    *,
    n_samples_per_arm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = [int(x) for x in indices]
    n_samples = int(n_samples_per_arm)
    if n_samples <= 0:
        raise ValueError("n_samples_per_arm must be positive.")

    kind = str(source["kind"])
    if kind == "joint":
        joint_probs = source["joint_probs"]
        combo_rewards = source["combo_rewards"]
        d = int(combo_rewards.shape[1])
        means = np.zeros((len(indices), d), dtype=float)
        for local_idx, arm_idx in enumerate(indices):
            draws = rng.choice(combo_rewards.shape[0], size=n_samples, p=joint_probs[arm_idx])
            means[local_idx] = np.mean(combo_rewards[draws], axis=0)
        return means

    if kind == "empirical":
        reward_values = source["reward_values"]
        first = reward_values[indices[0]]
        d = int(first.shape[1])
        means = np.zeros((len(indices), d), dtype=float)
        for local_idx, arm_idx in enumerate(indices):
            values = reward_values[arm_idx]
            draws = rng.integers(0, values.shape[0], size=n_samples)
            means[local_idx] = np.mean(values[draws], axis=0)
        return means

    if kind == "objective_empirical":
        objective_values = source["objective_values"]
        d = len(objective_values[indices[0]])
        means = np.zeros((len(indices), d), dtype=float)
        for local_idx, arm_idx in enumerate(indices):
            for obj_idx, values in enumerate(objective_values[arm_idx]):
                draws = rng.integers(0, values.shape[0], size=n_samples)
                means[local_idx, obj_idx] = float(np.mean(values[draws]))
        return means

    raise ValueError(f"Unknown reward source kind {kind!r}.")


def _contamination_score(
    *,
    true_mu: np.ndarray,
    global_indices: Sequence[int],
    source: Dict[str, object],
    n_samples_per_arm: int,
    n_bootstraps: int,
    gap_tol: float,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    opt_idx = pareto_nondominated_indices(true_mu)
    delta = pareto_arm_regrets(true_mu, opt_idx)
    positive = delta > gap_tol
    contaminated_counts: List[float] = []
    contaminated_mass: List[float] = []
    empirical_front_regrets: List[float] = []
    empirical_front_sizes: List[float] = []
    any_contaminated: List[float] = []
    for _ in range(int(n_bootstraps)):
        empirical_mu = _draw_empirical_subset_means(
            source,
            global_indices,
            n_samples_per_arm=int(n_samples_per_arm),
            rng=rng,
        )
        empirical_front = np.asarray(pareto_nondominated_indices(empirical_mu), dtype=int)
        front_delta = delta[empirical_front]
        front_positive = positive[empirical_front]
        contaminated_counts.append(float(np.sum(front_positive)))
        contaminated_mass.append(float(np.sum(front_delta[front_positive])))
        empirical_front_regrets.append(float(np.mean(front_delta)) if empirical_front.size else 0.0)
        empirical_front_sizes.append(float(empirical_front.size))
        any_contaminated.append(float(np.any(front_positive)))

    contamination_count_mean = _safe_mean(contaminated_counts)
    contamination_mass_mean = _safe_mean(contaminated_mass)
    empirical_front_regret_mean = _safe_mean(empirical_front_regrets)
    return {
        "contamination_count_mean": float(contamination_count_mean),
        "contamination_mass_mean": float(contamination_mass_mean),
        "empirical_front_regret_mean": float(empirical_front_regret_mean),
        "empirical_front_size_mean": float(_safe_mean(empirical_front_sizes)),
        "contaminated_front_probability": float(_safe_mean(any_contaminated)),
        "contamination_score": float(
            100000.0 * empirical_front_regret_mean
            + 100.0 * contamination_count_mean
            + 10000.0 * contamination_mass_mean
        ),
    }


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _candidate_subsets(
    *,
    n_arms: int,
    k: int,
    n_candidates: int,
    enumerate_limit: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    total = _n_choose_k(n_arms, k)
    if total <= enumerate_limit:
        for combo in itertools.combinations(range(n_arms), k):
            yield np.asarray(combo, dtype=int)
        return

    seen = set()
    deterministic = tuple(range(k))
    seen.add(deterministic)
    yield np.asarray(deterministic, dtype=int)
    while len(seen) < n_candidates:
        idx = tuple(int(x) for x in sorted(rng.choice(n_arms, size=k, replace=False).tolist()))
        if idx in seen:
            continue
        seen.add(idx)
        yield np.asarray(idx, dtype=int)


def _select_subsets(
    mu: np.ndarray,
    *,
    data: Optional[np.lib.npyio.NpzFile] = None,
    k_values: Sequence[int],
    n_candidates: int,
    enumerate_limit: int,
    n_subsets: int,
    subset_types: Sequence[str],
    t_horizon: int,
    seed: int,
    near_gap: float,
    tiny_gap: float,
    gap_tol: float,
    selection_mode: str = "geometry",
    contamination_samples_per_arm: int = 50,
    contamination_bootstraps: int = 64,
    contamination_prefilter: int = 500,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    scored: List[Dict[str, object]] = []
    for k in k_values:
        for indices in _candidate_subsets(
            n_arms=mu.shape[0],
            k=int(k),
            n_candidates=n_candidates,
            enumerate_limit=enumerate_limit,
            rng=rng,
        ):
            subset_mu = mu[indices]
            meta = _instance_metadata(
                subset_mu,
                t_horizon,
                near_gap=near_gap,
                tiny_gap=tiny_gap,
                gap_tol=gap_tol,
            )
            selection_scores = _subset_score(meta)
            selection_scores["random_score"] = float(rng.random())
            scored.append(
                {
                    "k": int(k),
                    "indices": [int(x) for x in indices.tolist()],
                    "metadata": meta,
                    "selection_scores": selection_scores,
                    "contamination_scores": {},
                }
            )

    if selection_mode in {"contamination", "contamination_balanced"}:
        if data is None:
            raise ValueError("Contamination selection requires the loaded NPZ data.")
        source = _reward_source_from_data(data)
        prefilter = sorted(
            scored,
            key=lambda item: (
                item["selection_scores"]["misleading_score"],
                item["metadata"]["misleading_near_positive_gap_arms"],
                item["metadata"]["misleading_near_burden"],
            ),
            reverse=True,
        )[: max(1, int(contamination_prefilter))]
        for rank, item in enumerate(prefilter):
            item["contamination_scores"] = _contamination_score(
                true_mu=mu[np.asarray(item["indices"], dtype=int)],
                global_indices=item["indices"],
                source=source,
                n_samples_per_arm=int(contamination_samples_per_arm),
                n_bootstraps=int(contamination_bootstraps),
                gap_tol=float(gap_tol),
                seed=int(seed + 50_000_000 + rank),
            )
    elif selection_mode != "geometry":
        raise ValueError(f"Unknown selection mode {selection_mode!r}.")

    selected: List[Dict[str, object]] = []
    used_indices = set()
    if selection_mode == "contamination":
        misleading_order = lambda item: (
            item["contamination_scores"].get("contamination_score", -1.0),
            item["contamination_scores"].get("empirical_front_regret_mean", -1.0),
            item["contamination_scores"].get("contamination_count_mean", -1.0),
            item["selection_scores"]["misleading_score"],
        )
    elif selection_mode == "contamination_balanced":
        misleading_order = lambda item: (
            item["contamination_scores"].get("contamination_score", -1.0)
            - 250.0 * float(item["metadata"]["separated_positive_gap_arms"])
            + 80.0 * float(item["metadata"]["misleading_near_positive_gap_arms"])
            - 0.5 * float(item["metadata"]["c_pucb_exact"]),
            item["metadata"]["misleading_near_positive_gap_arms"],
            -item["metadata"]["separated_positive_gap_arms"],
            item["contamination_scores"].get("contamination_score", -1.0),
        )
    else:
        misleading_order = lambda item: (
            item["selection_scores"]["misleading_score"],
            item["metadata"]["misleading_near_positive_gap_arms"],
            item["metadata"]["misleading_near_burden"],
        )
    order_specs = {
        "misleading_near_front": misleading_order,
        "friendly_near_front": lambda item: (
            item["selection_scores"]["friendly_score"],
            item["metadata"]["tiny_positive_gap_arms"],
            -item["metadata"]["misleading_near_positive_gap_arms"],
        ),
        "easy_separated": lambda item: (
            item["selection_scores"]["easy_score"],
            item["metadata"]["separated_positive_gap_arms"],
            item["metadata"]["g_dagger"],
            -item["metadata"]["near_positive_gap_arms"],
        ),
        "random_subsets": lambda item: (
            item["selection_scores"]["random_score"],
        ),
    }
    for subset_type in subset_types:
        if subset_type not in order_specs:
            raise ValueError(f"Unknown subset type {subset_type!r}.")
        ordered = sorted(scored, key=order_specs[subset_type], reverse=True)
        picked = 0
        for item in ordered:
            key = tuple(item["indices"])
            if key in used_indices:
                continue
            used_indices.add(key)
            selected.append(
                {
                    "subset_id": len(selected),
                    "subset_type": subset_type,
                    "subset_type_label": SUBSET_TYPE_LABELS[subset_type],
                    **item,
                }
            )
            picked += 1
            if picked >= n_subsets:
                break
    return selected


def _load_labels(data: np.lib.npyio.NpzFile, indices: Sequence[int]) -> Dict[str, object]:
    labels: Dict[str, object] = {}
    idx = np.asarray(indices, dtype=int)
    for key in ["arm_ids", "arm_names", "objective_names"]:
        if key in data:
            arr = np.asarray(data[key])
            labels[key] = [str(x) for x in arr[idx].tolist()] if key != "objective_names" else [str(x) for x in arr.tolist()]
    return labels


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    max_value = np.max(values)
    candidates = np.flatnonzero(np.isclose(values, max_value, atol=1e-12, rtol=0.0))
    return int(rng.choice(candidates))


def _random_argmax_excluding(values: np.ndarray, *, exclude: int, rng: np.random.Generator) -> int:
    mask = np.ones(len(values), dtype=bool)
    mask[int(exclude)] = False
    masked_values = np.asarray(values)[mask]
    idx_local = np.flatnonzero(np.isclose(masked_values, np.max(masked_values), atol=1e-12, rtol=0.0))
    return int(np.flatnonzero(mask)[rng.choice(idx_local)])


def _normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0 or counts.size <= 1:
        return 0.0
    prob = counts / total
    positive = prob[prob > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return entropy / float(np.log(counts.size))


class StreamingRealBandit:
    """Real-data reward sampler that records only final counts and sums."""

    def __init__(self, data: np.lib.npyio.NpzFile, indices: Sequence[int], rng: np.random.Generator):
        self.indices = np.asarray(indices, dtype=int)
        self.mu = np.asarray(data["mu"], dtype=float)[self.indices]
        self.k, self.d = self.mu.shape
        self.rng = rng
        self.pull_counts = np.zeros(self.k, dtype=np.int64)
        self.reward_sums = np.zeros((self.k, self.d), dtype=float)

        if "joint_probs" in data and "combo_rewards" in data:
            self.kind = "joint"
            self.joint_probs = np.asarray(data["joint_probs"], dtype=float)[self.indices]
            self.combo_rewards = np.asarray(data["combo_rewards"], dtype=float)
            self.reward_values = None
            return

        if "reward_values" in data:
            self.kind = "empirical"
            values = np.asarray(data["reward_values"], dtype=object)[self.indices]
            self.reward_values = [np.asarray(item, dtype=float) for item in values]
            self.objective_values = None
            self.joint_probs = None
            self.combo_rewards = None
            return

        if "objective_values" in data:
            self.kind = "objective_empirical"
            values = np.asarray(data["objective_values"], dtype=object)[self.indices]
            self.objective_values = [
                [np.asarray(component, dtype=float) for component in item]
                for item in values
            ]
            self.reward_values = None
            self.joint_probs = None
            self.combo_rewards = None
            return

        raise ValueError("Instance must contain joint_probs/combo_rewards, reward_values, or objective_values.")

    def pull(self, arm: int) -> np.ndarray:
        arm = int(arm)
        if self.kind == "joint":
            combo_idx = int(self.rng.choice(self.combo_rewards.shape[0], p=self.joint_probs[arm]))
            reward = self.combo_rewards[combo_idx].astype(float, copy=True)
        elif self.kind == "empirical":
            values = self.reward_values[arm]
            sample_idx = int(self.rng.integers(0, values.shape[0]))
            reward = values[sample_idx].astype(float, copy=True)
        else:
            reward = np.zeros(self.d, dtype=float)
            for j, values in enumerate(self.objective_values[arm]):
                sample_idx = int(self.rng.integers(0, values.shape[0]))
                reward[j] = float(values[sample_idx])

        self.pull_counts[arm] += 1
        self.reward_sums[arm] += reward
        return reward

    def add_expected_pulls(self, arm: int, n_pulls: int) -> None:
        """Fast-forward committed play using the true empirical mean vector."""
        if n_pulls <= 0:
            return
        arm = int(arm)
        n_pulls = int(n_pulls)
        self.pull_counts[arm] += n_pulls
        self.reward_sums[arm] += float(n_pulls) * self.mu[arm]

    def observed_means(self) -> np.ndarray:
        means = np.zeros_like(self.reward_sums)
        observed = self.pull_counts > 0
        means[observed] = self.reward_sums[observed] / self.pull_counts[observed, None]
        return means


def _width_bonus_scale(policy: str) -> float:
    if policy == "width_guided":
        return 1.0
    coefficient_prefix = "width_guided_c"
    if policy.startswith(coefficient_prefix):
        coefficient = float(policy[len(coefficient_prefix) :])
        if coefficient <= 0:
            raise ValueError("Width-guided coefficient must be positive.")
        return coefficient / 2.0
    prefix = "width_guided_b"
    if policy.startswith(prefix):
        return float(policy[len(prefix) :])
    raise ValueError(f"Policy {policy!r} is not a Width-guided variant.")


def _width_coefficient(policy: str) -> float:
    """Return c in beta=sqrt(c log(T) / N) for a Width-guided policy name."""
    return 2.0 * _width_bonus_scale(policy)


def _format_width_coefficient(policy: str) -> str:
    coefficient = _width_coefficient(policy)
    return f"{coefficient:g}"


def _annealing_decay(policy: str, rng: np.random.Generator) -> float:
    if policy == "annealing_pareto":
        return 0.4
    if policy == "annealing_pareto_random":
        return float(rng.uniform(np.nextafter(0.0, 1.0), 1.0))
    prefix = "annealing_pareto_decay"
    if policy.startswith(prefix):
        decay = float(policy[len(prefix) :])
        if not (0.0 < decay < 1.0):
            raise ValueError("Annealing-Pareto epsilon_decay must lie in (0, 1).")
        return decay
    raise ValueError(f"Policy {policy!r} is not an Annealing-Pareto variant.")


def _annealing_pareto_candidates(
    means: np.ndarray,
    epsilon_t: float,
    previous_set: Optional[np.ndarray] = None,
) -> np.ndarray:
    objective_best = np.max(means, axis=0)
    near_best = np.any(means >= objective_best[None, :] - float(epsilon_t), axis=1)
    near_idx = np.flatnonzero(near_best)
    candidate_idx = near_idx
    if previous_set is not None and len(previous_set) > 0:
        front = set(int(x) for x in pareto_nondominated_indices(means))
        retained = [int(x) for x in np.asarray(previous_set, dtype=int).tolist() if int(x) in front]
        if retained:
            candidate_idx = np.union1d(candidate_idx, np.asarray(retained, dtype=int))
    if candidate_idx.size == 0:
        candidate_idx = np.arange(means.shape[0], dtype=int)
    return candidate_idx.astype(int)


def _unique_weight_rows(weights: Sequence[np.ndarray]) -> np.ndarray:
    rows: List[np.ndarray] = []
    seen = set()
    for weight in weights:
        w = np.asarray(weight, dtype=float)
        if w.ndim != 1:
            continue
        total = float(np.sum(w))
        if total <= 0:
            continue
        w = w / total
        key = tuple(np.round(w, 12).tolist())
        if key in seen:
            continue
        seen.add(key)
        rows.append(w)
    if not rows:
        raise ValueError("Scalarized UCB weight set is empty.")
    return np.asarray(rows, dtype=float)


def _simplex_lattice_weights(d: int, denominator: int) -> np.ndarray:
    weights: List[np.ndarray] = []

    def rec(remaining: int, pos: int, prefix: List[int]) -> None:
        if pos == d - 1:
            weights.append(np.asarray(prefix + [remaining], dtype=float) / float(denominator))
            return
        for value in range(remaining + 1):
            rec(remaining - value, pos + 1, prefix + [value])

    rec(int(denominator), 0, [])
    return np.asarray(weights, dtype=float)


def _scalarized_weight_set(d: int, policy: str) -> np.ndarray:
    if policy in {"scalarized_ucb", "scalarized_ucb_equal"}:
        return np.ones((1, int(d)), dtype=float) / float(d)
    if policy != "scalarized_ucb_multi":
        raise ValueError(f"Policy {policy!r} is not a scalarized-UCB variant.")

    d = int(d)
    uniform = np.ones(d, dtype=float) / float(d)
    weights: List[np.ndarray] = [uniform]
    if d <= 6:
        weights.extend(_simplex_lattice_weights(d, denominator=4))
        return _unique_weight_rows(weights)

    eye = np.eye(d, dtype=float)
    weights.extend(eye)
    for j in range(d):
        weights.append(0.5 * uniform + 0.5 * eye[j])
        weights.append(0.1 * uniform + 0.9 * eye[j])
    for i in range(d):
        for j in range(i + 1, d):
            weights.append(0.5 * eye[i] + 0.5 * eye[j])
    return _unique_weight_rows(weights)


def _certificate_state(
    means: np.ndarray,
    counts: np.ndarray,
    *,
    coefficient: float,
    bonus_log: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    radius = np.sqrt(float(coefficient) * bonus_log / counts)
    ucb = means + radius[:, None]
    lcb = means - radius[:, None]
    d = means.shape[1]
    leaders = np.zeros(d, dtype=int)
    challengers = np.zeros(d, dtype=int)
    pair_width = np.zeros(d, dtype=float)
    certified = np.zeros(d, dtype=bool)
    for j in range(d):
        leader = _random_argmax(ucb[:, j], rng)
        challenger = _random_argmax_excluding(ucb[:, j], exclude=leader, rng=rng)
        leaders[j] = leader
        challengers[j] = challenger
        pair_width[j] = radius[leader] + radius[challenger]
        certified[j] = bool(lcb[leader, j] > ucb[challenger, j])
    return {
        "radius": radius,
        "ucb": ucb,
        "lcb": lcb,
        "leaders": leaders,
        "challengers": challengers,
        "pair_width": pair_width,
        "certified": certified,
    }


def _front_recovery_metrics(true_mu: np.ndarray, empirical_mu: np.ndarray) -> Dict[str, float]:
    true_front = set(pareto_nondominated_indices(true_mu))
    empirical_front = set(pareto_nondominated_indices(empirical_mu))
    tp = len(true_front & empirical_front)
    precision = tp / len(empirical_front) if empirical_front else 0.0
    recall = tp / len(true_front) if true_front else 0.0
    return {
        "front_precision": float(precision),
        "front_recall": float(recall),
    }


def _final_metrics(
    *,
    sampler: StreamingRealBandit,
    tail_counts: np.ndarray,
    final_regret: float,
    fairness_regret: float,
    gap_tol: float,
) -> Dict[str, float]:
    mu = sampler.mu
    opt_idx = pareto_nondominated_indices(mu)
    delta = pareto_arm_regrets(mu, opt_idx)
    terminal_counts = tail_counts if np.sum(tail_counts) > 0 else sampler.pull_counts
    terminal_arm = int(np.argmax(terminal_counts)) if np.sum(terminal_counts) > 0 else -1
    zero_mask = delta <= gap_tol
    pareto_mask = np.zeros(mu.shape[0], dtype=bool)
    pareto_mask[opt_idx] = True

    metrics = {
        "final_regret": float(final_regret),
        "terminal_zero_regret": float(terminal_arm >= 0 and zero_mask[terminal_arm]),
        "terminal_pareto_optimal": float(terminal_arm >= 0 and pareto_mask[terminal_arm]),
        "terminal_recommendation_arm": float(terminal_arm),
        "fairness_regret": float(fairness_regret),
        "front_coverage_entropy": _normalized_entropy(sampler.pull_counts[np.asarray(opt_idx, dtype=int)]),
    }
    metrics.update(_front_recovery_metrics(mu, sampler.observed_means()))
    return metrics


def _simulate_policy_on_subset(
    *,
    sampler: StreamingRealBandit,
    policy: str,
    t_horizon: int,
    policy_seed: int,
    gap_tol: float,
) -> Dict[str, float]:
    rng = np.random.default_rng(policy_seed)
    mu = sampler.mu
    k, d = sampler.k, sampler.d
    opt_idx = pareto_nondominated_indices(mu)
    delta = pareto_arm_regrets(mu, opt_idx)
    fair_value = np.min(mu, axis=1)
    fair_gap = float(np.max(fair_value)) - fair_value
    tail_start = int(np.floor(0.8 * t_horizon))
    tail_counts = np.zeros(k, dtype=np.int64)
    final_regret = 0.0
    fairness_regret = 0.0
    t = 0

    def record_pull(arm: int) -> None:
        nonlocal final_regret, fairness_regret, t
        sampler.pull(arm)
        final_regret += float(delta[int(arm)])
        fairness_regret += float(fair_gap[int(arm)])
        if t >= tail_start:
            tail_counts[int(arm)] += 1
        t += 1

    def fast_forward(arm: int, n_pulls: int) -> None:
        nonlocal final_regret, fairness_regret, t
        if n_pulls <= 0:
            return
        arm = int(arm)
        n_pulls = int(n_pulls)
        sampler.add_expected_pulls(arm, n_pulls)
        final_regret += float(n_pulls) * float(delta[arm])
        fairness_regret += float(n_pulls) * float(fair_gap[arm])
        if t + n_pulls > tail_start:
            tail_counts[arm] += int(t + n_pulls - max(t, tail_start))
        t += n_pulls

    for arm in range(k):
        if t >= t_horizon:
            return _final_metrics(
                sampler=sampler,
                tail_counts=tail_counts,
                final_regret=final_regret,
                fairness_regret=fairness_regret,
                gap_tol=gap_tol,
            )
        record_pull(arm)

    if policy.startswith("width_guided"):
        width_coefficient = _width_coefficient(policy)
        bonus_log = np.log(max(2, t_horizon))
        certified_objective: Optional[int] = None
        certified_leader: Optional[int] = None
        certification_round = float("nan")
        theory_same_certificate = float("nan")
        theory_any_certificate = float("nan")
        while t < t_horizon:
            counts = sampler.pull_counts.astype(float)
            means = sampler.reward_sums / counts[:, None]
            state = _certificate_state(
                means,
                counts,
                coefficient=width_coefficient,
                bonus_log=bonus_log,
                rng=rng,
            )
            radius = state["radius"]
            leaders = state["leaders"]
            challengers = state["challengers"]
            pair_width = state["pair_width"]
            certified = state["certified"]

            if certified_objective is None and np.any(certified):
                cert_idx = np.flatnonzero(certified)
                chosen_local = _random_argmax(pair_width[cert_idx], rng)
                certified_objective = int(cert_idx[chosen_local])
                certified_leader = int(leaders[certified_objective])
                certification_round = float(t + 1)

                theory_state = _certificate_state(
                    means,
                    counts,
                    coefficient=2.0,
                    bonus_log=bonus_log,
                    rng=rng,
                )
                theory_ucb = theory_state["ucb"]
                theory_lcb = theory_state["lcb"]
                other_mask = np.ones(k, dtype=bool)
                other_mask[int(certified_leader)] = False
                theory_same_certificate = float(
                    theory_lcb[int(certified_leader), certified_objective]
                    > np.max(theory_ucb[other_mask, certified_objective])
                )
                theory_any_certificate = float(np.any(theory_state["certified"]))

            if certified_objective is not None:
                fast_forward(int(certified_leader), t_horizon - t)
                break

            chosen_objective = _random_argmax(pair_width, rng)
            pair = np.array([leaders[chosen_objective], challengers[chosen_objective]], dtype=int)
            chosen = int(pair[_random_argmax(radius[pair], rng)])
            record_pull(chosen)

    elif policy == "pareto_ucb1":
        while t < t_horizon:
            counts = sampler.pull_counts.astype(float)
            means = sampler.reward_sums / counts[:, None]
            inside = max(2.0, t * (d * k) ** 0.25)
            bonus = np.sqrt(2.0 * np.log(inside) / counts)
            candidate_idx = pareto_nondominated_indices(means + bonus[:, None])
            record_pull(int(rng.choice(candidate_idx)))

    elif policy.startswith("annealing_pareto"):
        epsilon_decay = _annealing_decay(policy, rng)
        annealing_step = 0
        epsilon_pareto_set = np.asarray([], dtype=int)
        log_decay = math.log(float(epsilon_decay))
        scale = float(k * d)
        while t < t_horizon:
            means = sampler.reward_sums / sampler.pull_counts[:, None]
            epsilon_t = math.exp(annealing_step * log_decay) / scale
            candidate_idx = _annealing_pareto_candidates(means, epsilon_t, epsilon_pareto_set)
            epsilon_pareto_set = candidate_idx
            chosen = int(rng.choice(candidate_idx))
            annealing_step += 1
            record_pull(chosen)

    elif policy == "empirical_front_annealing":
        exploration_scale = 1.0
        decay_power = 0.5
        while t < t_horizon:
            epsilon_t = min(1.0, exploration_scale / ((t + 1.0) ** decay_power))
            if rng.random() < epsilon_t:
                chosen = int(rng.integers(0, k))
            else:
                candidate_idx = pareto_nondominated_indices(sampler.reward_sums / sampler.pull_counts[:, None])
                chosen = int(rng.choice(candidate_idx))
            record_pull(chosen)

    elif policy.startswith("scalarized_ucb"):
        weights = _scalarized_weight_set(d, policy)
        bonus_log = np.log(max(2, t_horizon))
        scalar_step = 0
        while t < t_horizon:
            counts = sampler.pull_counts.astype(float)
            means = sampler.reward_sums / counts[:, None]
            radius = np.sqrt(2.0 * bonus_log / counts)
            weight = weights[scalar_step % weights.shape[0]]
            scores = means @ weight + radius
            record_pull(_random_argmax(scores, rng))
            scalar_step += 1

    elif policy == "empirical_commit":
        candidate_idx = pareto_nondominated_indices(sampler.reward_sums / sampler.pull_counts[:, None])
        chosen = int(rng.choice(candidate_idx))
        fast_forward(chosen, t_horizon - t)

    else:
        raise ValueError(f"Unknown policy {policy!r}. Available defaults: {', '.join(sorted(POLICY_LABELS))}.")

    metrics = _final_metrics(
        sampler=sampler,
        tail_counts=tail_counts,
        final_regret=final_regret,
        fairness_regret=fairness_regret,
        gap_tol=gap_tol,
    )
    if policy.startswith("width_guided"):
        metrics.update(
            {
                "certified": float(certified_objective is not None),
                "certification_round": float(certification_round),
                "certified_objective": float(certified_objective) if certified_objective is not None else float("nan"),
                "certified_arm": float(certified_leader) if certified_leader is not None else float("nan"),
                "theory_radius_same_certificate_at_empirical_time": theory_same_certificate,
                "theory_radius_any_certificate_at_empirical_time": theory_any_certificate,
            }
        )
    return metrics


def _read_jobs(out_dir: Path) -> List[Dict[str, object]]:
    jobs_path = out_dir / "jobs.jsonl"
    jobs: List[Dict[str, object]] = []
    with open(jobs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def _job_output_path(out_dir: Path, job_index: int) -> Path:
    return out_dir / "job_outputs" / f"job_{job_index:05d}.json"


def _run_job_spec(out_dir_raw: str, job: Dict[str, object]) -> Dict[str, object]:
    out_dir = Path(out_dir_raw)
    output_path = _job_output_path(out_dir, int(job["job_index"]))
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    data = np.load(str(job["instance"]), allow_pickle=True)
    indices = [int(x) for x in job["indices"]]
    reward_rng = np.random.default_rng(int(job["reward_seed"]))
    sampler = StreamingRealBandit(data, indices, reward_rng)
    metrics = _simulate_policy_on_subset(
        sampler=sampler,
        policy=str(job["policy"]),
        t_horizon=int(job["T"]),
        policy_seed=int(job["policy_seed"]),
        gap_tol=float(job.get("gap_tol", 1e-12)),
    )
    payload = {
        **job,
        "metrics": {metric: float(metrics[metric]) for metric in ALL_METRICS if metric in metrics},
        "terminal_recommendation_arm": int(metrics.get("terminal_recommendation_arm", -1)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)
    tmp_path.replace(output_path)
    return payload


def _format_mean_std(summary: Dict[str, float], metric: str, *, precision: int = 2) -> str:
    mean = summary.get(f"{metric}_mean", float("nan"))
    std = summary.get(f"{metric}_std", float("nan"))
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def _format_mean(summary: Dict[str, float], metric: str, *, precision: int = 3) -> str:
    mean = summary.get(f"{metric}_mean", float("nan"))
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{precision}f}"


def _tables_dir(out_dir: Path) -> Path:
    path = out_dir / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _summaries_dir(out_dir: Path) -> Path:
    path = out_dir / "summaries"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _policies_for_type(
    subset_type: str,
    policies: Sequence[str],
    policies_by_type: Optional[Dict[str, Sequence[str]]] = None,
) -> List[str]:
    if policies_by_type:
        return list(policies_by_type.get(subset_type, policies))
    return list(policies)


def _write_main_table(
    out_dir: Path,
    aggregate: Dict[str, object],
    policies: Sequence[str],
    subset_types: Sequence[str],
    policies_by_type: Optional[Dict[str, Sequence[str]]] = None,
) -> None:
    table_subset_types = [subset_type for subset_type in subset_types if subset_type != "random_subsets"]
    if not table_subset_types:
        table_subset_types = list(subset_types)
    show_width_coefficients = len(_width_policies(policies)) > 1
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Real-data benchmark on held-out subsets. Subsets are grouped by empirical mean "
            "geometry before policy simulation. The main target is Pareto regret and detection of at least "
            "one Pareto-optimal arm; fairness is reported as a distinct objective.}"
        ),
        "\\label{tab:real-main}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        (
            "Subset type & Method & Pareto regret $\\downarrow$ & "
            "$\\Pr(\\widehat a_T\\in A^\\star)\\uparrow$ & "
            "Fairness regret $\\downarrow$ \\\\"
        ),
        "\\midrule",
    ]
    for type_idx, subset_type in enumerate(table_subset_types):
        type_label = _latex_escape(SUBSET_TYPE_LABELS.get(subset_type, subset_type))
        for policy in _policies_for_type(subset_type, policies, policies_by_type):
            if policy not in aggregate[subset_type]["policies"]:
                continue
            summary = aggregate[subset_type]["policies"][policy]
            lines.append(
                " & ".join(
                    [
                        type_label,
                        _latex_escape(_policy_label(policy, show_width_coefficient=show_width_coefficients)),
                        _format_mean_std(summary, "final_regret"),
                        _format_mean(summary, "terminal_pareto_optimal", precision=3),
                        _format_mean_std(summary, "fairness_regret"),
                    ]
                )
                + " \\\\"
            )
        if type_idx != len(table_subset_types) - 1:
            lines.append("\\midrule")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\begin{minipage}{0.98\\linewidth}",
            "\\scriptsize",
            (
                "Here $\\widehat a_T$ is the terminal recommendation, taken to be the arm most "
                "frequently played in the final 20\\% of rounds. The set $A^\\star$ "
                "is computed from the empirical mean vectors of the held-out subset. Fairness "
                "regret is the cumulative max-min regret "
                "$\\sum_{t=1}^T (\\max_a \\min_j \\mu_a^{(j)} - \\min_j \\mu_{A_t}^{(j)})$."
            ),
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    table_text = "\n".join(lines)
    tables_dir = _tables_dir(out_dir)
    for path in [tables_dir / "table_6_1_main.tex", out_dir / "table_6_1.tex"]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(table_text)


def _write_front_table(
    out_dir: Path,
    aggregate: Dict[str, object],
    policies: Sequence[str],
    subset_types: Sequence[str],
    policies_by_type: Optional[Dict[str, Sequence[str]]] = None,
) -> None:
    table_subset_types = [subset_type for subset_type in subset_types if subset_type != "random_subsets"]
    if not table_subset_types:
        table_subset_types = list(subset_types)
    show_width_coefficients = len(_width_policies(policies)) > 1
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Front-identification and coverage diagnostics on the same held-out subsets. "
            "These metrics evaluate full-front learning rather than Pareto-regret certification.}"
        ),
        "\\label{tab:real-front-coverage}",
        "\\scriptsize",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        (
            "Subset type & Method & Front precision $\\uparrow$ & "
            "Front recall $\\uparrow$ & Coverage entropy $\\uparrow$ \\\\"
        ),
        "\\midrule",
    ]
    for type_idx, subset_type in enumerate(table_subset_types):
        type_label = _latex_escape(SUBSET_TYPE_LABELS.get(subset_type, subset_type))
        for policy in _policies_for_type(subset_type, policies, policies_by_type):
            if policy not in aggregate[subset_type]["policies"]:
                continue
            summary = aggregate[subset_type]["policies"][policy]
            lines.append(
                " & ".join(
                    [
                        type_label,
                        _latex_escape(_policy_label(policy, show_width_coefficient=show_width_coefficients)),
                        _format_mean_std(summary, "front_precision", precision=3),
                        _format_mean_std(summary, "front_recall", precision=3),
                        _format_mean_std(summary, "front_coverage_entropy", precision=3),
                    ]
                )
                + " \\\\"
            )
        if type_idx != len(table_subset_types) - 1:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    with open(_tables_dir(out_dir) / "table_6_1_front_coverage.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _width_policies(policies: Sequence[str]) -> List[str]:
    return sorted(
        [policy for policy in policies if str(policy).startswith("width_guided")],
        key=lambda policy: _width_coefficient(str(policy)),
    )


def _metric_values(rows: Sequence[Dict[str, object]], metric: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and metric in metrics and metrics[metric] is not None:
            values.append(float(metrics[metric]))
    return values


def _empirical_width_policy_for_type(
    subset_type: str,
    policies: Sequence[str],
    policies_by_type: Optional[Dict[str, Sequence[str]]] = None,
    empirical_policy: Optional[str] = None,
) -> Optional[str]:
    if empirical_policy:
        return empirical_policy
    type_policies = list(policies_by_type[subset_type]) if policies_by_type else list(policies)
    candidates = [
        str(policy)
        for policy in type_policies
        if str(policy).startswith("width_guided_c")
    ]
    if not candidates:
        return None
    return _width_policies(candidates)[0]


def _metric_float(metrics: object, metric: str) -> float:
    if not isinstance(metrics, dict):
        return float("nan")
    value = metrics.get(metric)
    if value is None:
        return float("nan")
    return float(value)


def _format_scalar(value: float, *, precision: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.{precision}f}"


def _latex_escape(text: object) -> str:
    raw = str(text)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in raw)


def _format_median_iqr(values: Sequence[float], *, precision: int = 0) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "--"
    median = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    if precision == 0:
        return f"{median:.0f} [{q1:.0f}, {q3:.0f}]"
    return f"{median:.{precision}f} [{q1:.{precision}f}, {q3:.{precision}f}]"


def _write_width_sensitivity_table(
    out_dir: Path,
    rows: Sequence[Dict[str, object]],
    policies: Sequence[str],
    subset_types: Sequence[str],
    dataset_name: str,
) -> None:
    width_policies = _width_policies(policies)
    if not width_policies:
        return
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Width-guided radius-scale sensitivity on held-out subsets. "
            "The coefficient $c$ is defined by $\\beta_a(t)=\\sqrt{c\\log(T)/N_a(t-1)}$; "
            "$c=2$ is the theorem-matched coefficient.}"
        ),
        "\\label{tab:app-width-sensitivity}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        (
            "Dataset & Subset type & $c$ & Pareto regret $\\downarrow$ & "
            "$\\Pr(\\widehat a_T\\in A^\\star)\\uparrow$ & Cert. rate $\\uparrow$ & "
            "Median cert. round $\\downarrow$ \\\\"
        ),
        "\\midrule",
    ]
    wrote = False
    for subset_type in subset_types:
        type_label = _latex_escape(SUBSET_TYPE_LABELS.get(subset_type, subset_type))
        for policy in width_policies:
            policy_rows = [
                row
                for row in rows
                if row.get("subset_type") == subset_type and row.get("policy") == policy
            ]
            if not policy_rows:
                continue
            summary = _metric_summary([row["metrics"] for row in policy_rows], ALL_METRICS)
            lines.append(
                " & ".join(
                    [
                        _latex_escape(dataset_name),
                        type_label,
                        _format_width_coefficient(policy),
                        _format_mean_std(summary, "final_regret"),
                        _format_mean(summary, "terminal_pareto_optimal", precision=3),
                        _format_mean(summary, "certified", precision=3),
                        _format_scalar(summary.get("certification_round_median", float("nan")), precision=0),
                    ]
                )
                + " \\\\"
            )
            wrote = True
        lines.append("\\midrule")
    if not wrote:
        return
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    with open(_tables_dir(out_dir) / "table_6_1_width_sensitivity.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_certificate_validity_table(
    out_dir: Path,
    rows: Sequence[Dict[str, object]],
    subset_types: Sequence[str],
    dataset_name: str,
    policies: Sequence[str],
    policies_by_type: Optional[Dict[str, Sequence[str]]] = None,
    *,
    empirical_policy: Optional[str] = None,
    theory_policy: str = "width_guided",
) -> None:
    theory_by_key = {
        (row.get("subset_id"), row.get("run_idx"), row.get("reward_seed")): row
        for row in rows
        if row.get("policy") == theory_policy
    }
    if not theory_by_key:
        return
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Validity of empirical Width-guided certificates under the theorem-matched radius. "
            "The same-time column asks whether the empirical-coefficient certificate is also valid under "
            "$c=2$ at that stopping time; the eventual column is the matched $c=2$ certificate rate.}"
        ),
        "\\label{tab:app-cert-validity}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        (
            "Dataset & Subset type & Cert. rate at empirical $c$ & "
            "Same cert. valid under $c=2$ & Eventually cert. under $c=2$ & "
            "Median delay \\\\"
        ),
        "\\midrule",
    ]
    wrote = False
    for subset_type in subset_types:
        type_empirical_policy = _empirical_width_policy_for_type(
            subset_type,
            policies,
            policies_by_type,
            empirical_policy,
        )
        if type_empirical_policy is None:
            continue
        type_label = _latex_escape(SUBSET_TYPE_LABELS.get(subset_type, subset_type))
        type_rows = [
            row
            for row in rows
            if row.get("subset_type") == subset_type
            and row.get("policy") == type_empirical_policy
        ]
        if not type_rows:
            continue
        cert_rate = _safe_mean(_metric_values(type_rows, "certified"))
        same_time = _safe_mean(_metric_values(type_rows, "theory_radius_same_certificate_at_empirical_time"))
        eventual: List[float] = []
        delays: List[float] = []
        for row in type_rows:
            key = (row.get("subset_id"), row.get("run_idx"), row.get("reward_seed"))
            theory_row = theory_by_key.get(key)
            if theory_row is None:
                continue
            theory_metrics = theory_row.get("metrics", {})
            emp_metrics = row.get("metrics", {})
            eventual.append(_metric_float(theory_metrics, "certified"))
            emp_round = _metric_float(emp_metrics, "certification_round")
            theory_round = _metric_float(theory_metrics, "certification_round")
            if np.isfinite(emp_round) and np.isfinite(theory_round):
                delays.append(theory_round - emp_round)
        lines.append(
            " & ".join(
                [
                    _latex_escape(dataset_name),
                    type_label,
                    _format_scalar(cert_rate, precision=3),
                    _format_scalar(same_time, precision=3),
                    _format_scalar(_safe_mean(eventual), precision=3),
                    _format_median_iqr(delays, precision=0),
                ]
            )
            + " \\\\"
        )
        wrote = True
    if not wrote:
        return
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    with open(_tables_dir(out_dir) / "table_6_1_certificate_validity.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_certified_objective_table(
    out_dir: Path,
    rows: Sequence[Dict[str, object]],
    selected_subsets: Sequence[Dict[str, object]],
    subset_types: Sequence[str],
    dataset_name: str,
    policies: Sequence[str],
    policies_by_type: Optional[Dict[str, Sequence[str]]] = None,
    *,
    empirical_policy: Optional[str] = None,
) -> None:
    objective_names: List[str] = []
    for subset in selected_subsets:
        labels = subset.get("labels", {})
        if isinstance(labels, dict) and labels.get("objective_names"):
            objective_names = [str(x) for x in labels["objective_names"]]
            break
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Certified-objective diagnostics for Width-guided at the selected empirical coefficient.}",
        "\\label{tab:app-cert-objectives}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llllcc}",
        "\\toprule",
        "Dataset & Subset type & Top certified objectives & Objective entropy & Median cert. round & IQR \\\\",
        "\\midrule",
    ]
    wrote = False
    for subset_type in subset_types:
        type_empirical_policy = _empirical_width_policy_for_type(
            subset_type,
            policies,
            policies_by_type,
            empirical_policy,
        )
        if type_empirical_policy is None:
            continue
        type_rows = [
            row
            for row in rows
            if row.get("subset_type") == subset_type
            and row.get("policy") == type_empirical_policy
        ]
        cert_objectives = np.asarray(_metric_values(type_rows, "certified_objective"), dtype=float)
        cert_objectives = cert_objectives[np.isfinite(cert_objectives)].astype(int)
        if cert_objectives.size == 0:
            continue
        d = max(int(np.max(cert_objectives)) + 1, len(objective_names))
        counts = np.bincount(cert_objectives, minlength=d).astype(float)
        proportions = counts / float(np.sum(counts))
        top_count = len(proportions) if len(proportions) <= 6 else min(5, len(proportions))
        top_idx = np.argsort(proportions)[::-1][:top_count]
        top_parts = []
        for obj_idx in top_idx:
            if proportions[obj_idx] <= 0:
                continue
            label = objective_names[obj_idx] if obj_idx < len(objective_names) else f"Obj. {obj_idx + 1}"
            top_parts.append(f"{_latex_escape(label)}: {proportions[obj_idx]:.2f}")
        rounds = _metric_values(type_rows, "certification_round")
        q = np.asarray(rounds, dtype=float)
        q = q[np.isfinite(q)]
        iqr = "--"
        if q.size:
            q1, q3 = np.percentile(q, [25, 75])
            iqr = f"[{q1:.0f}, {q3:.0f}]"
        lines.append(
            " & ".join(
                [
                    _latex_escape(dataset_name),
                    _latex_escape(SUBSET_TYPE_LABELS.get(subset_type, subset_type)),
                    "; ".join(top_parts) if top_parts else "--",
                    _format_scalar(_normalized_entropy(counts), precision=3),
                    _format_scalar(float(np.median(q)) if q.size else float("nan"), precision=0),
                    iqr,
                ]
            )
            + " \\\\"
        )
        wrote = True
    if not wrote:
        return
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    with open(_tables_dir(out_dir) / "table_6_1_certified_objectives.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_random_subset_table(
    out_dir: Path,
    aggregate: Dict[str, object],
    policies: Sequence[str],
    dataset_name: str,
    policies_by_type: Optional[Dict[str, Sequence[str]]] = None,
) -> None:
    subset_type = "random_subsets"
    if subset_type not in aggregate:
        return
    show_width_coefficients = len(_width_policies(policies)) > 1
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Random-subset robustness check. Random subsets are selected before policy simulation and are not used for validation.}",
        "\\label{tab:real-random-subsets}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        (
            "Dataset & Method & Pareto regret $\\downarrow$ & "
            "$\\Pr(\\widehat a_T\\in A^\\star)\\uparrow$ & Fairness regret $\\downarrow$ \\\\"
        ),
        "\\midrule",
    ]
    wrote = False
    for policy in _policies_for_type(subset_type, policies, policies_by_type):
        if policy not in aggregate[subset_type]["policies"]:
            continue
        summary = aggregate[subset_type]["policies"][policy]
        if not np.isfinite(summary.get("final_regret_mean", float("nan"))):
            continue
        lines.append(
            " & ".join(
                [
                    _latex_escape(dataset_name),
                    _latex_escape(_policy_label(policy, show_width_coefficient=show_width_coefficients)),
                    _format_mean_std(summary, "final_regret"),
                    _format_mean(summary, "terminal_pareto_optimal", precision=3),
                    _format_mean_std(summary, "fairness_regret"),
                ]
            )
            + " \\\\"
        )
        wrote = True
    if not wrote:
        return
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    with open(_tables_dir(out_dir) / "table_6_1_random_subsets.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _aggregate_real_results(out_dir: Path, *, allow_missing: bool = False) -> Dict[str, object]:
    with open(out_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(out_dir / "selected_subsets.json", "r", encoding="utf-8") as f:
        selected_subsets = json.load(f)
    jobs = _read_jobs(out_dir)
    rows: List[Dict[str, object]] = []
    missing: List[int] = []
    for job in jobs:
        output_path = _job_output_path(out_dir, int(job["job_index"]))
        if not output_path.exists():
            missing.append(int(job["job_index"]))
            continue
        with open(output_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if missing and not allow_missing:
        raise RuntimeError(
            f"Missing {len(missing)} real-data jobs in {out_dir}; "
            f"first missing job indices: {missing[:10]}. "
            "Re-run the missing jobs or pass --allow-missing to write partial diagnostics."
        )

    subset_types = list(config["subset_types"])
    policies = list(config["policies"])
    policies_by_type = config.get("policies_by_type")
    aggregate: Dict[str, object] = {}
    for subset_type in subset_types:
        aggregate[subset_type] = {"policies": {}}
        type_policies = list(policies_by_type[subset_type]) if policies_by_type else policies
        for policy in type_policies:
            metric_rows = [
                row["metrics"]
                for row in rows
                if row["subset_type"] == subset_type and row["policy"] == policy
            ]
            aggregate[subset_type]["policies"][policy] = _metric_summary(metric_rows, ALL_METRICS)

    payload = {
        "config": config,
        "selected_subsets": selected_subsets,
        "n_jobs": len(jobs),
        "n_completed": len(rows),
        "missing_jobs": missing,
        "aggregate_by_type": aggregate,
    }
    summary_text = json.dumps(_json_safe(payload), indent=2, allow_nan=False)
    for path in [out_dir / "summary_6_1.json", _summaries_dir(out_dir) / "summary_6_1.json"]:
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary_text)
    dataset_name = str(config.get("dataset_name", ""))
    _write_main_table(out_dir, aggregate, policies, subset_types, policies_by_type)
    _write_front_table(out_dir, aggregate, policies, subset_types, policies_by_type)
    _write_width_sensitivity_table(out_dir, rows, policies, subset_types, dataset_name)
    _write_certificate_validity_table(
        out_dir,
        rows,
        subset_types,
        dataset_name,
        policies,
        policies_by_type,
    )
    _write_certified_objective_table(
        out_dir,
        rows,
        selected_subsets,
        subset_types,
        dataset_name,
        policies,
        policies_by_type,
    )
    _write_random_subset_table(out_dir, aggregate, policies, dataset_name, policies_by_type)
    return payload


def _clean_real_outputs(out_dir: Path) -> None:
    """Remove generated real-benchmark outputs that can make a new plan stale."""
    for name in ["config.json", "selected_subsets.json", "jobs.jsonl", "summary_6_1.json"]:
        path = out_dir / name
        if path.exists():
            path.unlink()
    for directory_name in ["job_outputs", "tables", "summaries", "slurm_logs"]:
        directory = out_dir / directory_name
        if directory.exists():
            for path in directory.glob("*"):
                if path.is_file():
                    path.unlink()


def _real_outputs_exist(out_dir: Path) -> List[str]:
    """Return generated files that can make a new real-data plan ambiguous."""
    existing = [
        name
        for name in ["config.json", "selected_subsets.json", "jobs.jsonl", "summary_6_1.json"]
        if (out_dir / name).exists()
    ]
    for directory_name in ["job_outputs", "tables", "summaries", "slurm_logs"]:
        directory = out_dir / directory_name
        if directory.exists() and any(directory.iterdir()):
            existing.append(f"{directory_name}/*")
    return existing


def _plan_real_benchmark(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    existing = _real_outputs_exist(out_dir)
    if existing and not args.force_clean:
        raise RuntimeError(
            f"Refusing to overwrite existing real-data outputs in {out_dir}: {existing[:10]}. "
            "Use --force-clean to remove generated outputs before planning a new run."
        )
    if existing and args.force_clean:
        _clean_real_outputs(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job_outputs").mkdir(parents=True, exist_ok=True)
    _tables_dir(out_dir)
    _summaries_dir(out_dir)
    data = np.load(args.instance, allow_pickle=True)
    mu = np.asarray(data["mu"], dtype=float)
    k_values = _split_csv(args.k_values, int)
    subset_types = _split_csv(args.subset_types, str)
    base_policies = _split_csv(args.policies, str)
    policies_by_type = _parse_policies_by_type(args.policies_by_type, subset_types) if args.policies_by_type else None
    policies = _unique_policy_order(
        [policies_by_type[subset_type] for subset_type in subset_types] if policies_by_type else [base_policies]
    )
    if args.frozen_subsets:
        with open(args.frozen_subsets, "r", encoding="utf-8") as f:
            selected = json.load(f)
        requested_types = set(subset_types)
        selected = [item for item in selected if item.get("subset_type") in requested_types]
        if not selected:
            raise ValueError("No frozen subsets match the requested subset types.")
    else:
        selected = _select_subsets(
            mu,
            data=data,
            k_values=k_values,
            n_candidates=int(args.n_candidates),
            enumerate_limit=int(args.enumerate_limit),
            n_subsets=int(args.n_subsets),
            subset_types=subset_types,
            t_horizon=int(args.T),
            seed=int(args.seed),
            near_gap=float(args.near_gap),
            tiny_gap=float(args.tiny_gap),
            gap_tol=float(args.gap_tol),
            selection_mode=str(args.selection_mode),
            contamination_samples_per_arm=int(args.contamination_samples_per_arm),
            contamination_bootstraps=int(args.contamination_bootstraps),
            contamination_prefilter=int(args.contamination_prefilter),
        )
    for item in selected:
        item["labels"] = _load_labels(data, item["indices"])

    config = {
        "dataset_name": args.dataset_name,
        "instance": str(Path(args.instance)),
        "T": int(args.T),
        "n_runs": int(args.n_runs),
        "seed": int(args.seed),
        "k_values": k_values,
        "n_subsets": int(args.n_subsets),
        "n_candidates": int(args.n_candidates),
        "enumerate_limit": int(args.enumerate_limit),
        "subset_types": subset_types,
        "policies": policies,
        "policies_by_type": policies_by_type,
        "frozen_subsets": str(Path(args.frozen_subsets)) if args.frozen_subsets else "",
        "near_gap": float(args.near_gap),
        "tiny_gap": float(args.tiny_gap),
        "gap_tol": float(args.gap_tol),
        "selection_mode": str(args.selection_mode),
        "contamination_samples_per_arm": int(args.contamination_samples_per_arm),
        "contamination_bootstraps": int(args.contamination_bootstraps),
        "contamination_prefilter": int(args.contamination_prefilter),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(config), f, indent=2, allow_nan=False)
    with open(out_dir / "selected_subsets.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(selected), f, indent=2, allow_nan=False)

    jobs_path = out_dir / "jobs.jsonl"
    job_index = 0
    with open(jobs_path, "w", encoding="utf-8") as f:
        for subset in selected:
            type_policies = policies_by_type[subset["subset_type"]] if policies_by_type else base_policies
            for run_idx in range(int(args.n_runs)):
                reward_seed = int(args.seed + 1_000_000 * int(subset["subset_id"]) + run_idx)
                for policy_idx, policy in enumerate(type_policies):
                    job = {
                        "job_index": job_index,
                        "dataset_name": args.dataset_name,
                        "instance": str(Path(args.instance)),
                        "T": int(args.T),
                        "subset_id": int(subset["subset_id"]),
                        "subset_type": subset["subset_type"],
                        "subset_type_label": subset["subset_type_label"],
                        "k": int(subset["k"]),
                        "indices": subset["indices"],
                        "run_idx": int(run_idx),
                        "policy": policy,
                        "reward_seed": reward_seed,
                        "policy_seed": int(args.seed + 10_000_000 * int(subset["subset_id"]) + 10_000 * run_idx + policy_idx),
                        "gap_tol": float(args.gap_tol),
                    }
                    f.write(json.dumps(_json_safe(job), allow_nan=False) + "\n")
                    job_index += 1
    print(f"Selected {len(selected)} subsets and wrote {job_index} jobs to {jobs_path}", flush=True)


def _run_local(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    jobs = _read_jobs(out_dir)
    if args.limit is not None:
        jobs = jobs[: int(args.limit)]
    _run_job_batch(out_dir, jobs, workers=int(args.workers), label="local")


def _run_shard(args: argparse.Namespace) -> None:
    if int(args.n_shards) <= 0:
        raise ValueError("--n-shards must be positive.")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.n_shards):
        raise ValueError("--shard-index must be in [0, n_shards - 1].")
    out_dir = Path(args.outdir)
    all_jobs = _read_jobs(out_dir)
    jobs = _jobs_for_shard(
        all_jobs,
        shard_index=int(args.shard_index),
        n_shards=int(args.n_shards),
        assignment=str(args.assignment),
        seed=int(args.assignment_seed),
    )
    if args.limit is not None:
        jobs = jobs[: int(args.limit)]
    print(
        f"starting shard {args.shard_index}/{args.n_shards} with {len(jobs)} jobs "
        f"and {int(args.workers)} workers",
        flush=True,
    )
    _run_job_batch(out_dir, jobs, workers=int(args.workers), label=f"shard {args.shard_index}")


def _jobs_for_shard(
    jobs: Sequence[Dict[str, object]],
    *,
    shard_index: int,
    n_shards: int,
    assignment: str,
    seed: int,
) -> List[Dict[str, object]]:
    if assignment == "modulo":
        return [job for job in jobs if int(job["job_index"]) % n_shards == shard_index]
    if assignment != "balanced":
        raise ValueError(f"Unknown shard assignment {assignment!r}.")

    rng = np.random.default_rng(seed)
    policy_order = []
    jobs_by_policy: Dict[str, List[Dict[str, object]]] = {}
    for job in jobs:
        policy = str(job["policy"])
        if policy not in jobs_by_policy:
            policy_order.append(policy)
            jobs_by_policy[policy] = []
        jobs_by_policy[policy].append(job)

    shard_jobs: List[Dict[str, object]] = []
    for policy in policy_order:
        group = list(jobs_by_policy[policy])
        order = rng.permutation(len(group))
        for pos, group_idx in enumerate(order.tolist()):
            if pos % n_shards == shard_index:
                shard_jobs.append(group[int(group_idx)])
    return sorted(shard_jobs, key=lambda job: int(job["job_index"]))


def _run_job_batch(out_dir: Path, jobs: Sequence[Dict[str, object]], *, workers: int, label: str) -> None:
    if not jobs:
        print(f"{label}: no jobs to run", flush=True)
        return
    if int(workers) <= 1:
        for job in jobs:
            result = _run_job_spec(str(out_dir), job)
            print(f"{label}: finished job {result['job_index']}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(_run_job_spec, str(out_dir), job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            print(f"{label}: finished job {result['job_index']}", flush=True)


def main() -> None:
    args = _parse_args()
    if args.command == "plan":
        _plan_real_benchmark(args)
    elif args.command == "run-job":
        out_dir = Path(args.outdir)
        jobs = _read_jobs(out_dir)
        if args.job_index < 0 or args.job_index >= len(jobs):
            raise IndexError(f"job-index must be in [0, {len(jobs) - 1}]")
        result = _run_job_spec(str(out_dir), jobs[int(args.job_index)])
        print(f"finished job {result['job_index']}", flush=True)
    elif args.command == "run-local":
        _run_local(args)
    elif args.command == "run-shard":
        _run_shard(args)
    elif args.command == "aggregate":
        payload = _aggregate_real_results(Path(args.outdir), allow_missing=bool(args.allow_missing))
        print(
            f"Aggregated {payload['n_completed']} / {payload['n_jobs']} jobs into "
            f"{Path(args.outdir) / 'summary_6_1.json'}",
            flush=True,
        )
    else:
        raise ValueError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
