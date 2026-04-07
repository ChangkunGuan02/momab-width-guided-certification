"""Shared utilities for the synthetic MOMAB experiments in the NeurIPS draft."""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


SCHEMA_VERSION = 1
EXPERIMENT_FAMILY_NAME = "synthetic_family_fixed_certification_gap_and_small_pareto_gaps"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = PROJECT_DIR / "computational_results" / "momab_synthetic_experiments"
DEFAULT_PLOTS_DIR = PROJECT_DIR / "plots"

MANIFEST_NAME = "manifest.json"
RUN_CACHE_NAME = "run_cache.npz"
TRAJECTORY_CACHE_NAME = "trajectory_cache.npz"
SUMMARY_NAME = "summary.json"
TRAJECTORY_SUMMARY_NAME = "trajectory_summary.json"
RESULTS_TABLE_NAME = "results_table.tex"
DERIVED_MANIFEST_NAME = "derived_manifest.json"

SIMULATION_SOURCE_FILES = [
    "momab_synthetic_core.py",
    "run_synthetic_experiments.py",
]

PLOTTING_SOURCE_FILES = [
    "momab_synthetic_core.py",
    "plot_synthetic_experiments.py",
]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def simulation_source_hashes() -> Dict[str, str]:
    """Hash the files that determine the raw simulation outputs."""
    code_dir = Path(__file__).resolve().parent
    return {name: file_sha256(code_dir / name) for name in SIMULATION_SOURCE_FILES}


def plotting_source_hashes() -> Dict[str, str]:
    """Hash the files that determine the derived summaries and figures."""
    code_dir = Path(__file__).resolve().parent
    return {name: file_sha256(code_dir / name) for name in PLOTTING_SOURCE_FILES}


def provenance_metadata() -> Dict[str, object]:
    """Collect runtime metadata for the raw simulation manifest."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "simulation_source_hashes": simulation_source_hashes(),
    }


def run_cache_key(index: int, field: str) -> str:
    """Create a stable key for packing per-setting arrays into an NPZ file."""
    return f"setting_{index}_{field}"


def _random_argmax(x: np.ndarray, rng: np.random.Generator) -> int:
    """Break exact ties in an argmax uniformly at random."""
    max_value = np.max(x)
    candidates = np.flatnonzero(np.isclose(x, max_value, atol=1e-12, rtol=0.0))
    return int(rng.choice(candidates))


def _random_argmax_excluding(x: np.ndarray, exclude: int, rng: np.random.Generator) -> int:
    """Sample an argmax after excluding a single index."""
    mask = np.ones(len(x), dtype=bool)
    mask[exclude] = False
    values = x[mask]
    max_value = np.max(values)
    idx_local = np.flatnonzero(np.isclose(values, max_value, atol=1e-12, rtol=0.0))
    idx_global = np.flatnonzero(mask)[idx_local]
    return int(rng.choice(idx_global))


def dominates(a: np.ndarray, b: np.ndarray, atol: float = 1e-12) -> bool:
    """Return whether vector ``a`` Pareto-dominates vector ``b``."""
    return np.all(a >= b - atol) and np.any(a > b + atol)


def pareto_nondominated_indices(vectors: np.ndarray, atol: float = 1e-12) -> List[int]:
    """Return the indices of the Pareto-nondominated rows of ``vectors``."""
    n_arms = vectors.shape[0]
    nondominated: List[int] = []
    for i in range(n_arms):
        dominated_flag = False
        for j in range(n_arms):
            if i == j:
                continue
            if dominates(vectors[j], vectors[i], atol=atol):
                dominated_flag = True
                break
        if not dominated_flag:
            nondominated.append(i)
    return nondominated


def pareto_arm_regrets(mu: np.ndarray, opt_idx: List[int], atol: float = 1e-12) -> np.ndarray:
    """Compute Pareto suboptimality gaps for all arms from their mean vectors."""
    n_arms, _ = mu.shape
    opt_mask = np.zeros(n_arms, dtype=bool)
    opt_mask[opt_idx] = True
    deltas = np.zeros(n_arms, dtype=float)
    for i in range(n_arms):
        if opt_mask[i]:
            continue
        eps_needed_all = []
        for h in opt_idx:
            diff = mu[h] - mu[i]
            if np.all(diff >= -atol) and np.any(diff > atol):
                eps_h = float(np.min(diff[diff > atol]))
                eps_needed_all.append(eps_h)
        deltas[i] = float(np.max(eps_needed_all)) if eps_needed_all else 0.0
    return deltas


def objective_winner_gaps(mu: np.ndarray) -> np.ndarray:
    """Return the winner-versus-runner-up gap for each objective."""
    n_arms, n_obj = mu.shape
    if n_arms < 2:
        raise ValueError("At least two arms are required.")
    gaps = np.zeros(n_obj, dtype=float)
    for j in range(n_obj):
        top_two = np.sort(mu[:, j])[-2:]
        gaps[j] = float(top_two[1] - top_two[0])
    return gaps


def _pucb_comparison_terms(mu: np.ndarray, t_horizon: int) -> Optional[Dict[str, object]]:
    """Collect the shared instance statistics used in the PUCB coefficient formulas."""
    opt_idx = pareto_nondominated_indices(mu)
    delta = pareto_arm_regrets(mu, opt_idx)
    subopt = delta[delta > 0]
    if subopt.size == 0:
        return None

    g_dagger = float(np.max(objective_winner_gaps(mu)))
    n_obj = mu.shape[1]
    log_ratio = np.log(max(2.0, t_horizon * (n_obj * len(opt_idx)) ** 0.25)) / np.log(max(2.0, t_horizon))
    return {
        "opt_idx": opt_idx,
        "subopt": subopt,
        "g_dagger": g_dagger,
        "log_ratio": float(log_ratio),
    }


def compute_exact_pucb_coefficient(mu: np.ndarray, t_horizon: int) -> float:
    """Compute the exact normalized Pareto-UCB1 leading coefficient."""
    terms = _pucb_comparison_terms(mu, t_horizon)
    if terms is None:
        return 0.0
    return float(
        8.0
        * (float(terms["g_dagger"]) / mu.shape[0])
        * np.sum(1.0 / np.asarray(terms["subopt"], dtype=float))
        * float(terms["log_ratio"])
    )


def compute_pucb_envelope_coefficient(mu: np.ndarray, t_horizon: int) -> float:
    """Compute the coarse envelope coefficient used in the paper discussion."""
    terms = _pucb_comparison_terms(mu, t_horizon)
    if terms is None:
        return 0.0
    opt_idx = list(terms["opt_idx"])
    subopt = np.asarray(terms["subopt"], dtype=float)
    delta_min = float(np.min(subopt))
    return float(
        8.0
        * ((mu.shape[0] - len(opt_idx)) / mu.shape[0])
        * (float(terms["g_dagger"]) / delta_min)
        * float(terms["log_ratio"])
    )


@dataclass
class PrecomputedBernoulliBandit:
    """Bandit environment backed by a pre-sampled Bernoulli reward table."""

    mu: np.ndarray
    reward_table: np.ndarray

    def __post_init__(self) -> None:
        """Normalize inputs and initialize arm pull counts."""
        self.mu = np.asarray(self.mu, dtype=float)
        self.reward_table = np.asarray(self.reward_table, dtype=float)
        self.pull_counts = np.zeros(self.mu.shape[0], dtype=int)

    @property
    def k(self) -> int:
        """Return the number of arms."""
        return int(self.mu.shape[0])

    @property
    def d(self) -> int:
        """Return the number of objectives."""
        return int(self.mu.shape[1])

    def pull(self, arm: int) -> np.ndarray:
        """Return the next pre-sampled reward vector for ``arm``."""
        arm = int(arm)
        idx = int(self.pull_counts[arm])
        reward = self.reward_table[arm, idx].copy()
        self.pull_counts[arm] += 1
        return reward


@dataclass
class BanditRunResult:
    """Container for one simulated bandit run."""

    cum_regret: np.ndarray
    opt_indices: List[int]
    delta: np.ndarray
    debug: Optional[Dict[str, np.ndarray]] = None


def run_pareto_ucb1(
    env: PrecomputedBernoulliBandit,
    t_horizon: int,
    *,
    seed: Optional[int] = None,
) -> BanditRunResult:
    """Run Pareto UCB1 on a precomputed stochastic environment."""
    rng = np.random.default_rng(seed)
    k, d = env.k, env.d
    mu = env.mu

    opt_idx = pareto_nondominated_indices(mu)
    delta = pareto_arm_regrets(mu, opt_idx)

    counts = np.zeros(k, dtype=int)
    sums = np.zeros((k, d), dtype=float)
    cum_regret = np.zeros(t_horizon, dtype=float)

    t = 0
    for arm in range(k):
        if t >= t_horizon:
            break
        reward = env.pull(arm)
        counts[arm] += 1
        sums[arm] += reward
        cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0.0) + delta[arm]
        t += 1

    while t < t_horizon:
        n = t
        means = sums / counts[:, None]
        inside = max(2.0, n * (4.0 * d * k) ** 0.25)
        bonus = np.sqrt(2.0 * np.log(inside) / counts)
        ucb_vectors = means + bonus[:, None]

        candidate_idx = pareto_nondominated_indices(ucb_vectors)
        chosen = int(rng.choice(candidate_idx))
        reward = env.pull(chosen)
        counts[chosen] += 1
        sums[chosen] += reward
        cum_regret[t] = cum_regret[t - 1] + delta[chosen]
        t += 1

    return BanditRunResult(
        cum_regret=cum_regret,
        opt_indices=opt_idx,
        delta=delta,
        debug=None,
    )


def run_width_guided_policy(
    env: PrecomputedBernoulliBandit,
    t_horizon: int,
    *,
    seed: Optional[int] = None,
    return_debug: bool = False,
) -> BanditRunResult:
    """Run the width-guided first-certification policy from the paper."""
    rng = np.random.default_rng(seed)
    k, d = env.k, env.d
    mu = env.mu
    bonus_log = np.log(max(2, t_horizon))

    opt_idx = pareto_nondominated_indices(mu)
    delta = pareto_arm_regrets(mu, opt_idx)
    winner_gaps = objective_winner_gaps(mu)
    champion_objective = int(np.argmax(winner_gaps))

    counts = np.zeros(k, dtype=int)
    sums = np.zeros((k, d), dtype=float)
    cum_regret = np.zeros(t_horizon, dtype=float)
    debug: Optional[Dict[str, np.ndarray]] = None

    if return_debug:
        debug = {
            "chosen_objective": -np.ones(t_horizon, dtype=int),
            "is_champion_objective": np.zeros(t_horizon, dtype=bool),
            "pair_leader": -np.ones(t_horizon, dtype=int),
            "pair_challenger": -np.ones(t_horizon, dtype=int),
            "selected_arm": -np.ones(t_horizon, dtype=int),
            "objective_certified": np.zeros(t_horizon, dtype=bool),
            "pair_width": np.zeros(t_horizon, dtype=float),
        }

    t = 0
    for arm in range(k):
        if t >= t_horizon:
            break
        reward = env.pull(arm)
        counts[arm] += 1
        sums[arm] += reward
        cum_regret[t] = (cum_regret[t - 1] if t > 0 else 0.0) + delta[arm]
        if debug is not None:
            debug["selected_arm"][t] = arm
        t += 1

    certified_objective: Optional[int] = None
    certified_leader: Optional[int] = None

    while t < t_horizon:
        means = sums / counts[:, None]
        radius = np.sqrt(2.0 * bonus_log / counts)
        ucb = means + radius[:, None]
        lcb = means - radius[:, None]

        leaders = np.zeros(d, dtype=int)
        challengers = np.zeros(d, dtype=int)
        pair_width = np.zeros(d, dtype=float)
        certified = np.zeros(d, dtype=bool)

        for j in range(d):
            # Each objective maintains its own optimistic top-two race.
            leader = _random_argmax(ucb[:, j], rng)
            challenger = _random_argmax_excluding(ucb[:, j], exclude=leader, rng=rng)
            leaders[j] = leader
            challengers[j] = challenger
            pair_width[j] = radius[leader] + radius[challenger]
            certified[j] = bool(lcb[leader, j] >= ucb[challenger, j])

        if certified_objective is None and np.any(certified):
            # Once any objective certifies its leader, the policy commits forever.
            cert_idx = np.flatnonzero(certified)
            pick_local = _random_argmax(pair_width[cert_idx], rng)
            certified_objective = int(cert_idx[pick_local])
            certified_leader = int(leaders[certified_objective])

        if certified_objective is not None:
            chosen_objective = int(certified_objective)
            chosen_arm = int(certified_leader)
        else:
            # Before certification, sample the widest unresolved objective-level race.
            chosen_objective = int(_random_argmax(pair_width, rng))
            pair = np.array([leaders[chosen_objective], challengers[chosen_objective]], dtype=int)
            chosen_arm = int(pair[_random_argmax(radius[pair], rng)])

        reward = env.pull(chosen_arm)
        counts[chosen_arm] += 1
        sums[chosen_arm] += reward
        cum_regret[t] = cum_regret[t - 1] + delta[chosen_arm]

        if debug is not None:
            debug["chosen_objective"][t] = chosen_objective
            debug["is_champion_objective"][t] = chosen_objective == champion_objective
            debug["pair_leader"][t] = leaders[chosen_objective]
            debug["pair_challenger"][t] = challengers[chosen_objective]
            debug["selected_arm"][t] = chosen_arm
            debug["objective_certified"][t] = certified_objective is not None
            debug["pair_width"][t] = pair_width[chosen_objective]
        t += 1

    return BanditRunResult(
        cum_regret=cum_regret,
        opt_indices=opt_idx,
        delta=delta,
        debug=debug,
    )


def build_synthetic_instance(delta: float, crowd_size: int, total_arms: int = 20) -> np.ndarray:
    """Build the synthetic two-frontier-arm instance family used in the paper."""
    if not (0.0 < delta < 0.20):
        raise ValueError("delta must lie in (0, 0.20).")
    if not (1 <= crowd_size <= total_arms - 2):
        raise ValueError("crowd_size must leave room for the two Pareto-optimal arms.")

    p = 0.25
    g = 0.55
    eta = 0.20
    filler_level = 0.05

    # The first two arms form the Pareto frontier and fix the certification gap.
    front = np.array(
        [
            [p + g, p],
            [p, p + g],
        ],
        dtype=float,
    )
    # Crowd arms stay close to the frontier through one coordinate, but never
    # become runner-up on the certifying objective.
    crowd = np.tile(
        np.array([[p - eta, p + g - delta]], dtype=float),
        (crowd_size, 1),
    )
    filler_count = total_arms - front.shape[0] - crowd_size
    filler = np.tile(np.array([[filler_level, filler_level]], dtype=float), (filler_count, 1))
    return np.vstack([front, crowd, filler])


def build_final_regret_settings() -> List[Dict[str, object]]:
    """Return the settings used in the final-regret comparison table and figure."""
    experiments: List[Dict[str, object]] = []
    for delta in [0.12, 0.08, 0.04, 0.02, 0.01]:
        experiments.append(
            {
                "family": "delta_variation",
                "label": f"({delta:.2f}, 1)",
                "delta": float(delta),
                "crowd_size": 1,
            }
        )
    for crowd_size in [4, 8, 12]:
        experiments.append(
            {
                "family": "crowd_variation",
                "label": f"(0.02, {crowd_size})",
                "delta": 0.02,
                "crowd_size": int(crowd_size),
            }
        )
    return experiments


def build_trajectory_settings() -> List[Dict[str, object]]:
    """Return the representative settings used in the trajectory figure."""
    return [
        {
            "label": "(0.12, 1)",
            "description": r"$(\delta,m)=(0.12,1)$",
            "family": "delta_variation",
            "delta": 0.12,
            "crowd_size": 1,
        },
        {
            "label": "(0.02, 8)",
            "description": r"$(\delta,m)=(0.02,8)$",
            "family": "crowd_variation",
            "delta": 0.02,
            "crowd_size": 8,
        },
    ]


def summarize_instance_metadata(mu: np.ndarray, t_horizon: int) -> Dict[str, object]:
    """Compute paper-facing metadata for one synthetic instance."""
    opt_idx = pareto_nondominated_indices(mu)
    delta = pareto_arm_regrets(mu, opt_idx)
    subopt = delta[delta > 0]
    gaps = objective_winner_gaps(mu)
    return {
        "k": int(mu.shape[0]),
        "d": int(mu.shape[1]),
        "pareto_size": int(len(opt_idx)),
        "gaps_by_objective": [float(x) for x in gaps.tolist()],
        "g_dagger": float(np.max(gaps)),
        "delta_min_p": float(np.min(subopt)) if subopt.size > 0 else 0.0,
        "sum_inv_delta_p": float(np.sum(1.0 / subopt)) if subopt.size > 0 else 0.0,
        "c_pucb_exact": float(compute_exact_pucb_coefficient(mu, t_horizon)),
        "c_pucb_envelope": float(compute_pucb_envelope_coefficient(mu, t_horizon)),
    }
