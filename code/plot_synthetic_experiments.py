"""Plotting/reporting CLI for the synthetic MOMAB experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from momab_synthetic_core import (
    DEFAULT_OUTDIR,
    DEFAULT_PLOTS_DIR,
    DERIVED_MANIFEST_NAME,
    MANIFEST_NAME,
    RESULTS_TABLE_NAME,
    RUN_CACHE_NAME,
    SCHEMA_VERSION,
    SUMMARY_NAME,
    TRAJECTORY_CACHE_NAME,
    TRAJECTORY_SUMMARY_NAME,
    plotting_source_hashes,
    run_cache_key,
    simulation_source_hashes,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the plotting/reporting CLI."""
    parser = argparse.ArgumentParser(
        description="Generate tables and manuscript plots from cached Pareto UCB1 vs width-guided simulations."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Directory containing manifest.json, run_cache.npz, and trajectory_cache.npz.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help=(
            "Directory in which the PDF figures will be written. "
            "Default: ../plots/ when using the canonical outdir, otherwise <outdir>/plots/."
        ),
    )
    return parser.parse_args()


def _import_plotting_backend():
    """Import Matplotlib lazily so --help stays lightweight."""
    mpl_config_dir = Path(__file__).resolve().parents[1] / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, matplotlib.__version__


def _load_manifest(out_dir: Path) -> Dict[str, object]:
    """Load and validate the simulation manifest."""
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible cache schema in {manifest_path}. "
            f"Expected schema_version={SCHEMA_VERSION}, found {manifest.get('schema_version')}."
        )
    return manifest


def _validate_simulation_sources(manifest: Dict[str, object]) -> None:
    """Reject stale raw caches whose simulation-side code hashes no longer match."""
    cached_hashes = (
        manifest.get("provenance", {}).get("simulation_source_hashes", {})
        if isinstance(manifest.get("provenance"), dict)
        else {}
    )
    current_hashes = simulation_source_hashes()
    mismatches = []
    for name, current_hash in current_hashes.items():
        cached_hash = cached_hashes.get(name)
        if cached_hash != current_hash:
            mismatches.append(name)
    if mismatches:
        joined = ", ".join(sorted(mismatches))
        raise ValueError(
            "Cached simulation results do not match the current simulation code "
            f"for: {joined}. Please rerun run_synthetic_experiments.py before plotting."
        )


def _load_caches(out_dir: Path) -> Tuple[Dict[str, object], np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    """Load the raw NPZ caches declared by the manifest."""
    manifest = _load_manifest(out_dir)
    _validate_simulation_sources(manifest)

    files = manifest.get("files", {}) if isinstance(manifest.get("files"), dict) else {}
    run_cache_name = str(files.get("run_cache", RUN_CACHE_NAME))
    trajectory_cache_name = str(files.get("trajectory_cache", TRAJECTORY_CACHE_NAME))
    run_cache_path = out_dir / run_cache_name
    trajectory_cache_path = out_dir / trajectory_cache_name
    if not run_cache_path.exists():
        raise FileNotFoundError(f"Missing run cache: {run_cache_path}")
    if not trajectory_cache_path.exists():
        raise FileNotFoundError(f"Missing trajectory cache: {trajectory_cache_path}")

    return manifest, np.load(run_cache_path), np.load(trajectory_cache_path)


def _sample_std(values: np.ndarray) -> float:
    """Return the sample standard deviation of a one-dimensional array."""
    ddof = 1 if values.size > 1 else 0
    return float(np.std(values, ddof=ddof))


def _sample_se(values: np.ndarray) -> np.ndarray:
    """Return the pointwise sample standard error for path-valued runs."""
    ddof = 1 if values.shape[0] > 1 else 0
    denom = max(float(np.sqrt(values.shape[0])), 1.0)
    return np.std(values, axis=0, ddof=ddof) / denom


def _safe_mean(values: np.ndarray) -> float:
    """Return the mean of a possibly empty array."""
    return float(np.mean(values)) if values.size else float("nan")


def _safe_median(values: np.ndarray) -> float:
    """Return the median of a possibly empty array."""
    return float(np.median(values)) if values.size else float("nan")


def _make_json_serializable(value):
    """Convert nested NumPy-rich structures into strict JSON-compatible data."""
    if isinstance(value, dict):
        return {key: _make_json_serializable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return _make_json_serializable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _format_horizon_for_latex(t_horizon: int) -> str:
    """Format the experiment horizon for LaTeX table output."""
    return str(int(t_horizon))


def _summarize_final_regret_runs(manifest: Dict[str, object], run_cache: np.lib.npyio.NpzFile) -> List[Dict[str, object]]:
    """Aggregate final-regret runs into paper-facing summary rows."""
    rows: List[Dict[str, object]] = []
    for setting in manifest["final_settings"]:
        idx = int(setting["index"])
        p_regret = np.asarray(run_cache[run_cache_key(idx, "pareto_regret_final")], dtype=float)
        w_regret = np.asarray(run_cache[run_cache_key(idx, "width_regret_final")], dtype=float)
        cert_flag = np.asarray(run_cache[run_cache_key(idx, "certified_flag")], dtype=float)
        cert_round = np.asarray(run_cache[run_cache_key(idx, "certified_round")], dtype=float)
        observed_cert_round = cert_round[np.isfinite(cert_round)]

        row = {
            "family": setting["family"],
            "label": setting["label"],
            "delta": float(setting["delta"]),
            "crowd_size": int(setting["crowd_size"]),
            "k": int(setting["k"]),
            "d": int(setting["d"]),
            "pareto_size": int(setting["pareto_size"]),
            "gaps_by_objective": list(setting["gaps_by_objective"]),
            "g_dagger": float(setting["g_dagger"]),
            "delta_min_p": float(setting["delta_min_p"]),
            "sum_inv_delta_p": float(setting["sum_inv_delta_p"]),
            "c_pucb_exact": float(setting["c_pucb_exact"]),
            "c_pucb_envelope": float(setting["c_pucb_envelope"]),
            "pareto_ucb1_regret_mean": _safe_mean(p_regret),
            "pareto_ucb1_regret_std": _sample_std(p_regret),
            "width_guided_regret_mean": _safe_mean(w_regret),
            "width_guided_regret_std": _sample_std(w_regret),
            "width_certified_rate": _safe_mean(cert_flag),
            "width_certified_runs": int(np.sum(cert_flag)),
            "width_certified_round_mean_observed": _safe_mean(observed_cert_round),
            "width_certified_round_median_observed": _safe_median(observed_cert_round),
            "regret_ratio_pucb_over_width": float(np.mean(p_regret) / max(np.mean(w_regret), 1e-12)),
        }
        rows.append(row)
    return rows


def _summarize_trajectory_runs(
    manifest: Dict[str, object],
    trajectory_cache: np.lib.npyio.NpzFile,
) -> List[Dict[str, object]]:
    """Aggregate trajectory runs into figure-ready summaries."""
    items: List[Dict[str, object]] = []
    t_horizon = int(manifest["config"]["T"])
    rounds = np.arange(1, t_horizon + 1, dtype=float)
    for setting in manifest["trajectory_settings"]:
        idx = int(setting["index"])
        p_paths = np.asarray(trajectory_cache[run_cache_key(idx, "pareto_regret_paths")], dtype=float)
        w_paths = np.asarray(trajectory_cache[run_cache_key(idx, "width_regret_paths")], dtype=float)
        cert_round = np.asarray(trajectory_cache[run_cache_key(idx, "certified_round")], dtype=float)

        # Non-certified runs remain censored (NaN) in the raw cache. For the
        # curve we convert them into a certification fraction over time.
        cert_fraction = np.zeros(t_horizon, dtype=float)
        for t_idx, round_number in enumerate(rounds):
            cert_fraction[t_idx] = float(np.mean(np.isfinite(cert_round) & (cert_round <= round_number)))

        observed_cert_round = cert_round[np.isfinite(cert_round)]
        items.append(
            {
                "index": idx,
                "label": setting["label"],
                "description": setting["description"],
                "family": setting["family"],
                "delta": float(setting["delta"]),
                "crowd_size": int(setting["crowd_size"]),
                "g_dagger": float(setting["g_dagger"]),
                "delta_min_p": float(setting["delta_min_p"]),
                "c_pucb_exact": float(setting["c_pucb_exact"]),
                "rounds": rounds.copy(),
                "pareto_ucb1_regret_mean": p_paths.mean(axis=0),
                "pareto_ucb1_regret_se": _sample_se(p_paths),
                "width_guided_regret_mean": w_paths.mean(axis=0),
                "width_guided_regret_se": _sample_se(w_paths),
                "width_certified_fraction": cert_fraction,
                "width_certified_runs": int(np.sum(np.isfinite(cert_round))),
                "width_certified_round_mean_observed": _safe_mean(observed_cert_round),
                "width_certified_round_median_observed": _safe_median(observed_cert_round),
            }
        )
    return items


def _write_summary_outputs(out_dir: Path, manifest: Dict[str, object], rows: List[Dict[str, object]]) -> None:
    """Write the JSON summary and LaTeX table derived from final-regret runs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable({"config": manifest["config"], "results": rows}), f, indent=2)

    t_horizon = int(manifest["config"]["T"])
    with open(out_dir / RESULTS_TABLE_NAME, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Synthetic family with fixed certification gap and small Pareto gaps "
            f"at $T={_format_horizon_for_latex(t_horizon)}$; $g^\\dagger=0.55$ throughout and "
            "$\\delta=\\Delta_{\\min}^{\\mathrm{P}}$. Entries report mean $\\pm$ standard deviation over runs.}\n"
        )
        f.write("\\label{tab:neurips-pucb-vs-width}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{cccccc}\n")
        f.write("\\hline\n")
        f.write("$\\delta$ & $m$ & $C_{\\mathrm{PUCB}}$ & Pareto UCB1 regret & Width-guided regret & Cert. rate \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{row['delta_min_p']:.3f} & "
                f"{int(row['crowd_size'])} & "
                f"{row['c_pucb_exact']:.2f} & "
                f"{row['pareto_ucb1_regret_mean']:.2f} $\\pm$ {row['pareto_ucb1_regret_std']:.2f} & "
                f"{row['width_guided_regret_mean']:.2f} $\\pm$ {row['width_guided_regret_std']:.2f} & "
                f"{100.0 * row['width_certified_rate']:.1f}\\% \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def _write_trajectory_summary(out_dir: Path, manifest: Dict[str, object], items: List[Dict[str, object]]) -> None:
    """Write the JSON trajectory summary used for inspection and auditability."""
    payload = {
        "config": manifest["config"],
        "settings": [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "rounds",
                    "pareto_ucb1_regret_mean",
                    "pareto_ucb1_regret_se",
                    "width_guided_regret_mean",
                    "width_guided_regret_se",
                    "width_certified_fraction",
                }
            }
            for item in items
        ],
    }
    with open(out_dir / TRAJECTORY_SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(payload), f, indent=2)


def _write_derived_manifest(
    out_dir: Path,
    manifest: Dict[str, object],
    plots_dir: Path,
    matplotlib_version: str,
) -> None:
    """Record provenance for the derived summaries and figures."""
    relative_plots_dir = Path(os.path.relpath(plots_dir, start=out_dir))
    derived_manifest = {
        "source_manifest": MANIFEST_NAME,
        "source_files": manifest.get("files", {}),
        "plotting_source_hashes": plotting_source_hashes(),
        "matplotlib_version": matplotlib_version,
        "derived_outputs": {
            "summary": SUMMARY_NAME,
            "trajectory_summary": TRAJECTORY_SUMMARY_NAME,
            "results_table": RESULTS_TABLE_NAME,
            "plots_dir": str(relative_plots_dir),
        },
    }
    with open(out_dir / DERIVED_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(derived_manifest, f, indent=2)


def _write_comparison_figure(rows: List[Dict[str, object]], plots_dir: Path, plt) -> None:
    """Render the two-panel comparison figure used in the paper."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    delta_rows = sorted(
        [row for row in rows if row["family"] == "delta_variation"],
        key=lambda row: float(row["delta"]),
        reverse=True,
    )
    crowd_rows = []
    for row in rows:
        if row["family"] == "crowd_variation":
            crowd_rows.append(row)
        elif row["family"] == "delta_variation" and int(row["crowd_size"]) == 1 and abs(float(row["delta"]) - 0.02) < 1e-12:
            # Include the shared (delta=0.02, crowd_size=1) anchor so the crowd-size
            # panel starts from the same baseline used in the paper discussion.
            crowd_rows.append(row)
    crowd_rows = sorted(crowd_rows, key=lambda row: int(row["crowd_size"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    legend_handles = None
    legend_labels = None

    if delta_rows:
        ax = axes[0]
        x = np.array([float(row["delta_min_p"]) for row in delta_rows], dtype=float)
        regret_p = np.array([float(row["pareto_ucb1_regret_mean"]) for row in delta_rows], dtype=float)
        regret_w = np.array([float(row["width_guided_regret_mean"]) for row in delta_rows], dtype=float)
        c_exact = np.array([float(row["c_pucb_exact"]) for row in delta_rows], dtype=float)

        ax.plot(x, regret_p, marker="o", linewidth=2.0, color="#c23b22", label="Pareto UCB1 regret")
        ax.plot(x, regret_w, marker="s", linewidth=2.0, color="#1b6ca8", label="Width-guided regret")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$\Delta_{\min}^{\mathrm{P}}$")
        ax.set_ylabel("Final Pareto regret")
        ax.set_title(r"Varying $\delta$ ($m=1$)")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

        ax2 = ax.twinx()
        ax2.plot(
            x,
            c_exact,
            marker="^",
            linewidth=1.8,
            linestyle="--",
            color="#444444",
            label=r"$C_{\mathrm{PUCB}}$",
        )
        ax2.set_ylabel(r"$C_{\mathrm{PUCB}}$")

        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        if legend_handles is None:
            legend_handles, legend_labels = lines, labels

    if crowd_rows:
        ax = axes[1]
        x = np.array([int(row["crowd_size"]) for row in crowd_rows], dtype=int)
        regret_p = np.array([float(row["pareto_ucb1_regret_mean"]) for row in crowd_rows], dtype=float)
        regret_w = np.array([float(row["width_guided_regret_mean"]) for row in crowd_rows], dtype=float)
        c_exact = np.array([float(row["c_pucb_exact"]) for row in crowd_rows], dtype=float)

        ax.plot(x, regret_p, marker="o", linewidth=2.0, color="#c23b22", label="Pareto UCB1 regret")
        ax.plot(x, regret_w, marker="s", linewidth=2.0, color="#1b6ca8", label="Width-guided regret")
        ax.set_xlabel("Number of crowd arms")
        ax.set_ylabel("Final Pareto regret")
        ax.set_title(r"Varying $m$ ($\delta=0.02$)")
        ax.set_xticks(x)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)

        ax2 = ax.twinx()
        ax2.plot(
            x,
            c_exact,
            marker="^",
            linewidth=1.8,
            linestyle="--",
            color="#444444",
            label=r"$C_{\mathrm{PUCB}}$",
        )
        ax2.set_ylabel(r"$C_{\mathrm{PUCB}}$")

        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        if legend_handles is None:
            legend_handles, legend_labels = lines, labels

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
            fontsize=9,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(plots_dir / "comparison_plots.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_trajectory_figure(items: List[Dict[str, object]], plots_dir: Path, plt) -> None:
    """Render the trajectory figure that visualizes certification over time."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    if not items:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0), sharex="col")
    colors = {"pareto": "#c23b22", "width": "#1b6ca8"}
    legend_handles = None
    legend_labels = None

    for row_idx, item in enumerate(items):
        rounds = np.asarray(item["rounds"], dtype=float)

        ax_regret = axes[row_idx, 0]
        regret_p = np.asarray(item["pareto_ucb1_regret_mean"], dtype=float)
        regret_p_se = np.asarray(item["pareto_ucb1_regret_se"], dtype=float)
        regret_w = np.asarray(item["width_guided_regret_mean"], dtype=float)
        regret_w_se = np.asarray(item["width_guided_regret_se"], dtype=float)

        ax_regret.plot(rounds, regret_p, color=colors["pareto"], linewidth=2.0, label="Pareto UCB1")
        ax_regret.fill_between(rounds, regret_p - regret_p_se, regret_p + regret_p_se, color=colors["pareto"], alpha=0.15)
        ax_regret.plot(rounds, regret_w, color=colors["width"], linewidth=2.0, label="Width-guided")
        ax_regret.fill_between(rounds, regret_w - regret_w_se, regret_w + regret_w_se, color=colors["width"], alpha=0.15)

        median_cert = float(item["width_certified_round_median_observed"])
        if np.isfinite(median_cert):
            ax_regret.axvline(
                median_cert,
                color="#444444",
                linestyle="--",
                linewidth=1.2,
                label="Median cert. round",
            )

        ax_regret.set_ylabel("Cumulative Pareto regret")
        ax_regret.set_title(rf"$(\delta,m)={item['label']}$")
        ax_regret.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        if legend_handles is None:
            legend_handles = ax_regret.get_lines()
            legend_labels = [line.get_label() for line in legend_handles]

        ax_cert = axes[row_idx, 1]
        cert_frac = np.asarray(item["width_certified_fraction"], dtype=float)
        ax_cert.plot(rounds, cert_frac, color=colors["width"], linewidth=2.2)
        if np.isfinite(median_cert):
            ax_cert.axvline(
                median_cert,
                color="#444444",
                linestyle="--",
                linewidth=1.2,
            )
        ax_cert.set_ylim(-0.02, 1.02)
        ax_cert.set_ylabel("Certified-run fraction")
        ax_cert.set_title("Width-guided certification")
        ax_cert.grid(alpha=0.25, linestyle="--", linewidth=0.7)

    axes[1, 0].set_xlabel("Round")
    axes[1, 1].set_xlabel("Round")
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
            fontsize=9,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(plots_dir / "trajectory_plots.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Regenerate derived summaries and manuscript figures from raw caches."""
    args = _parse_args()
    out_dir = Path(args.outdir)
    if args.plots_dir is None:
        canonical_outdir = DEFAULT_OUTDIR.resolve()
        plots_dir = DEFAULT_PLOTS_DIR if out_dir.resolve() == canonical_outdir else out_dir / "plots"
    else:
        plots_dir = Path(args.plots_dir)
    plt, matplotlib_version = _import_plotting_backend()

    manifest, run_cache, trajectory_cache = _load_caches(out_dir)
    rows = _summarize_final_regret_runs(manifest, run_cache)
    trajectory_items = _summarize_trajectory_runs(manifest, trajectory_cache)

    _write_summary_outputs(out_dir, manifest, rows)
    _write_trajectory_summary(out_dir, manifest, trajectory_items)
    _write_comparison_figure(rows, plots_dir, plt)
    _write_trajectory_figure(trajectory_items, plots_dir, plt)
    _write_derived_manifest(out_dir, manifest, plots_dir, matplotlib_version)

    print(
        f"Saved derived summaries to {out_dir} and manuscript plots to {plots_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
