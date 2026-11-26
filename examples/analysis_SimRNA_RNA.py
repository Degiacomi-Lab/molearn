"""SimRNA-only model analysis.

Open this file in VS Code and use ``Run Cell`` on the ``# %%`` sections, or run
the complete analysis with ``python examples/analysis_SimRNA_RNA.py``.
"""

# %% Imports
import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from molearn.analysis import (
    MolearnAnalysis,
    add_connect_directory_RNA,
    reconstruct_all_atom_trafl_RNA,
    refine_pdb_with_simrna_RNA,
    write_transition_trafl_RNA,
)
from molearn.data import SimRNAData_RNA
from molearn.models import DistanceMatrix_AE_RNA
from molearn.models.foldingnet import AutoEncoder


# %% Configuration
SIMRNA_CG_CLASH_DISTANCE_RNA = 3.0


parser_RNA = argparse.ArgumentParser(description=__doc__)
parser_RNA.add_argument("--training-root", type=Path, default=None,
                        help="Directory containing training statistics and checkpoint folder.")
parser_RNA.add_argument("--checkpoint", type=Path, default=None,
                        help="Explicit .ckpt file; overrides automatic checkpoint discovery.")
parser_RNA.add_argument("--statistics", type=Path, default=None,
                        help="Explicit training statistics JSON file.")
parser_RNA.add_argument("--data-root", type=Path, default=None,
                        help="Base directory for relative trajectory paths in the manifest.")
parser_RNA.add_argument("--output-root", type=Path, default=None,
                        help="Directory in which to create the analysis directory.")
parser_RNA.add_argument("--ensemble-config", type=Path, default=None)
parser_RNA.add_argument("--ensembles", nargs="+", default=["apo", "holo"])
parser_RNA.add_argument(
    "--model", choices=("foldingnet", "distance-map"),
    default="foldingnet",
    help="Model family used for automatic checkpoint/statistics discovery.",
)
parser_RNA.add_argument("--start-state", default="apo")
parser_RNA.add_argument("--end-state", default="holo")
parser_RNA.add_argument("--frame-step", type=int, default=10)
parser_RNA.add_argument(
    "--trajectory-part",
    choices=("auto", "development", "test", "all"),
    default="auto",
    help=(
        "Trajectory interval to analyse. 'auto' uses the held-out test "
        "interval when it is recorded in the training statistics, otherwise "
        "all frames."
    ),
)
parser_RNA.add_argument("--batch-size", type=int, default=16)
parser_RNA.add_argument(
    "--transition-states", type=int, default=10,
    help="Number of generated path structures, including both endpoints.",
)
parser_RNA.add_argument(
    "--detail-analysis", action="store_true",
    help=(
        "Generate detailed raw-latent, separation, correlation, energy, and "
        "path-projection figures. By default only the PCA latent projection "
        "is generated from the latent-space diagnostics."
    ),
)
parser_RNA.add_argument(
    "--show", action="store_true",
    help="Display generated figures interactively as well as saving them.",
)
parser_RNA.add_argument(
    "--simrna-refine", action="store_true",
    help="Refine every generated path structure with restrained SimRNA (-P).",
)
parser_RNA.add_argument(
    "--simrna-iterations", type=int, default=1000,
    help="Number of SimRNA iterations per generated path structure.",
)
parser_RNA.add_argument(
    "--simrna-executable", type=Path, default=None,
    help="Path to official SimRNA (default: <repository>/SimRNA/SimRNA).",
)
parser_RNA.add_argument(
    "--simrna-data-directory", type=Path, default=None,
    help="SimRNA energy-data directory (default: data beside the executable).",
)
parser_RNA.add_argument(
    "--simrna-workers", type=int, default=0,
    help=(
        "Parallel SimRNA processes; 0 detects the available CPU allocation "
        "automatically (default: 0)."
    ),
)
# ``parse_known_args`` keeps the file usable as VS Code/Jupyter cells, whose
# kernels inject their own command-line arguments.
args_RNA, _unknown_RNA = parser_RNA.parse_known_args()

ensemble_names_RNA = args_RNA.ensembles
start_state_RNA = args_RNA.start_state
end_state_RNA = args_RNA.end_state
frame_step_RNA = args_RNA.frame_step
batch_size_RNA = args_RNA.batch_size
transition_states_RNA = args_RNA.transition_states
detail_analysis_RNA = args_RNA.detail_analysis
show_figures_RNA = args_RNA.show


def finish_figure_RNA(figure_RNA):
    """Optionally display a saved figure, then release its memory."""
    if show_figures_RNA:
        plt.show()
    plt.close(figure_RNA)

if (
    frame_step_RNA < 1 or batch_size_RNA < 1
    or transition_states_RNA < 2 or args_RNA.simrna_iterations < 1
    or args_RNA.simrna_workers < 0
):
    raise ValueError(
        "frame step, batch size, and SimRNA iterations must be positive; "
        "at least two transition states are required; SimRNA workers cannot "
        "be negative"
    )

cwd_RNA = Path.cwd().resolve()
repository_root_RNA = cwd_RNA.parent if cwd_RNA.name == "examples" else cwd_RNA
example_dir_RNA = repository_root_RNA / "examples"
training_root_RNA = (
    args_RNA.training_root.expanduser().resolve()
    if args_RNA.training_root is not None else example_dir_RNA
)
data_root_RNA = (
    args_RNA.data_root.expanduser().resolve()
    if args_RNA.data_root is not None else repository_root_RNA
)
analysis_root_RNA = (
    args_RNA.output_root.expanduser().resolve()
    if args_RNA.output_root is not None else training_root_RNA
)
manifest_path_RNA = (
    args_RNA.ensemble_config.expanduser().resolve()
    if args_RNA.ensemble_config is not None
    else example_dir_RNA / "simrna_ensembles.json"
)
model_name_RNA = "_".join(ensemble_names_RNA)
if args_RNA.model == "distance-map":
    checkpoint_name_RNA = (
        f"distance_matrix_SimRNA_RNA_{model_name_RNA}_aligned_checkpoints"
    )
    statistics_name_RNA = (
        f"data_statistics_DistanceMatrix_SimRNA_RNA_"
        f"{model_name_RNA}_aligned.json"
    )
    analysis_name_RNA = (
        f"analysis_DistanceMatrix_SimRNA_RNA_{model_name_RNA}_"
        f"{start_state_RNA}_to_{end_state_RNA}"
    )
else:
    checkpoint_name_RNA = (
        f"foldingnet_SimRNA_RNA_{model_name_RNA}_aligned_checkpoints"
    )
    statistics_name_RNA = (
        f"data_statistics_SimRNA_RNA_{model_name_RNA}_aligned.json"
    )
    analysis_name_RNA = (
        f"analysis_SimRNA_RNA_{model_name_RNA}_"
        f"{start_state_RNA}_to_{end_state_RNA}"
    )
checkpoint_dir_RNA = training_root_RNA / checkpoint_name_RNA
statistics_path_RNA = (
    args_RNA.statistics.expanduser().resolve()
    if args_RNA.statistics is not None
    else training_root_RNA / statistics_name_RNA
)
output_dir_RNA = analysis_root_RNA / analysis_name_RNA
output_dir_RNA.mkdir(parents=True, exist_ok=True)

if start_state_RNA not in ensemble_names_RNA or end_state_RNA not in ensemble_names_RNA:
    raise ValueError("Start and end states must occur in ensemble_names_RNA")


# %% Load the SimRNA manifest and trained checkpoint
with open(manifest_path_RNA) as infile_RNA:
    manifest_RNA = json.load(infile_RNA)

converged_RNA = checkpoint_dir_RNA / "checkpoint_converged.ckpt"
last_RNA = checkpoint_dir_RNA / "last.ckpt"
checkpoint_path_RNA = (
    args_RNA.checkpoint.expanduser().resolve()
    if args_RNA.checkpoint is not None
    else (converged_RNA if converged_RNA.exists() else last_RNA)
)
if not checkpoint_path_RNA.exists():
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir_RNA}")
if not statistics_path_RNA.exists():
    raise FileNotFoundError(statistics_path_RNA)

device_RNA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_RNA = torch.load(
    checkpoint_path_RNA, map_location=device_RNA, weights_only=True
)
with open(statistics_path_RNA) as infile_RNA:
    statistics_RNA = json.load(infile_RNA)
alignment_reference_RNA = np.asarray(
    statistics_RNA["alignment_reference_RNA"], dtype=float
)
trajectory_part_RNA = args_RNA.trajectory_part
if trajectory_part_RNA == "auto":
    trajectory_part_RNA = (
        "test" if "test_frame_start_RNA" in statistics_RNA else "all"
    )
if trajectory_part_RNA == "development":
    frame_start_RNA = int(
        statistics_RNA.get("development_frame_start_RNA", 0)
    )
    frame_stop_RNA = statistics_RNA.get("development_frame_stop_RNA")
    if frame_stop_RNA is None:
        raise ValueError(
            "Training statistics do not define a development-frame boundary"
        )
    frame_stop_RNA = int(frame_stop_RNA)
elif trajectory_part_RNA == "test":
    if "test_frame_start_RNA" not in statistics_RNA:
        raise ValueError(
            "Training statistics do not define a held-out test boundary"
        )
    frame_start_RNA = int(statistics_RNA["test_frame_start_RNA"])
    frame_stop_RNA = None
else:
    frame_start_RNA = None
    frame_stop_RNA = None
print(
    "Trajectory part:", trajectory_part_RNA,
    "raw-frame interval:", f"[{frame_start_RNA or 0}, "
    f"{frame_stop_RNA if frame_stop_RNA is not None else 'end'})",
)
network_kwargs_RNA = checkpoint_RNA["network_kwargs"]
distance_map_model_RNA = "residue_count_RNA" in network_kwargs_RNA
if distance_map_model_RNA:
    model_RNA = DistanceMatrix_AE_RNA(**network_kwargs_RNA).to(device_RNA)
    out_points_RNA = (
        int(network_kwargs_RNA["residue_count_RNA"])
        * int(network_kwargs_RNA.get("sites_per_residue_RNA", 5))
    )
    loaded_model_name_RNA = "DistanceMatrix_AE_RNA"
else:
    model_RNA = AutoEncoder(**network_kwargs_RNA).to(device_RNA)
    out_points_RNA = int(network_kwargs_RNA["out_points"])
    loaded_model_name_RNA = "FoldingNet"
model_RNA.load_state_dict(checkpoint_RNA["model_state_dict"])
model_RNA.eval()
print("Checkpoint:", checkpoint_path_RNA)
print("Epoch:", checkpoint_RNA.get("epoch"))
print("Model:", loaded_model_name_RNA)
print("Expected points:", out_points_RNA)
print("Device:", device_RNA)


# %% Stream the requested SimRNA ensembles
def resolve_manifest_path_RNA(filename_RNA):
    path_RNA = Path(filename_RNA).expanduser()
    return path_RNA if path_RNA.is_absolute() else data_root_RNA / path_RNA


def load_simrna_ensemble_RNA(state_RNA):
    source_RNA = manifest_RNA[state_RNA]
    data_RNA = SimRNAData_RNA()
    data_RNA.import_simrna_RNA(
        trc_file=resolve_manifest_path_RNA(source_RNA["trc"]),
        hdr_file=resolve_manifest_path_RNA(source_RNA["hdr"]),
        pdb_file=resolve_manifest_path_RNA(source_RNA["pdb"]),
    )
    data_RNA.atomselect_RNA("all")
    data_RNA.prepare_dataset_RNA(
        mean=checkpoint_RNA["mean"],
        std=checkpoint_RNA["std"],
        start=frame_start_RNA,
        stop=frame_stop_RNA,
        step=frame_step_RNA,
        align_RNA=True,
        alignment_reference_RNA=alignment_reference_RNA,
    )
    if data_RNA.dataset.shape[1] != out_points_RNA:
        raise ValueError(
            f"{state_RNA} has {data_RNA.dataset.shape[1]} points; "
            f"checkpoint expects {out_points_RNA}"
        )
    print(state_RNA, tuple(data_RNA.dataset.shape))
    return data_RNA


datasets_RNA = {
    state_RNA: load_simrna_ensemble_RNA(state_RNA)
    for state_RNA in ensemble_names_RNA
}


# %% Register all ensembles
analysis_RNA = MolearnAnalysis()
analysis_RNA.set_network(model_RNA)
analysis_RNA.batch_size = batch_size_RNA
for state_RNA, data_RNA in datasets_RNA.items():
    analysis_RNA.set_dataset(key=state_RNA, data=data_RNA)
print("Trainable parameters:", analysis_RNA.num_trainable_params())


# %% Reconstruction RMSD
rmsd_RNA = {
    state_RNA: np.asarray(analysis_RNA.get_error(state_RNA))
    for state_RNA in ensemble_names_RNA
}
summary_RNA = []
for state_RNA, values_RNA in rmsd_RNA.items():
    summary_RNA.append(
        {
            "ensemble": state_RNA,
            "frames": len(values_RNA),
            "mean_rmsd_A": float(values_RNA.mean()),
            "median_rmsd_A": float(np.median(values_RNA)),
            "p95_rmsd_A": float(np.percentile(values_RNA, 95)),
            "max_rmsd_A": float(values_RNA.max()),
        }
    )
with open(output_dir_RNA / "reconstruction_summary.csv", "w", newline="") as out_RNA:
    writer_RNA = csv.DictWriter(out_RNA, fieldnames=summary_RNA[0].keys())
    writer_RNA.writeheader()
    writer_RNA.writerows(summary_RNA)
for row_RNA in summary_RNA:
    print(row_RNA)

fig_RNA, axis_RNA = plt.subplots(figsize=(8, 5))
for state_RNA, values_RNA in rmsd_RNA.items():
    axis_RNA.hist(values_RNA, bins=50, alpha=0.55, label=state_RNA)
axis_RNA.set(xlabel="Reconstruction RMSD (Å)", ylabel="Frames")
axis_RNA.set_title("SimRNA reconstruction error")
axis_RNA.legend()
fig_RNA.tight_layout()
fig_RNA.savefig(output_dir_RNA / "reconstruction_rmsd.png", dpi=150)
finish_figure_RNA(fig_RNA)


# %% Latent-space diagnostics
encoded_RNA = {
    state_RNA: analysis_RNA.get_encoded(state_RNA).detach().cpu().numpy()
    for state_RNA in ensemble_names_RNA
}
latent_dimension_count_RNA = next(iter(encoded_RNA.values())).shape[1]
if any(
    values_RNA.ndim != 2
    or values_RNA.shape[1] != latent_dimension_count_RNA
    for values_RNA in encoded_RNA.values()
):
    raise ValueError("All encoded ensembles must have the same 2D latent shape")


def wasserstein_distance_1d_RNA(first_RNA, second_RNA):
    """Return the exact 1D empirical Wasserstein distance without SciPy."""
    first_RNA = np.sort(np.asarray(first_RNA, dtype=float))
    second_RNA = np.sort(np.asarray(second_RNA, dtype=float))
    if not len(first_RNA) or not len(second_RNA):
        raise ValueError("Cannot compare empty latent distributions")
    combined_RNA = np.sort(np.concatenate([first_RNA, second_RNA]))
    deltas_RNA = np.diff(combined_RNA)
    if not len(deltas_RNA):
        return 0.0
    first_cdf_RNA = np.searchsorted(
        first_RNA, combined_RNA[:-1], side="right"
    ) / len(first_RNA)
    second_cdf_RNA = np.searchsorted(
        second_RNA, combined_RNA[:-1], side="right"
    ) / len(second_RNA)
    return float(np.sum(np.abs(first_cdf_RNA - second_cdf_RNA) * deltas_RNA))


all_latent_RNA = np.concatenate(
    [encoded_RNA[state_RNA] for state_RNA in ensemble_names_RNA], axis=0
)
start_encoded_RNA = encoded_RNA[start_state_RNA]
end_encoded_RNA = encoded_RNA[end_state_RNA]
raw_wasserstein_RNA = np.zeros(latent_dimension_count_RNA, dtype=float)
standardized_wasserstein_RNA = np.zeros(latent_dimension_count_RNA, dtype=float)
combined_variance_RNA = all_latent_RNA.var(axis=0)
for dimension_RNA in range(latent_dimension_count_RNA):
    raw_wasserstein_RNA[dimension_RNA] = wasserstein_distance_1d_RNA(
        start_encoded_RNA[:, dimension_RNA],
        end_encoded_RNA[:, dimension_RNA],
    )
    combined_std_RNA = all_latent_RNA[:, dimension_RNA].std()
    standardized_wasserstein_RNA[dimension_RNA] = (
        raw_wasserstein_RNA[dimension_RNA] / combined_std_RNA
        if combined_std_RNA > np.finfo(float).eps else 0.0
    )
dimension_order_RNA = np.argsort(standardized_wasserstein_RNA)[::-1]

separation_rows_RNA = []
for rank_RNA, dimension_RNA in enumerate(dimension_order_RNA, start=1):
    row_RNA = {
        "separation_rank": rank_RNA,
        "latent_dimension": int(dimension_RNA + 1),
        "wasserstein_distance": float(raw_wasserstein_RNA[dimension_RNA]),
        "standardized_wasserstein_distance": float(
            standardized_wasserstein_RNA[dimension_RNA]
        ),
        "combined_variance": float(combined_variance_RNA[dimension_RNA]),
    }
    for state_RNA in ensemble_names_RNA:
        values_RNA = encoded_RNA[state_RNA][:, dimension_RNA]
        row_RNA[f"{state_RNA}_mean"] = float(values_RNA.mean())
        row_RNA[f"{state_RNA}_std"] = float(values_RNA.std())
    separation_rows_RNA.append(row_RNA)
with open(
    output_dir_RNA / "latent_dimension_separation.csv", "w", newline=""
) as out_RNA:
    writer_RNA = csv.DictWriter(
        out_RNA, fieldnames=separation_rows_RNA[0].keys()
    )
    writer_RNA.writeheader()
    writer_RNA.writerows(separation_rows_RNA)

# Plot the raw latent-coordinate diagnostics only when explicitly requested.
if detail_analysis_RNA:
    # One overlaid distribution plot for every latent coordinate (8 x 4 for 32D).
    histogram_columns_RNA = min(4, latent_dimension_count_RNA)
    histogram_rows_RNA = int(np.ceil(
        latent_dimension_count_RNA / histogram_columns_RNA
    ))
    fig_RNA, axes_RNA = plt.subplots(
        histogram_rows_RNA,
        histogram_columns_RNA,
        figsize=(4 * histogram_columns_RNA, 2.6 * histogram_rows_RNA),
        squeeze=False,
    )
    for dimension_RNA, axis_RNA in enumerate(axes_RNA.flat):
        if dimension_RNA >= latent_dimension_count_RNA:
            axis_RNA.set_visible(False)
            continue
        combined_dimension_RNA = all_latent_RNA[:, dimension_RNA]
        low_RNA, high_RNA = (
            combined_dimension_RNA.min(), combined_dimension_RNA.max()
        )
        if np.isclose(low_RNA, high_RNA):
            low_RNA -= 0.5
            high_RNA += 0.5
        bins_RNA = np.linspace(low_RNA, high_RNA, 41)
        for state_RNA in ensemble_names_RNA:
            axis_RNA.hist(
                encoded_RNA[state_RNA][:, dimension_RNA],
                bins=bins_RNA,
                density=True,
                histtype="step",
                linewidth=1.4,
                label=state_RNA,
            )
        axis_RNA.set_title(
            f"Dimension {dimension_RNA + 1}  "
            f"(W/std={standardized_wasserstein_RNA[dimension_RNA]:.2f})",
            fontsize=9,
        )
        axis_RNA.tick_params(labelsize=7)
        if dimension_RNA % histogram_columns_RNA == 0:
            axis_RNA.set_ylabel("Density", fontsize=8)
        if dimension_RNA // histogram_columns_RNA == histogram_rows_RNA - 1:
            axis_RNA.set_xlabel("Latent value", fontsize=8)
    handles_RNA, labels_RNA = axes_RNA.flat[0].get_legend_handles_labels()
    fig_RNA.legend(
        handles_RNA, labels_RNA, loc="upper center",
        bbox_to_anchor=(0.5, 0.985), ncol=len(ensemble_names_RNA),
    )
    fig_RNA.suptitle(f"All {latent_dimension_count_RNA} latent distributions")
    fig_RNA.tight_layout(rect=(0, 0, 1, 0.965))
    fig_RNA.savefig(
        output_dir_RNA / "latent_dimension_histograms.png", dpi=200
    )
    finish_figure_RNA(fig_RNA)

    fig_RNA, axis_RNA = plt.subplots(
        figsize=(9, max(5, 0.27 * latent_dimension_count_RNA))
    )
    plot_order_RNA = dimension_order_RNA[::-1]
    axis_RNA.barh(
        np.arange(latent_dimension_count_RNA),
        standardized_wasserstein_RNA[plot_order_RNA],
        color="tab:purple",
    )
    axis_RNA.set_yticks(
        np.arange(latent_dimension_count_RNA),
        [str(index_RNA + 1) for index_RNA in plot_order_RNA],
    )
    axis_RNA.set(
        xlabel="Wasserstein distance / combined standard deviation",
        ylabel="Latent dimension",
        title=f"Latent separation: {start_state_RNA} versus {end_state_RNA}",
    )
    fig_RNA.tight_layout()
    fig_RNA.savefig(
        output_dir_RNA / "latent_dimension_separation.png", dpi=200
    )
    finish_figure_RNA(fig_RNA)

    # Plot the two individual latent coordinates with the strongest separation.
    if latent_dimension_count_RNA >= 2:
        best_x_RNA, best_y_RNA = dimension_order_RNA[:2]
        fig_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
        for state_RNA, latent_RNA in encoded_RNA.items():
            axis_RNA.scatter(
                latent_RNA[:, best_x_RNA], latent_RNA[:, best_y_RNA],
                s=7, alpha=0.35, label=state_RNA,
            )
        axis_RNA.set(
            xlabel=f"Latent dimension {best_x_RNA + 1}",
            ylabel=f"Latent dimension {best_y_RNA + 1}",
            title="Two most separating latent dimensions",
        )
        axis_RNA.legend()
        fig_RNA.tight_layout()
        fig_RNA.savefig(
            output_dir_RNA / "latent_space_most_separating_dimensions.png",
            dpi=200,
        )
        finish_figure_RNA(fig_RNA)

# PCA incorporates all latent coordinates instead of displaying only the first two.
latent_mean_RNA = all_latent_RNA.mean(axis=0)
centered_latent_RNA = all_latent_RNA - latent_mean_RNA
_left_RNA, singular_values_RNA, pca_components_RNA = np.linalg.svd(
    centered_latent_RNA, full_matrices=False
)
pca_variances_RNA = singular_values_RNA**2 / max(len(all_latent_RNA) - 1, 1)
pca_variance_total_RNA = pca_variances_RNA.sum()
pca_ratios_RNA = (
    pca_variances_RNA / pca_variance_total_RNA
    if pca_variance_total_RNA > 0 else np.zeros_like(pca_variances_RNA)
)
effective_dimension_RNA = (
    float(pca_variance_total_RNA**2 / np.sum(pca_variances_RNA**2))
    if np.any(pca_variances_RNA) else 0.0
)
pca_rows_RNA = [
    {
        "principal_component": index_RNA + 1,
        "variance": float(pca_variances_RNA[index_RNA]),
        "explained_variance_fraction": float(pca_ratios_RNA[index_RNA]),
        "cumulative_explained_variance_fraction": float(
            pca_ratios_RNA[:index_RNA + 1].sum()
        ),
    }
    for index_RNA in range(len(pca_variances_RNA))
]
with open(
    output_dir_RNA / "latent_pca_explained_variance.csv", "w", newline=""
) as out_RNA:
    writer_RNA = csv.DictWriter(out_RNA, fieldnames=pca_rows_RNA[0].keys())
    writer_RNA.writeheader()
    writer_RNA.writerows(pca_rows_RNA)

if len(pca_components_RNA) >= 2:
    fig_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
    for state_RNA, latent_RNA in encoded_RNA.items():
        projected_RNA = (
            latent_RNA - latent_mean_RNA
        ) @ pca_components_RNA[:2].T
        axis_RNA.scatter(
            projected_RNA[:, 0], projected_RNA[:, 1],
            s=7, alpha=0.35, label=state_RNA,
        )
    axis_RNA.set(
        xlabel=f"PC1 ({100 * pca_ratios_RNA[0]:.1f}% variance)",
        ylabel=f"PC2 ({100 * pca_ratios_RNA[1]:.1f}% variance)",
        title=f"PCA projection of all {latent_dimension_count_RNA} dimensions",
    )
    axis_RNA.legend()
    fig_RNA.tight_layout()
    fig_RNA.savefig(output_dir_RNA / "latent_space_pca.png", dpi=200)
    finish_figure_RNA(fig_RNA)

if detail_analysis_RNA:
    fig_RNA, axis_RNA = plt.subplots(figsize=(10, 4.5))
    axis_RNA.bar(
        np.arange(1, latent_dimension_count_RNA + 1), combined_variance_RNA,
        color="tab:blue",
    )
    axis_RNA.set(
        xlabel="Latent dimension", ylabel="Variance",
        title="Variance used by each latent dimension",
    )
    axis_RNA.set_xticks(np.arange(1, latent_dimension_count_RNA + 1))
    axis_RNA.tick_params(axis="x", labelsize=7)
    fig_RNA.tight_layout()
    fig_RNA.savefig(output_dir_RNA / "latent_dimension_variance.png", dpi=200)
    finish_figure_RNA(fig_RNA)

    if latent_dimension_count_RNA >= 2:
        correlation_RNA = np.nan_to_num(
            np.corrcoef(all_latent_RNA, rowvar=False), nan=0.0
        )
        fig_RNA, axis_RNA = plt.subplots(figsize=(9, 8))
        image_RNA = axis_RNA.imshow(
            correlation_RNA, cmap="coolwarm", vmin=-1.0, vmax=1.0,
            interpolation="nearest",
        )
        ticks_RNA = np.arange(latent_dimension_count_RNA)
        axis_RNA.set_xticks(ticks_RNA, ticks_RNA + 1, fontsize=6)
        axis_RNA.set_yticks(ticks_RNA, ticks_RNA + 1, fontsize=6)
        axis_RNA.set(
            xlabel="Latent dimension", ylabel="Latent dimension",
            title="Latent-dimension correlation",
        )
        fig_RNA.colorbar(image_RNA, ax=axis_RNA, label="Pearson correlation")
        fig_RNA.tight_layout()
        fig_RNA.savefig(
            output_dir_RNA / "latent_dimension_correlation.png", dpi=200
        )
        finish_figure_RNA(fig_RNA)

latent_diagnostics_RNA = {
    "latent_dimensions": int(latent_dimension_count_RNA),
    "frames_per_ensemble": {
        state_RNA: int(len(encoded_RNA[state_RNA]))
        for state_RNA in ensemble_names_RNA
    },
    "separation_states": [start_state_RNA, end_state_RNA],
    "most_separating_dimensions": [
        int(index_RNA + 1) for index_RNA in dimension_order_RNA[:2]
    ],
    "effective_dimension_participation_ratio": effective_dimension_RNA,
    "pca_first_two_explained_variance_fraction": float(
        pca_ratios_RNA[:2].sum()
    ),
}
with open(output_dir_RNA / "latent_space_diagnostics.json", "w") as out_RNA:
    json.dump(latent_diagnostics_RNA, out_RNA, indent=2)
print("Latent-space diagnostics:", latent_diagnostics_RNA)


# %% Optional raw latent-space and energy projections
if detail_analysis_RNA:
    fig_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
    for state_RNA, latent_RNA in encoded_RNA.items():
        axis_RNA.scatter(
            latent_RNA[:, 0], latent_RNA[:, 1],
            s=7, alpha=0.35, label=state_RNA,
        )
    axis_RNA.set(xlabel="Latent dimension 1", ylabel="Latent dimension 2")
    axis_RNA.set_title("SimRNA latent space")
    axis_RNA.legend()
    fig_RNA.tight_layout()
    fig_RNA.savefig(output_dir_RNA / "latent_space.png", dpi=150)
    finish_figure_RNA(fig_RNA)

    fig_RNA, axes_RNA = plt.subplots(
        1, len(ensemble_names_RNA),
        figsize=(7 * len(ensemble_names_RNA), 5), squeeze=False,
    )
    for axis_RNA, state_RNA in zip(axes_RNA[0], ensemble_names_RNA):
        headers_RNA = datasets_RNA[state_RNA].frame_headers_RNA[0]
        energies_RNA = np.asarray([
            float(header_RNA["E_TOTAL"]) for header_RNA in headers_RNA
        ])
        latent_RNA = encoded_RNA[state_RNA]
        scatter_RNA = axis_RNA.scatter(
            latent_RNA[:, 0], latent_RNA[:, 1], c=energies_RNA,
            s=8, alpha=0.6, cmap="viridis",
        )
        axis_RNA.set_title(f"{state_RNA}: SimRNA E_TOTAL")
        axis_RNA.set(
            xlabel="Latent dimension 1", ylabel="Latent dimension 2"
        )
        fig_RNA.colorbar(scatter_RNA, ax=axis_RNA, label="E_TOTAL")
    fig_RNA.tight_layout()
    fig_RNA.savefig(output_dir_RNA / "latent_energy.png", dpi=150)
    finish_figure_RNA(fig_RNA)


# %% Original-versus-decoded radius of gyration
def radius_of_gyration_RNA(coords_RNA):
    centered_RNA = coords_RNA - coords_RNA.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered_RNA**2, axis=2), axis=1))


fig_RNA, axis_RNA = plt.subplots(figsize=(8, 5))
for state_RNA in ensemble_names_RNA:
    original_RNA = analysis_RNA.get_dataset(
        state_RNA, scale=True
    ).detach().cpu().numpy()
    decoded_RNA = analysis_RNA.get_decoded(
        state_RNA, scale=True
    ).detach().cpu().numpy()
    axis_RNA.hist(
        radius_of_gyration_RNA(original_RNA), bins=50, histtype="step",
        linewidth=1.5, label=f"{state_RNA} original"
    )
    axis_RNA.hist(
        radius_of_gyration_RNA(decoded_RNA), bins=50, histtype="step",
        linewidth=1.5, label=f"{state_RNA} decoded"
    )
axis_RNA.set(xlabel="Radius of gyration (Å)", ylabel="Frames")
axis_RNA.set_title("SimRNA compactness")
axis_RNA.legend()
fig_RNA.tight_layout()
fig_RNA.savefig(output_dir_RNA / "radius_of_gyration.png", dpi=150)
finish_figure_RNA(fig_RNA)


# %% Generate start-to-end latent path and connected PDB files
start_latent_RNA = encoded_RNA[start_state_RNA]
end_latent_RNA = encoded_RNA[end_state_RNA]
start_center_RNA = start_latent_RNA.mean(axis=0)
end_center_RNA = end_latent_RNA.mean(axis=0)
start_index_RNA = np.linalg.norm(start_latent_RNA - start_center_RNA, axis=1).argmin()
end_index_RNA = np.linalg.norm(end_latent_RNA - end_center_RNA, axis=1).argmin()
latent_path_RNA = np.linspace(
    start_latent_RNA[start_index_RNA],
    end_latent_RNA[end_index_RNA],
    num=transition_states_RNA,
)
path_dir_RNA = output_dir_RNA / f"{start_state_RNA}_to_{end_state_RNA}_path"
path_dir_RNA.mkdir(parents=True, exist_ok=True)
if distance_map_model_RNA:
    generated_path_RNA = analysis_RNA.generate(crd=latent_path_RNA)
    generated_path_RNA = datasets_RNA[
        start_state_RNA
    ].align_coordinates_RNA(
        generated_path_RNA,
        reference=alignment_reference_RNA,
    )
    for frame_index_RNA, coordinates_RNA in enumerate(
        generated_path_RNA
    ):
        analysis_RNA._pdb_file(
            coordinates_RNA,
            str(path_dir_RNA / f"s_{frame_index_RNA}.pdb"),
        )
else:
    generated_path_RNA = analysis_RNA.generate(
        crd=latent_path_RNA, pdb_path=path_dir_RNA
    )
connected_pdbs_RNA = add_connect_directory_RNA(path_dir_RNA)

if detail_analysis_RNA:
    fig_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
    for state_RNA, latent_RNA in encoded_RNA.items():
        axis_RNA.scatter(
            latent_RNA[:, 0], latent_RNA[:, 1],
            s=7, alpha=0.2, label=state_RNA,
        )
    axis_RNA.plot(
        latent_path_RNA[:, 0], latent_path_RNA[:, 1], "k.-", label="Path"
    )
    axis_RNA.set(xlabel="Latent dimension 1", ylabel="Latent dimension 2")
    axis_RNA.set_title(f"Generated {start_state_RNA}-to-{end_state_RNA} path")
    axis_RNA.legend()
    fig_RNA.tight_layout()
    fig_RNA.savefig(output_dir_RNA / "linear_path.png", dpi=150)
    finish_figure_RNA(fig_RNA)

    if latent_dimension_count_RNA >= 2:
        fig_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
        for state_RNA, latent_RNA in encoded_RNA.items():
            axis_RNA.scatter(
                latent_RNA[:, best_x_RNA], latent_RNA[:, best_y_RNA],
                s=7, alpha=0.2, label=state_RNA,
            )
        axis_RNA.plot(
            latent_path_RNA[:, best_x_RNA], latent_path_RNA[:, best_y_RNA],
            "k.-", label="Path",
        )
        axis_RNA.set(
            xlabel=f"Latent dimension {best_x_RNA + 1}",
            ylabel=f"Latent dimension {best_y_RNA + 1}",
            title="Transition in the two most separating dimensions",
        )
        axis_RNA.legend()
        fig_RNA.tight_layout()
        fig_RNA.savefig(
            output_dir_RNA / "linear_path_most_separating_dimensions.png",
            dpi=200,
        )
        finish_figure_RNA(fig_RNA)

if len(pca_components_RNA) >= 2:
    latent_path_pca_RNA = (
        latent_path_RNA - latent_mean_RNA
    ) @ pca_components_RNA[:2].T
    fig_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
    for state_RNA, latent_RNA in encoded_RNA.items():
        projected_RNA = (
            latent_RNA - latent_mean_RNA
        ) @ pca_components_RNA[:2].T
        axis_RNA.scatter(
            projected_RNA[:, 0], projected_RNA[:, 1],
            s=7, alpha=0.2, label=state_RNA,
        )
    axis_RNA.plot(
        latent_path_pca_RNA[:, 0], latent_path_pca_RNA[:, 1],
        "k.-", label="Path",
    )
    axis_RNA.set(
        xlabel=f"PC1 ({100 * pca_ratios_RNA[0]:.1f}% variance)",
        ylabel=f"PC2 ({100 * pca_ratios_RNA[1]:.1f}% variance)",
        title="Transition projected from the complete latent space",
    )
    axis_RNA.legend()
    fig_RNA.tight_layout()
    fig_RNA.savefig(output_dir_RNA / "linear_path_pca.png", dpi=200)
    finish_figure_RNA(fig_RNA)
print("Generated connected PDB files:", len(connected_pdbs_RNA))


# %% Local coarse-grained connection validation
def connection_indices_RNA(data_RNA):
    residues_RNA = []
    lookup_RNA = {}
    for index_RNA, (name_RNA, resname_RNA, resid_RNA) in enumerate(
        data_RNA.get_atominfo()
    ):
        key_RNA = (int(resid_RNA), str(resname_RNA))
        if key_RNA not in lookup_RNA:
            lookup_RNA[key_RNA] = {}
            residues_RNA.append(lookup_RNA[key_RNA])
        lookup_RNA[key_RNA][str(name_RNA)] = index_RNA
    pairs_RNA = {
        "P-C4'": [], "C4'-P(next)": [], "C4'-N": [],
        "N-C2": [], "C2-C4/C6": [], "C4/C6-N": [],
    }
    for number_RNA, atoms_RNA in enumerate(residues_RNA):
        n_RNA = atoms_RNA.get("N9", atoms_RNA.get("N1"))
        outer_RNA = atoms_RNA.get("C6", atoms_RNA.get("C4"))
        candidates_RNA = {
            "P-C4'": (atoms_RNA.get("P"), atoms_RNA.get("C4'")),
            "C4'-N": (atoms_RNA.get("C4'"), n_RNA),
            "N-C2": (n_RNA, atoms_RNA.get("C2")),
            "C2-C4/C6": (atoms_RNA.get("C2"), outer_RNA),
            "C4/C6-N": (outer_RNA, n_RNA),
        }
        for label_RNA, pair_RNA in candidates_RNA.items():
            if None not in pair_RNA:
                pairs_RNA[label_RNA].append(pair_RNA)
        if number_RNA + 1 < len(residues_RNA):
            next_RNA = residues_RNA[number_RNA + 1]
            pair_RNA = (atoms_RNA.get("C4'"), next_RNA.get("P"))
            if None not in pair_RNA:
                pairs_RNA["C4'-P(next)"].append(pair_RNA)
    return pairs_RNA


def distances_RNA(coords_RNA, pairs_RNA):
    first_RNA = np.asarray([pair_RNA[0] for pair_RNA in pairs_RNA])
    second_RNA = np.asarray([pair_RNA[1] for pair_RNA in pairs_RNA])
    return np.linalg.norm(
        coords_RNA[:, first_RNA] - coords_RNA[:, second_RNA], axis=2
    )


pairs_RNA = connection_indices_RNA(datasets_RNA[start_state_RNA])
reference_coords_RNA = np.concatenate([
    analysis_RNA.get_dataset(state_RNA, scale=True).detach().cpu().numpy()
    for state_RNA in ensemble_names_RNA
])
fig_RNA, axes_RNA = plt.subplots(
    len(pairs_RNA), 1, figsize=(9, 15), sharex=True
)
rows_RNA = []
steps_RNA = np.arange(len(generated_path_RNA))
for axis_RNA, (label_RNA, indices_RNA) in zip(axes_RNA, pairs_RNA.items()):
    reference_RNA = distances_RNA(reference_coords_RNA, indices_RNA).ravel()
    path_values_RNA = distances_RNA(generated_path_RNA, indices_RNA)
    low_RNA, median_RNA, high_RNA = np.percentile(reference_RNA, [5, 50, 95])
    axis_RNA.axhspan(low_RNA, high_RNA, color="tab:green", alpha=0.2)
    axis_RNA.axhline(median_RNA, color="tab:green", linestyle="--")
    axis_RNA.plot(steps_RNA, path_values_RNA.mean(axis=1), "ko-")
    axis_RNA.fill_between(
        steps_RNA, path_values_RNA.min(axis=1), path_values_RNA.max(axis=1),
        color="black", alpha=0.15
    )
    axis_RNA.set_ylabel(f"{label_RNA}\nDistance (Å)")
    for step_RNA, values_RNA in enumerate(path_values_RNA):
        rows_RNA.append({
            "path_step": step_RNA,
            "bond_type": label_RNA,
            "path_mean_A": float(values_RNA.mean()),
            "path_min_A": float(values_RNA.min()),
            "path_max_A": float(values_RNA.max()),
            "reference_p05_A": float(low_RNA),
            "reference_median_A": float(median_RNA),
            "reference_p95_A": float(high_RNA),
            "fraction_outside_reference_5_95": float(
                ((values_RNA < low_RNA) | (values_RNA > high_RNA)).mean()
            ),
        })
axes_RNA[-1].set_xlabel(f"Path step ({start_state_RNA} → {end_state_RNA})")
fig_RNA.suptitle("SimRNA local geometry along generated path")
fig_RNA.tight_layout()
fig_RNA.savefig(output_dir_RNA / "bond_distances_path.png", dpi=150)
finish_figure_RNA(fig_RNA)

with open(output_dir_RNA / "bond_distances_path.csv", "w", newline="") as out_RNA:
    writer_RNA = csv.DictWriter(out_RNA, fieldnames=rows_RNA[0].keys())
    writer_RNA.writeheader()
    writer_RNA.writerows(rows_RNA)


# %% Detect locally distorted residues and optionally refine them with SimRNA
def residue_geometry_scores_RNA(
    coords_RNA, edge_pairs_RNA, edge_means_RNA, edge_stds_RNA,
    atom_to_residue_RNA, residue_count_RNA,
):
    """Return each residue's largest standardized incident-bond deviation."""
    edge_values_RNA = distances_RNA(coords_RNA, edge_pairs_RNA)
    deviations_RNA = np.abs(
        edge_values_RNA - edge_means_RNA[None, :]
    ) / edge_stds_RNA[None, :]
    scores_RNA = np.zeros((len(coords_RNA), residue_count_RNA), dtype=float)
    for edge_index_RNA, (first_RNA, second_RNA) in enumerate(edge_pairs_RNA):
        for residue_index_RNA in {
            atom_to_residue_RNA[first_RNA], atom_to_residue_RNA[second_RNA]
        }:
            scores_RNA[:, residue_index_RNA] = np.maximum(
                scores_RNA[:, residue_index_RNA],
                deviations_RNA[:, edge_index_RNA],
            )
    return scores_RNA


atom_to_residue_RNA = {}
atom_names_RNA = []
residue_labels_RNA = []
residue_lookup_RNA = {}
for atom_index_RNA, atom_RNA in enumerate(
    datasets_RNA[start_state_RNA]._mol.atoms
):
    residue_key_RNA = (
        str(getattr(atom_RNA, "chainID", "")),
        int(atom_RNA.resid),
        str(atom_RNA.resname),
    )
    if residue_key_RNA not in residue_lookup_RNA:
        residue_lookup_RNA[residue_key_RNA] = len(residue_labels_RNA)
        residue_labels_RNA.append(residue_key_RNA)
    atom_to_residue_RNA[atom_index_RNA] = residue_lookup_RNA[residue_key_RNA]
    atom_names_RNA.append(str(atom_RNA.name))


def residue_clash_metrics_RNA(
    coords_RNA, atom_residues_RNA, atom_names_RNA, residue_labels_RNA,
    clash_distance_RNA,
):
    """Measure non-neighbor nucleotide bead overlaps by frame and residue."""
    atom_residues_RNA = np.asarray(atom_residues_RNA, dtype=int)
    first_RNA, second_RNA = np.triu_indices(len(atom_residues_RNA), k=1)
    first_residue_all_RNA = atom_residues_RNA[first_RNA]
    second_residue_all_RNA = atom_residues_RNA[second_RNA]
    different_residue_RNA = first_residue_all_RNA != second_residue_all_RNA
    residue_chains_RNA = np.asarray([
        label_RNA[0] for label_RNA in residue_labels_RNA
    ])
    same_chain_RNA = (
        residue_chains_RNA[first_residue_all_RNA]
        == residue_chains_RNA[second_residue_all_RNA]
    )
    immediate_neighbor_RNA = (
        same_chain_RNA
        & (np.abs(first_residue_all_RNA - second_residue_all_RNA) == 1)
    )
    eligible_RNA = different_residue_RNA & ~immediate_neighbor_RNA
    first_RNA = first_RNA[eligible_RNA]
    second_RNA = second_RNA[eligible_RNA]
    if not len(first_RNA):
        raise ValueError("At least two RNA nucleotides are required for clash detection")

    first_residue_RNA = atom_residues_RNA[first_RNA]
    second_residue_RNA = atom_residues_RNA[second_RNA]
    frame_count_RNA = len(coords_RNA)
    residue_count_RNA = int(atom_residues_RNA.max()) + 1
    minimum_RNA = np.full((frame_count_RNA, residue_count_RNA), np.inf)
    counts_RNA = np.zeros((frame_count_RNA, residue_count_RNA), dtype=int)
    partners_RNA = np.full((frame_count_RNA, residue_count_RNA), -1, dtype=int)
    atom_pairs_RNA = np.full(
        (frame_count_RNA, residue_count_RNA), "", dtype=object
    )

    for frame_index_RNA, frame_RNA in enumerate(np.asarray(coords_RNA)):
        values_RNA = np.linalg.norm(
            frame_RNA[first_RNA] - frame_RNA[second_RNA], axis=1
        )
        np.minimum.at(
            minimum_RNA[frame_index_RNA], first_residue_RNA, values_RNA
        )
        np.minimum.at(
            minimum_RNA[frame_index_RNA], second_residue_RNA, values_RNA
        )
        colliding_RNA = np.flatnonzero(values_RNA < clash_distance_RNA)
        np.add.at(
            counts_RNA[frame_index_RNA], first_residue_RNA[colliding_RNA], 1
        )
        np.add.at(
            counts_RNA[frame_index_RNA], second_residue_RNA[colliding_RNA], 1
        )
        for pair_index_RNA in colliding_RNA:
            atom_1_RNA = first_RNA[pair_index_RNA]
            atom_2_RNA = second_RNA[pair_index_RNA]
            residue_1_RNA = first_residue_RNA[pair_index_RNA]
            residue_2_RNA = second_residue_RNA[pair_index_RNA]
            distance_RNA = values_RNA[pair_index_RNA]
            if distance_RNA <= minimum_RNA[frame_index_RNA, residue_1_RNA]:
                partners_RNA[frame_index_RNA, residue_1_RNA] = residue_2_RNA
                atom_pairs_RNA[frame_index_RNA, residue_1_RNA] = (
                    f"{atom_names_RNA[atom_1_RNA]}--"
                    f"{atom_names_RNA[atom_2_RNA]}"
                )
            if distance_RNA <= minimum_RNA[frame_index_RNA, residue_2_RNA]:
                partners_RNA[frame_index_RNA, residue_2_RNA] = residue_1_RNA
                atom_pairs_RNA[frame_index_RNA, residue_2_RNA] = (
                    f"{atom_names_RNA[atom_2_RNA]}--"
                    f"{atom_names_RNA[atom_1_RNA]}"
                )
    return {
        "minimum_distances_RNA": minimum_RNA,
        "clash_counts_RNA": counts_RNA,
        "has_clash_RNA": counts_RNA > 0,
        "partner_residues_RNA": partners_RNA,
        "atom_pairs_RNA": atom_pairs_RNA,
        "pair_indices_RNA": (first_RNA, second_RNA),
    }


def immediate_neighbors_RNA(primary_residues_RNA, residue_labels_RNA):
    """Return one sequence neighbor on either side, without crossing chains."""
    primary_RNA = {int(index_RNA) for index_RNA in primary_residues_RNA}
    neighbors_RNA = set()
    for residue_index_RNA in primary_RNA:
        chain_RNA = residue_labels_RNA[residue_index_RNA][0]
        for candidate_RNA in (residue_index_RNA - 1, residue_index_RNA + 1):
            if (
                0 <= candidate_RNA < len(residue_labels_RNA)
                and residue_labels_RNA[candidate_RNA][0] == chain_RNA
                and candidate_RNA not in primary_RNA
            ):
                neighbors_RNA.add(candidate_RNA)
    return sorted(neighbors_RNA)

edge_pairs_RNA = [
    pair_RNA for indices_RNA in pairs_RNA.values() for pair_RNA in indices_RNA
]
reference_edge_values_RNA = distances_RNA(reference_coords_RNA, edge_pairs_RNA)
edge_means_RNA = reference_edge_values_RNA.mean(axis=0)
# Avoid amplifying insignificant numerical noise in nearly rigid bonds.
edge_stds_RNA = np.maximum(reference_edge_values_RNA.std(axis=0), 0.05)
reference_local_scores_RNA = residue_geometry_scores_RNA(
    reference_coords_RNA,
    edge_pairs_RNA,
    edge_means_RNA,
    edge_stds_RNA,
    atom_to_residue_RNA,
    len(residue_labels_RNA),
)
bad_score_cutoff_RNA = float(np.percentile(reference_local_scores_RNA, 95))
path_local_scores_RNA = residue_geometry_scores_RNA(
    generated_path_RNA,
    edge_pairs_RNA,
    edge_means_RNA,
    edge_stds_RNA,
    atom_to_residue_RNA,
    len(residue_labels_RNA),
)
atom_residue_indices_RNA = np.asarray([
    atom_to_residue_RNA[index_RNA] for index_RNA in range(len(atom_names_RNA))
])
path_clash_metrics_RNA = residue_clash_metrics_RNA(
    generated_path_RNA,
    atom_residue_indices_RNA,
    atom_names_RNA,
    residue_labels_RNA,
    SIMRNA_CG_CLASH_DISTANCE_RNA,
)
maximum_geometry_bad_residues_RNA = max(
    1, int(np.ceil(0.05 * len(residue_labels_RNA)))
)
bad_residues_by_step_RNA = []
neighbor_residues_by_step_RNA = []
for frame_index_RNA, frame_scores_RNA in enumerate(path_local_scores_RNA):
    frame_clashes_RNA = path_clash_metrics_RNA["has_clash_RNA"][frame_index_RNA]
    geometry_candidates_RNA = np.flatnonzero(
        frame_scores_RNA > bad_score_cutoff_RNA
    )
    ranked_geometry_RNA = geometry_candidates_RNA[
        np.argsort(frame_scores_RNA[geometry_candidates_RNA])[::-1]
    ]
    bad_RNA = sorted(set(
        np.flatnonzero(frame_clashes_RNA).tolist()
        + ranked_geometry_RNA[:maximum_geometry_bad_residues_RNA].tolist()
    ))
    bad_residues_by_step_RNA.append(bad_RNA)
    neighbor_residues_by_step_RNA.append(
        immediate_neighbors_RNA(bad_RNA, residue_labels_RNA)
    )

local_score_rows_RNA = []
for path_step_RNA, frame_scores_RNA in enumerate(path_local_scores_RNA):
    bad_set_RNA = set(bad_residues_by_step_RNA[path_step_RNA])
    neighbor_set_RNA = set(neighbor_residues_by_step_RNA[path_step_RNA])
    for residue_index_RNA, score_RNA in enumerate(frame_scores_RNA):
        chain_RNA, resid_RNA, resname_RNA = residue_labels_RNA[residue_index_RNA]
        partner_index_RNA = int(
            path_clash_metrics_RNA["partner_residues_RNA"]
            [path_step_RNA, residue_index_RNA]
        )
        if partner_index_RNA >= 0:
            partner_chain_RNA, partner_resid_RNA, partner_resname_RNA = (
                residue_labels_RNA[partner_index_RNA]
            )
        else:
            partner_chain_RNA, partner_resid_RNA, partner_resname_RNA = "", "", ""
        geometry_bad_RNA = score_RNA > bad_score_cutoff_RNA
        has_clash_RNA = bool(
            path_clash_metrics_RNA["has_clash_RNA"]
            [path_step_RNA, residue_index_RNA]
        )
        candidate_reason_RNA = "+".join([
            reason_RNA for reason_RNA, selected_RNA in (
                ("local_geometry", geometry_bad_RNA),
                ("clash", has_clash_RNA),
            ) if selected_RNA
        ])
        if residue_index_RNA in bad_set_RNA:
            restraint_class_RNA = "bad"
            occupancy_RNA, bfactor_RNA = 0.50, 10.00
        elif residue_index_RNA in neighbor_set_RNA:
            restraint_class_RNA = "neighbor"
            occupancy_RNA, bfactor_RNA = 0.25, 5.00
        else:
            restraint_class_RNA = "fixed"
            occupancy_RNA, bfactor_RNA = 0.00, 0.00
        local_score_rows_RNA.append({
            "path_step": path_step_RNA,
            "residue_index": residue_index_RNA,
            "chain": chain_RNA,
            "resid": resid_RNA,
            "resname": resname_RNA,
            "local_geometry_score": float(score_RNA),
            "training_p95_cutoff": bad_score_cutoff_RNA,
            "minimum_inter_nucleotide_distance_A": float(
                path_clash_metrics_RNA["minimum_distances_RNA"]
                [path_step_RNA, residue_index_RNA]
            ),
            "clash_distance_cutoff_A": SIMRNA_CG_CLASH_DISTANCE_RNA,
            "clash_count": int(
                path_clash_metrics_RNA["clash_counts_RNA"]
                [path_step_RNA, residue_index_RNA]
            ),
            "has_clash": int(has_clash_RNA),
            "closest_clash_partner_index": (
                partner_index_RNA if partner_index_RNA >= 0 else ""
            ),
            "closest_clash_partner_chain": partner_chain_RNA,
            "closest_clash_partner_resid": partner_resid_RNA,
            "closest_clash_partner_resname": partner_resname_RNA,
            "closest_clash_atom_pair": (
                path_clash_metrics_RNA["atom_pairs_RNA"]
                [path_step_RNA, residue_index_RNA]
            ),
            "candidate_reason": candidate_reason_RNA,
            "maximum_geometry_bad_residues": maximum_geometry_bad_residues_RNA,
            "total_bad_residues_in_frame": len(bad_set_RNA),
            "is_bad": int(residue_index_RNA in bad_set_RNA),
            "is_neighbor": int(residue_index_RNA in neighbor_set_RNA),
            "restraint_class": restraint_class_RNA,
            "occupancy": occupancy_RNA,
            "bfactor": bfactor_RNA,
        })
with open(
    output_dir_RNA / "simrna_local_geometry_scores.csv", "w", newline=""
) as out_RNA:
    writer_RNA = csv.DictWriter(
        out_RNA, fieldnames=local_score_rows_RNA[0].keys()
    )
    writer_RNA.writeheader()
    writer_RNA.writerows(local_score_rows_RNA)

if args_RNA.simrna_refine:
    simrna_executable_RNA = (
        args_RNA.simrna_executable.expanduser().resolve()
        if args_RNA.simrna_executable is not None
        else repository_root_RNA / "SimRNA" / "SimRNA"
    )
    simrna_data_directory_RNA = (
        args_RNA.simrna_data_directory.expanduser().resolve()
        if args_RNA.simrna_data_directory is not None else None
    )
    refined_dir_RNA = output_dir_RNA / (
        f"{start_state_RNA}_to_{end_state_RNA}_path_simrna_refined"
    )
    refined_dir_RNA.mkdir(parents=True, exist_ok=True)

    def path_number_RNA(path_RNA):
        try:
            return 0, int(path_RNA.stem.rsplit("_", 1)[-1])
        except ValueError:
            return 1, path_RNA.name

    ordered_pdbs_RNA = sorted(connected_pdbs_RNA, key=path_number_RNA)
    if len(ordered_pdbs_RNA) != len(bad_residues_by_step_RNA):
        raise ValueError(
            f"Found {len(ordered_pdbs_RNA)} path PDB files for "
            f"{len(bad_residues_by_step_RNA)} generated coordinate frames"
        )

    def available_cpu_count_RNA():
        """Respect process affinity and scheduler CPU limits when available."""
        try:
            affinity_count_RNA = len(os.sched_getaffinity(0))
        except AttributeError:
            affinity_count_RNA = os.cpu_count() or 1
        scheduler_counts_RNA = []
        for variable_RNA in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
            try:
                value_RNA = int(os.environ.get(variable_RNA, ""))
            except ValueError:
                continue
            if value_RNA > 0:
                scheduler_counts_RNA.append(value_RNA)
        return max(1, min([affinity_count_RNA] + scheduler_counts_RNA))

    available_cpus_RNA = available_cpu_count_RNA()
    requested_workers_RNA = (
        args_RNA.simrna_workers
        if args_RNA.simrna_workers > 0 else available_cpus_RNA
    )
    worker_count_RNA = max(
        1, min(requested_workers_RNA, available_cpus_RNA, len(ordered_pdbs_RNA))
    )
    print(
        f"Refining {len(ordered_pdbs_RNA)} generated structures with "
        f"{args_RNA.simrna_iterations} SimRNA iterations each; detected "
        f"{available_cpus_RNA} available CPUs and using "
        f"{worker_count_RNA} parallel workers"
    )

    def run_refinement_RNA(path_step_RNA, input_pdb_RNA):
        bad_residues_RNA = bad_residues_by_step_RNA[path_step_RNA]
        neighbor_residues_RNA = neighbor_residues_by_step_RNA[path_step_RNA]
        output_pdb_RNA = refined_dir_RNA / (
            f"{input_pdb_RNA.stem}_simrna_refined.pdb"
        )
        result_RNA = refine_pdb_with_simrna_RNA(
            input_pdb_RNA=input_pdb_RNA,
            output_pdb_RNA=output_pdb_RNA,
            executable_RNA=simrna_executable_RNA,
            iterations_RNA=args_RNA.simrna_iterations,
            data_directory_RNA=simrna_data_directory_RNA,
            run_name_RNA=f"molearn_path_{path_step_RNA:04d}",
            bad_residue_indices_RNA=bad_residues_RNA,
            neighbor_residue_indices_RNA=neighbor_residues_RNA,
        )
        row_RNA = {
            "path_step": path_step_RNA,
            "input_pdb": str(input_pdb_RNA),
            "refined_pdb": str(output_pdb_RNA),
            "bad_residues": result_RNA["bad_residues_RNA"],
            "neighbor_residues": result_RNA["neighbor_residues_RNA"],
            "minimum_unrestrained_energy": result_RNA["minimum_energy_RNA"],
            "energy_including_restraints": result_RNA[
                "minimum_restrained_energy_RNA"
            ],
            "minimum_temperature": result_RNA["minimum_temperature_RNA"],
            "minimum_frame": result_RNA["minimum_frame_RNA"],
            "minimum_write_number": result_RNA["minimum_write_number_RNA"],
            "minimum_replica_number": result_RNA["minimum_replica_number_RNA"],
            "frames_written": result_RNA["frames_written_RNA"],
            "iterations": result_RNA["iterations_RNA"],
            "maximum_frames": result_RNA["maximum_frames_RNA"],
            "write_frequency": result_RNA["write_frequency_RNA"],
            "random_seed": result_RNA["random_seed_RNA"],
            "parallel_workers": worker_count_RNA,
            "simrna_returncode": result_RNA["simrna_returncode_RNA"],
        }
        return row_RNA, result_RNA

    refinement_rows_by_step_RNA = [None] * len(ordered_pdbs_RNA)
    refinement_results_by_step_RNA = [None] * len(ordered_pdbs_RNA)
    with ThreadPoolExecutor(max_workers=worker_count_RNA) as executor_RNA:
        futures_RNA = {
            executor_RNA.submit(
                run_refinement_RNA, path_step_RNA, input_pdb_RNA
            ): path_step_RNA
            for path_step_RNA, input_pdb_RNA in enumerate(ordered_pdbs_RNA)
        }
        for future_RNA in as_completed(futures_RNA):
            path_step_RNA = futures_RNA[future_RNA]
            row_RNA, result_RNA = future_RNA.result()
            refinement_rows_by_step_RNA[path_step_RNA] = row_RNA
            refinement_results_by_step_RNA[path_step_RNA] = result_RNA
            print(
                f"  {Path(row_RNA['input_pdb']).name}: "
                f"{row_RNA['bad_residues']} bad + "
                f"{row_RNA['neighbor_residues']} neighbor residues; "
                f"minimum unrestrained energy "
                f"{row_RNA['minimum_unrestrained_energy']:.6f} at "
                f"frame {row_RNA['minimum_frame']}"
            )
    refinement_rows_RNA = list(refinement_rows_by_step_RNA)
    add_connect_directory_RNA(refined_dir_RNA)

    transition_trafl_RNA = output_dir_RNA / (
        f"{start_state_RNA}_to_{end_state_RNA}_path_simrna_refined.trafl"
    )
    write_transition_trafl_RNA(
        refinement_results_by_step_RNA, transition_trafl_RNA
    )
    all_atom_dir_RNA = output_dir_RNA / (
        f"{start_state_RNA}_to_{end_state_RNA}_path_simrna_refined_all_atom"
    )
    all_atom_names_RNA = [
        f"{pdb_RNA.stem}_simrna_refined_AA.pdb"
        for pdb_RNA in ordered_pdbs_RNA
    ]
    all_atom_pdbs_RNA = reconstruct_all_atom_trafl_RNA(
        template_pdb_RNA=ordered_pdbs_RNA[0],
        transition_trafl_RNA=transition_trafl_RNA,
        converter_executable_RNA=(
            simrna_executable_RNA.parent / "SimRNA_trafl2pdbs"
        ),
        output_directory_RNA=all_atom_dir_RNA,
        data_directory_RNA=simrna_data_directory_RNA,
        output_names_RNA=all_atom_names_RNA,
    )
    all_atom_rows_RNA = [
        {
            "path_step": path_step_RNA,
            "refined_cg_pdb": row_RNA["refined_pdb"],
            "all_atom_pdb": str(all_atom_pdb_RNA),
        }
        for path_step_RNA, (row_RNA, all_atom_pdb_RNA) in enumerate(
            zip(refinement_rows_RNA, all_atom_pdbs_RNA)
        )
    ]
    with open(
        output_dir_RNA / "simrna_all_atom_reconstruction.csv",
        "w", newline="",
    ) as out_RNA:
        writer_RNA = csv.DictWriter(
            out_RNA, fieldnames=all_atom_rows_RNA[0].keys()
        )
        writer_RNA.writeheader()
        writer_RNA.writerows(all_atom_rows_RNA)

    def read_pdb_coordinates_RNA(pdb_file_RNA):
        coordinates_RNA = []
        for line_RNA in Path(pdb_file_RNA).read_text().splitlines():
            if line_RNA.startswith(("ATOM  ", "HETATM")):
                coordinates_RNA.append([
                    float(line_RNA[30:38]),
                    float(line_RNA[38:46]),
                    float(line_RNA[46:54]),
                ])
        coordinates_RNA = np.asarray(coordinates_RNA, dtype=float)
        if coordinates_RNA.shape != generated_path_RNA.shape[1:]:
            raise ValueError(
                f"Unexpected refined coordinate shape {coordinates_RNA.shape} "
                f"in {pdb_file_RNA}; expected {generated_path_RNA.shape[1:]}"
            )
        return coordinates_RNA

    refined_path_RNA = np.asarray([
        read_pdb_coordinates_RNA(
            refined_dir_RNA / f"{pdb_RNA.stem}_simrna_refined.pdb"
        )
        for pdb_RNA in ordered_pdbs_RNA
    ])

    refined_clash_metrics_RNA = residue_clash_metrics_RNA(
        refined_path_RNA,
        atom_residue_indices_RNA,
        atom_names_RNA,
        residue_labels_RNA,
        SIMRNA_CG_CLASH_DISTANCE_RNA,
    )
    before_clash_pairs_RNA = (
        path_clash_metrics_RNA["clash_counts_RNA"].sum(axis=1) // 2
    )
    after_clash_pairs_RNA = (
        refined_clash_metrics_RNA["clash_counts_RNA"].sum(axis=1) // 2
    )
    before_clashing_residues_RNA = (
        path_clash_metrics_RNA["has_clash_RNA"].sum(axis=1)
    )
    after_clashing_residues_RNA = (
        refined_clash_metrics_RNA["has_clash_RNA"].sum(axis=1)
    )
    before_minimum_RNA = path_clash_metrics_RNA[
        "minimum_distances_RNA"
    ].min(axis=1)
    after_minimum_RNA = refined_clash_metrics_RNA[
        "minimum_distances_RNA"
    ].min(axis=1)
    clash_comparison_rows_RNA = []
    for path_step_RNA in range(len(refined_path_RNA)):
        clash_comparison_rows_RNA.append({
            "path_step": path_step_RNA,
            "clash_cutoff_A": SIMRNA_CG_CLASH_DISTANCE_RNA,
            "before_clash_pairs": int(before_clash_pairs_RNA[path_step_RNA]),
            "after_clash_pairs": int(after_clash_pairs_RNA[path_step_RNA]),
            "clash_pair_reduction": int(
                before_clash_pairs_RNA[path_step_RNA]
                - after_clash_pairs_RNA[path_step_RNA]
            ),
            "before_clashing_residues": int(
                before_clashing_residues_RNA[path_step_RNA]
            ),
            "after_clashing_residues": int(
                after_clashing_residues_RNA[path_step_RNA]
            ),
            "clashing_residue_reduction": int(
                before_clashing_residues_RNA[path_step_RNA]
                - after_clashing_residues_RNA[path_step_RNA]
            ),
            "before_minimum_inter_nucleotide_distance_A": float(
                before_minimum_RNA[path_step_RNA]
            ),
            "after_minimum_inter_nucleotide_distance_A": float(
                after_minimum_RNA[path_step_RNA]
            ),
            "minimum_distance_improvement_A": float(
                after_minimum_RNA[path_step_RNA]
                - before_minimum_RNA[path_step_RNA]
            ),
        })
    with open(
        output_dir_RNA / "clashes_refinement_comparison.csv", "w", newline=""
    ) as out_RNA:
        writer_RNA = csv.DictWriter(
            out_RNA, fieldnames=clash_comparison_rows_RNA[0].keys()
        )
        writer_RNA.writeheader()
        writer_RNA.writerows(clash_comparison_rows_RNA)

    fig_RNA, axes_RNA = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes_RNA[0].plot(
        steps_RNA, before_clash_pairs_RNA, "o-", color="tab:red",
        label="Before refinement",
    )
    axes_RNA[0].plot(
        steps_RNA, after_clash_pairs_RNA, "o-", color="tab:blue",
        label="After refinement",
    )
    axes_RNA[0].set_ylabel("Clashing bead pairs")
    axes_RNA[0].legend()
    axes_RNA[1].plot(
        steps_RNA, before_clashing_residues_RNA, "o-", color="tab:red"
    )
    axes_RNA[1].plot(
        steps_RNA, after_clashing_residues_RNA, "o-", color="tab:blue"
    )
    axes_RNA[1].set_ylabel("Clashing nucleotides")
    axes_RNA[2].axhline(
        SIMRNA_CG_CLASH_DISTANCE_RNA, color="black", linestyle="--",
        label=f"Hard-coded cutoff ({SIMRNA_CG_CLASH_DISTANCE_RNA:.1f} Å)",
    )
    axes_RNA[2].plot(
        steps_RNA, before_minimum_RNA, "o-", color="tab:red"
    )
    axes_RNA[2].plot(
        steps_RNA, after_minimum_RNA, "o-", color="tab:blue"
    )
    axes_RNA[2].set_ylabel("Minimum distance (Å)")
    axes_RNA[2].set_xlabel(
        f"Path step ({start_state_RNA} → {end_state_RNA})"
    )
    axes_RNA[2].legend()
    fig_RNA.suptitle("Coarse-grained clashes before and after refinement")
    fig_RNA.tight_layout()
    fig_RNA.savefig(
        output_dir_RNA / "clashes_refinement_comparison.png", dpi=150
    )
    finish_figure_RNA(fig_RNA)

    comparison_rows_RNA = []
    fig_RNA, axes_RNA = plt.subplots(
        len(pairs_RNA), 1, figsize=(10, 16), sharex=True
    )
    for axis_RNA, (label_RNA, indices_RNA) in zip(
        axes_RNA, pairs_RNA.items()
    ):
        reference_RNA = distances_RNA(reference_coords_RNA, indices_RNA).ravel()
        before_RNA = distances_RNA(generated_path_RNA, indices_RNA)
        after_RNA = distances_RNA(refined_path_RNA, indices_RNA)
        low_RNA, median_RNA, high_RNA = np.percentile(
            reference_RNA, [5, 50, 95]
        )
        axis_RNA.axhspan(low_RNA, high_RNA, color="tab:green", alpha=0.18)
        axis_RNA.axhline(median_RNA, color="tab:green", linestyle="--")
        axis_RNA.plot(
            steps_RNA, before_RNA.mean(axis=1), "o-", color="tab:red",
            label="Before refinement",
        )
        axis_RNA.plot(
            steps_RNA, after_RNA.mean(axis=1), "o-", color="tab:blue",
            label="After refinement",
        )
        axis_RNA.set_ylabel(f"{label_RNA}\nDistance (Å)")
        for step_RNA, (before_values_RNA, after_values_RNA) in enumerate(
            zip(before_RNA, after_RNA)
        ):
            before_deviation_RNA = float(
                np.abs(before_values_RNA - median_RNA).mean()
            )
            after_deviation_RNA = float(
                np.abs(after_values_RNA - median_RNA).mean()
            )
            before_outside_RNA = float((
                (before_values_RNA < low_RNA)
                | (before_values_RNA > high_RNA)
            ).mean())
            after_outside_RNA = float((
                (after_values_RNA < low_RNA)
                | (after_values_RNA > high_RNA)
            ).mean())
            comparison_rows_RNA.append({
                "path_step": step_RNA,
                "bond_type": label_RNA,
                "before_mean_A": float(before_values_RNA.mean()),
                "after_mean_A": float(after_values_RNA.mean()),
                "reference_median_A": float(median_RNA),
                "before_mean_abs_deviation_A": before_deviation_RNA,
                "after_mean_abs_deviation_A": after_deviation_RNA,
                "deviation_improvement_A": (
                    before_deviation_RNA - after_deviation_RNA
                ),
                "before_fraction_outside_reference_5_95": before_outside_RNA,
                "after_fraction_outside_reference_5_95": after_outside_RNA,
                "outside_fraction_improvement": (
                    before_outside_RNA - after_outside_RNA
                ),
            })
    axes_RNA[0].legend()
    axes_RNA[-1].set_xlabel(
        f"Path step ({start_state_RNA} → {end_state_RNA})"
    )
    fig_RNA.suptitle("Bond geometry before and after SimRNA refinement")
    fig_RNA.tight_layout()
    fig_RNA.savefig(
        output_dir_RNA / "bond_distances_refinement_comparison.png", dpi=150
    )
    finish_figure_RNA(fig_RNA)
    with open(
        output_dir_RNA / "bond_distances_refinement_comparison.csv",
        "w", newline="",
    ) as out_RNA:
        writer_RNA = csv.DictWriter(
            out_RNA, fieldnames=comparison_rows_RNA[0].keys()
        )
        writer_RNA.writeheader()
        writer_RNA.writerows(comparison_rows_RNA)

    with open(
        output_dir_RNA / "simrna_refinement_summary.csv", "w", newline=""
    ) as out_RNA:
        writer_RNA = csv.DictWriter(
            out_RNA, fieldnames=refinement_rows_RNA[0].keys()
        )
        writer_RNA.writeheader()
        writer_RNA.writerows(refinement_rows_RNA)
    print("Refined PDB files:", refined_dir_RNA)
    print("Refined transition TRAFL:", transition_trafl_RNA)
    print("All-atom reconstructed PDB files:", all_atom_dir_RNA)
    print(
        "Temporary SimRNA files, including restraint PDBs, reduced PDBs, "
        ".bonds, and .ss_detected files, were deleted."
    )
print("Analysis output:", output_dir_RNA)
