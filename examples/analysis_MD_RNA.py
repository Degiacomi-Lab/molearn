"""Analyse a five-site RNA distance-matrix model trained from MD trajectories.

The default comparison uses the saved test plan: held-out apo frames and the
fully unseen LUM trajectory.  It produces reconstruction, latent-space,
compactness, transition-path, local-geometry, and connected-PDB outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from molearn.analysis import MolearnAnalysis, add_connect_directory_RNA
from molearn.data import PDBData_RNA
from molearn.models import DistanceMatrix_AE_RNA


def parse_args_RNA():
    parser_RNA = argparse.ArgumentParser(description=__doc__)
    parser_RNA.add_argument(
        "--training-root", type=Path, required=True,
        help="Directory containing the MD statistics and checkpoint folder.",
    )
    parser_RNA.add_argument("--checkpoint", type=Path, default=None)
    parser_RNA.add_argument("--statistics", type=Path, default=None)
    parser_RNA.add_argument("--data-root", type=Path, default=None)
    parser_RNA.add_argument("--output-root", type=Path, default=None)
    parser_RNA.add_argument("--ensemble-config", type=Path, default=None)
    parser_RNA.add_argument(
        "--training-ensembles", nargs="+", default=["apo", "CTI"],
        help="Names used to discover training files.",
    )
    parser_RNA.add_argument(
        "--ensembles", nargs="+", default=["apo", "LUM"],
        help="MD ensembles to compare.",
    )
    parser_RNA.add_argument("--start-state", default="apo")
    parser_RNA.add_argument("--end-state", default="LUM")
    parser_RNA.add_argument(
        "--trajectory-part", choices=("test", "development", "all"),
        default="test",
    )
    parser_RNA.add_argument("--frame-step", type=int, default=10)
    parser_RNA.add_argument("--batch-size", type=int, default=32)
    parser_RNA.add_argument(
        "--transition-states", type=int, default=10,
        help=(
            "Number of intermediate transition structures; the start and "
            "end structures are added separately (default: 10)."
        ),
    )
    parser_RNA.add_argument(
        "--detail-analysis", action="store_true",
        help="Also create per-dimension histograms, correlations, and raw plots.",
    )
    parser_RNA.add_argument(
        "--show", action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser_RNA.parse_args()


def finish_figure_RNA(figure_RNA, show_RNA):
    if show_RNA:
        plt.show()
    plt.close(figure_RNA)


def write_csv_RNA(path_RNA, rows_RNA):
    if not rows_RNA:
        return
    with open(path_RNA, "w", newline="") as outfile_RNA:
        writer_RNA = csv.DictWriter(
            outfile_RNA, fieldnames=rows_RNA[0].keys()
        )
        writer_RNA.writeheader()
        writer_RNA.writerows(rows_RNA)


def best_checkpoint_RNA(directory_RNA):
    converged_RNA = directory_RNA / "checkpoint_converged.ckpt"
    if converged_RNA.exists():
        return converged_RNA
    candidates_RNA = []
    pattern_RNA = re.compile(r"_loss([-+0-9.eE]+)\.ckpt$")
    for path_RNA in directory_RNA.glob("checkpoint_epoch*_loss*.ckpt"):
        match_RNA = pattern_RNA.search(path_RNA.name)
        if match_RNA:
            candidates_RNA.append((float(match_RNA.group(1)), path_RNA))
    if candidates_RNA:
        return min(candidates_RNA, key=lambda item_RNA: item_RNA[0])[1]
    last_RNA = directory_RNA / "last.ckpt"
    if last_RNA.exists():
        return last_RNA
    raise FileNotFoundError(f"No checkpoint found in {directory_RNA}")


def wasserstein_distance_1d_RNA(first_RNA, second_RNA):
    first_RNA = np.sort(np.asarray(first_RNA, dtype=float))
    second_RNA = np.sort(np.asarray(second_RNA, dtype=float))
    combined_RNA = np.sort(np.concatenate((first_RNA, second_RNA)))
    deltas_RNA = np.diff(combined_RNA)
    if not len(deltas_RNA):
        return 0.0
    first_cdf_RNA = np.searchsorted(
        first_RNA, combined_RNA[:-1], side="right"
    ) / len(first_RNA)
    second_cdf_RNA = np.searchsorted(
        second_RNA, combined_RNA[:-1], side="right"
    ) / len(second_RNA)
    return float(np.sum(
        np.abs(first_cdf_RNA - second_cdf_RNA) * deltas_RNA
    ))


def radius_of_gyration_RNA(coordinates_RNA):
    centered_RNA = coordinates_RNA - coordinates_RNA.mean(
        axis=1, keepdims=True
    )
    return np.sqrt(np.mean(np.sum(centered_RNA**2, axis=2), axis=1))


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
    for position_RNA, atoms_RNA in enumerate(residues_RNA):
        nitrogen_RNA = atoms_RNA.get("N9", atoms_RNA.get("N1"))
        outer_RNA = atoms_RNA.get("C6", atoms_RNA.get("C4"))
        candidates_RNA = {
            "P-C4'": (atoms_RNA.get("P"), atoms_RNA.get("C4'")),
            "C4'-N": (atoms_RNA.get("C4'"), nitrogen_RNA),
            "N-C2": (nitrogen_RNA, atoms_RNA.get("C2")),
            "C2-C4/C6": (atoms_RNA.get("C2"), outer_RNA),
            "C4/C6-N": (outer_RNA, nitrogen_RNA),
        }
        for label_RNA, pair_RNA in candidates_RNA.items():
            if None not in pair_RNA:
                pairs_RNA[label_RNA].append(pair_RNA)
        if position_RNA + 1 < len(residues_RNA):
            next_RNA = residues_RNA[position_RNA + 1]
            pair_RNA = (atoms_RNA.get("C4'"), next_RNA.get("P"))
            if None not in pair_RNA:
                pairs_RNA["C4'-P(next)"].append(pair_RNA)
    return pairs_RNA


def distances_RNA(coordinates_RNA, pairs_RNA):
    first_RNA = np.asarray([pair_RNA[0] for pair_RNA in pairs_RNA])
    second_RNA = np.asarray([pair_RNA[1] for pair_RNA in pairs_RNA])
    return np.linalg.norm(
        coordinates_RNA[:, first_RNA] - coordinates_RNA[:, second_RNA],
        axis=2,
    )


def main_RNA():
    args_RNA = parse_args_RNA()
    if min(args_RNA.frame_step, args_RNA.batch_size) < 1:
        raise ValueError("Frame step and batch size must be positive")
    if args_RNA.transition_states < 0:
        raise ValueError("--transition-states cannot be negative")
    if args_RNA.start_state not in args_RNA.ensembles:
        raise ValueError("--start-state must occur in --ensembles")
    if args_RNA.end_state not in args_RNA.ensembles:
        raise ValueError("--end-state must occur in --ensembles")

    repository_root_RNA = Path(__file__).resolve().parents[1]
    example_directory_RNA = Path(__file__).resolve().parent
    training_root_RNA = args_RNA.training_root.expanduser().resolve()
    data_root_RNA = (
        args_RNA.data_root.expanduser().resolve()
        if args_RNA.data_root is not None else repository_root_RNA
    )
    output_root_RNA = (
        args_RNA.output_root.expanduser().resolve()
        if args_RNA.output_root is not None else training_root_RNA
    )
    manifest_path_RNA = (
        args_RNA.ensemble_config.expanduser().resolve()
        if args_RNA.ensemble_config is not None
        else example_directory_RNA / "rna_ensembles.json"
    )
    training_name_RNA = "_".join(args_RNA.training_ensembles)
    checkpoint_directory_RNA = training_root_RNA / (
        f"distance_matrix_MD_RNA_{training_name_RNA}_aligned_checkpoints"
    )
    statistics_path_RNA = (
        args_RNA.statistics.expanduser().resolve()
        if args_RNA.statistics is not None
        else training_root_RNA / (
            f"data_statistics_DistanceMatrix_MD_RNA_"
            f"{training_name_RNA}_aligned.json"
        )
    )
    checkpoint_path_RNA = (
        args_RNA.checkpoint.expanduser().resolve()
        if args_RNA.checkpoint is not None
        else best_checkpoint_RNA(checkpoint_directory_RNA)
    )
    output_directory_RNA = output_root_RNA / (
        f"analysis_DistanceMatrix_MD_RNA_{training_name_RNA}_"
        f"{args_RNA.start_state}_to_{args_RNA.end_state}"
    )
    output_directory_RNA.mkdir(parents=True, exist_ok=True)

    with open(manifest_path_RNA) as infile_RNA:
        manifest_RNA = json.load(infile_RNA)
    with open(statistics_path_RNA) as infile_RNA:
        statistics_RNA = json.load(infile_RNA)
    unknown_RNA = [
        state_RNA for state_RNA in args_RNA.ensembles
        if state_RNA not in manifest_RNA
    ]
    if unknown_RNA:
        raise ValueError(f"Unknown ensembles: {unknown_RNA}")

    device_RNA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_RNA = torch.load(
        checkpoint_path_RNA, map_location=device_RNA, weights_only=True
    )
    network_kwargs_RNA = checkpoint_RNA["network_kwargs"]
    model_RNA = DistanceMatrix_AE_RNA(**network_kwargs_RNA).to(device_RNA)
    model_RNA.load_state_dict(checkpoint_RNA["model_state_dict"])
    model_RNA.eval()
    expected_points_RNA = (
        int(network_kwargs_RNA["residue_count_RNA"])
        * int(network_kwargs_RNA.get("sites_per_residue_RNA", 5))
    )
    alignment_reference_RNA = np.asarray(
        statistics_RNA["alignment_reference_RNA"], dtype=float
    )

    def resolve_RNA(filename_RNA):
        path_RNA = Path(filename_RNA).expanduser()
        return path_RNA if path_RNA.is_absolute() else data_root_RNA / path_RNA

    def interval_RNA(state_RNA):
        if args_RNA.trajectory_part == "all":
            return 0, None
        development_stop_RNA = int(
            statistics_RNA["development_frame_stop_RNA"]
        )
        if args_RNA.trajectory_part == "development":
            if state_RNA not in statistics_RNA.get(
                "training_ensemble_names_RNA", []
            ):
                raise ValueError(
                    f"{state_RNA} was not a training/development ensemble"
                )
            return int(statistics_RNA.get("development_frame_start_RNA", 0)), development_stop_RNA
        intervals_RNA = statistics_RNA.get("test_frame_intervals_RNA", {})
        if state_RNA not in intervals_RNA:
            raise ValueError(f"No saved test interval for {state_RNA}")
        start_RNA, stop_RNA = intervals_RNA[state_RNA]
        return int(start_RNA), None if stop_RNA is None else int(stop_RNA)

    datasets_RNA = {}
    intervals_used_RNA = {}
    for state_RNA in args_RNA.ensembles:
        source_RNA = manifest_RNA[state_RNA]
        start_RNA, stop_RNA = interval_RNA(state_RNA)
        data_RNA = PDBData_RNA(standardise=False)
        data_RNA.import_trajectories_RNA(
            filenames=[resolve_RNA(source_RNA["trajectory"])],
            topologies=[resolve_RNA(source_RNA["topology"])],
        )
        data_RNA.atomselect_five_site_RNA()
        data_RNA.prepare_dataset_RNA(
            mean=statistics_RNA["mean"],
            std=statistics_RNA["std"],
            start=start_RNA,
            stop=stop_RNA,
            step=args_RNA.frame_step,
            align_RNA=True,
            alignment_reference_RNA=alignment_reference_RNA,
        )
        if data_RNA.dataset.shape[1] != expected_points_RNA:
            raise ValueError(
                f"{state_RNA} has {data_RNA.dataset.shape[1]} sites; "
                f"checkpoint expects {expected_points_RNA}"
            )
        datasets_RNA[state_RNA] = data_RNA
        intervals_used_RNA[state_RNA] = {
            "start": start_RNA, "stop": stop_RNA,
            "step": args_RNA.frame_step,
            "selected_frames": len(data_RNA.dataset),
        }

    analysis_RNA = MolearnAnalysis(batch_size=args_RNA.batch_size)
    analysis_RNA.set_network(model_RNA)
    for state_RNA, data_RNA in datasets_RNA.items():
        analysis_RNA.set_dataset(state_RNA, data_RNA)

    print("Checkpoint:", checkpoint_path_RNA)
    print("Epoch:", checkpoint_RNA.get("epoch"))
    print("Device:", device_RNA)
    print("Intervals:", intervals_used_RNA)

    # Reconstruction error and distributions.
    rmsd_RNA = {
        state_RNA: np.asarray(analysis_RNA.get_error(state_RNA))
        for state_RNA in args_RNA.ensembles
    }
    reconstruction_rows_RNA = [
        {
            "ensemble": state_RNA,
            "frames": len(values_RNA),
            "mean_rmsd_A": float(values_RNA.mean()),
            "median_rmsd_A": float(np.median(values_RNA)),
            "p95_rmsd_A": float(np.percentile(values_RNA, 95)),
            "max_rmsd_A": float(values_RNA.max()),
        }
        for state_RNA, values_RNA in rmsd_RNA.items()
    ]
    write_csv_RNA(
        output_directory_RNA / "reconstruction_summary.csv",
        reconstruction_rows_RNA,
    )
    figure_RNA, axis_RNA = plt.subplots(figsize=(8, 5))
    for state_RNA, values_RNA in rmsd_RNA.items():
        axis_RNA.hist(values_RNA, bins=50, alpha=0.55, label=state_RNA)
    axis_RNA.set(xlabel="Reconstruction RMSD (Å)", ylabel="Frames")
    axis_RNA.set_title("MD RNA reconstruction error")
    axis_RNA.legend()
    figure_RNA.tight_layout()
    figure_RNA.savefig(output_directory_RNA / "reconstruction_rmsd.png", dpi=200)
    finish_figure_RNA(figure_RNA, args_RNA.show)

    # Latent distributions, separation, PCA, and effective dimensionality.
    encoded_RNA = {
        state_RNA: analysis_RNA.get_encoded(state_RNA).cpu().numpy()
        for state_RNA in args_RNA.ensembles
    }
    all_latent_RNA = np.concatenate(list(encoded_RNA.values()), axis=0)
    latent_dimensions_RNA = all_latent_RNA.shape[1]
    start_latent_RNA = encoded_RNA[args_RNA.start_state]
    end_latent_RNA = encoded_RNA[args_RNA.end_state]
    variances_RNA = all_latent_RNA.var(axis=0)
    separation_RNA = np.zeros(latent_dimensions_RNA)
    raw_wasserstein_RNA = np.zeros(latent_dimensions_RNA)
    for dimension_RNA in range(latent_dimensions_RNA):
        raw_wasserstein_RNA[dimension_RNA] = wasserstein_distance_1d_RNA(
            start_latent_RNA[:, dimension_RNA],
            end_latent_RNA[:, dimension_RNA],
        )
        standard_deviation_RNA = all_latent_RNA[:, dimension_RNA].std()
        if standard_deviation_RNA > np.finfo(float).eps:
            separation_RNA[dimension_RNA] = (
                raw_wasserstein_RNA[dimension_RNA]
                / standard_deviation_RNA
            )
    dimension_order_RNA = np.argsort(separation_RNA)[::-1]
    separation_rows_RNA = []
    for rank_RNA, dimension_RNA in enumerate(dimension_order_RNA, start=1):
        row_RNA = {
            "separation_rank": rank_RNA,
            "latent_dimension": int(dimension_RNA + 1),
            "wasserstein_distance": float(raw_wasserstein_RNA[dimension_RNA]),
            "standardized_wasserstein_distance": float(
                separation_RNA[dimension_RNA]
            ),
            "combined_variance": float(variances_RNA[dimension_RNA]),
        }
        for state_RNA in args_RNA.ensembles:
            values_RNA = encoded_RNA[state_RNA][:, dimension_RNA]
            row_RNA[f"{state_RNA}_mean"] = float(values_RNA.mean())
            row_RNA[f"{state_RNA}_std"] = float(values_RNA.std())
        separation_rows_RNA.append(row_RNA)
    write_csv_RNA(
        output_directory_RNA / "latent_dimension_separation.csv",
        separation_rows_RNA,
    )

    latent_mean_RNA = all_latent_RNA.mean(axis=0)
    centered_latent_RNA = all_latent_RNA - latent_mean_RNA
    _, singular_values_RNA, components_RNA = np.linalg.svd(
        centered_latent_RNA, full_matrices=False
    )
    pca_variances_RNA = singular_values_RNA**2 / max(
        len(all_latent_RNA) - 1, 1
    )
    variance_total_RNA = pca_variances_RNA.sum()
    pca_ratios_RNA = (
        pca_variances_RNA / variance_total_RNA
        if variance_total_RNA > 0 else np.zeros_like(pca_variances_RNA)
    )
    effective_dimension_RNA = (
        float(variance_total_RNA**2 / np.sum(pca_variances_RNA**2))
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
    write_csv_RNA(
        output_directory_RNA / "latent_pca_explained_variance.csv",
        pca_rows_RNA,
    )
    if len(components_RNA) >= 2:
        figure_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
        for state_RNA, latent_RNA in encoded_RNA.items():
            projected_RNA = (
                latent_RNA - latent_mean_RNA
            ) @ components_RNA[:2].T
            axis_RNA.scatter(
                projected_RNA[:, 0], projected_RNA[:, 1],
                s=8, alpha=0.35, label=state_RNA,
            )
        axis_RNA.set(
            xlabel=f"PC1 ({100 * pca_ratios_RNA[0]:.1f}% variance)",
            ylabel=f"PC2 ({100 * pca_ratios_RNA[1]:.1f}% variance)",
            title=f"PCA projection of all {latent_dimensions_RNA} dimensions",
        )
        axis_RNA.legend()
        figure_RNA.tight_layout()
        figure_RNA.savefig(output_directory_RNA / "latent_space_pca.png", dpi=200)
        finish_figure_RNA(figure_RNA, args_RNA.show)

    if args_RNA.detail_analysis:
        columns_RNA = min(4, latent_dimensions_RNA)
        rows_RNA = int(np.ceil(latent_dimensions_RNA / columns_RNA))
        figure_RNA, axes_RNA = plt.subplots(
            rows_RNA, columns_RNA,
            figsize=(4 * columns_RNA, 2.6 * rows_RNA), squeeze=False,
        )
        for dimension_RNA, axis_RNA in enumerate(axes_RNA.flat):
            if dimension_RNA >= latent_dimensions_RNA:
                axis_RNA.set_visible(False)
                continue
            for state_RNA in args_RNA.ensembles:
                axis_RNA.hist(
                    encoded_RNA[state_RNA][:, dimension_RNA], bins=40,
                    density=True, histtype="step", linewidth=1.4,
                    label=state_RNA,
                )
            axis_RNA.set_title(
                f"Dimension {dimension_RNA + 1} "
                f"(W/std={separation_RNA[dimension_RNA]:.2f})",
                fontsize=9,
            )
        handles_RNA, labels_RNA = axes_RNA.flat[0].get_legend_handles_labels()
        figure_RNA.legend(
            handles_RNA, labels_RNA, loc="upper center",
            ncol=len(args_RNA.ensembles),
        )
        figure_RNA.tight_layout(rect=(0, 0, 1, 0.97))
        figure_RNA.savefig(
            output_directory_RNA / "latent_dimension_histograms.png", dpi=200
        )
        finish_figure_RNA(figure_RNA, args_RNA.show)

        figure_RNA, axis_RNA = plt.subplots(figsize=(10, 4.5))
        axis_RNA.bar(np.arange(1, latent_dimensions_RNA + 1), variances_RNA)
        axis_RNA.set(
            xlabel="Latent dimension", ylabel="Variance",
            title="Variance used by each latent dimension",
        )
        figure_RNA.tight_layout()
        figure_RNA.savefig(
            output_directory_RNA / "latent_dimension_variance.png", dpi=200
        )
        finish_figure_RNA(figure_RNA, args_RNA.show)

        correlation_RNA = np.nan_to_num(
            np.corrcoef(all_latent_RNA, rowvar=False), nan=0.0
        )
        figure_RNA, axis_RNA = plt.subplots(figsize=(9, 8))
        image_RNA = axis_RNA.imshow(
            correlation_RNA, cmap="coolwarm", vmin=-1.0, vmax=1.0
        )
        axis_RNA.set_title("Latent-dimension correlation")
        figure_RNA.colorbar(image_RNA, ax=axis_RNA, label="Pearson correlation")
        figure_RNA.tight_layout()
        figure_RNA.savefig(
            output_directory_RNA / "latent_dimension_correlation.png", dpi=200
        )
        finish_figure_RNA(figure_RNA, args_RNA.show)

        best_x_RNA, best_y_RNA = dimension_order_RNA[:2]
        figure_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
        for state_RNA, latent_RNA in encoded_RNA.items():
            axis_RNA.scatter(
                latent_RNA[:, best_x_RNA], latent_RNA[:, best_y_RNA],
                s=8, alpha=0.35, label=state_RNA,
            )
        axis_RNA.set(
            xlabel=f"Latent dimension {best_x_RNA + 1}",
            ylabel=f"Latent dimension {best_y_RNA + 1}",
            title="Two most separating latent dimensions",
        )
        axis_RNA.legend()
        figure_RNA.tight_layout()
        figure_RNA.savefig(
            output_directory_RNA / "latent_space_most_separating_dimensions.png",
            dpi=200,
        )
        finish_figure_RNA(figure_RNA, args_RNA.show)

    diagnostics_RNA = {
        "latent_dimensions": latent_dimensions_RNA,
        "frames_per_ensemble": {
            state_RNA: len(values_RNA)
            for state_RNA, values_RNA in encoded_RNA.items()
        },
        "most_separating_dimensions": [
            int(index_RNA + 1) for index_RNA in dimension_order_RNA[:2]
        ],
        "effective_dimension_participation_ratio": effective_dimension_RNA,
        "pca_first_two_explained_variance_fraction": float(
            pca_ratios_RNA[:2].sum()
        ),
    }
    with open(
        output_directory_RNA / "latent_space_diagnostics.json", "w"
    ) as outfile_RNA:
        json.dump(diagnostics_RNA, outfile_RNA, indent=2)

    # Original versus reconstructed compactness.
    figure_RNA, axis_RNA = plt.subplots(figsize=(8, 5))
    for state_RNA in args_RNA.ensembles:
        original_RNA = analysis_RNA.get_dataset(
            state_RNA, scale=True
        ).cpu().numpy()
        decoded_RNA = analysis_RNA.get_decoded(
            state_RNA, scale=True
        ).cpu().numpy()
        axis_RNA.hist(
            radius_of_gyration_RNA(original_RNA), bins=50,
            histtype="step", linewidth=1.5,
            label=f"{state_RNA} original",
        )
        axis_RNA.hist(
            radius_of_gyration_RNA(decoded_RNA), bins=50,
            histtype="step", linewidth=1.5,
            label=f"{state_RNA} decoded",
        )
    axis_RNA.set(xlabel="Radius of gyration (Å)", ylabel="Frames")
    axis_RNA.set_title("MD RNA compactness")
    axis_RNA.legend()
    figure_RNA.tight_layout()
    figure_RNA.savefig(output_directory_RNA / "radius_of_gyration.png", dpi=200)
    finish_figure_RNA(figure_RNA, args_RNA.show)

    # Linear path between representative start/end conformations in full latent space.
    start_center_RNA = start_latent_RNA.mean(axis=0)
    end_center_RNA = end_latent_RNA.mean(axis=0)
    start_index_RNA = np.linalg.norm(
        start_latent_RNA - start_center_RNA, axis=1
    ).argmin()
    end_index_RNA = np.linalg.norm(
        end_latent_RNA - end_center_RNA, axis=1
    ).argmin()
    latent_path_RNA = np.linspace(
        start_latent_RNA[start_index_RNA],
        end_latent_RNA[end_index_RNA],
        num=args_RNA.transition_states + 2,
    )
    generated_path_RNA = analysis_RNA.generate(crd=latent_path_RNA)
    generated_path_RNA = datasets_RNA[
        args_RNA.start_state
    ].align_coordinates_RNA(
        generated_path_RNA, reference=alignment_reference_RNA
    )
    path_directory_RNA = output_directory_RNA / (
        f"{args_RNA.start_state}_to_{args_RNA.end_state}_path"
    )
    path_directory_RNA.mkdir(parents=True, exist_ok=True)
    for frame_RNA, coordinates_RNA in enumerate(generated_path_RNA):
        analysis_RNA._pdb_file(
            coordinates_RNA,
            str(path_directory_RNA / f"s_{frame_RNA:03d}.pdb"),
        )
    connected_RNA = add_connect_directory_RNA(path_directory_RNA)

    if len(components_RNA) >= 2:
        path_pca_RNA = (
            latent_path_RNA - latent_mean_RNA
        ) @ components_RNA[:2].T
        figure_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
        for state_RNA, latent_RNA in encoded_RNA.items():
            projected_RNA = (
                latent_RNA - latent_mean_RNA
            ) @ components_RNA[:2].T
            axis_RNA.scatter(
                projected_RNA[:, 0], projected_RNA[:, 1],
                s=8, alpha=0.2, label=state_RNA,
            )
        axis_RNA.plot(path_pca_RNA[:, 0], path_pca_RNA[:, 1], "k.-", label="Path")
        axis_RNA.set(
            xlabel=f"PC1 ({100 * pca_ratios_RNA[0]:.1f}% variance)",
            ylabel=f"PC2 ({100 * pca_ratios_RNA[1]:.1f}% variance)",
            title="Transition projected from the complete latent space",
        )
        axis_RNA.legend()
        figure_RNA.tight_layout()
        figure_RNA.savefig(output_directory_RNA / "linear_path_pca.png", dpi=200)
        finish_figure_RNA(figure_RNA, args_RNA.show)

    if args_RNA.detail_analysis:
        best_x_RNA, best_y_RNA = dimension_order_RNA[:2]
        figure_RNA, axis_RNA = plt.subplots(figsize=(7, 6))
        for state_RNA, latent_RNA in encoded_RNA.items():
            axis_RNA.scatter(
                latent_RNA[:, best_x_RNA], latent_RNA[:, best_y_RNA],
                s=8, alpha=0.2, label=state_RNA,
            )
        axis_RNA.plot(
            latent_path_RNA[:, best_x_RNA], latent_path_RNA[:, best_y_RNA],
            "k.-", label="Path",
        )
        axis_RNA.set(
            xlabel=f"Latent dimension {best_x_RNA + 1}",
            ylabel=f"Latent dimension {best_y_RNA + 1}",
            title="Transition in the most separating dimensions",
        )
        axis_RNA.legend()
        figure_RNA.tight_layout()
        figure_RNA.savefig(
            output_directory_RNA / "linear_path_most_separating_dimensions.png",
            dpi=200,
        )
        finish_figure_RNA(figure_RNA, args_RNA.show)

    # Compare path-local geometry against the MD test ensembles.
    connection_pairs_RNA = connection_indices_RNA(
        datasets_RNA[args_RNA.start_state]
    )
    reference_coordinates_RNA = np.concatenate([
        analysis_RNA.get_dataset(state_RNA, scale=True).cpu().numpy()
        for state_RNA in args_RNA.ensembles
    ])
    figure_RNA, axes_RNA = plt.subplots(
        len(connection_pairs_RNA), 1, figsize=(9, 15), sharex=True
    )
    geometry_rows_RNA = []
    steps_RNA = np.arange(len(generated_path_RNA))
    for axis_RNA, (label_RNA, pairs_RNA) in zip(
        axes_RNA, connection_pairs_RNA.items()
    ):
        reference_values_RNA = distances_RNA(
            reference_coordinates_RNA, pairs_RNA
        ).ravel()
        path_values_RNA = distances_RNA(generated_path_RNA, pairs_RNA)
        low_RNA, median_RNA, high_RNA = np.percentile(
            reference_values_RNA, [5, 50, 95]
        )
        axis_RNA.axhspan(low_RNA, high_RNA, color="tab:green", alpha=0.2)
        axis_RNA.axhline(median_RNA, color="tab:green", linestyle="--")
        axis_RNA.plot(steps_RNA, path_values_RNA.mean(axis=1), "ko-")
        axis_RNA.fill_between(
            steps_RNA, path_values_RNA.min(axis=1),
            path_values_RNA.max(axis=1), color="black", alpha=0.15,
        )
        axis_RNA.set_ylabel(f"{label_RNA}\nDistance (Å)")
        for step_RNA, values_RNA in enumerate(path_values_RNA):
            geometry_rows_RNA.append({
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
    axes_RNA[-1].set_xlabel(
        f"Path step ({args_RNA.start_state} → {args_RNA.end_state})"
    )
    figure_RNA.suptitle("MD RNA local geometry along generated path")
    figure_RNA.tight_layout()
    figure_RNA.savefig(output_directory_RNA / "bond_distances_path.png", dpi=200)
    finish_figure_RNA(figure_RNA, args_RNA.show)
    write_csv_RNA(
        output_directory_RNA / "bond_distances_path.csv", geometry_rows_RNA
    )

    summary_RNA = {
        "checkpoint": str(checkpoint_path_RNA),
        "checkpoint_epoch": checkpoint_RNA.get("epoch"),
        "trajectory_part": args_RNA.trajectory_part,
        "intervals": intervals_used_RNA,
        "reconstruction": reconstruction_rows_RNA,
        "latent": diagnostics_RNA,
        "intermediate_transition_states": args_RNA.transition_states,
        "transition_states": len(generated_path_RNA),
        "connected_pdb_files": len(connected_RNA),
    }
    with open(output_directory_RNA / "analysis_summary.json", "w") as outfile_RNA:
        json.dump(summary_RNA, outfile_RNA, indent=2)
    print("Reconstruction:", reconstruction_rows_RNA)
    print("Latent diagnostics:", diagnostics_RNA)
    print(
        "Generated path structures:", len(generated_path_RNA),
        f"(start + {args_RNA.transition_states} intermediate + end)",
    )
    print("Generated connected PDB files:", len(connected_RNA))
    print("Analysis directory:", output_directory_RNA)


if __name__ == "__main__":
    main_RNA()
