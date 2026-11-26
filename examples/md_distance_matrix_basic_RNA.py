"""Train the five-site RNA distance-matrix autoencoder on MD trajectories.

By default, apo and CTI frames train/validate the model.  The later apo frames
and the entirely unseen LUM trajectory are reserved for subsequent testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from molearn.data import PDBData_RNA
from molearn.models import DistanceMatrix_AE_RNA
from molearn.trainers import (
    DistanceMatrix_AE_Trainer_Config_RNA,
    DistanceMatrix_AE_Trainer_RNA,
)


def parse_args_RNA():
    parser_RNA = argparse.ArgumentParser(description=__doc__)
    parser_RNA.add_argument(
        "--train-ensembles", nargs="+", default=["apo", "CTI"],
        help="MD ensembles used for training and validation.",
    )
    parser_RNA.add_argument(
        "--test-ensembles", nargs="+", default=["apo", "LUM"],
        help="Ensembles recorded for later evaluation (not fitted here).",
    )
    parser_RNA.add_argument("--ensemble-config", type=Path, default=None)
    parser_RNA.add_argument("--data-root", type=Path, default=None)
    parser_RNA.add_argument("--output-root", type=Path, default=None)
    parser_RNA.add_argument(
        "--development-frames", type=int, default=22500,
        help="Use raw frames [0, N) from each training ensemble.",
    )
    parser_RNA.add_argument(
        "--frame-step", type=int, default=1,
        help="Read every Nth development frame; 1 uses every frame.",
    )
    parser_RNA.add_argument("--validation-split", type=float, default=0.1)
    parser_RNA.add_argument("--batch-size", type=int, default=4)
    parser_RNA.add_argument("--max-epochs", type=int, default=None)
    parser_RNA.add_argument("--latent-dimension", type=int, default=32)
    parser_RNA.add_argument("--initial-channels", type=int, default=32)
    parser_RNA.add_argument("--channel-multiplier", type=float, default=2.0)
    parser_RNA.add_argument("--minimum-map-size", type=int, default=9)
    parser_RNA.add_argument("--learning-rate", type=float, default=1e-3)
    parser_RNA.add_argument("--weight-decay", type=float, default=1e-4)
    parser_RNA.add_argument("--dm-weight", type=float, default=1.0)
    parser_RNA.add_argument("--local-k", type=int, default=4)
    parser_RNA.add_argument("--local-weight", type=float, default=0.9)
    parser_RNA.add_argument("--nonlocal-weight", type=float, default=0.1)
    parser_RNA.add_argument(
        "--openmm-cg-weight", type=float, default=0.0,
        help="Optional harmonic five-site energy weight (default: disabled).",
    )
    parser_RNA.add_argument("--openmm-bond-k", type=float, default=5000.0)
    parser_RNA.add_argument(
        "--dry-run", action="store_true",
        help="Run one forward/backward batch without fitting.",
    )
    return parser_RNA.parse_args()


def validate_args_RNA(args_RNA):
    positive_RNA = {
        "development frames": args_RNA.development_frames,
        "frame step": args_RNA.frame_step,
        "batch size": args_RNA.batch_size,
        "latent dimension": args_RNA.latent_dimension,
        "initial channels": args_RNA.initial_channels,
        "channel multiplier": args_RNA.channel_multiplier,
        "minimum map size": args_RNA.minimum_map_size,
        "learning rate": args_RNA.learning_rate,
        "OpenMM bond k": args_RNA.openmm_bond_k,
    }
    invalid_RNA = [
        name_RNA for name_RNA, value_RNA in positive_RNA.items()
        if value_RNA <= 0
    ]
    if invalid_RNA:
        raise ValueError(
            "These options must be positive: " + ", ".join(invalid_RNA)
        )
    if args_RNA.max_epochs is not None and args_RNA.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if not 0.0 < args_RNA.validation_split < 1.0:
        raise ValueError("--validation-split must be between zero and one")
    if args_RNA.local_k < 0:
        raise ValueError("--local-k cannot be negative")
    if any(
        value_RNA < 0 for value_RNA in (
            args_RNA.weight_decay,
            args_RNA.dm_weight,
            args_RNA.local_weight,
            args_RNA.nonlocal_weight,
            args_RNA.openmm_cg_weight,
        )
    ):
        raise ValueError("Loss weights and weight decay cannot be negative")
    if len(set(args_RNA.train_ensembles)) != len(args_RNA.train_ensembles):
        raise ValueError("Training ensemble names must be unique")
    if len(set(args_RNA.test_ensembles)) != len(args_RNA.test_ensembles):
        raise ValueError("Test ensemble names must be unique")


def main_RNA():
    args_RNA = parse_args_RNA()
    validate_args_RNA(args_RNA)

    repository_root_RNA = Path(__file__).resolve().parents[1]
    example_directory_RNA = Path(__file__).resolve().parent
    data_root_RNA = (
        args_RNA.data_root.expanduser().resolve()
        if args_RNA.data_root is not None else repository_root_RNA
    )
    output_root_RNA = (
        args_RNA.output_root.expanduser().resolve()
        if args_RNA.output_root is not None else example_directory_RNA
    )
    output_root_RNA.mkdir(parents=True, exist_ok=True)
    manifest_path_RNA = (
        args_RNA.ensemble_config.expanduser().resolve()
        if args_RNA.ensemble_config is not None
        else example_directory_RNA / "rna_ensembles.json"
    )
    with open(manifest_path_RNA) as infile_RNA:
        manifest_RNA = json.load(infile_RNA)

    requested_RNA = args_RNA.train_ensembles + args_RNA.test_ensembles
    unknown_RNA = [
        name_RNA for name_RNA in requested_RNA
        if name_RNA not in manifest_RNA
    ]
    if unknown_RNA:
        raise ValueError(
            f"Unknown ensembles {unknown_RNA}; available: {list(manifest_RNA)}"
        )

    def resolve_RNA(filename_RNA):
        path_RNA = Path(filename_RNA).expanduser()
        return path_RNA if path_RNA.is_absolute() else data_root_RNA / path_RNA

    training_sources_RNA = [
        manifest_RNA[name_RNA] for name_RNA in args_RNA.train_ensembles
    ]
    data_RNA = PDBData_RNA(standardise=False)
    data_RNA.import_trajectories_RNA(
        filenames=[
            resolve_RNA(source_RNA["trajectory"])
            for source_RNA in training_sources_RNA
        ],
        topologies=[
            resolve_RNA(source_RNA["topology"])
            for source_RNA in training_sources_RNA
        ],
    )
    available_frames_RNA = [
        len(molecule_RNA.trajectory) for molecule_RNA in data_RNA._mols_RNA
    ]
    too_short_RNA = [
        name_RNA for name_RNA, frames_RNA in zip(
            args_RNA.train_ensembles, available_frames_RNA
        ) if frames_RNA < args_RNA.development_frames
    ]
    if too_short_RNA:
        raise ValueError(
            f"Development boundary {args_RNA.development_frames} exceeds "
            f"available frames for {too_short_RNA}"
        )

    data_RNA.atomselect_five_site_RNA()
    data_RNA.ensemble_names_RNA = list(args_RNA.train_ensembles)
    data_RNA.training_ensemble_names_RNA = list(args_RNA.train_ensembles)
    data_RNA.test_ensemble_names_RNA = list(args_RNA.test_ensembles)
    data_RNA.test_frame_intervals_RNA = {
        name_RNA: [
            args_RNA.development_frames
            if name_RNA in args_RNA.train_ensembles else 0,
            None,
        ]
        for name_RNA in args_RNA.test_ensembles
    }
    data_RNA.prepare_dataset_RNA(
        stop=args_RNA.development_frames,
        step=args_RNA.frame_step,
        align_RNA=True,
    )
    if data_RNA.dataset.shape[1] % 5:
        raise ValueError("The selected MD topology is not five sites per RNA")
    residue_count_RNA = data_RNA.dataset.shape[1] // 5
    data_RNA.development_frame_start_RNA = 0
    data_RNA.development_frame_stop_RNA = args_RNA.development_frames
    data_RNA.validation_split_RNA = args_RNA.validation_split

    run_name_RNA = "_".join(args_RNA.train_ensembles)
    checkpoint_directory_RNA = output_root_RNA / (
        f"distance_matrix_MD_RNA_{run_name_RNA}_aligned_checkpoints"
    )
    statistics_path_RNA = output_root_RNA / (
        f"data_statistics_DistanceMatrix_MD_RNA_{run_name_RNA}_aligned.json"
    )
    data_RNA.write_statistics_RNA(statistics_path_RNA)

    device_RNA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_RNA = DistanceMatrix_AE_Trainer_Config_RNA(
        dm_weight=args_RNA.dm_weight,
        local_k=args_RNA.local_k,
        local_weight_RNA=args_RNA.local_weight,
        nonlocal_weight_RNA=args_RNA.nonlocal_weight,
    )
    trainer_RNA = DistanceMatrix_AE_Trainer_RNA(
        residue_count_RNA=residue_count_RNA,
        device=device_RNA,
        config_RNA=config_RNA,
    )
    trainer_RNA.set_data_RNA(
        data_RNA,
        batch_size=args_RNA.batch_size,
        validation_split=args_RNA.validation_split,
        manual_seed=25,
        save_indices=True,
        indices_dir=str(checkpoint_directory_RNA),
    )
    trainer_RNA.prepare_openmm_simrna_RNA(
        weight_RNA=args_RNA.openmm_cg_weight,
        bond_k_RNA=args_RNA.openmm_bond_k,
    )
    trainer_RNA.set_autoencoder(
        DistanceMatrix_AE_RNA,
        residue_count_RNA=residue_count_RNA,
        sites_per_residue_RNA=5,
        latent_dimension=args_RNA.latent_dimension,
        init_c=args_RNA.initial_channels,
        m=args_RNA.channel_multiplier,
        min_size=args_RNA.minimum_map_size,
    )
    trainer_RNA.prepare_optimiser(
        lr=args_RNA.learning_rate,
        weight_decay=args_RNA.weight_decay,
    )

    print(f"Training ensembles: {args_RNA.train_ensembles}")
    print(f"Available raw frames: {available_frames_RNA}")
    print(f"Selected training frames per ensemble: {data_RNA.frames_per_trajectory_RNA}")
    print(f"Test ensembles: {args_RNA.test_ensembles}")
    print(f"Test raw-frame intervals: {data_RNA.test_frame_intervals_RNA}")
    print(f"RNA residues/sites: {residue_count_RNA}/{data_RNA.dataset.shape[1]}")
    matrix_side_RNA = data_RNA.dataset.shape[1]
    print(f"Full distance matrix: (1, {matrix_side_RNA}, {matrix_side_RNA})")
    print(
        f"Development total: {len(data_RNA.dataset)}; "
        f"training: {len(data_RNA.train_dataset)}; "
        f"validation: {len(data_RNA.valid_dataset)}"
    )
    print(f"Device: {device_RNA}")
    print(f"Statistics: {statistics_path_RNA}")
    print(f"Checkpoint directory: {checkpoint_directory_RNA}")

    if args_RNA.dry_run:
        batch_RNA = next(iter(trainer_RNA.train_dataloader))[0].to(device_RNA)
        result_RNA = trainer_RNA.common_step_RNA(batch_RNA)
        result_RNA["loss"].backward()
        print(f"Dry-run batch: {tuple(batch_RNA.shape)}")
        print(f"Dry-run distance-map loss: {result_RNA['dm_loss'].item():.6f}")
        print(f"Dry-run weighted DM RMSE: {result_RNA['dm_rmse_A'].item():.6f} A")
        return

    run_kwargs_RNA = {
        "log_filename": "log.dat",
        "log_folder": str(checkpoint_directory_RNA),
        "checkpoint_folder": str(checkpoint_directory_RNA),
        "verbose": True,
    }
    if args_RNA.max_epochs is None:
        result_RNA = trainer_RNA.run_until_converge(
            patience=16, **run_kwargs_RNA
        )
    else:
        result_RNA = trainer_RNA.run(
            epochs=args_RNA.max_epochs, **run_kwargs_RNA
        )
    print(result_RNA)


if __name__ == "__main__":
    main_RNA()
