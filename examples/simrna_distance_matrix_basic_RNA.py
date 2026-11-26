"""Train RNA using one triangular matrix over all five-site SimRNA coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from molearn.data import SimRNAData_RNA
from molearn.models import DistanceMatrix_AE_RNA
from molearn.trainers import (
    DistanceMatrix_AE_Trainer_Config_RNA,
    DistanceMatrix_AE_Trainer_RNA,
)


def parse_args_RNA():
    parser_RNA = argparse.ArgumentParser(description=__doc__)
    parser_RNA.add_argument(
        "--ensembles", nargs="+", default=["apo", "holo"], metavar="NAME",
        help="Manifest ensembles in alignment-reference-first order.",
    )
    parser_RNA.add_argument(
        "--ensemble-config", type=Path, default=None,
        help="JSON manifest (default: examples/simrna_ensembles.json).",
    )
    parser_RNA.add_argument(
        "--data-root", type=Path, default=None,
        help="Base directory for relative manifest paths (default: repository).",
    )
    parser_RNA.add_argument(
        "--output-root", type=Path, default=None,
        help="Checkpoint/statistics directory (default: examples).",
    )
    parser_RNA.add_argument(
        "--frame-step", type=int, default=1,
        help="Read every Nth TRC frame; use 1 for all frames.",
    )
    parser_RNA.add_argument(
        "--development-frames", type=int, default=7007,
        help=(
            "Use raw frames [0, N) from every ensemble for training and "
            "validation; later frames remain held out for testing "
            "(default: 7007)."
        ),
    )
    parser_RNA.add_argument(
        "--validation-split", type=float, default=0.1,
        help="Fraction of the development pool used for validation.",
    )
    parser_RNA.add_argument(
        "--max-epochs", type=int, default=None,
        help="Fixed epoch count; omit to use validation patience.",
    )
    parser_RNA.add_argument("--batch-size", type=int, default=16)
    parser_RNA.add_argument(
        "--latent-dimension", type=int, default=32,
        help="Latent-space dimensions (default: 32).",
    )
    parser_RNA.add_argument(
        "--initial-channels", type=int, default=32,
        help="Channels after the first encoder/decoder layer.",
    )
    parser_RNA.add_argument(
        "--channel-multiplier", type=float, default=2.0,
    )
    parser_RNA.add_argument("--minimum-map-size", type=int, default=9)
    parser_RNA.add_argument("--learning-rate", type=float, default=1e-3)
    parser_RNA.add_argument("--weight-decay", type=float, default=1e-4)
    parser_RNA.add_argument("--dm-weight", type=float, default=1.0)
    parser_RNA.add_argument(
        "--local-k", "--local-residue-span", dest="local_k",
        type=int, default=4,
        help=(
            "Flattened site-index separation receiving the local distance "
            "weight, matching the protein trainer (default: 4)."
        ),
    )
    parser_RNA.add_argument("--local-weight", type=float, default=0.9)
    parser_RNA.add_argument("--nonlocal-weight", type=float, default=0.1)
    parser_RNA.add_argument(
        "--openmm-cg-weight", type=float, default=0.0,
        help="Optional five-site harmonic OpenMM loss weight.",
    )
    parser_RNA.add_argument("--openmm-bond-k", type=float, default=5000.0)
    parser_RNA.add_argument(
        "--simrna-sp-weight", type=float, default=0.0,
        help=(
            "Shared loss weight for the local SimRNA bond, angle, and "
            "eta/theta statistical-potential terms (default: disabled)."
        ),
    )
    parser_RNA.add_argument(
        "--dry-run", action="store_true",
        help="Load one batch and run forward/backward without training.",
    )
    return parser_RNA.parse_args()


def validate_args_RNA(args_RNA):
    positive_RNA = {
        "frame step": args_RNA.frame_step,
        "development frames": args_RNA.development_frames,
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
    if args_RNA.local_k < 0:
        raise ValueError("--local-k cannot be negative")
    if not 0.0 < args_RNA.validation_split < 1.0:
        raise ValueError("--validation-split must be between zero and one")
    if any(
        value_RNA < 0
        for value_RNA in (
            args_RNA.dm_weight,
            args_RNA.local_weight,
            args_RNA.nonlocal_weight,
            args_RNA.openmm_cg_weight,
            args_RNA.simrna_sp_weight,
            args_RNA.weight_decay,
        )
    ):
        raise ValueError("Loss weights and weight decay cannot be negative")
    if len(set(args_RNA.ensembles)) != len(args_RNA.ensembles):
        raise ValueError("Each ensemble may be selected only once")


def main_RNA():
    args_RNA = parse_args_RNA()
    validate_args_RNA(args_RNA)

    repository_root_RNA = Path(__file__).resolve().parents[1]
    example_directory_RNA = Path(__file__).resolve().parent
    data_root_RNA = (
        args_RNA.data_root.expanduser().resolve()
        if args_RNA.data_root is not None
        else repository_root_RNA
    )
    output_root_RNA = (
        args_RNA.output_root.expanduser().resolve()
        if args_RNA.output_root is not None
        else example_directory_RNA
    )
    output_root_RNA.mkdir(parents=True, exist_ok=True)
    manifest_path_RNA = (
        args_RNA.ensemble_config.expanduser().resolve()
        if args_RNA.ensemble_config is not None
        else example_directory_RNA / "simrna_ensembles.json"
    )
    with open(manifest_path_RNA) as infile_RNA:
        manifest_RNA = json.load(infile_RNA)
    unknown_RNA = [
        name_RNA for name_RNA in args_RNA.ensembles
        if name_RNA not in manifest_RNA
    ]
    if unknown_RNA:
        raise ValueError(
            f"Unknown ensembles {unknown_RNA}; available: {list(manifest_RNA)}"
        )

    def resolve_RNA(filename_RNA):
        path_RNA = Path(filename_RNA).expanduser()
        return (
            path_RNA if path_RNA.is_absolute()
            else data_root_RNA / path_RNA
        )

    sources_RNA = [
        manifest_RNA[name_RNA] for name_RNA in args_RNA.ensembles
    ]
    data_RNA = SimRNAData_RNA(standardise=False)
    data_RNA.import_simrna_trajectories_RNA(
        [resolve_RNA(source_RNA["trc"]) for source_RNA in sources_RNA],
        [resolve_RNA(source_RNA["hdr"]) for source_RNA in sources_RNA],
        [resolve_RNA(source_RNA["pdb"]) for source_RNA in sources_RNA],
    )
    data_RNA.atomselect_RNA("all")
    data_RNA.ensemble_names_RNA = list(args_RNA.ensembles)
    data_RNA.prepare_dataset_RNA(
        stop=args_RNA.development_frames,
        step=args_RNA.frame_step,
        align_RNA=True,
    )
    if data_RNA.dataset.shape[1] % 5:
        raise ValueError(
            "The SimRNA dataset does not contain exactly five sites per residue"
        )
    residue_count_RNA = data_RNA.dataset.shape[1] // 5
    data_RNA.development_frame_start_RNA = 0
    data_RNA.development_frame_stop_RNA = args_RNA.development_frames
    data_RNA.test_frame_start_RNA = args_RNA.development_frames
    data_RNA.validation_split_RNA = args_RNA.validation_split

    run_name_RNA = "_".join(args_RNA.ensembles)
    checkpoint_directory_RNA = (
        output_root_RNA
        / f"distance_matrix_SimRNA_RNA_{run_name_RNA}_aligned_checkpoints"
    )
    statistics_path_RNA = (
        output_root_RNA
        / f"data_statistics_DistanceMatrix_SimRNA_RNA_"
        f"{run_name_RNA}_aligned.json"
    )
    data_RNA.write_statistics_RNA(statistics_path_RNA)

    device_RNA = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
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
    trainer_RNA.prepare_simrna_sp_RNA(
        weight_RNA=args_RNA.simrna_sp_weight,
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

    print(f"Ensembles: {args_RNA.ensembles}")
    print(f"Frames per ensemble: {data_RNA.frames_per_trajectory_RNA}")
    print(f"RNA residues: {residue_count_RNA}")
    print(f"RNA points: {data_RNA.dataset.shape[1]}")
    print("Distance-matrix channels before folding: 1")
    print(
        "Full distance matrix per frame: "
        f"(1, {5 * residue_count_RNA}, {5 * residue_count_RNA})"
    )
    packed_side_RNA = (5 * residue_count_RNA + 1) // 2
    print(
        "Losslessly folded triangular encoder input per frame: "
        f"(2, {packed_side_RNA}, {packed_side_RNA})"
    )
    print(
        "Development raw-frame interval per ensemble: "
        f"[0, {args_RNA.development_frames})"
    )
    print(
        "Held-out test raw-frame interval per ensemble: "
        f"[{args_RNA.development_frames}, end)"
    )
    print(
        f"Selected development frames: {len(data_RNA.dataset)}; "
        f"training: {len(data_RNA.train_dataset)}; "
        f"validation: {len(data_RNA.valid_dataset)}"
    )
    print(f"Latent dimensions: {args_RNA.latent_dimension}")
    print(f"Device: {device_RNA}")
    print(f"Checkpoint directory: {checkpoint_directory_RNA}")

    if args_RNA.dry_run:
        batch_RNA = next(iter(trainer_RNA.train_dataloader))[0].to(
            device_RNA
        )
        trainer_RNA.autoencoder.train()
        result_RNA = trainer_RNA.common_step_RNA(batch_RNA)
        result_RNA["loss"].backward()
        print(f"Dry-run batch: {tuple(batch_RNA.shape)}")
        print(
            "Dry-run full distance matrix: "
            f"{tuple(trainer_RNA.autoencoder.coords_to_distance_matrix_RNA(batch_RNA).shape)}"
        )
        print(f"Dry-run distance-map loss: {result_RNA['dm_loss'].item():.6f}")
        print(
            "Dry-run weighted distance RMSE: "
            f"{result_RNA['dm_rmse_A'].item():.6f} A"
        )
        print(
            "Dry-run OpenMM CG energy: "
            f"{result_RNA['openmm_simrna_energy'].item():.6f}"
        )
        print(
            "Dry-run local SimRNA SP energy: "
            f"{result_RNA['simrna_sp_energy'].item():.6f}"
        )
        return

    run_kwargs_RNA = {
        "log_filename": "log.dat",
        "log_folder": str(checkpoint_directory_RNA),
        "checkpoint_folder": str(checkpoint_directory_RNA),
        "verbose": True,
    }
    if args_RNA.max_epochs is None:
        fit_result_RNA = trainer_RNA.run_until_converge(
            patience=16, **run_kwargs_RNA
        )
    else:
        fit_result_RNA = trainer_RNA.run(
            epochs=args_RNA.max_epochs, **run_kwargs_RNA
        )
    print(fit_result_RNA)


if __name__ == "__main__":
    main_RNA()
