"""Train a reconstruction-only FoldingNet directly from SimRNA TRC files."""

import argparse
import json
from pathlib import Path

import torch

from molearn.data import SimRNAData_RNA
from molearn.models.foldingnet import AutoEncoder
from molearn.trainers import Trainer_RNA


def parse_args_RNA():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensembles",
        nargs="+",
        default=["apo", "holo"],
        metavar="NAME",
        help="Manifest ensembles in alignment-reference-first order.",
    )
    parser.add_argument(
        "--ensemble-config",
        type=Path,
        default=None,
        help="JSON manifest (default: examples/simrna_ensembles.json).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Read every Nth TRC frame (default: 10).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Run a fixed number of epochs instead of early stopping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--openmm-cg-weight", type=float, default=0.0,
        help="Weight for the optional OpenMM harmonic five-site baseline.",
    )
    parser.add_argument(
        "--openmm-bond-k", type=float, default=5000.0,
        help="CG harmonic bond constant in kJ mol-1 nm-2.",
    )
    parser.add_argument(
        "--simrna-sp-weight", type=float, default=0.0,
        help=(
            "Shared loss weight for local SimRNA bond, angle, and eta/theta "
            "statistical-potential terms."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and perform one reconstruction pass without training.",
    )
    return parser.parse_args()


def main_RNA():
    args = parse_args_RNA()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if (
        args.openmm_cg_weight < 0
        or args.simrna_sp_weight < 0
        or args.openmm_bond_k <= 0
    ):
        raise ValueError("Energy weights must be non-negative and bond k positive")
    if len(set(args.ensembles)) != len(args.ensembles):
        raise ValueError("Each ensemble may be selected only once")

    repository_root = Path(__file__).resolve().parents[1]
    example_dir = Path(__file__).resolve().parent
    manifest_path_RNA = args.ensemble_config or example_dir / "simrna_ensembles.json"
    with open(manifest_path_RNA) as infile_RNA:
        manifest_RNA = json.load(infile_RNA)
    unknown_RNA = [name for name in args.ensembles if name not in manifest_RNA]
    if unknown_RNA:
        raise ValueError(
            f"Unknown ensembles {unknown_RNA}; available: {list(manifest_RNA)}"
        )

    def resolve_RNA(filename_RNA):
        path_RNA = Path(filename_RNA).expanduser()
        return path_RNA if path_RNA.is_absolute() else repository_root / path_RNA

    sources_RNA = [manifest_RNA[name] for name in args.ensembles]
    trc_files_RNA = [resolve_RNA(source["trc"]) for source in sources_RNA]
    hdr_files_RNA = [resolve_RNA(source["hdr"]) for source in sources_RNA]
    pdb_files_RNA = [resolve_RNA(source["pdb"]) for source in sources_RNA]

    run_name_RNA = "_".join(args.ensembles)
    output_RNA = (
        example_dir
        / f"foldingnet_SimRNA_RNA_{run_name_RNA}_aligned_checkpoints"
    )
    statistics_RNA = (
        example_dir / f"data_statistics_SimRNA_RNA_{run_name_RNA}_aligned.json"
    )

    data_RNA = SimRNAData_RNA()
    data_RNA.import_simrna_trajectories_RNA(
        trc_files_RNA, hdr_files_RNA, pdb_files_RNA
    )
    # The supplied SimRNA PDB already contains the intended five sites per
    # nucleotide. Dummy pseudo-torsion sites occur only in TRC and are excluded.
    data_RNA.atomselect_RNA("all")
    data_RNA.ensemble_names_RNA = list(args.ensembles)
    data_RNA.prepare_dataset_RNA(step=args.frame_step, align_RNA=True)
    data_RNA.write_statistics_RNA(statistics_RNA)

    print(f"Ensembles: {args.ensembles}")
    print(f"Frames per ensemble: {data_RNA.frames_per_trajectory_RNA}")
    print(f"Physical RNA points: {data_RNA.dataset.shape[1]}")
    print(f"SimRNA dummy sites excluded: {data_RNA.unmapped_trc_atoms_RNA}")
    print(f"Shared alignment reference: {args.ensembles[0]} frame 0")
    print(f"Checkpoint directory: {output_RNA}")

    device_RNA = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device_RNA}")
    trainer_RNA = Trainer_RNA(device=device_RNA)
    trainer_RNA.set_data_RNA(
        data_RNA,
        batch_size=args.batch_size,
        validation_split=0.1,
        manual_seed=25,
        save_indices=False,
    )
    trainer_RNA.prepare_openmm_simrna_RNA(
        weight_RNA=args.openmm_cg_weight,
        bond_k_RNA=args.openmm_bond_k,
    )
    trainer_RNA.prepare_simrna_sp_RNA(weight_RNA=args.simrna_sp_weight)
    trainer_RNA.set_autoencoder(
        AutoEncoder, out_points=data_RNA.dataset.shape[1]
    )
    trainer_RNA.prepare_optimiser()
    print(f"OpenMM five-site CG weight: {args.openmm_cg_weight}")
    print(f"Local SimRNA SP weight: {args.simrna_sp_weight}")

    if args.dry_run:
        batch_RNA = next(iter(trainer_RNA.train_dataloader))[0].to(device_RNA)
        trainer_RNA.autoencoder.eval()
        with torch.no_grad():
            result_RNA = trainer_RNA.common_step(batch_RNA)
        print(f"Dry-run batch shape: {tuple(batch_RNA.shape)}")
        print(f"Dry-run reconstruction MSE: {result_RNA['mse_loss'].item():.6f}")
        print(
            "Dry-run OpenMM CG energy (kJ/mol): "
            f"{result_RNA['openmm_simrna_energy'].item():.6f}"
        )
        print(
            "Dry-run local SimRNA SP energy: "
            f"{result_RNA['simrna_sp_energy'].item():.6f}"
        )
        return

    run_kwargs_RNA = dict(
        log_filename="log.dat",
        log_folder=str(output_RNA),
        checkpoint_folder=str(output_RNA),
        verbose=True,
    )
    if args.max_epochs is None:
        result_RNA = trainer_RNA.run_until_converge(
            patience=16, **run_kwargs_RNA
        )
    else:
        result_RNA = trainer_RNA.run(
            epochs=args.max_epochs, **run_kwargs_RNA
        )
    print(result_RNA)


if __name__ == "__main__":
    main_RNA()
