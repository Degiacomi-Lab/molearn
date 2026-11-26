"""Minimal reconstruction-only FoldingNet training example for RNA.

Run this script from any directory. Ensembles are defined in
``rna_ensembles.json`` and selected with ``--ensembles``.
"""

import argparse
import json
from pathlib import Path

import torch

from molearn.data import PDBData_RNA
from molearn.models.foldingnet import AutoEncoder
from molearn.trainers import Trainer_RNA


def parse_args_RNA():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=10,
        help="Read every Nth trajectory frame (default: 10; use 1 for all frames).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Run a fixed number of epochs instead of training until convergence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and perform one reconstruction pass without training.",
    )
    parser.add_argument(
        "--ensembles",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Manifest ensembles to combine, in reference-first order "
        "(default: apo CTI).",
    )
    parser.add_argument(
        "--ensemble-config",
        type=Path,
        default=None,
        help="JSON ensemble manifest (default: examples/rna_ensembles.json).",
    )
    parser.add_argument(
        "--holo-state",
        choices=("CTI", "LUM"),
        default=None,
        help="Legacy alias for --ensembles apo STATE.",
    )
    return parser.parse_args()


def main_RNA():
    args = parse_args_RNA()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be a positive integer")

    repository_root = Path(__file__).resolve().parents[1]
    example_dir = Path(__file__).resolve().parent
    if args.holo_state is not None and args.ensembles is not None:
        raise ValueError("Use either --ensembles or --holo-state, not both")
    ensemble_names_RNA = (
        ["apo", args.holo_state]
        if args.holo_state is not None
        else (args.ensembles or ["apo", "CTI"])
    )
    if len(set(ensemble_names_RNA)) != len(ensemble_names_RNA):
        raise ValueError("Each ensemble name may be selected only once")

    manifest_path_RNA = args.ensemble_config or example_dir / "rna_ensembles.json"
    with open(manifest_path_RNA) as infile_RNA:
        manifest_RNA = json.load(infile_RNA)
    unknown_RNA = [name for name in ensemble_names_RNA if name not in manifest_RNA]
    if unknown_RNA:
        raise ValueError(
            f"Unknown ensembles {unknown_RNA}; available: {list(manifest_RNA)}"
        )

    def resolve_path_RNA(filename_RNA):
        path_RNA = Path(filename_RNA).expanduser()
        return path_RNA if path_RNA.is_absolute() else repository_root / path_RNA

    topologies_RNA = [
        resolve_path_RNA(manifest_RNA[name]["topology"])
        for name in ensemble_names_RNA
    ]
    trajectories_RNA = [
        resolve_path_RNA(manifest_RNA[name]["trajectory"])
        for name in ensemble_names_RNA
    ]
    non_apo_RNA = [name for name in ensemble_names_RNA if name.lower() != "apo"]
    run_name_RNA = "_".join(non_apo_RNA or ensemble_names_RNA)
    # Keep these separate from checkpoints trained on independently oriented
    # apo/holo coordinates; those old models have learned the rigid rotation.
    output = example_dir / f"foldingnet_RNA_{run_name_RNA}_aligned_checkpoints"
    statistics = example_dir / f"data_statistics_RNA_{run_name_RNA}_aligned.json"

    input_files = topologies_RNA + trajectories_RNA
    missing = [path for path in input_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing yjdf input files:\n  " + "\n  ".join(map(str, missing))
        )

    data = PDBData_RNA()
    data.import_trajectories_RNA(
        filenames=[str(path) for path in trajectories_RNA],
        topologies=[str(path) for path in topologies_RNA],
    )
    # data.atomselect_RNA(
    #     "resname A C G U A3 and (name P or name C4')"
    # )
    data.atomselect_RNA(
    """
    (
        (resname A G A3)
        and
        (name P or name C4' or name N9 or name C2 or name C6)
    )
    or
    (
        (resname C U)
        and
        (name P or name C4' or name N1 or name C2 or name C4)
    )
    """
)
    # The first frame of the first named ensemble becomes the common rigid-body
    # reference. Put "apo" first when it should define the coordinate frame.
    data.ensemble_names_RNA = ensemble_names_RNA
    data.prepare_dataset_RNA(step=args.frame_step, align_RNA=True)
    data.write_statistics_RNA(str(statistics))

    print("Selected RNA atom counts:")
    for name, indices in data.indices.items():
        print(f"  {name}: {len(indices)}")
    print(f"Frames per ensemble: {data.frames_per_trajectory_RNA}")
    print(f"Ensembles: {ensemble_names_RNA}")
    print(f"Shared alignment reference: {ensemble_names_RNA[0]} frame 0")
    print(f"Checkpoint directory: {output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Reconstruction loss only: no OpenMM trainer and no prepare_physics call.
    trainer = Trainer_RNA(device=device)
    trainer.set_data_RNA(
        data,
        batch_size=16,
        validation_split=0.1,
        manual_seed=25,
        save_indices=False,
    )

    trainer.set_autoencoder(AutoEncoder, out_points=data.dataset.shape[1])
    trainer.prepare_optimiser()

    if args.dry_run:
        batch = next(iter(trainer.train_dataloader))[0].to(device)
        trainer.autoencoder.eval()
        with torch.no_grad():
            result = trainer.common_step(batch)
        print(f"Dry-run batch shape: {tuple(batch.shape)}")
        print(f"Dry-run reconstruction MSE: {result['mse_loss'].item():.6f}")
        return

    run_kwargs = dict(
        log_filename="log.dat",
        log_folder=str(output),
        checkpoint_folder=str(output),
        verbose=True,
    )
    if args.max_epochs is None:
        fit_results = trainer.run_until_converge(patience=16, **run_kwargs)
    else:
        fit_results = trainer.run(epochs=args.max_epochs, **run_kwargs)
    print(fit_results)


if __name__ == "__main__":
    main_RNA()
