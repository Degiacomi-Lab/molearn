"""Optimize the SimRNA RNA distance-matrix autoencoder, one worker per GPU."""

import argparse
import csv
import json
import os
from pathlib import Path

import optuna
import torch

from molearn.data import SimRNAData_RNA
from molearn.models import DistanceMatrix_AE_RNA
from molearn.trainers import (
    DistanceMatrix_AE_Trainer_Config_RNA,
    DistanceMatrix_AE_Trainer_RNA,
)


def parse_args_RNA():
    parser_RNA = argparse.ArgumentParser(description=__doc__)
    parser_RNA.add_argument("--storage", required=True)
    parser_RNA.add_argument(
        "--study-name", default="molearn_simrna_distance_matrix_converged"
    )
    parser_RNA.add_argument("--n-trials", type=int, default=10)
    parser_RNA.add_argument(
        "--max-epochs", type=int, default=500,
        help="Safety ceiling per trial; convergence can stop training earlier.",
    )
    parser_RNA.add_argument(
        "--patience", type=int, default=40,
        help="Stop after this many epochs without validation-loss improvement.",
    )
    parser_RNA.add_argument("--ensembles", nargs="+", default=["apo", "holo"])
    parser_RNA.add_argument("--ensemble-config", type=Path, required=True)
    parser_RNA.add_argument("--data-root", type=Path, required=True)
    parser_RNA.add_argument("--output-root", type=Path, required=True)
    parser_RNA.add_argument("--frame-step", type=int, default=1)
    parser_RNA.add_argument("--development-frames", type=int, default=7007)
    parser_RNA.add_argument("--validation-split", type=float, default=0.1)
    parser_RNA.add_argument("--split-seed", type=int, default=25)
    parser_RNA.add_argument("--openmm-bond-k", type=float, default=5000.0)
    parser_RNA.add_argument(
        "--simrna-sp-weight", type=float, default=0.0,
        help=(
            "Fixed shared weight for local SimRNA statistical-potential "
            "terms. When positive, the older OpenMM harmonic term is disabled."
        ),
    )
    parser_RNA.add_argument("--objective-energy-weight", type=float, default=1e-6)
    return parser_RNA.parse_args()


def best_checkpoint_metrics_RNA(result_RNA):
    """Return the log row corresponding to the selected best checkpoint."""

    if not result_RNA.best_checkpoint:
        raise RuntimeError("Training did not produce a best checkpoint")
    checkpoint_RNA = torch.load(
        result_RNA.best_checkpoint, map_location="cpu", weights_only=True
    )
    best_epoch_RNA = int(checkpoint_RNA["epoch"])
    with open(result_RNA.log_file, newline="") as infile_RNA:
        rows_RNA = list(csv.DictReader(infile_RNA))
    for row_RNA in rows_RNA:
        if int(float(row_RNA["epoch"])) == best_epoch_RNA:
            return {
                key_RNA: float(value_RNA)
                for key_RNA, value_RNA in row_RNA.items()
            }, best_epoch_RNA
    raise RuntimeError(
        f"Epoch {best_epoch_RNA} from the best checkpoint was not found in "
        f"{result_RNA.log_file}"
    )


def load_data_RNA(args_RNA):
    manifest_path_RNA = args_RNA.ensemble_config.expanduser().resolve()
    with open(manifest_path_RNA) as infile_RNA:
        manifest_RNA = json.load(infile_RNA)

    def resolve_RNA(filename_RNA):
        path_RNA = Path(filename_RNA).expanduser()
        return path_RNA if path_RNA.is_absolute() else args_RNA.data_root / path_RNA

    unknown_RNA = [name_RNA for name_RNA in args_RNA.ensembles
                   if name_RNA not in manifest_RNA]
    if unknown_RNA:
        raise ValueError(f"Unknown ensembles: {unknown_RNA}")
    sources_RNA = [manifest_RNA[name_RNA] for name_RNA in args_RNA.ensembles]
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
        raise ValueError("SimRNA data must contain five sites per nucleotide")
    data_RNA.development_frame_start_RNA = 0
    data_RNA.development_frame_stop_RNA = args_RNA.development_frames
    data_RNA.test_frame_start_RNA = args_RNA.development_frames
    data_RNA.validation_split_RNA = args_RNA.validation_split
    return data_RNA


def main_RNA():
    args_RNA = parse_args_RNA()
    if any(value_RNA < 1 for value_RNA in (
        args_RNA.n_trials, args_RNA.max_epochs, args_RNA.patience,
        args_RNA.frame_step,
        args_RNA.development_frames,
    )):
        raise ValueError(
            "n-trials, max-epochs, patience, frame-step, and "
            "development-frames must be positive"
        )
    if not 0.0 < args_RNA.validation_split < 1.0:
        raise ValueError("validation-split must be between zero and one")
    if args_RNA.simrna_sp_weight < 0:
        raise ValueError("--simrna-sp-weight cannot be negative")
    args_RNA.data_root = args_RNA.data_root.expanduser().resolve()
    args_RNA.output_root = args_RNA.output_root.expanduser().resolve()
    args_RNA.output_root.mkdir(parents=True, exist_ok=True)
    data_RNA = load_data_RNA(args_RNA)
    residue_count_RNA = data_RNA.dataset.shape[1] // 5
    device_RNA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Worker PID:", os.getpid(), "device:", device_RNA)

    def objective_RNA(trial_RNA):
        parameters_RNA = {
            "batch_size": trial_RNA.suggest_categorical(
                "batch_size", [2, 4, 8, 16]
            ),
            "latent_dimension": trial_RNA.suggest_categorical(
                "latent_dimension", [2, 4, 8, 16, 32]
            ),
            "learning_rate": trial_RNA.suggest_float(
                "learning_rate", 1e-5, 1e-3, log=True
            ),
            "weight_decay": trial_RNA.suggest_float(
                "weight_decay", 1e-6, 1e-3, log=True
            ),
            "initial_channels": trial_RNA.suggest_categorical(
                "initial_channels", [16, 32, 64]
            ),
            "channel_multiplier": trial_RNA.suggest_categorical(
                "channel_multiplier", [1.5, 2.0]
            ),
            "local_weight": trial_RNA.suggest_float(
                "local_weight", 0.5, 1.0, step=0.05
            ),
            "openmm_cg_weight": (
                0.0 if args_RNA.simrna_sp_weight > 0
                else trial_RNA.suggest_float(
                    "openmm_cg_weight", 1e-8, 1e-5, log=True
                )
            ),
            "simrna_sp_weight": args_RNA.simrna_sp_weight,
            "openmm_bond_k": args_RNA.openmm_bond_k,
        }
        parameters_RNA["nonlocal_weight"] = (
            1.0 - parameters_RNA["local_weight"]
        )
        trial_dir_RNA = (
            args_RNA.output_root / args_RNA.study_name
            / f"trial_{trial_RNA.number:05d}"
        )
        trial_dir_RNA.mkdir(parents=True, exist_ok=True)
        data_RNA.write_statistics_RNA(
            trial_dir_RNA / "data_statistics_DistanceMatrix_SimRNA_RNA.json"
        )
        with open(trial_dir_RNA / "parameters.json", "w") as outfile_RNA:
            json.dump(parameters_RNA, outfile_RNA, indent=2)

        torch.manual_seed(args_RNA.split_seed)
        config_RNA = DistanceMatrix_AE_Trainer_Config_RNA(
            dm_weight=1.0,
            local_k=4,
            local_weight_RNA=parameters_RNA["local_weight"],
            nonlocal_weight_RNA=parameters_RNA["nonlocal_weight"],
        )
        trainer_RNA = DistanceMatrix_AE_Trainer_RNA(
            residue_count_RNA=residue_count_RNA,
            device=device_RNA,
            config_RNA=config_RNA,
        )
        trainer_RNA.set_data_RNA(
            data_RNA,
            batch_size=parameters_RNA["batch_size"],
            validation_split=args_RNA.validation_split,
            manual_seed=args_RNA.split_seed,
            save_indices=False,
        )
        trainer_RNA.prepare_openmm_simrna_RNA(
            weight_RNA=parameters_RNA["openmm_cg_weight"],
            bond_k_RNA=parameters_RNA["openmm_bond_k"],
        )
        trainer_RNA.prepare_simrna_sp_RNA(
            weight_RNA=parameters_RNA["simrna_sp_weight"]
        )
        trainer_RNA.set_autoencoder(
            DistanceMatrix_AE_RNA,
            residue_count_RNA=residue_count_RNA,
            sites_per_residue_RNA=5,
            latent_dimension=parameters_RNA["latent_dimension"],
            init_c=parameters_RNA["initial_channels"],
            m=parameters_RNA["channel_multiplier"],
            min_size=9,
        )
        trainer_RNA.prepare_optimiser(
            lr=parameters_RNA["learning_rate"],
            weight_decay=parameters_RNA["weight_decay"],
        )
        result_RNA = trainer_RNA.run_until_converge(
            patience=args_RNA.patience,
            max_epochs=args_RNA.max_epochs,
            log_filename="log.dat",
            log_folder=str(trial_dir_RNA),
            checkpoint_folder=str(trial_dir_RNA),
            verbose=False,
        )
        metrics_RNA, best_epoch_RNA = best_checkpoint_metrics_RNA(result_RNA)
        dm_loss_RNA = float(metrics_RNA["valid_dm_loss"])
        dm_rmse_A_RNA = float(metrics_RNA["valid_dm_rmse_A"])
        energy_RNA = float(metrics_RNA["valid_openmm_simrna_energy"])
        sp_energy_RNA = float(metrics_RNA["valid_simrna_sp_energy"])
        objective_value_RNA = (
            dm_rmse_A_RNA
            + args_RNA.objective_energy_weight
            * (energy_RNA + sp_energy_RNA)
        )
        trial_RNA.set_user_attr("valid_dm_loss", dm_loss_RNA)
        trial_RNA.set_user_attr("valid_dm_rmse_A", dm_rmse_A_RNA)
        trial_RNA.set_user_attr("valid_openmm_simrna_energy", energy_RNA)
        trial_RNA.set_user_attr("valid_simrna_sp_energy", sp_energy_RNA)
        trial_RNA.set_user_attr("best_epoch", best_epoch_RNA)
        trial_RNA.set_user_attr("epochs_run", result_RNA.epochs_run)
        trial_RNA.set_user_attr("converged", trainer_RNA.progress.converged)
        trial_RNA.set_user_attr(
            "stop_reason",
            "converged" if trainer_RNA.progress.converged else "max_epochs",
        )
        trial_RNA.set_user_attr(
            "nonlocal_weight", parameters_RNA["nonlocal_weight"]
        )
        trial_RNA.set_user_attr("checkpoint", result_RNA.best_checkpoint)
        return objective_value_RNA

    storage_RNA = optuna.storages.RDBStorage(
        url=args_RNA.storage, heartbeat_interval=60, grace_period=180
    )
    study_RNA = optuna.create_study(
        study_name=args_RNA.study_name,
        storage=storage_RNA,
        direction="minimize",
        load_if_exists=True,
    )
    study_RNA.optimize(objective_RNA, n_trials=args_RNA.n_trials)
    print("Best trial:", study_RNA.best_trial.number)
    print("Best value:", study_RNA.best_value)
    print("Best parameters:", study_RNA.best_params)


if __name__ == "__main__":
    main_RNA()
