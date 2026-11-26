"""Post-process generated five-site RNA structures with official SimRNA."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


SIMRNA_MAXIMUM_REFINEMENT_FRAMES_RNA = 1000
SIMRNA_REFINEMENT_RANDOM_SEED_RNA = 42


def simrna_write_frequency_RNA(
    iterations_RNA,
    maximum_frames_RNA=SIMRNA_MAXIMUM_REFINEMENT_FRAMES_RNA,
):
    """Choose a write interval that caps the TRAFL including its initial frame."""
    iterations_RNA = int(iterations_RNA)
    maximum_frames_RNA = int(maximum_frames_RNA)
    if iterations_RNA < 1:
        raise ValueError("iterations_RNA must be positive")
    if maximum_frames_RNA < 2:
        raise ValueError("maximum_frames_RNA must be at least 2")
    return max(1, math.ceil(iterations_RNA / (maximum_frames_RNA - 1)))


def minimum_energy_trafl_frame_RNA(trafl_file_RNA):
    """Return the lowest unrestrained-energy frame from an official TRAFL.

    Official SimRNA writes two lines per frame. The five header values are the
    write number, replica number, energy including restraints, energy without
    restraints, and temperature. Refinement selects by the fourth value, as
    does the official ``trafl_extract_lowestE_frame.py`` helper.
    """
    trafl_file_RNA = Path(trafl_file_RNA)
    best_energy_RNA = np.inf
    best_restrained_energy_RNA = None
    best_temperature_RNA = None
    best_write_number_RNA = None
    best_replica_number_RNA = None
    best_coordinates_RNA = None
    best_coordinate_line_RNA = None
    best_frame_RNA = None
    frames_RNA = 0
    with trafl_file_RNA.open() as infile_RNA:
        while True:
            header_line_RNA = infile_RNA.readline()
            if not header_line_RNA:
                break
            if not header_line_RNA.strip():
                continue
            coordinate_line_RNA = infile_RNA.readline()
            if not coordinate_line_RNA:
                raise ValueError(f"Incomplete final frame in {trafl_file_RNA}")
            header_values_RNA = header_line_RNA.split()
            if len(header_values_RNA) != 5:
                raise ValueError(
                    f"Official SimRNA TRAFL frame {frames_RNA + 1} in "
                    f"{trafl_file_RNA} has {len(header_values_RNA)} header "
                    "values; expected 5"
                )
            try:
                write_number_RNA = int(header_values_RNA[0])
                replica_number_RNA = int(header_values_RNA[1])
                restrained_energy_RNA = float(header_values_RNA[2])
                energy_RNA = float(header_values_RNA[3])
                temperature_RNA = float(header_values_RNA[4])
            except ValueError as error_RNA:
                raise ValueError(
                    f"Invalid official SimRNA TRAFL header at frame "
                    f"{frames_RNA + 1} in {trafl_file_RNA}"
                ) from error_RNA
            flat_RNA = np.fromstring(coordinate_line_RNA, sep=" ", dtype=float)
            if not flat_RNA.size or flat_RNA.size % 3:
                raise ValueError(
                    f"Invalid coordinates at frame {frames_RNA + 1} in "
                    f"{trafl_file_RNA}"
                )
            if energy_RNA < best_energy_RNA:
                best_energy_RNA = energy_RNA
                best_restrained_energy_RNA = restrained_energy_RNA
                best_temperature_RNA = temperature_RNA
                best_write_number_RNA = write_number_RNA
                best_replica_number_RNA = replica_number_RNA
                best_coordinates_RNA = flat_RNA.reshape(-1, 3)
                best_coordinate_line_RNA = coordinate_line_RNA.strip()
                best_frame_RNA = frames_RNA + 1
            frames_RNA += 1

    if best_coordinates_RNA is None:
        raise ValueError(f"No frames found in {trafl_file_RNA}")
    return {
        "coordinates_RNA": best_coordinates_RNA,
        "coordinate_line_RNA": best_coordinate_line_RNA,
        "energy_RNA": float(best_energy_RNA),
        "restrained_energy_RNA": float(best_restrained_energy_RNA),
        "temperature_RNA": float(best_temperature_RNA),
        "write_number_RNA": int(best_write_number_RNA),
        "replica_number_RNA": int(best_replica_number_RNA),
        "frame_RNA": int(best_frame_RNA),
        "frames_RNA": int(frames_RNA),
    }


def minimum_energy_frame_RNA(trafl_file_RNA):
    """Compatibility name for :func:`minimum_energy_trafl_frame_RNA`."""
    return minimum_energy_trafl_frame_RNA(trafl_file_RNA)


def write_transition_trafl_RNA(refinement_results_RNA, output_trafl_RNA):
    """Write ordered selected refinement frames as one official TRAFL file."""
    refinement_results_RNA = list(refinement_results_RNA)
    if not refinement_results_RNA:
        raise ValueError("At least one refinement result is required")
    output_trafl_RNA = Path(output_trafl_RNA)
    records_RNA = []
    coordinate_count_RNA = None
    for frame_number_RNA, result_RNA in enumerate(
        refinement_results_RNA, start=1
    ):
        coordinate_line_RNA = result_RNA["minimum_coordinate_line_RNA"].strip()
        coordinates_RNA = np.fromstring(coordinate_line_RNA, sep=" ")
        if not coordinates_RNA.size or coordinates_RNA.size % 3:
            raise ValueError(
                f"Invalid selected TRAFL coordinates for frame {frame_number_RNA}"
            )
        if coordinate_count_RNA is None:
            coordinate_count_RNA = coordinates_RNA.size
        elif coordinates_RNA.size != coordinate_count_RNA:
            raise ValueError(
                "Selected TRAFL frames do not have matching coordinate counts"
            )
        records_RNA.append(
            f"{frame_number_RNA} "
            f"{int(result_RNA['minimum_replica_number_RNA'])} "
            f"{float(result_RNA['minimum_restrained_energy_RNA']):.6f} "
            f"{float(result_RNA['minimum_energy_RNA']):.6f} "
            f"{float(result_RNA['minimum_temperature_RNA']):.6f}\n"
            f" {coordinate_line_RNA}\n"
        )
    output_trafl_RNA.parent.mkdir(parents=True, exist_ok=True)
    output_trafl_RNA.write_text("".join(records_RNA))
    return output_trafl_RNA


def reconstruct_all_atom_trafl_RNA(
    template_pdb_RNA,
    transition_trafl_RNA,
    converter_executable_RNA,
    output_directory_RNA,
    data_directory_RNA=None,
    output_names_RNA=None,
):
    """Reconstruct all TRAFL frames and retain only their all-atom PDBs.

    The official converter's reduced PDB and ``.ss_detected`` by-products are
    generated in a temporary directory and removed automatically.
    """
    template_pdb_RNA = Path(template_pdb_RNA).expanduser().resolve()
    transition_trafl_RNA = Path(transition_trafl_RNA).expanduser().resolve()
    converter_executable_RNA = Path(
        converter_executable_RNA
    ).expanduser().resolve()
    output_directory_RNA = Path(output_directory_RNA).expanduser().resolve()
    data_directory_RNA = (
        Path(data_directory_RNA).expanduser().resolve()
        if data_directory_RNA is not None
        else converter_executable_RNA.parent / "data"
    )
    for required_RNA in (
        template_pdb_RNA, transition_trafl_RNA, converter_executable_RNA
    ):
        if not required_RNA.is_file():
            raise FileNotFoundError(required_RNA)
    if not data_directory_RNA.is_dir():
        raise FileNotFoundError(data_directory_RNA)

    with tempfile.TemporaryDirectory(
        prefix="molearn-simrna-all-atom-"
    ) as tmp_RNA:
        tmp_path_RNA = Path(tmp_RNA)
        shutil.copy2(template_pdb_RNA, tmp_path_RNA / "template.pdb")
        shutil.copy2(transition_trafl_RNA, tmp_path_RNA / "transition.trafl")
        os.symlink(
            data_directory_RNA, tmp_path_RNA / "data", target_is_directory=True
        )
        command_RNA = [
            str(converter_executable_RNA),
            "template.pdb", "transition.trafl", "AA", ":",
        ]
        process_RNA = subprocess.run(
            command_RNA, cwd=tmp_path_RNA, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        generated_RNA = sorted(tmp_path_RNA.glob("transition-*_AA.pdb"))
        if process_RNA.returncode != 0 or not generated_RNA:
            tail_RNA = "\n".join(process_RNA.stdout.splitlines()[-30:])
            raise RuntimeError(
                "SimRNA_trafl2pdbs failed with return code "
                f"{process_RNA.returncode}:\n{tail_RNA}"
            )
        if output_names_RNA is None:
            output_names_RNA = [path_RNA.name for path_RNA in generated_RNA]
        else:
            output_names_RNA = list(output_names_RNA)
        if len(generated_RNA) != len(output_names_RNA):
            raise ValueError(
                f"SimRNA_trafl2pdbs generated {len(generated_RNA)} all-atom "
                f"PDB files; expected {len(output_names_RNA)}"
            )
        output_directory_RNA.mkdir(parents=True, exist_ok=True)
        outputs_RNA = []
        for generated_pdb_RNA, output_name_RNA in zip(
            generated_RNA, output_names_RNA
        ):
            output_pdb_RNA = output_directory_RNA / Path(output_name_RNA).name
            shutil.copy2(generated_pdb_RNA, output_pdb_RNA)
            outputs_RNA.append(output_pdb_RNA)
    return outputs_RNA


def write_refined_pdb_RNA(
    template_pdb_RNA, coordinates_RNA, output_pdb_RNA,
    energy_RNA=None, restrained_energy_RNA=None, temperature_RNA=None,
    frame_RNA=None,
):
    """Write coordinates into a PDB while preserving occupancy and B-factor."""
    template_pdb_RNA = Path(template_pdb_RNA)
    output_pdb_RNA = Path(output_pdb_RNA)
    lines_RNA = template_pdb_RNA.read_text().splitlines()
    atom_line_indices_RNA = [
        index_RNA for index_RNA, line_RNA in enumerate(lines_RNA)
        if line_RNA.startswith(("ATOM  ", "HETATM"))
    ]
    coordinates_RNA = np.asarray(coordinates_RNA)
    if len(coordinates_RNA) != len(atom_line_indices_RNA):
        raise ValueError(
            f"SimRNA returned {len(coordinates_RNA)} coordinates but "
            f"{template_pdb_RNA} contains {len(atom_line_indices_RNA)} atoms"
        )
    for coordinate_RNA, line_index_RNA in zip(
        coordinates_RNA, atom_line_indices_RNA
    ):
        line_RNA = lines_RNA[line_index_RNA].ljust(80)
        x_RNA, y_RNA, z_RNA = coordinate_RNA
        lines_RNA[line_index_RNA] = (
            f"{line_RNA[:30]}{x_RNA:8.3f}{y_RNA:8.3f}{z_RNA:8.3f}"
            f"{line_RNA[54:]}"
        ).rstrip()
    remarks_RNA = []
    if energy_RNA is not None:
        remarks_RNA.append(
            f"REMARK SIMRNA MINIMUM UNRESTRAINED ENERGY "
            f"{float(energy_RNA):.6f}"
        )
    if restrained_energy_RNA is not None:
        remarks_RNA.append(
            f"REMARK SIMRNA ENERGY INCLUDING RESTRAINTS "
            f"{float(restrained_energy_RNA):.6f}"
        )
    if temperature_RNA is not None:
        remarks_RNA.append(
            f"REMARK SIMRNA TEMPERATURE {float(temperature_RNA):.6f}"
        )
    if frame_RNA is not None:
        remarks_RNA.append(f"REMARK SIMRNA MINIMUM FRAME {int(frame_RNA)}")
    output_pdb_RNA.parent.mkdir(parents=True, exist_ok=True)
    output_pdb_RNA.write_text("\n".join(remarks_RNA + lines_RNA) + "\n")
    return output_pdb_RNA


def write_pinning_pdb_RNA(
    input_pdb_RNA,
    output_pdb_RNA,
    bad_residue_indices_RNA,
    neighbor_residue_indices_RNA=None,
    bad_occupancy_RNA=0.50,
    bad_bfactor_RNA=10.00,
    neighbor_occupancy_RNA=0.25,
    neighbor_bfactor_RNA=5.00,
):
    """Encode movable and fixed regions for SimRNA ``-P`` restraints.

    Residues use their zero-based order in the PDB. Bad residues receive
    occupancy/B-factor ``0.50/10.00``. Their immediate sequence neighbors
    receive the weaker values ``0.25/5.00`` and every other residue receives
    ``0.00/0.00``. A bad-residue assignment takes precedence if the two sets
    overlap.
    """
    input_pdb_RNA = Path(input_pdb_RNA)
    output_pdb_RNA = Path(output_pdb_RNA)
    lines_RNA = input_pdb_RNA.read_text().splitlines()
    bad_RNA = {int(index_RNA) for index_RNA in bad_residue_indices_RNA}
    neighbor_RNA = {
        int(index_RNA) for index_RNA in (neighbor_residue_indices_RNA or [])
    }.difference(bad_RNA)
    residue_lookup_RNA = {}
    for line_index_RNA, line_RNA in enumerate(lines_RNA):
        if not line_RNA.startswith(("ATOM  ", "HETATM")):
            continue
        key_RNA = (
            line_RNA[21:22], line_RNA[22:26], line_RNA[26:27],
            line_RNA[17:21],
        )
        if key_RNA not in residue_lookup_RNA:
            residue_lookup_RNA[key_RNA] = len(residue_lookup_RNA)
        residue_index_RNA = residue_lookup_RNA[key_RNA]
        if residue_index_RNA in bad_RNA:
            occupancy_RNA, bfactor_RNA = bad_occupancy_RNA, bad_bfactor_RNA
        elif residue_index_RNA in neighbor_RNA:
            occupancy_RNA = neighbor_occupancy_RNA
            bfactor_RNA = neighbor_bfactor_RNA
        else:
            occupancy_RNA, bfactor_RNA = 0.0, 0.0
        padded_RNA = line_RNA.ljust(80)
        lines_RNA[line_index_RNA] = (
            f"{padded_RNA[:54]}{occupancy_RNA:6.2f}{bfactor_RNA:6.2f}"
            f"{padded_RNA[66:]}"
        ).rstrip()
    invalid_RNA = (bad_RNA | neighbor_RNA).difference(
        range(len(residue_lookup_RNA))
    )
    if invalid_RNA:
        raise ValueError(
            f"Bad residue indices outside the PDB: {sorted(invalid_RNA)}"
        )
    output_pdb_RNA.parent.mkdir(parents=True, exist_ok=True)
    output_pdb_RNA.write_text("\n".join(lines_RNA) + "\n")
    return output_pdb_RNA


def refine_pdb_with_simrna_RNA(
    input_pdb_RNA,
    output_pdb_RNA,
    executable_RNA,
    iterations_RNA=1000,
    data_directory_RNA=None,
    run_name_RNA="molearn_refinement",
    bad_residue_indices_RNA=None,
    neighbor_residue_indices_RNA=None,
    random_seed_RNA=SIMRNA_REFINEMENT_RANDOM_SEED_RNA,
):
    """Run restrained SimRNA refinement and retain only its lowest-energy PDB.

    The trajectory write interval is calculated automatically so that the
    temporary TRAFL contains no more than 1,000 frames, including frame zero.
    All native SimRNA output is created inside a temporary directory and
    removed after the minimum-energy TRAFL frame has been transferred to
    ``output_pdb_RNA``.
    """
    write_frequency_RNA = simrna_write_frequency_RNA(
        iterations_RNA, SIMRNA_MAXIMUM_REFINEMENT_FRAMES_RNA
    )
    input_pdb_RNA = Path(input_pdb_RNA).expanduser().resolve()
    output_pdb_RNA = Path(output_pdb_RNA).expanduser().resolve()
    executable_RNA = Path(executable_RNA).expanduser().resolve()
    random_seed_RNA = int(random_seed_RNA)
    run_name_RNA = Path(str(run_name_RNA)).name
    if not run_name_RNA or run_name_RNA in {".", ".."}:
        raise ValueError("run_name_RNA must be a valid output basename")
    data_directory_RNA = (
        Path(data_directory_RNA).expanduser().resolve()
        if data_directory_RNA is not None else executable_RNA.parent / "data"
    )
    if not input_pdb_RNA.is_file():
        raise FileNotFoundError(input_pdb_RNA)
    if not executable_RNA.is_file():
        raise FileNotFoundError(executable_RNA)
    if not data_directory_RNA.is_dir():
        raise FileNotFoundError(data_directory_RNA)

    with tempfile.TemporaryDirectory(prefix="molearn-simrna-refine-") as tmp_RNA:
        tmp_path_RNA = Path(tmp_RNA)
        temporary_input_RNA = tmp_path_RNA / "input.pdb"
        if bad_residue_indices_RNA is None:
            shutil.copy2(input_pdb_RNA, temporary_input_RNA)
        else:
            write_pinning_pdb_RNA(
                input_pdb_RNA,
                temporary_input_RNA,
                bad_residue_indices_RNA,
                neighbor_residue_indices_RNA=neighbor_residue_indices_RNA,
            )
        os.symlink(data_directory_RNA, tmp_path_RNA / "data", target_is_directory=True)
        (tmp_path_RNA / "refinement.cfg").write_text(
            f"NUMBER_OF_ITERATIONS {int(iterations_RNA)}\n"
            f"TRA_WRITE_IN_EVERY_N_ITERATIONS {write_frequency_RNA}\n"
            """
INIT_TEMP 0.90
FINAL_TEMP 0.90

BONDS_WEIGHT 1.0
ANGLES_WEIGHT 1.0
TORS_ANGLES_WEIGHT 0.0
ETA_THETA_WEIGHT 0.40"""
        )
        command_RNA = [
            str(executable_RNA), "-P", "input.pdb",
            "-c", "refinement.cfg", "-o", str(run_name_RNA),
            "-R", str(random_seed_RNA),
        ]
        process_RNA = subprocess.run(
            command_RNA, cwd=tmp_path_RNA, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        trafl_file_RNA = tmp_path_RNA / f"{run_name_RNA}.trafl"
        if process_RNA.returncode != 0 or not trafl_file_RNA.is_file():
            tail_RNA = "\n".join(process_RNA.stdout.splitlines()[-30:])
            raise RuntimeError(
                f"SimRNA failed with return code {process_RNA.returncode}:\n{tail_RNA}"
            )
        minimum_RNA = minimum_energy_trafl_frame_RNA(trafl_file_RNA)
        write_refined_pdb_RNA(
            temporary_input_RNA, minimum_RNA["coordinates_RNA"], output_pdb_RNA,
            energy_RNA=minimum_RNA["energy_RNA"],
            restrained_energy_RNA=minimum_RNA["restrained_energy_RNA"],
            temperature_RNA=minimum_RNA["temperature_RNA"],
            frame_RNA=minimum_RNA["frame_RNA"],
        )

    return {
        "input_pdb_RNA": str(input_pdb_RNA),
        "output_pdb_RNA": str(output_pdb_RNA),
        "minimum_energy_RNA": minimum_RNA["energy_RNA"],
        "minimum_coordinate_line_RNA": minimum_RNA["coordinate_line_RNA"],
        "minimum_restrained_energy_RNA": minimum_RNA["restrained_energy_RNA"],
        "minimum_temperature_RNA": minimum_RNA["temperature_RNA"],
        "minimum_write_number_RNA": minimum_RNA["write_number_RNA"],
        "minimum_replica_number_RNA": minimum_RNA["replica_number_RNA"],
        "minimum_frame_RNA": minimum_RNA["frame_RNA"],
        "frames_written_RNA": minimum_RNA["frames_RNA"],
        "iterations_RNA": int(iterations_RNA),
        "maximum_frames_RNA": SIMRNA_MAXIMUM_REFINEMENT_FRAMES_RNA,
        "write_frequency_RNA": int(write_frequency_RNA),
        "random_seed_RNA": random_seed_RNA,
        "bad_residues_RNA": (
            None if bad_residue_indices_RNA is None
            else len(set(bad_residue_indices_RNA))
        ),
        "neighbor_residues_RNA": (
            None if neighbor_residue_indices_RNA is None
            else len(
                set(neighbor_residue_indices_RNA)
                .difference(set(bad_residue_indices_RNA or []))
            )
        ),
        "simrna_returncode_RNA": int(process_RNA.returncode),
    }
