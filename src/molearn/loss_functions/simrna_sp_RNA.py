"""Differentiable local SimRNA statistical potentials for five-site RNA.

This module implements only the sequence-local part of the SimRNA score:

* P--C4' and C4'--P distances;
* P--C4'--P and C4'--P--C4' angles; and
* the coupled eta/theta pseudo-torsion potential.

Base pairing, stacking, base--backbone interactions, and excluded volume are
deliberately not included.  Consequently, evaluation scales linearly with the
number of nucleotides and remains practical inside every training step.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


LOCAL_TABLES_RNA = (
    "dist_CP.data.new_hist",
    "dist_PC.data.new_hist",
    "angle_PCP.data.new_hist",
    "angle_CPC.data.new_hist",
    "eta_theta.data.new_hist",
)


class LinearTable1D_RNA(nn.Module):
    """Linearly interpolate a uniformly sampled one-dimensional table."""

    def __init__(self, values_RNA, origin_RNA, spacing_RNA):
        super().__init__()
        self.register_buffer("values_RNA", torch.as_tensor(values_RNA).float())
        self.register_buffer("origin_RNA", torch.tensor(float(origin_RNA)))
        self.register_buffer("spacing_RNA", torch.tensor(float(spacing_RNA)))

    def forward(self, coordinates_RNA):
        position_RNA = (
            (coordinates_RNA - self.origin_RNA) / self.spacing_RNA
        ).clamp(0, len(self.values_RNA) - 1)
        lower_RNA = torch.floor(position_RNA).long().clamp(
            max=len(self.values_RNA) - 2
        )
        fraction_RNA = position_RNA - lower_RNA.to(position_RNA.dtype)
        return (
            self.values_RNA[lower_RNA] * (1 - fraction_RNA)
            + self.values_RNA[lower_RNA + 1] * fraction_RNA
        )


class PeriodicLinearTable2D_RNA(nn.Module):
    """Bilinearly interpolate a uniformly sampled table on a periodic torus."""

    def __init__(self, values_RNA, origin_RNA, spacing_RNA):
        super().__init__()
        values_RNA = torch.as_tensor(values_RNA).float()
        if values_RNA.ndim != 2 or values_RNA.shape[0] != values_RNA.shape[1]:
            raise ValueError("Periodic SimRNA table must be square")
        self.register_buffer("values_RNA", values_RNA)
        self.register_buffer("origin_RNA", torch.as_tensor(origin_RNA).float())
        self.register_buffer("spacing_RNA", torch.as_tensor(spacing_RNA).float())

    def forward(self, coordinates_RNA):
        size_RNA = self.values_RNA.shape[0]
        position_RNA = (
            (coordinates_RNA - self.origin_RNA) / self.spacing_RNA
        ).remainder(size_RNA)
        lower_RNA = torch.floor(position_RNA).long()
        upper_RNA = (lower_RNA + 1).remainder(size_RNA)
        fraction_RNA = position_RNA - lower_RNA.to(position_RNA.dtype)
        x0_RNA, y0_RNA = lower_RNA.unbind(dim=-1)
        x1_RNA, y1_RNA = upper_RNA.unbind(dim=-1)
        tx_RNA, ty_RNA = fraction_RNA.unbind(dim=-1)
        return (
            self.values_RNA[x0_RNA, y0_RNA]
            * (1 - tx_RNA)
            * (1 - ty_RNA)
            + self.values_RNA[x1_RNA, y0_RNA] * tx_RNA * (1 - ty_RNA)
            + self.values_RNA[x0_RNA, y1_RNA] * (1 - tx_RNA) * ty_RNA
            + self.values_RNA[x1_RNA, y1_RNA] * tx_RNA * ty_RNA
        )


def _load_1d_table_RNA(path_RNA):
    with Path(path_RNA).open() as infile_RNA:
        header_RNA = [float(value_RNA) for value_RNA in next(infile_RNA).split()]
        rows_RNA = [
            [float(value_RNA) for value_RNA in line_RNA.split()]
            for line_RNA in infile_RNA
            if line_RNA.strip()
        ]
    if len(header_RNA) != 3 or len(rows_RNA) != int(header_RNA[2]):
        raise ValueError(f"Invalid SimRNA 1D table: {path_RNA}")
    coordinates_RNA = [row_RNA[0] for row_RNA in rows_RNA]
    spacing_RNA = coordinates_RNA[1] - coordinates_RNA[0]
    if any(
        not math.isclose(
            coordinate_RNA,
            coordinates_RNA[0] + index_RNA * spacing_RNA,
            abs_tol=2e-5,
        )
        for index_RNA, coordinate_RNA in enumerate(coordinates_RNA)
    ):
        raise ValueError(f"Non-uniform SimRNA 1D table: {path_RNA}")
    return LinearTable1D_RNA(
        [row_RNA[1] for row_RNA in rows_RNA],
        coordinates_RNA[0],
        spacing_RNA,
    )


def _load_eta_theta_table_RNA(path_RNA):
    with Path(path_RNA).open() as infile_RNA:
        header_RNA = [float(value_RNA) for value_RNA in next(infile_RNA).split()]
        rows_RNA = [
            [float(value_RNA) for value_RNA in line_RNA.split()]
            for line_RNA in infile_RNA
            if line_RNA.strip()
        ]
    if len(header_RNA) != 5:
        raise ValueError(f"Invalid SimRNA eta/theta table: {path_RNA}")
    size_RNA = int(header_RNA[4])
    grid_RNA = torch.zeros(size_RNA, size_RNA)
    if len(rows_RNA) != size_RNA * size_RNA:
        raise ValueError(f"Incomplete SimRNA eta/theta table: {path_RNA}")
    for x_RNA, y_RNA, value_RNA in rows_RNA:
        grid_RNA[int(x_RNA), int(y_RNA)] = value_RNA
    return PeriodicLinearTable2D_RNA(
        grid_RNA,
        [header_RNA[0], header_RNA[2]],
        [
            (header_RNA[1] - header_RNA[0]) / size_RNA,
            (header_RNA[3] - header_RNA[2]) / size_RNA,
        ],
    )


def _angle_RNA(first_RNA, middle_RNA, last_RNA, epsilon_RNA=1e-7):
    first_vector_RNA = first_RNA - middle_RNA
    last_vector_RNA = last_RNA - middle_RNA
    denominator_RNA = (
        torch.linalg.vector_norm(first_vector_RNA, dim=-1)
        * torch.linalg.vector_norm(last_vector_RNA, dim=-1)
    ).clamp_min(epsilon_RNA)
    cosine_RNA = (
        (first_vector_RNA * last_vector_RNA).sum(dim=-1) / denominator_RNA
    ).clamp(-1 + epsilon_RNA, 1 - epsilon_RNA)
    return torch.acos(cosine_RNA)


def _dihedral_RNA(a_RNA, b_RNA, c_RNA, d_RNA, epsilon_RNA=1e-8):
    middle_RNA = c_RNA - b_RNA
    middle_RNA = middle_RNA / torch.linalg.vector_norm(
        middle_RNA, dim=-1, keepdim=True
    ).clamp_min(epsilon_RNA)
    left_RNA = a_RNA - b_RNA
    right_RNA = d_RNA - c_RNA
    left_RNA = left_RNA - (
        left_RNA * middle_RNA
    ).sum(dim=-1, keepdim=True) * middle_RNA
    right_RNA = right_RNA - (
        right_RNA * middle_RNA
    ).sum(dim=-1, keepdim=True) * middle_RNA
    x_RNA = (left_RNA * right_RNA).sum(dim=-1)
    y_RNA = (
        torch.linalg.cross(middle_RNA, left_RNA, dim=-1) * right_RNA
    ).sum(dim=-1)
    return torch.atan2(y_RNA, x_RNA).remainder(2 * math.pi)


class SimRNALocalStatisticalPotential_RNA(nn.Module):
    """Evaluate equal-weight local SimRNA terms per nucleotide."""

    def __init__(self, data_RNA, parameter_directory_RNA=None):
        super().__init__()
        if parameter_directory_RNA is None:
            parameter_directory_RNA = (
                Path(__file__).resolve().parents[1]
                / "parameters"
                / "simrna_sp_RNA"
            )
        parameter_directory_RNA = Path(parameter_directory_RNA)
        missing_RNA = [
            name_RNA
            for name_RNA in LOCAL_TABLES_RNA
            if not (parameter_directory_RNA / name_RNA).is_file()
        ]
        if missing_RNA:
            raise FileNotFoundError(
                "Missing local SimRNA tables: " + ", ".join(missing_RNA)
            )

        self.dist_CP_RNA = _load_1d_table_RNA(
            parameter_directory_RNA / "dist_CP.data.new_hist"
        )
        self.dist_PC_RNA = _load_1d_table_RNA(
            parameter_directory_RNA / "dist_PC.data.new_hist"
        )
        self.angle_PCP_RNA = _load_1d_table_RNA(
            parameter_directory_RNA / "angle_PCP.data.new_hist"
        )
        self.angle_CPC_RNA = _load_1d_table_RNA(
            parameter_directory_RNA / "angle_CPC.data.new_hist"
        )
        self.eta_theta_RNA = _load_eta_theta_table_RNA(
            parameter_directory_RNA / "eta_theta.data.new_hist"
        )

        self.coordinate_std_RNA = abs(float(data_RNA.std))
        self.coordinate_mean_RNA = float(data_RNA.mean)
        atoms_RNA = list(data_RNA._mol.atoms)
        if len(atoms_RNA) % 5:
            raise ValueError("Local SimRNA loss requires five sites per residue")

        expected_RNA = (("P",), ("C4'",), ("N1", "N9"), ("C2",), ("C4", "C6"))
        chain_keys_RNA = []
        for start_RNA in range(0, len(atoms_RNA), 5):
            residue_RNA = atoms_RNA[start_RNA : start_RNA + 5]
            names_RNA = [str(atom_RNA.name) for atom_RNA in residue_RNA]
            if any(
                name_RNA not in allowed_RNA
                for name_RNA, allowed_RNA in zip(names_RNA, expected_RNA)
            ):
                raise ValueError(
                    "Expected P, C4', N1/N9, C2, C4/C6; "
                    f"received {names_RNA}"
                )
            residue_keys_RNA = {
                (
                    str(getattr(atom_RNA, "segid", "")),
                    str(getattr(atom_RNA, "chainID", "")),
                    int(atom_RNA.resid),
                )
                for atom_RNA in residue_RNA
            }
            if len(residue_keys_RNA) != 1:
                raise ValueError("Each SimRNA five-site block must be one residue")
            chain_keys_RNA.append(next(iter(residue_keys_RNA))[:2])

        self.residue_count_RNA = len(chain_keys_RNA)
        adjacent_RNA = [
            index_RNA
            for index_RNA in range(self.residue_count_RNA - 1)
            if chain_keys_RNA[index_RNA] == chain_keys_RNA[index_RNA + 1]
        ]
        torsion_centers_RNA = [
            index_RNA
            for index_RNA in range(1, self.residue_count_RNA - 1)
            if (
                chain_keys_RNA[index_RNA - 1]
                == chain_keys_RNA[index_RNA]
                == chain_keys_RNA[index_RNA + 1]
            )
        ]
        self.register_buffer(
            "adjacent_RNA", torch.tensor(adjacent_RNA, dtype=torch.long)
        )
        self.register_buffer(
            "torsion_centers_RNA",
            torch.tensor(torsion_centers_RNA, dtype=torch.long),
        )

    def components_RNA(self, coordinates_RNA):
        expected_shape_RNA = (self.residue_count_RNA * 5, 3)
        if (
            coordinates_RNA.ndim != 3
            or coordinates_RNA.shape[1:] != expected_shape_RNA
        ):
            raise ValueError(
                f"Expected [batch, {expected_shape_RNA[0]}, 3] coordinates"
            )

        physical_RNA = (
            coordinates_RNA * self.coordinate_std_RNA
            + self.coordinate_mean_RNA
        ).reshape(-1, self.residue_count_RNA, 5, 3)
        phosphate_RNA = physical_RNA[:, :, 0]
        c4_RNA = physical_RNA[:, :, 1]

        pc_RNA = torch.linalg.vector_norm(phosphate_RNA - c4_RNA, dim=-1)
        bond_RNA = self.dist_PC_RNA(pc_RNA).sum(dim=1)
        zero_RNA = coordinates_RNA.new_zeros(coordinates_RNA.shape[0])

        if self.adjacent_RNA.numel():
            left_RNA = self.adjacent_RNA
            right_RNA = left_RNA + 1
            cp_RNA = torch.linalg.vector_norm(
                c4_RNA[:, left_RNA] - phosphate_RNA[:, right_RNA], dim=-1
            )
            bond_RNA = bond_RNA + self.dist_CP_RNA(cp_RNA).sum(dim=1)
            angle_RNA = (
                self.angle_PCP_RNA(
                    _angle_RNA(
                        phosphate_RNA[:, left_RNA],
                        c4_RNA[:, left_RNA],
                        phosphate_RNA[:, right_RNA],
                    )
                ).sum(dim=1)
                + self.angle_CPC_RNA(
                    _angle_RNA(
                        c4_RNA[:, left_RNA],
                        phosphate_RNA[:, right_RNA],
                        c4_RNA[:, right_RNA],
                    )
                ).sum(dim=1)
            )
        else:
            angle_RNA = zero_RNA

        if self.torsion_centers_RNA.numel():
            center_RNA = self.torsion_centers_RNA
            eta_RNA = _dihedral_RNA(
                c4_RNA[:, center_RNA - 1],
                phosphate_RNA[:, center_RNA],
                c4_RNA[:, center_RNA],
                phosphate_RNA[:, center_RNA + 1],
            )
            theta_RNA = _dihedral_RNA(
                phosphate_RNA[:, center_RNA],
                c4_RNA[:, center_RNA],
                phosphate_RNA[:, center_RNA + 1],
                c4_RNA[:, center_RNA + 1],
            )
            torsion_RNA = self.eta_theta_RNA(
                torch.stack((eta_RNA, theta_RNA), dim=-1)
            ).sum(dim=1)
        else:
            torsion_RNA = zero_RNA

        scale_RNA = float(self.residue_count_RNA)
        bond_RNA = bond_RNA / scale_RNA
        angle_RNA = angle_RNA / scale_RNA
        torsion_RNA = torsion_RNA / scale_RNA
        return {
            "simrna_sp_bond_energy": bond_RNA,
            "simrna_sp_angle_energy": angle_RNA,
            "simrna_sp_torsion_energy": torsion_RNA,
            "simrna_sp_energy": bond_RNA + angle_RNA + torsion_RNA,
        }

    def forward(self, coordinates_RNA):
        return self.components_RNA(coordinates_RNA)["simrna_sp_energy"]
