"""Trainer for the triangular five-site RNA distance-matrix autoencoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .trainer_RNA import Trainer_RNA


@dataclass
class DistanceMatrix_AE_Trainer_Config_RNA:
    """Protein-compatible distance-matrix loss configuration for RNA."""

    dm_weight: float = 1.0
    local_k: int = 4
    local_weight_RNA: float = 0.9
    nonlocal_weight_RNA: float = 0.1


class DistanceMatrix_AE_Trainer_RNA(Trainer_RNA):
    """Train fixed-length five-site RNA through one triangular distance matrix."""

    site_names_RNA = (
        ("P",),
        ("C4'",),
        ("N1", "N9"),
        ("C2",),
        ("C4", "C6"),
    )

    def __init__(
        self,
        residue_count_RNA,
        device=None,
        config_RNA=None,
        json_log=False,
    ):
        super().__init__(device=device, json_log=json_log)
        self.residue_count_RNA = int(residue_count_RNA)
        if self.residue_count_RNA < 1:
            raise ValueError("residue_count_RNA must be positive")
        self.sites_per_residue_RNA = 5
        self.number_of_points_RNA = self.residue_count_RNA * 5
        self.config_RNA = (
            config_RNA
            if config_RNA is not None
            else DistanceMatrix_AE_Trainer_Config_RNA()
        )
        if self.config_RNA.dm_weight < 0:
            raise ValueError("dm_weight cannot be negative")
        if self.config_RNA.local_k < 0:
            raise ValueError("local_k cannot be negative")
        if (
            self.config_RNA.local_weight_RNA < 0
            or self.config_RNA.nonlocal_weight_RNA < 0
        ):
            raise ValueError("RNA distance-map weights cannot be negative")
        self.dm_weight = float(self.config_RNA.dm_weight)
        self.local_k = int(self.config_RNA.local_k)
        self.local_weight = float(self.config_RNA.local_weight_RNA)
        self.nonlocal_weight = float(self.config_RNA.nonlocal_weight_RNA)
        self.W = self._get_weight_matrix_RNA()
        # Compatibility name used by RNA notebooks and diagnostics.
        self.distance_weights_RNA = self.W

    def _get_weight_matrix_RNA(self):
        """Build the same full symmetric weight matrix as the protein trainer."""
        indices_RNA = torch.arange(self.number_of_points_RNA, device=self.device)
        separation_RNA = torch.abs(indices_RNA[:, None] - indices_RNA[None, :])
        weights_RNA = torch.full(
            (self.number_of_points_RNA, self.number_of_points_RNA),
            self.nonlocal_weight,
            device=self.device,
        )
        weights_RNA[separation_RNA <= self.local_k] = self.local_weight
        return weights_RNA[None, None]

    def _validate_five_site_order_RNA(self, data_RNA):
        """Require residue-major P, C4-prime, N, C2, outer site ordering."""
        if data_RNA.dataset.shape[1] != self.number_of_points_RNA:
            raise ValueError(
                f"Trainer expects {self.number_of_points_RNA} points "
                f"({self.residue_count_RNA} residues x 5), but the dataset "
                f"contains {data_RNA.dataset.shape[1]}"
            )
        atoms_RNA = data_RNA._mol.atoms
        names_RNA = np.asarray(atoms_RNA.names, dtype=object)
        residue_indices_RNA = np.asarray(atoms_RNA.resindices)
        ordered_residues_RNA = list(dict.fromkeys(residue_indices_RNA.tolist()))
        if len(ordered_residues_RNA) != self.residue_count_RNA:
            raise ValueError(
                f"Trainer expects {self.residue_count_RNA} RNA residues, "
                f"but the selected topology contains {len(ordered_residues_RNA)}"
            )

        for position_RNA, residue_index_RNA in enumerate(
            ordered_residues_RNA
        ):
            atom_indices_RNA = np.flatnonzero(
                residue_indices_RNA == residue_index_RNA
            )
            expected_indices_RNA = np.arange(
                position_RNA * 5, position_RNA * 5 + 5
            )
            if not np.array_equal(atom_indices_RNA, expected_indices_RNA):
                raise ValueError(
                    "Five-site RNA atoms must be contiguous and residue-major"
                )
            residue_names_RNA = names_RNA[atom_indices_RNA]
            for site_RNA, allowed_RNA in zip(
                residue_names_RNA, self.site_names_RNA
            ):
                if str(site_RNA) not in allowed_RNA:
                    raise ValueError(
                        f"Unexpected five-site RNA order in residue "
                        f"{position_RNA + 1}: received "
                        f"{residue_names_RNA.tolist()}; expected "
                        "P, C4', N1/N9, C2, C4/C6"
                    )

    def set_data_RNA(self, data_RNA, **kwargs):
        super().set_data_RNA(data_RNA, **kwargs)
        self._validate_five_site_order_RNA(data_RNA)

    set_data = set_data_RNA

    def _get_dm_loss_RNA(self, decoded_RNA, target_RNA):
        """Use the same weighted full-matrix Euclidean loss as protein."""
        difference_RNA = decoded_RNA - target_RNA
        weighted_squared_RNA = difference_RNA.square() * self.W
        per_sample_RNA = weighted_squared_RNA.sum(dim=(1, 2, 3)).sqrt()
        return per_sample_RNA.mean()

    def _get_dm_rmse_A_RNA(self, dm_loss_RNA):
        """Convert aggregate DM loss to general weighted RMSE in Angstrom."""
        weight_sum_RNA = self.W.sum()
        if weight_sum_RNA.item() == 0.0:
            return dm_loss_RNA.new_zeros(())
        coordinate_scale_RNA = dm_loss_RNA.new_tensor(
            abs(float(getattr(self, "std", 1.0)))
        )
        return dm_loss_RNA * coordinate_scale_RNA / weight_sum_RNA.sqrt()

    def common_step_RNA(self, batch_RNA):
        self._internal = {}
        target_matrix_RNA = (
            self.autoencoder.coords_to_distance_matrix_RNA(batch_RNA)
        )
        encoded_RNA = self.autoencoder.encode_distance_matrix_RNA(
            target_matrix_RNA
        )
        decoded_RNA = self.autoencoder.decode(encoded_RNA)
        decoded_matrix_RNA = (
            self.autoencoder.coords_to_distance_matrix_RNA(decoded_RNA)
        )
        dm_loss_RNA = self._get_dm_loss_RNA(
            decoded_matrix_RNA, target_matrix_RNA
        )
        dm_rmse_A_RNA = self._get_dm_rmse_A_RNA(dm_loss_RNA)

        potential_RNA = getattr(self, "openmm_simrna_loss_RNA", None)
        if potential_RNA is None:
            energy_RNA = dm_loss_RNA.new_zeros(())
        else:
            energy_RNA = potential_RNA(decoded_RNA).mean()

        sp_metrics_RNA = self._simrna_sp_metrics_RNA(
            decoded_RNA, dm_loss_RNA.new_zeros(())
        )

        self._internal["encoded"] = encoded_RNA
        self._internal["decoded"] = decoded_RNA
        loss_RNA = (
            self.dm_weight * dm_loss_RNA
            + getattr(self, "openmm_simrna_weight_RNA", 0.0) * energy_RNA
            + getattr(self, "simrna_sp_weight_RNA", 0.0)
            * sp_metrics_RNA["simrna_sp_energy"]
        )
        results_RNA = {
            "dm_loss": dm_loss_RNA,
            "dm_rmse_A": dm_rmse_A_RNA,
            "openmm_simrna_energy": energy_RNA,
            "loss": loss_RNA,
        }
        results_RNA.update(sp_metrics_RNA)
        return results_RNA

    common_step = common_step_RNA
    train_step = common_step_RNA
    valid_step = common_step_RNA
