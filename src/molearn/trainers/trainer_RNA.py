"""Reconstruction-only trainer entry point for RNA models."""

from __future__ import annotations

from .trainer import Trainer


class Trainer_RNA(Trainer):
    """Trainer variant that does not require protein backbone indices."""

    def set_data_RNA(self, data, **kwargs):
        """Attach RNA data and create training/validation data loaders."""
        self.set_dataloader(*data.get_dataloader(**kwargs))
        self._data = data
        self.mol = data.mol
        self.standardise = data.standardise
        self.std = data.std
        self.mean = data.mean
        self.indices_RNA = {
            name: indices.to(device=self.device)
            for name, indices in data.indices.items()
        }

    def prepare_simrna_sp_RNA(self, weight_RNA=0.0):
        """Enable the local SimRNA statistical-potential loss.

        The single weight applies to the sum of the distance, angle, and
        eta/theta terms.  The potential itself reports energy per nucleotide.
        """
        if weight_RNA < 0:
            raise ValueError("SimRNA statistical-potential weight cannot be negative")
        self.simrna_sp_weight_RNA = float(weight_RNA)
        if self.simrna_sp_weight_RNA == 0.0:
            self.simrna_sp_loss_RNA = None
            return

        from molearn.loss_functions import SimRNALocalStatisticalPotential_RNA

        self.simrna_sp_loss_RNA = SimRNALocalStatisticalPotential_RNA(
            self._data
        ).to(self.device)

    def _simrna_sp_metrics_RNA(self, decoded_RNA, zero_RNA):
        potential_RNA = getattr(self, "simrna_sp_loss_RNA", None)
        if potential_RNA is None:
            return {
                "simrna_sp_bond_energy": zero_RNA,
                "simrna_sp_angle_energy": zero_RNA,
                "simrna_sp_torsion_energy": zero_RNA,
                "simrna_sp_energy": zero_RNA,
            }
        return {
            name_RNA: value_RNA.mean()
            for name_RNA, value_RNA in potential_RNA.components_RNA(
                decoded_RNA
            ).items()
        }

    def common_step_RNA(self, batch_RNA):
        results_RNA = super().common_step(batch_RNA)
        sp_metrics_RNA = self._simrna_sp_metrics_RNA(
            self._internal["decoded"], results_RNA["mse_loss"].new_zeros(())
        )
        results_RNA.update(sp_metrics_RNA)
        results_RNA["loss"] = (
            results_RNA["mse_loss"]
            + getattr(self, "simrna_sp_weight_RNA", 0.0)
            * sp_metrics_RNA["simrna_sp_energy"]
        )
        return results_RNA

    common_step = common_step_RNA
    train_step = common_step_RNA
    valid_step = common_step_RNA

    def run_until_converge(self, patience=16, **kwargs):
        """Run with convergence tracking and remember its display limit."""
        self._display_patience_RNA = patience
        return super().run_until_converge(patience=patience, **kwargs)

    def run(self, epochs=100, **kwargs):
        """Run for a fixed epoch count without a convergence counter limit."""
        self._display_patience_RNA = None
        return super().run(epochs=epochs, **kwargs)

    def log(self, log_file, log_dict, verbose=None):
        """Print one compact RNA epoch line while retaining normal file logs."""
        should_print_RNA = self.verbose if verbose is None else verbose
        if should_print_RNA:
            patience_RNA = getattr(self, "_display_patience_RNA", None)
            patience_text_RNA = (
                str(patience_RNA) if patience_RNA is not None else "disabled"
            )
            print(
                f"epoch: {int(log_dict.get('epoch', self.epoch))}, "
                f"train_mse_loss: {log_dict.get('train_mse_loss', float('nan')):.6f}, "
                f"train_SimRNA_local: "
                f"{log_dict.get('train_simrna_sp_energy', float('nan')):.6f}, "
                f"train_loss: {log_dict.get('train_loss', float('nan')):.6f}, "
                f"valid_mse_loss: {log_dict.get('valid_mse_loss', float('nan')):.6f}, "
                f"valid_SimRNA_local: "
                f"{log_dict.get('valid_simrna_sp_energy', float('nan')):.6f}, "
                f"valid_loss: {log_dict.get('valid_loss', float('nan')):.6f}, "
                f"no_improve: {self.progress.epochs_since_improve}/"
                f"{patience_text_RNA}, "
                f"total_seconds: {log_dict.get('total_seconds', float('nan')):.2f}"
            )

        previous_verbose_RNA = self.verbose
        self.verbose = False
        try:
            return super().log(log_file, log_dict, verbose=False)
        finally:
            self.verbose = previous_verbose_RNA

    # Retain the conventional Trainer API as well as the explicit RNA name.
    set_data = set_data_RNA
