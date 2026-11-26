"""Single distance-matrix autoencoder for five-site RNA.

The five sites of every nucleotide remain in residue-major order:
``P, C4', N1/N9, C2, C4/C6``.  All ``5L`` sites are combined into one
``5L x 5L`` matrix.  Because that matrix is symmetric, the encoder uses the
lossless triangular packing implemented by :class:`DistanceMatrix_AE`.
"""

from __future__ import annotations

from .distance_matrix_ae import DistanceMatrix_AE


class DistanceMatrix_AE_RNA(DistanceMatrix_AE):
    """Distance-matrix autoencoder for fixed-length five-site RNA.

    ``fold_symmetric=True`` stores every independent off-diagonal distance
    exactly once.  The packed encoder input has two channels and side length
    ``ceil(5L / 2)``; it can be unfolded back into the original matrix without
    losing information.
    """

    def __init__(
        self,
        residue_count_RNA: int,
        sites_per_residue_RNA: int = 5,
        latent_dimension: int = 2,
        init_c: int = 32,
        m: float = 2,
        min_size: int = 9,
        fold_symmetric: bool = True,
        verbose: bool = False,
    ):
        self.residue_count_RNA = int(residue_count_RNA)
        self.sites_per_residue_RNA = int(sites_per_residue_RNA)
        self.latent_dimension = int(latent_dimension)
        if self.residue_count_RNA < 1:
            raise ValueError("residue_count_RNA must be positive")
        if self.sites_per_residue_RNA != 5:
            raise ValueError(
                "DistanceMatrix_AE_RNA currently requires five sites per residue"
            )
        if self.latent_dimension < 1:
            raise ValueError("latent_dimension must be positive")
        if not fold_symmetric:
            raise ValueError(
                "DistanceMatrix_AE_RNA requires triangular folding to avoid "
                "symmetric duplication"
            )

        self.number_of_points_RNA = (
            self.residue_count_RNA * self.sites_per_residue_RNA
        )
        super().__init__(
            dm_dim=self.number_of_points_RNA,
            latent_dim=self.latent_dimension,
            init_c=init_c,
            m=m,
            min_size=min_size,
            fold_symmetric=True,
            verbose=verbose,
        )

    def coords_to_distance_matrix_RNA(self, coordinates_RNA):
        """RNA-named alias of the inherited protein ``coords_to_dm`` method."""
        return self.coords_to_dm(coordinates_RNA)

    def encode_distance_matrix_RNA(self, distance_matrix_RNA):
        expected_RNA = (
            1,
            self.number_of_points_RNA,
            self.number_of_points_RNA,
        )
        if tuple(distance_matrix_RNA.shape[1:]) != expected_RNA:
            raise ValueError(
                f"Expected RNA distance matrix [batch, {expected_RNA[0]}, "
                f"{expected_RNA[1]}, {expected_RNA[2]}], received "
                f"{tuple(distance_matrix_RNA.shape)}"
            )
        return self.encode_dm(distance_matrix_RNA)

    def encode(self, coordinates_RNA):
        if coordinates_RNA.shape[1] != self.number_of_points_RNA:
            raise ValueError(
                f"Expected {self.number_of_points_RNA} RNA points, "
                f"received {coordinates_RNA.shape[1]}"
            )
        return self.encode_distance_matrix_RNA(self.coords_to_dm(coordinates_RNA))
