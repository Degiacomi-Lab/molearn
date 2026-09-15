"""2-D convolutional autoencoder over the inter-atomic distance matrix.

The encoder consumes a distance matrix and the decoder emits Cartesian coordinates, so
the model is invariant to rotation and translation by construction.

A distance matrix is symmetric with a zero diagonal, so only ``n(n-1)/2`` of its ``n^2``
entries are independent. ``fold_symmetric`` packs it losslessly into two half-size
channels.
"""
import torch
from torch import nn


class Encoder(nn.Module):
    """Strided Conv2d stack over the distance matrix, global-pooled to a latent vector."""

    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        assert len(dims) == len(channels), "dims/channels length mismatch"

        self.convs = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            ))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.finallayer = nn.Linear(channels[-1], latent_dim)

    def forward(self, x):
        # x: [B, in_ch, d, d]
        for conv in self.convs:
            x = conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.finallayer(x)


class Decoder(nn.Module):
    """ConvTranspose1d stack from the latent vector to ``[B, 3, n_atoms]``.

    The upsampling schedule mirrors the encoder's downsampling, with
    ``output_padding`` correcting the odd sizes that floor division introduces.
    """

    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        assert len(dims) == len(channels), "dims/channels length mismatch"

        self.from_latent = nn.Linear(latent_dim, channels[-1] * dims[-1])
        self.dims = dims
        self.channels = channels

        dims_rev, ch_rev = dims[::-1], channels[::-1]
        layers = []
        for i in range(len(dims_rev) - 1):
            h_in, h_out = dims_rev[i], dims_rev[i + 1]
            is_last = (i == len(dims_rev) - 2)
            out_ch = 3 if is_last else ch_rev[i + 1]
            op_h = h_out - 2 * h_in          # ConvTranspose1d(k=4, s=2, p=1) gives 2*h_in
            layers.append(nn.ConvTranspose1d(ch_rev[i], out_ch, 4, 2, 1, op_h, bias=True))
            if not is_last:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), -1)
        h = self.from_latent(z)
        h = h.view(h.size(0), -1, self.dims[-1])
        return self.convs(h)                 # [B, 3, n_atoms]


class DistanceMatrix_AE(nn.Module):
    """Distance-matrix encoder paired with a Cartesian-coordinate decoder.

    :param dm_dim: number of atoms, i.e. the side length of the distance matrix.
    :param latent_dim: size of the latent vector.
    :param init_c: channels after the first downsample.
    :param m: channel growth factor per downsample.
    :param min_size: stop downsampling below this spatial size.
    :param fold_symmetric: pack the symmetric matrix into two half-size channels.
    """

    def __init__(self, dm_dim, latent_dim=2, init_c=32, m=2, min_size=9,
                 fold_symmetric=True, verbose=False):
        super().__init__()
        self.dm_dim = dm_dim
        self.fold_symmetric = fold_symmetric

        # the decoder always reconstructs the full n atoms
        dec_dims, dec_channels = self._compute_dims_channels(dm_dim, init_c, m, min_size)

        if fold_symmetric:
            enc_side = (dm_dim + 1) // 2
            enc_dims, enc_channels = self._compute_dims_channels(
                enc_side, init_c, m, min_size)
            enc_channels = [2] + enc_channels[1:]
        else:
            enc_dims, enc_channels = dec_dims, dec_channels

        if verbose:
            print(f"encoder: dims={enc_dims}, channels={enc_channels}")
            print(f"decoder: dims={dec_dims}, channels={dec_channels}")

        self.dims, self.channels = dec_dims, dec_channels
        self.encoder = Encoder(latent_dim, enc_dims, enc_channels)
        self.decoder = Decoder(latent_dim, dec_dims, dec_channels)

    @staticmethod
    def _compute_dims_channels(dm_dim, init_c, m, min_size):
        """Spatial sizes and channel counts, mirroring the encoder's downsampling."""
        dims, channels = [dm_dim], [1]
        curr, ch = dm_dim, init_c
        while curr >= min_size:
            channels.append(ch)
            curr = (curr + 2 * 1 - 4) // 2 + 1
            dims.append(curr)
            ch = int(ch * m)
        return dims, channels

    @staticmethod
    def coords_to_dm(coord):
        """``[B, n, 3]`` -> ``[B, 1, n, n]`` pairwise distances."""
        n = coord.size(1)
        G = torch.bmm(coord, coord.transpose(1, 2))
        Gt = torch.diagonal(G, dim1=-2, dim2=-1)[:, None, :].repeat(1, n, 1)
        dm = Gt + Gt.transpose(1, 2) - 2 * G
        return torch.sqrt(torch.clamp(dm, min=1e-12))[:, None, :, :]

    @staticmethod
    def fold_dm(dm):
        """``[B, 1, n, n]`` symmetric -> ``[B, 2, m, m]`` with ``m = ceil(n/2)``.

        Splitting the matrix into blocks ``[[A, B], [B.T, D]]``, the independent content
        is ``B`` plus the strict upper triangles of the symmetric ``A`` and ``D``.
        Channel 0 carries ``B``; channel 1 carries ``A``'s upper triangle above its own
        diagonal and ``D``'s below it. Odd ``n`` is zero-padded by one.
        """
        B, _, n, _ = dm.shape
        if n % 2:
            dm = torch.nn.functional.pad(dm, (0, 1, 0, 1))
            n += 1
        m = n // 2
        M = dm[:, 0]
        A, Boff, D = M[:, :m, :m], M[:, :m, m:], M[:, m:, m:]
        iu = torch.triu_indices(m, m, offset=1, device=dm.device)
        packed = torch.zeros(B, m, m, dtype=dm.dtype, device=dm.device)
        packed[:, iu[0], iu[1]] = A[:, iu[0], iu[1]]
        packed[:, iu[1], iu[0]] = D[:, iu[0], iu[1]]
        return torch.stack([Boff, packed], dim=1)

    @staticmethod
    def unfold_dm(folded, n):
        """Inverse of :meth:`fold_dm`, returning ``[B, 1, n, n]``."""
        B = folded.shape[0]
        m = folded.shape[-1]
        Boff, packed = folded[:, 0], folded[:, 1]
        iu = torch.triu_indices(m, m, offset=1, device=folded.device)
        A = torch.zeros(B, m, m, dtype=folded.dtype, device=folded.device)
        D = torch.zeros_like(A)
        A[:, iu[0], iu[1]] = packed[:, iu[0], iu[1]]
        A = A + A.transpose(1, 2)
        D[:, iu[0], iu[1]] = packed[:, iu[1], iu[0]]
        D = D + D.transpose(1, 2)
        top = torch.cat([A, Boff], dim=2)
        bot = torch.cat([Boff.transpose(1, 2), D], dim=2)
        return torch.cat([top, bot], dim=1).unsqueeze(1)[:, :, :n, :n]

    def encode_dm(self, dm):
        """``[B, 1, n, n]`` -> ``[B, latent_dim]``, folding first when enabled."""
        return self.encoder(self.fold_dm(dm) if self.fold_symmetric else dm)

    def encode(self, x):
        """``[B, n, 3]`` coordinates -> ``[B, latent_dim]``."""
        return self.encode_dm(self.coords_to_dm(x))

    def decode(self, z):
        """``[B, latent_dim]`` -> ``[B, n, 3]`` coordinates."""
        return self.decoder(z).permute(0, 2, 1)

    def forward(self, x):
        return self.decode(self.encode(x))
