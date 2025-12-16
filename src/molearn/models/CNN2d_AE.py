import torch
from torch import nn
import math
from typing import List


class Encoder(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        """
        dm_dim : dimensionality of the input distance matrix 
        latent_dim : dimensionality of z
        init_c : number of filters in the first conv block
        m  : channel up-scaling factor
        """

        super().__init__()
        assert len(dims) == len(channels), "dims/channels length mismatch"

        # Build one Conv(4,2,1) -> BN -> LeakyReLU block for each downsample step
        self.convs = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=4, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            ))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        final_ch = channels[-1]
        self.finallayer = nn.Linear(final_ch, latent_dim)

    def forward(self, x):
        # x: [B, 1, dm_dim, dm_dim]
        for conv in self.convs:
            x = conv(x)

        x = self.global_pool(x)     # → [B, final_ch, 1, 1]
        x = x.view(x.size(0), -1)   # → [B, final_ch]
        z = self.finallayer(x)      # → [B, latent_dim]
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        """
        Builds a stack of ConvTranspose2d layers that upsamples from 1→dm_dim
        purely by factors of 2 (kernel=4,stride=2,pad=1) and exact output_padding adjustments.

        Args:
          dm_dim        : target #atoms (height of output)
          latent_dim    : size of z
          init_c : #filters after first upsample
          channel_mult  : how channels grow each step
          min_size      : stop when feature‐height < min_size
        """

        super().__init__()
        assert len(dims) == len(channels), "dims/channels length mismatch"


        self.from_latent = nn.Linear(latent_dim, channels[-1]*dims[-1])

        self.dims = dims
        self.channels = channels

        dims_rev = dims[::-1]
        ch_rev   = channels[::-1]

        layers = []
        for i in range(len(dims_rev) - 1):
            h_in, h_out = dims_rev[i], dims_rev[i+1]
            in_ch       = ch_rev[i]
            default_out = ch_rev[i+1]
            is_last     = (i == len(dims_rev)-2)

            out_ch = 3 if is_last else default_out
            op_h = h_out - 2 * h_in

            layers.append(nn.ConvTranspose1d(in_ch, out_ch, 4, 2, 1, op_h, bias=True))
            if not is_last:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*layers)

    def forward(self, z):
        # z: [B, latent_dim]  (or [B,latent_dim,1,1])
        z = z.view(z.size(0), -1)              # → [B, latent_dim]
        h = self.from_latent(z)                # → [B, channels[-1]*dims[-1]]
        h = h.view(h.size(0), -1, self.dims[-1])  # → [B, channels[-1], dims[-1]]
        out = self.convs(h)                    # → [B, 3, dm_dim]
        return out


class AutoEncoder(nn.Module):
    def __init__(self, dm_dim, latent_dim=2, init_c=32, m=2, min_size=9):
        super(AutoEncoder, self).__init__()

        dims, channels = self._compute_dims_channels(dm_dim, init_c, m, min_size)
        print(f"VAE: dims={dims}, channels={channels}")
        self.dims = dims
        self.channels = channels

        self.encoder = Encoder(latent_dim, dims, channels)
        self.decoder = Decoder(latent_dim, dims, channels)

    def _compute_dims_channels(self, dm_dim, init_c, m, min_size):
        """
        Returns two lists:
           dims     = [dm_dim, d1, d2, ..., d_n < min_size]
           channels = [1, c1, c2, ..., c_n]
        exactly mirroring the encoder’s down‐sampling loop.
        """
        dims = [dm_dim]
        channels = [1]
        curr = dm_dim
        ch = init_c

        while curr >= min_size:
            channels.append(ch)
            # same formula as in Encoder:
            curr = (curr + 2*1 - 4) // 2 + 1
            dims.append(curr)
            ch = int(ch * m)  # ensure integer filter‐counts

        return dims, channels
    
    @staticmethod
    def coords_to_dm(coord):
        # [B, n, 3]
        n = coord.size(1)
        G = torch.bmm(coord, coord.transpose(1, 2))
        Gt =  torch.diagonal(G, dim1=-2, dim2=-1)[:, None, :]
        Gt = Gt.repeat(1, n, 1)
        dm = Gt + Gt.transpose(1, 2) - 2*G
        dm_clamped = torch.clamp(dm, min=1e-12)
        dm = torch.sqrt(dm_clamped)[:, None, :, :]
        # Output: [B, 1, n, n]
        return dm

    def encode(self, x):
        dm = self.coords_to_dm(x)       # [B, n, 3] - > [B, 1, n, n]  
        z = self.encoder(dm)
        return z
        
    def decode(self, z):
        coords = self.decoder(z)        # [B, 3, n, 1]
        coords = coords.squeeze(-1).permute(0, 2, 1)    # [B, n, 3]
        return coords

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return decoded