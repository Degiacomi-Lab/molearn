import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet1d(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv1d(c_in, 64)
        self.down1 = Down1d(64, 128)
        self.sa1 = SelfAttention1d(128, 32**2)
        self.down2 = Down1d(128, 256)
        self.sa2 = SelfAttention1d(256, 16**2)
        self.down3 = Down1d(256, 256)
        self.sa3 = SelfAttention1d(256, 8**2)

        self.bot1 = DoubleConv1d(256, 512)
        self.bot2 = DoubleConv1d(512, 512)
        self.bot3 = DoubleConv1d(512, 256)

        self.up1 = Up1d(512, 128)
        self.sa4 = SelfAttention1d(128, 16**2)
        self.up2 = Up1d(256, 64)
        self.sa5 = SelfAttention1d(64, 32**2)
        self.up3 = Up1d(128, 64)
        self.sa6 = SelfAttention1d(64, 64**2)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        '''
        sin embedding
        '''
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        '''
        :param x: noised images
        :param t: time step
        '''
        t
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        N = x.shape[2]    
        x1 = torch.nn.functional.pad(x, (0,4096-N))
        x1 = self.inc(x1)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)[:,:,:N]
    
        output = self.outc(x)
        return output


class SelfAttention1d(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention1d, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size)



class DoubleConv1d(nn.Module):
    '''
    Wrapper for two convolutional layers
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down1d(nn.Module):
    '''
    down sample block
    will reduce the sample by 2
    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(4),
            DoubleConv1d(in_channels, in_channels, residual=True),
            DoubleConv1d(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up1d(nn.Module):
    '''
    up sampling block
    '''
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=4, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv1d(in_channels, in_channels, residual=True),
            DoubleConv1d(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb
