from __future__ import annotations
import torch
import torch.nn as nn

import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerCoordGen(nn.Module):
    """
    Transformer for coordinate generation
    """

    def __init__(
        self,
        input_dim=3,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
    ):
        super(TransformerCoordGen, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt, tgt_mask=None):
        src = self.input_embedding(src)
        tgt = self.input_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc_out(output)


def generate_square_subsequent_mask(sz: int):
    """
    the additive mask for the tgt sequence
    :param int sz: sequence size
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


# Generation function
def generate_coordinates(
    model,
    start_sequence: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    device: torch.device,
    num_steps: int = 10,
):
    """
    use model to autoregessively generate coordinaes
    :param molearn.models.transformer.TransformerCoordGen model: the transformer to
    genereate the coordinates
    :param np.ndarray[tuple[int, int], np.dtype[np.float64]] start_sequence: the
    given coordinates to initialize the coordinate generation
    :param torch.device device
    :param int num_steps: how many coordinates should be generated
    """
    model.eval()
    generated = start_sequence.tolist()
    with torch.no_grad():
        src = torch.tensor(start_sequence, dtype=torch.float32).unsqueeze(1).to(device)

        for _ in range(num_steps):
            tgt = torch.tensor(generated, dtype=torch.float32).unsqueeze(1).to(device)
            next_point = model(src, tgt)
            generated.append(next_point[-1, 0].cpu().numpy().tolist())

    return np.array(generated)


def gen_xyz(
    generated_coords: np.ndarray[tuple[int, int], np.dtype[np.float64]], path: str
):
    """
    generates an xyz file based on an array of coordinates

    :param np.ndarray[tuple[int, int], np.dtype[np.float64]] generated_coords: coordinates
    :param str path: path where the xyz file shoudl be stored
    """
    with open(path, "w+") as cfile:
        cfile.write(f"{len(generated_coords)}\n")
        for j in generated_coords:
            cfile.write(f"C\t{j[0]}\t{j[1]}\t{j[2]}\n")


if __name__ == "__main__":
    pass
