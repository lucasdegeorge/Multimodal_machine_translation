import torch
import torch.nn as nn
import numpy as np
import math
import json

with open("parameters.json", "r") as f:
    parameters = json.load(f)
    device = parameters["device"]


class PositionalEncoder(nn.Module):  # fully modified compared to the UMMT repo one.
    """
    inputs (json):
        seq_len: the length of the sequence
        d_model: the dimension of the model
    """

    def __init__(self):
        super().__init__()

        seq_len = parameters["seq_len"]
        d_model = parameters["d_model"]

        pe_matrix = torch.zeros(seq_len, d_model)

        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** (2 * i / d_model)))

        pe_matrix = pe_matrix.unsqueeze(0)
        self.positional_encoding = pe_matrix.to(device=device).requires_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(parameters["d_model"])
        x = x + self.positional_encoding
        return x


class Encoder(nn.Module):
    """
    inputs (in the paramaters.json file)):
        d_model: dimension of the model
        n_heads: number of heads in the multihead attention layers
        dim_feedforward: dimension of the feedforward layer
        dropout: dropout rate
    """

    def __init__(self):
        super().__init__()

        d_model = parameters["d_model"]
        n_heads = parameters["n_heads"]
        dim_feedforward = parameters["dim_feedforward"]
        dropout = parameters["dropout"]

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
