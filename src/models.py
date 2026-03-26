from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RNNRegressor(nn.Module):
    """Vanilla RNN regressor for sequence-to-one prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)       # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        pred = self.head(last)     # [B, 1]
        return pred, None


class GRURegressor(nn.Module):
    """GRU regressor for sequence-to-one prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)       # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        pred = self.head(last)     # [B, 1]
        return pred, None


class LSTMRegressor(nn.Module):
    """LSTM regressor for sequence-to-one prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        pred = self.head(last)     # [B, 1]
        return pred, None


class TemporalAttention(nn.Module):
    """Temporal attention over recurrent outputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, sequence_outputs):
        # sequence_outputs: [B, T, H]
        attn_logits = self.score(sequence_outputs).squeeze(-1)   # [B, T]
        attn_weights = torch.softmax(attn_logits, dim=1)         # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), sequence_outputs).squeeze(1)  # [B, H]
        return context, attn_weights


class AttentionLSTMRegressor(nn.Module):
    """LSTM regressor with temporal attention."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                   # [B, T, H]
        context, attn_weights = self.attention(out)
        pred = self.head(context)               # [B, 1]
        return pred, attn_weights


class TransformerRegressor(nn.Module):
    """Transformer encoder regressor for sequence-to-one prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_heads: int = 4,
        ff_dim: int = 128,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}) for transformer."
            )

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.position_encoding(x)
        out = self.encoder(x)
        pooled = self.norm(out.mean(dim=1))
        pred = self.head(pooled)
        return pred, None


def build_model(
    model_name: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    transformer_heads: int = 4,
    transformer_ff_dim: int = 128,
):
    model_name = model_name.lower()

    if model_name == "rnn":
        return RNNRegressor(input_size, hidden_size, num_layers, dropout)
    if model_name == "gru":
        return GRURegressor(input_size, hidden_size, num_layers, dropout)
    if model_name == "lstm":
        return LSTMRegressor(input_size, hidden_size, num_layers, dropout)
    if model_name in ("attention", "lstm_attention", "attn"):
        return AttentionLSTMRegressor(input_size, hidden_size, num_layers, dropout)
    if model_name in ("transformer", "tx", "transformer_encoder"):
        return TransformerRegressor(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            num_heads=transformer_heads,
            ff_dim=transformer_ff_dim,
        )

    raise ValueError(f"Unknown model: {model_name}")
