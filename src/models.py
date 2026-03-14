from __future__ import annotations

import torch
import torch.nn as nn


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


def build_model(
    model_name: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
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

    raise ValueError(f"Unknown model: {model_name}")