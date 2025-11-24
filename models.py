
from typing import Tuple

import torch
from torch import nn


class BaselineLSTM(nn.Module):
    """
    Many-to-many LSTM that predicts a multi-step horizon given a history window.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 24,
    ):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # (batch, hidden)
        expanded = last_hidden.unsqueeze(1).repeat(1, self.horizon, 1)
        pred = self.proj(expanded)
        return pred


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return outputs, (h, c)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, hidden):
        # encoder_outputs: (batch, seq_len, hidden)
        # hidden: (batch, hidden) - last decoder hidden
        hidden = hidden.unsqueeze(1)  # (batch, 1, hidden)
        score = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attn_weights = torch.softmax(self.v(score), dim=1)  # (batch, seq_len, 1)
        context = (attn_weights * encoder_outputs).sum(dim=1)  # (batch, hidden)
        return context, attn_weights


class DecoderWithAttention(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(n_features + hidden_size, hidden_size)
        self.attn = BahdanauAttention(hidden_size)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, encoder_outputs, hidden_state, cell_state, decoder_inputs):
        """
        Autoregressive decoding with teacher forcing.
        encoder_outputs: (batch, seq_len, hidden)
        hidden_state, cell_state: (num_layers=1, batch, hidden)
        decoder_inputs: (batch, horizon, n_features)
        """
        batch_size, horizon, n_features = decoder_inputs.shape
        h = hidden_state[-1]
        c = cell_state[-1]

        outputs = []
        x_t = decoder_inputs[:, 0, :]

        for t in range(horizon):
            context, _ = self.attn(encoder_outputs, h)
            lstm_input = torch.cat([x_t, context], dim=-1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            pred = self.out(h)
            outputs.append(pred.unsqueeze(1))
            if t + 1 < horizon:
                x_t = decoder_inputs[:, t + 1, :]

        return torch.cat(outputs, dim=1)


class Seq2SeqAttentionModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        horizon: int = 24,
    ):
        super().__init__()
        self.horizon = horizon
        self.encoder = Encoder(n_features, hidden_size)
        self.decoder = DecoderWithAttention(n_features, hidden_size)

    def forward(self, src, tgt):
        enc_outputs, (h, c) = self.encoder(src)
        out = self.decoder(enc_outputs, h, c, tgt)
        return out
