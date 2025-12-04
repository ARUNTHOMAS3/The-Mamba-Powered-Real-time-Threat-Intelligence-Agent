# models/transformer_baseline.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer inputs.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)     # even indices
        pe[:, 1::2] = torch.cos(position * div_term)     # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TransformerBaseline(nn.Module):
    """
    Simple baseline Transformer model for sequence classification.
    Used to compare against Mamba-based model.
    """

    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """

        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encodings
        x = self.positional(x)

        # Transformer encoder
        enc = self.encoder(x)  # (b, seq, d_model)

        # Use last token representation
        last_token = enc[:, -1, :]  # (b, d_model)

        # Classification head
        out = self.classifier(last_token).squeeze(-1)

        return out
