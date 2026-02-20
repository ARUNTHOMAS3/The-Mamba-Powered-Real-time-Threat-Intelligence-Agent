"""
Benchmark Model Architectures for IDS Comparison Study

All models share the same interface:
    __init__(input_dim, d_model=128, n_layers=2)
    forward(x) -> logits  # x: (batch, seq_len, features), logits: (batch, 1)

Models:
    1. MambaClassifier   - State Space Model (from tabular_models.py)
    2. LSTMClassifier    - Long Short-Term Memory (from tabular_models.py)
    3. GRUClassifier     - Gated Recurrent Unit
    4. TransformerClassifier - Multi-Head Self-Attention Encoder
    5. CNNLSTMClassifier - 1D-CNN + LSTM Hybrid
    6. TCNClassifier     - Temporal Convolutional Network (Dilated Causal Conv)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tabular_models import MambaClassifier, LSTMClassifier


# ============================================================
# GRU Classifier
# ============================================================
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        # Use n_layers+1 to compensate for fewer params per layer vs Mamba
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers + 1, batch_first=True)
        # 2-layer MLP projection to reach parity (GRU has 3 gates vs LSTM's 4)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.proj(out)
        out = self.norm(out)
        return self.head(out[:, -1, :])


# ============================================================
# Transformer Classifier
# ============================================================
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=128, n_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Create causal mask for autoregressive processing
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(x, mask=causal_mask)
        out = self.norm(out)
        return self.head(out[:, -1, :])  # Classify based on last token


# ============================================================
# CNN-LSTM Classifier
# ============================================================
class CNNLSTMClassifier(nn.Module):
    """1D-CNN for local feature extraction -> LSTM for temporal modeling."""
    def __init__(self, input_dim, d_model=128, n_layers=2, kernel_size=3):
        super().__init__()
        # CNN feature extractor (3 conv layers for richer features)
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_norm = nn.BatchNorm1d(d_model)
        self.conv_act = nn.GELU()
        
        # LSTM sequence modeler
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)  # Projection for capacity parity
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_act(self.conv1(x))
        x = self.conv_act(self.conv2(x))
        x = self.conv_act(self.conv_norm(self.conv3(x)))
        x = x.transpose(1, 2)  # Back to (batch, seq_len, d_model)
        
        # LSTM
        out, _ = self.lstm(x)
        out = self.proj(out)
        out = self.norm(out)
        return self.head(out[:, -1, :])


# ============================================================
# TCN (Temporal Convolutional Network) Classifier
# ============================================================
class CausalConv1d(nn.Module):
    """Causal (left-padded) 1D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove future padding (keep only causal part)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block with two dilated causal convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        res = self.residual(x)
        out = self.dropout(self.act(self.norm1(self.conv1(x))))
        out = self.dropout(self.act(self.norm2(self.conv2(out))))
        return out + res


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network with exponentially dilated causal convolutions."""
    def __init__(self, input_dim, d_model=128, n_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Stack TCN blocks with exponentially increasing dilation
        # Use n_layers+2 to compensate for fewer params per block vs Mamba
        n_tcn_layers = n_layers + 2
        layers = []
        for i in range(n_tcn_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(d_model, d_model, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        
        self.proj = nn.Linear(d_model, d_model)  # Extra projection for capacity
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len) for Conv1d
        x = self.tcn(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, d_model)
        x = self.proj(x)
        x = self.norm(x)
        return self.head(x[:, -1, :])  # Classify based on last timestep


# ============================================================
# Model Registry
# ============================================================
MODEL_REGISTRY = {
    'Mamba': MambaClassifier,
    'LSTM': LSTMClassifier,
    'GRU': GRUClassifier,
    'Transformer': TransformerClassifier,
    'CNN-LSTM': CNNLSTMClassifier,
    'TCN': TCNClassifier,
}


def get_model(name, input_dim, d_model=128, n_layers=2):
    """
    Factory function to get a model by name.
    
    Args:
        name (str): Model name
        input_dim (int): Number of input features
        d_model (int): Hidden dimension
        n_layers (int): Number of layers
    
    Returns:
        nn.Module instance
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[name](input_dim=input_dim, d_model=d_model, n_layers=n_layers)


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())
