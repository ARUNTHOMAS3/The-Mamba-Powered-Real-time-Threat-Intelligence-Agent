# models/mamba_backbone.py
import torch
import torch.nn as nn

# Try to import a real SSM/Mamba implementation; if not present, use a light fallback RNN-like module.
try:
    from s4 import S4  # example; replace with actual package name if installed
    HAS_S4 = True
except Exception:
    HAS_S4 = False

class FallbackSSM(nn.Module):
    """Lightweight SSM-like fallback using gated RNN for local experiments."""
    def __init__(self, d_input, d_model):
        super().__init__()
        self.rnn = nn.GRU(d_input, d_model, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (b, seq, d_input)
        out, _ = self.rnn(x)
        return self.proj(out)  # (b, seq, d_model)

class MambaBackbone(nn.Module):
    def __init__(self, d_input, d_model=128, d_state=64):
        super().__init__()
        self.d_model = d_model
        if HAS_S4:
            # Example usage if S4/Mamba installed — adapt to actual API.
            self.ssm = S4(d_model, l_max=None)
            self.project = nn.Linear(d_input, d_model)
        else:
            self.ssm = FallbackSSM(d_input, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (b, seq_len, d_input)
        if HAS_S4:
            x_proj = self.project(x)
            y = self.ssm(x_proj)  # depends on API
        else:
            y = self.ssm(x)
        return self.layernorm(y)  # (b, seq, d_model)
