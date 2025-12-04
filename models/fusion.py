# models/fusion.py
import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, log_vec, osint_vec, cve_vec):
        # Accept either (b,d) single-step vectors or (b,seq,d) and use last step.
        if log_vec.dim() == 3:
            log_vec = log_vec[:, -1, :]
        if osint_vec.dim() == 3:
            osint_vec = osint_vec[:, -1, :]
        if cve_vec.dim() == 3:
            cve_vec = cve_vec[:, -1, :]
        x = torch.cat([log_vec, osint_vec, cve_vec], dim=-1)
        return self.act(self.norm(self.fc(x)))
