# models/tci_head.py
import torch
import torch.nn as nn

class TCILayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
        self.sig = nn.Sigmoid()

    def forward(self, z):
        # z: (b, d)
        return self.sig(self.fc(z)).squeeze(-1)  # (b,)
