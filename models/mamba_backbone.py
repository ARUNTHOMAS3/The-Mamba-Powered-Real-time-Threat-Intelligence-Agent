# models/mamba_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaConfig:
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model

class PureMambaBlock(nn.Module):
    """
    A pure PyTorch implementation of the Mamba block (S6) for CPU usage.
    It simulates the selective scan mechanism.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model

        # Projects input to hidden state (x) and gate (z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Conv for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Projection for x to B, C, delta
        # dt_rank = math.ceil(d_model / 16)
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Projection for delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D / HiPPO matrix A (approximated as diagonal for Mamba)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, L, D)
        (b, l, d) = x.shape
        
        # 1. Project to x and z
        xz = self.in_proj(x) # (B, L, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1) # (B, L, d_inner) each

        # 2. Conv1d
        x_branch = x_branch.transpose(1, 2) # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :l] # Causal padding
        x_branch = x_branch.transpose(1, 2) # (B, L, d_inner)
        x_branch = self.act(x_branch)

        # 3. Discretization & SSM (Scan)
        # We need to compute B, C, dt from x_branch
        x_dbl = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        
        # Scanning
        # This is a simplified "slow" scan loop in pure PyTorch
        # A_log corresponds to -A in some implementations, we take standard approximation
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Prepare hidden state
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        y = []
        
        for t in range(l):
            # Current inputs
            xt = x_branch[:, t, :] # (B, d_inner)
            dt_t = dt[:, t, :]     # (B, d_inner)
            bt = B[:, t, :]        # (B, d_state)
            ct = C[:, t, :]        # (B, d_state)
            
            # Discretize A -> dA, B -> dB
            # exp(A * dt)
            # A broadcast: (1, d_inner, d_state) * (B, d_inner, 1)
            dA = torch.exp(A * dt_t.unsqueeze(-1)) # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * bt.unsqueeze(1) # (B, d_inner, d_state)
            
            # State update
            # h_t = dA * h_{t-1} + dB * x_t
            h = dA * h + dB * xt.unsqueeze(-1)
            
            # Output
            # y_t = C_t * h_t + D * x_t
            # C broadcast: (B, 1, d_state) * (B, d_inner, d_state) -> sum dim -1
            y_t = torch.sum(h * ct.unsqueeze(1), dim=-1) + self.D * xt
            y.append(y_t)
            
        y = torch.stack(y, dim=1) # (B, L, d_inner)
        
        # 4. Gating
        y = y * self.act(z_branch)
        out = self.out_proj(y)
        
        return out


class MambaBackbone(nn.Module):
    def __init__(self, d_input, d_model=128, d_state=16):
        super().__init__()
        self.d_model = d_model
        
        # Project input to d_model first if needed
        self.embedding = nn.Linear(d_input, d_model)
        
        # Use our Pure PyTorch Mamba Block
        self.mamba_block = PureMambaBlock(d_model, d_state=d_state)
        
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (b, seq_len, d_input)
        x_emb = self.embedding(x)
        y = self.mamba_block(x_emb)
        return self.layernorm(y)
