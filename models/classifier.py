import torch
import torch.nn as nn
from models.mamba_backbone import MambaBackbone
from models.fusion import FusionLayer

class ThreatModel(nn.Module):
    def __init__(self, input_dims, d_model=128):
        super().__init__()
        # input_dims is a tuple: (log_dim, text_dim, cve_dim)
        # We access them by index [0], [1], [2]
        
        self.backbone_log = MambaBackbone(input_dims[0], d_model)
        self.backbone_text = MambaBackbone(input_dims[1], d_model)
        self.backbone_cve = MambaBackbone(input_dims[2], d_model)
        
        self.fusion = FusionLayer(d_model)
        
        # Threat Certainty Index (TCI) Head
        # Heads now output logits; apply sigmoid outside when needed
        self.tci_head = nn.Linear(d_model, 1)
        
        # Binary Classifier Head (logits)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x_log, x_text, x_cve):
        enc_log = self.backbone_log(x_log)
        enc_text = self.backbone_text(x_text)
        enc_cve = self.backbone_cve(x_cve)
        
        z = self.fusion(enc_log, enc_text, enc_cve)
        
        tci_logit = self.tci_head(z)
        score_logit = self.classifier(z)
        
        return {
            "tci_logit": tci_logit.squeeze(),
            "tci": torch.sigmoid(tci_logit).squeeze(),
            "score_logit": score_logit.squeeze(),
            "score": torch.sigmoid(score_logit).squeeze(),
        }
