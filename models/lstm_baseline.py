import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dims, d_model=128):
        super().__init__()
        # input_dims is (32, 64, 16)
        
        # We project features to d_model size
        self.log_proj = nn.Linear(input_dims[0], d_model)
        self.text_proj = nn.Linear(input_dims[1], d_model)
        self.cve_proj = nn.Linear(input_dims[2], d_model)
        
        # LSTM Layer (The older technology)
        # We concatenate inputs: d_model * 3
        self.lstm = nn.LSTM(input_size=d_model*3, hidden_size=d_model, batch_first=True)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_log, x_text, x_cve):
        # Project inputs
        p_log = self.log_proj(x_log)
        p_text = self.text_proj(x_text)
        p_cve = self.cve_proj(x_cve)
        
        # Combine (Concatenate)
        combined = torch.cat([p_log, p_text, p_cve], dim=-1)
        
        # LSTM Processing
        out, (h_n, c_n) = self.lstm(combined)
        
        # Take last time step
        last_hidden = out[:, -1, :]
        
        score = self.classifier(last_hidden)
        # Return same dict format as Mamba for compatibility
        return {"score": score.squeeze(), "tci": score.squeeze()} 
