import torch
import hashlib
import numpy as np
import re

class RealTimeVectorizer:
    """
    Advanced Vectorizer that handles both Network Packets and System Logs.
    """
    def __init__(self, device):
        self.device = device

    def _text_to_vector(self, text, dim):
        # 1. Clean text to generalize (remove specific numbers/IPs for patterns)
        clean_text = re.sub(r'\d', 'N', str(text)) 
        
        # 2. Deterministic Hash -> Seed
        seed = int(hashlib.sha256(clean_text.encode('utf-8')).hexdigest(), 16) % (2**32)
        np.random.seed(seed)
        
        # 3. Generate Vector
        vector = np.random.normal(0, 1, (1, 128, dim))
        return torch.tensor(vector, dtype=torch.float32).to(self.device)

    def process_log_line(self, log_line):
        """
        Takes a raw string (Network log OR System log) and returns 3 tensors.
        """
        # Feature 1: Log Structure (32 dim)
        x_log = self._text_to_vector(log_line, 32)
        
        # Feature 2: Semantic Context (64 dim)
        x_text = self._text_to_vector(log_line, 64)
        
        # Feature 3: Threat Signal (16 dim)
        # If text matches known threat keywords, we inject a signal spike
        threat_keywords = ['cve', 'exploit', 'attack', 'failed', 'denied', 'botnet', 'malicious']
        if any(kw in log_line.lower() for kw in threat_keywords):
            x_cve = self._text_to_vector(log_line, 16) + 2.0
        else:
            x_cve = self._text_to_vector(log_line, 16)
            
        return x_log, x_text, x_cve
