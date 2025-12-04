import streamlit as st
import torch
import sys
import os
import time
import pandas as pd
import numpy as np

# Import the Real Bridge we just made
from real_bridge import LATEST_DATA, BAD_IPS

# Setup Paths & Model
sys.path.append(os.path.abspath("."))
from models.classifier import ThreatModel
from utils.utils import load_config
from utils.vectorizer import RealTimeVectorizer

st.set_page_config(layout="wide", page_title="REAL-TIME THREAT SENTINEL")

# Load Resources
@st.cache_resource
def load_resources():
    cfg = load_config("configs/default.yaml")
    device = torch.device(cfg.get("device", "cpu"))
    
    model = ThreatModel(input_dims=(32, 64, 16), d_model=cfg["model"]["d_model"])
    # Load your trained weights
    if os.path.exists("outputs/supervised_epoch5.pt"):
        model.load_state_dict(torch.load("outputs/supervised_epoch5.pt", map_location=device))
    
    vectorizer = RealTimeVectorizer(device)
    return model, vectorizer, device

model, vectorizer, device = load_resources()

# UI Header
st.title("🛡️ ACTIVE DEFENSE: Real-Time Network & Log Monitor")
st.markdown(f"**Threat Intelligence Feed:** {len(BAD_IPS)} active botnet IPs loaded.")

# Live Metrics
col1, col2, col3 = st.columns(3)
m_net = col1.empty()
m_log = col2.empty()
m_tci = col3.empty()

# Charts
chart_spot = st.empty()
tci_history = []

st.markdown("### 📡 Live Packet Stream")
packet_text = st.empty()

if st.button("🚀 Connect to Live Sensors"):
    while True:
        # 1. GET REAL DATA from the Bridge
        net_msg = LATEST_DATA["network"]
        log_msg = LATEST_DATA["log"]
        is_known_threat = LATEST_DATA["threat_match"]
        
        # 2. VECTORIZE (Turn real text into tensors)
        # We combine network info and log info into one context
        full_context = f"{net_msg} | {log_msg}"
        x_log, x_text, x_cve = vectorizer.process_log_line(full_context)
        
        # 3. MAMBA ANALYSIS
        with torch.no_grad():
            out = model(x_log, x_text, x_cve)
            model_tci = out["tci"].item()
            
        # 4. FUSION (Model Score + Real Threat Feed)
        # If the IP is in our Blocklist, FORCE the score to 1.0 (Critical)
        final_tci = 1.0 if is_known_threat else model_tci
        
        # 5. UPDATE UI
        tci_history.append(final_tci)
        if len(tci_history) > 50: tci_history.pop(0)
        
        m_net.metric("Network Activity", LATEST_DATA["source_ip"])
        
        if is_known_threat:
            m_tci.metric("Threat Index", "1.0 (CRITICAL)", delta="KNOWN BOTNET DETECTED", delta_color="inverse")
            st.error(f"🚨 ALERT: Traffic detected from known malicious IP: {LATEST_DATA['source_ip']}")
        else:
            m_tci.metric("Threat Index", f"{final_tci:.3f}", delta="Normal")

        packet_text.code(net_msg)
        
        # Draw Chart
        chart_data = pd.DataFrame(tci_history, columns=["Threat Score"])
        chart_spot.line_chart(chart_data)
        
        time.sleep(0.5)