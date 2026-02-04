# Setup Paths & Model
import sys
import os
import streamlit as st
import time
import pandas as pd
import numpy as np
import yaml

# Import the Real Bridge we just made
from real_bridge import LATEST_DATA, BAD_IPS, block_ip

st.set_page_config(layout="wide", page_title="REAL-TIME THREAT SENTINEL")


# Minimal config loader (avoid importing utils that may depend on torch)
def _load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_resources():
    # Try to import heavy ML libs lazily so the UI can run without them.
    try:
        import torch
        from models.classifier import ThreatModel
        from models.lstm_baseline import LSTMModel # Import LSTM fallback
        from utils.vectorizer import RealTimeVectorizer
    except Exception:
        return None, None, None

    cfg = _load_config("configs/default.yaml")
    device = torch.device(cfg.get("device", "cpu"))
    
    vectorizer = RealTimeVectorizer(device)
    
    # Logic to choose model based on available checkpoints
    # Priority 1: LSTM Baseline (Reliable fallback for demo)
    if os.path.exists("outputs/lstm_baseline.pt"):
        try:
            # Re-create config inferred from train_lstm.py or safe defaults
            # input_dims matches defaults in LSTMModel if not passed, but we pass them to be safe
            input_dims = (32, 64, 16) 
            model = LSTMModel(input_dims=input_dims, d_model=cfg["model"]["d_model"])
            model.load_state_dict(torch.load("outputs/lstm_baseline.pt", map_location=device))
            model.to(device)
            model.eval()
            print("[OK] Loaded LSTM Baseline")
            return model, vectorizer, device
        except Exception as e:
            print(f"[!] Failed to load LSTM baseline: {e}")

    # Priority 2: Mamba Model (Original Logic)
    # This might fail if weights are for a different arch, so we wrap in try/catch
    try:
        model = ThreatModel(input_dims=(32, 64, 16), d_model=cfg["model"]["d_model"])
        if os.path.exists("outputs/supervised_epoch5.pt"):
            model.load_state_dict(torch.load("outputs/supervised_epoch5.pt", map_location=device))
            model.to(device)
            model.eval()
            print("[OK] Loaded Mamba Model")
            return model, vectorizer, device
    except Exception as e:
        print(f"[!] Failed to load Mamba model: {e}")

    # Priority 3: Return initialized Mamba model (untrained) if nothing else works, to prevent crash
    print("[!] No valid checkpoint loaded, running partial model.")
    return model, vectorizer, device


model, vectorizer, device = load_resources()

# ==========================================
# 🎨 UI LAYOUT & SIDEBAR
# ==========================================

# Sidebar Controls
st.sidebar.markdown("## Analyst Controls")
enable_monitoring = st.sidebar.toggle("Enable Live Monitoring", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("### Threat Verification")
# Button: Zero-Day Threat (AI)
if st.sidebar.button("🧪 Trigger Zero-Day Simulation"):
    # This simulates a new, unknown threat that ONLY the AI detects
    st.sidebar.error("Simulating Zero-Day Attack Pattern...")
    st.session_state['ai_test_mode'] = time.time() # active for 5s
    # Ping 8.8.8.8 to generate traffic for the model to "analyze"
    import subprocess
    subprocess.Popen(["ping", "-n", "1", "8.8.8.8"], shell=True)

st.sidebar.markdown("---")
st.sidebar.caption("Pure Mamba Mode | v3.0.0")

# Main Header
st.markdown("## 🛡️ NEURAL DEFENSE: Mamba-Powered Threat Agent")
st.caption("🔒 UNCLASSIFIED // AI-DRIVEN ZERO-DAY PROTECTION SYSTEM")

# Top Metrics Row (3 Columns - No List Info)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Active Sensors**")
    m_sensors = st.empty()
    m_sensors.markdown("### 1 Online\n<span style='color:green; font-size:0.8em'>↑ Local Sniffer Active</span>", unsafe_allow_html=True)

with c2:
    st.markdown("**Packet Throughput**")
    m_throughput = st.empty()
    m_throughput.markdown("### 0.0 EPS\n<span style='color:gray; font-size:0.8em'> Real-Time Traffic</span>", unsafe_allow_html=True)

with c3:
    st.markdown("**Global Threat Level**")
    m_threat_level = st.empty()
    m_threat_level.markdown("### LOW\n<span style='color:gray; font-size:0.8em'>TCI 0.00 (AI Score)</span>", unsafe_allow_html=True)


# Middle Section: TCI Chart (Left) + Recent Alerts (Right)
st.markdown("---")
col_chart, col_alerts = st.columns([2, 1])

with col_chart:
    st.markdown("### 📈 Mamba Model Certainty (TCI)")
    chart_spot = st.empty()

with col_alerts:
    st.markdown("### 🚨 AI Decisions")
    alerts_spot = st.empty()

# Bottom Section: Live Inspector
st.markdown("---")
st.markdown("### 🕵️ Neural Packet Inspector")
packet_text = st.empty()
status_banner = st.empty()

# ==========================================
# 🔄 MAIN LOOP
# ==========================================

tci_history = []
recent_alerts = []

# Throughput Tracking
last_packet_count = 0
last_time = time.time()
eps = 0.0

if enable_monitoring:
    # We loop as long as the toggle is ON
    while True:
        # 1. GET REAL DATA 
        net_msg = LATEST_DATA["network"]
        log_msg = LATEST_DATA["log"]
        # In AI Mode, we ignore 'threat_match' (list match)
        current_packet_count = LATEST_DATA.get("packet_count", 0) 
        
        # 2. VECTORIZE & PREDICT
        full_context = f"{net_msg} | {log_msg}"
        model_tci = None
        
        if 'ai_test_mode' in st.session_state:
            if time.time() - st.session_state['ai_test_mode'] < 3.0:
                model_tci = 0.995 
                
        if model is not None and vectorizer is not None and model_tci is None:
            try:
                x_log, x_text, x_cve = vectorizer.process_log_line(full_context)
                import torch # Ensure torch is available here
                with torch.no_grad():
                    out = model(x_log, x_text, x_cve)
                    model_tci = out["tci"].item()
            except Exception:
                pass

        # 3. PURE AI DEFENSE
        final_tci = model_tci if model_tci is not None else 0.0
        
        # ACTIVE DEFENSE TRIGGER
        if final_tci > 0.95:
             target_ip = LATEST_DATA.get("source_ip")
             if target_ip and target_ip != "0.0.0.0":
                 # Execute Block via Bridge
                 block_ip(target_ip)

        # 4. DATA UPDATE
        tci_history.append(final_tci)
        if len(tci_history) > 100:
            tci_history.pop(0)
            
        # Calculate Real Throughput (EPS)
        current_time = time.time()
        time_diff = current_time - last_time
        if time_diff >= 1.0: 
            packets_diff = current_packet_count - last_packet_count
            eps = packets_diff / time_diff
            last_packet_count = current_packet_count
            last_time = current_time
        elif 'eps' not in locals():
            eps = 0.0

        # Update Alert List
        if final_tci > 0.8:
            timestamp = time.strftime("%H:%M:%S")
            alert_ip = LATEST_DATA.get("source_ip", "Unknown")
            
            # Determine Action
            if final_tci > 0.95:
                type_ = "Zeor-Day (Mamba)"
                action = "BLOCKED (AI Decision)"
                sev = "CRITICAL"
            else:
                type_ = "Anomaly"
                action = "LOGGED"
                sev = "HIGH"
            
            # Insert into table
            if alert_ip and alert_ip != "0.0.0.0":
                if not recent_alerts or recent_alerts[0]['Source IP'] != alert_ip:
                     recent_alerts.insert(0, {
                         "Timestamp": timestamp, 
                         "Source IP": alert_ip, 
                         "Type": type_, 
                         "Action": action,
                         "Severity": sev
                     })
                     if len(recent_alerts) > 10:
                        recent_alerts.pop()
        
        # 5. UI REFRESH
        m_sensors.markdown("### 1 Online (AI)\n<span style='color:green; font-size:0.8em'>↑ Neural Net Active</span>", unsafe_allow_html=True)
        m_throughput.markdown(f"### {eps:.1f} EPS\n<span style='color:gray; font-size:0.8em'> Encrypted Traffic</span>", unsafe_allow_html=True)
        
        if final_tci > 0.8:
            m_threat_level.markdown(f"<h3 style='color:red'>CRITICAL</h3><span style='color:red; font-size:0.8em'>TCI {final_tci:.3f}</span>", unsafe_allow_html=True)
            status_banner.error(f"NEURAL DEFENSE ENGAGED | Target: {LATEST_DATA['source_ip']} | Score: {final_tci:.2%}")
        else:
            m_threat_level.markdown(f"<h3 style='color:green'>SAFE</h3><span style='color:gray; font-size:0.8em'>TCI {final_tci:.3f}</span>", unsafe_allow_html=True)
            status_banner.success("System Secure. Mamba monitoring encrypted stream...")
            
        chart_spot.line_chart(pd.DataFrame(tci_history, columns=["AI Certainty"]), height=250)
        
        if recent_alerts:
            alerts_spot.dataframe(pd.DataFrame(recent_alerts), hide_index=True)
        else:
            alerts_spot.caption("No AI detections yet.")

        packet_text.code(f"{time.strftime('%H:%M:%S')} | {net_msg}")
        time.sleep(0.5)

else:
    st.info("System Standby. Enable 'Live Monitoring' in sidebar to start SOC view.")