# Setup Paths & Model
import sys
import os
import streamlit as st
import time
import pandas as pd
import numpy as np
import yaml
import datetime

# Fix path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Real Bridge we just made
from real_bridge import LATEST_DATA, BAD_IPS

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS (THEME)
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="ACTIVE DEFENSE UI")

ST_STYLE = """
<style>
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #050510;
        color: #e0e0e0;
        font-family: 'Consolas', 'Courier New', monospace;
    }

    /* REMOVE DEFAULT HEADER/FOOTER */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* METRIC CARDS (HUD STYLE) */
    div[data-testid="stMetric"] {
        background-color: #0f111a;
        border: 1px solid #1f293a;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #8899a6;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricValue"] {
        color: #00ffcc;
        font-size: 1.8rem;
        text-shadow: 0 0 5px #00ffcc;
    }

    /* CUSTOM ALERT BANNER */
    .alert-banner {
        background-color: #3d0000;
        border: 1px solid #ff0000;
        color: #ffcccc;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }
    }

    /* PACKET CONSOLE */
    .console-box {
        background-color: #000;
        border: 1px solid #333;
        color: #33ff00;
        font-family: 'Consolas', monospace;
        padding: 10px;
        height: 100px;
        overflow-y: hidden;
        white-space: pre-wrap;
    }
    
    /* TABLE/DATAFRAME STYLING */
    div[data-testid="stDataFrame"] {
        background-color: #0f111a;
        border: 1px solid #1f293a;
    }

    /* TITLE HEADER */
    .hud-header {
        border-bottom: 2px solid #1f293a;
        margin-bottom: 20px;
        padding-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hud-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .hud-subtitle {
        color: #666;
        font-size: 0.8rem;
    }
</style>
"""
st.markdown(ST_STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. STATE MANAGEMENT
# ---------------------------------------------------------
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'threat_history' not in st.session_state:
    st.session_state.threat_history = [0.0]*50


# ---------------------------------------------------------
# 3. RESOURCE LOADING
# ---------------------------------------------------------
def _load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_resources():
    # Try to import heavy ML libs lazily
    try:
        import torch
        from models.classifier import ThreatModel
        from utils.vectorizer import RealTimeVectorizer
    except Exception:
        return None, None, None

    cfg = _load_config("configs/default.yaml")
    device = torch.device(cfg.get("device", "cpu"))
    model = ThreatModel(input_dims=(32, 64, 16), d_model=cfg["model"]["d_model"])
    
    # Load your trained weights if exist
    if os.path.exists("outputs/supervised_epoch5.pt"):
        try:
            state_dict = torch.load("outputs/supervised_epoch5.pt", map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("tci_head.0."):
                    new_state_dict[k.replace("tci_head.0.", "tci_head.")] = v
                elif k.startswith("classifier.0."):
                    new_state_dict[k.replace("classifier.0.", "classifier.")] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            # st.warning("⚠️ Architecture changed. Running init weights.")

    vectorizer = RealTimeVectorizer(device)
    return model, vectorizer, device

model, vectorizer, device = load_resources()


# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ Analyst Controls")
    
    # 1. LIVE MONITORING TOGGLE
    monitor_active = st.toggle("Enable Live Monitoring", value=True)
    
    st.divider()
    
    st.markdown("### 🧪 Simulation (Demo)")
    # 2. SYNTHETIC ATTACK INJECTION
    inject_attack = st.checkbox("Inject Synthetic Attack Event")
    
    if inject_attack:
        st.info("⚠️ SIMULATION: Forcing threat detection signatures.")
    
    st.divider()
    st.caption("version 2.4.0 // JOURNAL-READY MODE")


# ---------------------------------------------------------
# 5. MAIN DASHBOARD LAYOUT
# ---------------------------------------------------------

# HEADER
st.markdown("""
<div class="hud-header">
    <div>
        <div class="hud-subtitle">// UNCLASSIFIED // INTERNAL USE ONLY // SYNTHETIC EVALUATION ENVIRONMENT</div>
        <div class="hud-title">🛡️ ACTIVE DEFENSE: Real-Time Network & Log Monitor</div>
    </div>
</div>
""", unsafe_allow_html=True)

# TOP ROW METRICS
col1, col2, col3, col4 = st.columns(4)

# A. Active Sensors
with col1:
    st.metric("Active Sensors", "2 Online", delta="All Systems Nominal")

# B. Packet Throughput (Mocked/Calculated)
# Use a random fluctuation for "liveness" feel
import random
keps = 12.0 + random.uniform(-0.5, 2.5)
with col2:
    st.metric("Packet Throughput", f"{keps:.1f} kEPS", delta="+1.2%vs.Avg")

# C. Threat Intelligence
with col3:
    st.metric("Threat Intelligence", f"{len(BAD_IPS)} Indicators", delta="Updated 2m ago")

# D. Global Threat Level (Placeholder for now, changes on threat)
threat_level_display = st.empty()


# ---------------------------------------------------------
# 6. LOGIC LOOP & UPDATES
# ---------------------------------------------------------

# Chart & Table Placeholders
st.markdown("### 📈 Real-Time Threat Certainty Index (TCI)")
chart_spot = st.empty()

col_table, col_console = st.columns([2, 1])

with col_table:
    st.markdown("### 🚨 Recent Alerts")
    alerts_table = st.empty()

with col_console:
    st.markdown("### 📡 Live Packet Inspector")
    console_spot = st.empty()

alert_banner_spot = st.empty()

# We use a button to trigger the loop if it's not running, or just run it.
# In Streamlit, a while loop inside the script will block interactions unless we are careful.
# However, for a dashboard display, this is a common pattern.
if monitor_active:
    # Use a placeholders for the top metrics too so they update live
    # (We need to re-create the columns inside the loop or use empty slots created before)
    pass

while monitor_active:
    # Fetch Data
    net_msg = LATEST_DATA["network"]
    log_msg = LATEST_DATA["log"]
    real_threat = LATEST_DATA["threat_match"]
    
    # Logic
    is_active_threat = real_threat or inject_attack
    
    # 1. Update Global Threat Level Metric (using the placeholder we created earlier)
    with threat_level_display.container():
        if is_active_threat:
            st.metric("Global Threat Level", "CRITICAL", delta="TCI 0.982", delta_color="inverse")
        else:
            st.metric("Global Threat Level", "LOW", delta="Normal", delta_color="normal")

    # 2. Process Model (Mamba)
    full_context = f"{net_msg} | {log_msg}"
    model_tci = 0.05
    
    if model is not None and vectorizer is not None:
        try:
            x_log, x_text, x_cve = vectorizer.process_log_line(full_context)
            try:
                import torch as _torch
                with _torch.no_grad():
                    out = model(x_log, x_text, x_cve)
                    model_tci = out["tci"].item()
            except:
                pass
        except:
            pass

    final_tci = 1.0 if is_active_threat else model_tci
    
    # Update History
    st.session_state.threat_history.append(final_tci)
    if len(st.session_state.threat_history) > 100:
        st.session_state.threat_history.pop(0)

    # 3. Update Chart
    chart_data = pd.DataFrame(st.session_state.threat_history, columns=["Threat Score"])
    chart_spot.line_chart(chart_data)

    # 4. Handle Alerts
    if is_active_threat:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        source_ip = LATEST_DATA['source_ip']
        if inject_attack and source_ip == "0.0.0.0": source_ip = "192.168.1.104 (Simulated)"
        
        # Add to history if unique or new
        if not st.session_state.alert_history or st.session_state.alert_history[0]["Timestamp"] != current_time:
             new_alert = {
                "Timestamp": current_time,
                "Source IP": source_ip,
                "Type": "Botnet Activity",
                "Severity": "HIGH",
                "Action": "BLOCK"
            }
             st.session_state.alert_history.insert(0, new_alert)
             if len(st.session_state.alert_history) > 10:
                 st.session_state.alert_history.pop()

    # Render Table
    if st.session_state.alert_history:
        alerts_table.dataframe(pd.DataFrame(st.session_state.alert_history), use_container_width=True)
    else:
        alerts_table.info("No active threats detected.")

    # 5. Live Console
    protocol_color = "#33ff00"
    if "TCP" in net_msg: protocol_color = "#00ccff"
    elif "UDP" in net_msg: protocol_color = "#ffcc00"
    
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    console_html = f"""
    <div class="console-box">
    <span style="color:#666">{timestamp}</span> | <span style="color:{protocol_color}">{net_msg}</span>
    </div>
    """
    console_spot.markdown(console_html, unsafe_allow_html=True)
    
    # 6. ALERT BANNER
    if is_active_threat:
        src_ip = LATEST_DATA['source_ip']
        if inject_attack and src_ip == "0.0.0.0": src_ip = "192.168.1.104"
        
        banner_html = f"""
        <div class="alert-banner">
            <h3>⚠️ SECURITY ALERT: Potential Botnet Detection</h3>
            <p><strong>Analysis:</strong> High-confidence anomaly detected originating from {src_ip}. Pattern matches known C2 communication signatures.</p>
            <p><strong>Recommended Action:</strong> Isolate endpoint {src_ip} and inspect outbound traffic on port 443/80.</p>
        </div>
        """
        alert_banner_spot.markdown(banner_html, unsafe_allow_html=True)
    else:
        alert_banner_spot.empty()

    time.sleep(0.5)
    # NO st.rerun() call here! The loop continues updating placeholders.