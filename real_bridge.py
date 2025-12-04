import time
import threading
import subprocess
import requests
import sys
from scapy.all import sniff, IP, TCP, UDP, ICMP, conf

# --- 1. CONFIGURATION ---
conf.L3socket = conf.L3socket

# Global storage
LATEST_DATA = {
    "log": "System Active...",
    "network": "Listening for traffic...",
    "threat_match": False,
    "source_ip": "0.0.0.0",
    "threat_timestamp": 0
}

# --- 2. DOWNLOAD THREAT FEED (FIXED HEADER ISSUE) ---
print("📉 Downloading Feodo Tracker IPs...")
try:
    url = "https://feodotracker.abuse.ch/downloads/ipblocklist.csv"
    response = requests.get(url, timeout=5)
    content = response.content.decode('utf-8')
    lines = [l for l in content.splitlines() if not l.startswith('#')]
    
    BAD_IPS = set()
    for l in lines:
        parts = l.split(',')
        if len(parts) > 1:
            # Clean up the IP string
            ip = parts[1].replace('"', '').strip()
            # SKIP THE HEADER "dst_ip"
            if ip and ip != 'dst_ip': 
                BAD_IPS.add(ip)
    
    print(f"✅ LOADED {len(BAD_IPS)} MALICIOUS IPs.")
    print("🔻 TARGET IP (Attack this IP to test):")
    
    if len(BAD_IPS) > 0:
        # Get a reliable target
        TARGET_IP = list(BAD_IPS)[-1] 
        print(f"   👉 {TARGET_IP}")
    else:
        print("   ⚠️ No IPs loaded. Check internet.")
        TARGET_IP = "8.8.8.8"
except Exception as e:
    print(f"⚠️ Feed Error: {e}")
    BAD_IPS = set()
    TARGET_IP = "1.1.1.1"

# --- 3. SNIFFER ---
def packet_callback(packet):
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst
        
        # Check Threat
        is_threat = (src in BAD_IPS) or (dst in BAD_IPS)
        
        protocol = "TCP" if TCP in packet else "UDP" if UDP in packet else "ICMP"
        LATEST_DATA["network"] = f"[{protocol}] {src} -> {dst}"
        
        if is_threat:
            print(f"\n[🚨 ALERT] TRAFFIC MATCH: {dst} is in BAD_IPS!")
            LATEST_DATA["threat_match"] = True
            LATEST_DATA["source_ip"] = dst if dst in BAD_IPS else src
            LATEST_DATA["threat_timestamp"] = time.time()
        else:
            if time.time() - LATEST_DATA["threat_timestamp"] > 5:
                LATEST_DATA["threat_match"] = False

def start_sniffing():
    print("🔎 Sniffer Started...")
    try:
        sniff(filter="ip", prn=packet_callback, store=0)
    except Exception as e:
        print(f"Sniffer Error: {e}")

# --- 4. LOG MONITOR ---
def monitor_windows_logs():
    cmd = "Get-EventLog -LogName System -Newest 1 | Select-Object -ExpandProperty Message"
    last_log = ""
    while True:
        try:
            res = subprocess.run(["powershell.exe", "-Command", cmd], capture_output=True, text=True)
            msg = res.stdout.strip()
            if msg and msg != last_log:
                LATEST_DATA["log"] = msg
                last_log = msg
        except:
            pass
        time.sleep(2)

# Start Threads
t1 = threading.Thread(target=start_sniffing, daemon=True)
t2 = threading.Thread(target=monitor_windows_logs, daemon=True)
t1.start()
t2.start()
