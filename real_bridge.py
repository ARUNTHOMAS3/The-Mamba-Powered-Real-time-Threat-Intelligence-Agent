import time
import threading
import subprocess
import requests
import sys
from scapy.all import sniff, IP, TCP, UDP, ICMP, conf

# --- 1. CONFIGURATION ---
try:
    from scapy.all import conf
    conf.use_pcap = True # Force Npcap usage
    # conf.L3socket = conf.L3socket # Optional: Uncomment if Layer 3 injection is needed
    print(f"[INFO] Scapy Configuration: use_pcap={conf.use_pcap}")
except ImportError:
    pass

from scapy.arch import get_if_list


# Global storage
LATEST_DATA = {
    "log": "System Active...",
    "network": "Listening for traffic...",
    "threat_match": False,
    "source_ip": "0.0.0.0",
    "threat_timestamp": 0,
    "packet_count": 0,
    "threat_ip": None
}

# --- 2. THREAT FEED (DISABLED - AI ONLY MODE) ---
print("AI-ONLY MODE: Ignoring Static Threat Lists.")
BAD_IPS = set() 
TARGET_IP = "8.8.8.8" # Default verify target since we have no bad list

# --- 3. SNIFFER ---
def packet_callback(packet):
    if IP in packet:
        src = packet[IP].src
        dst = packet[IP].dst
        
        # In AI-only mode, the bridge does NOT decide what is a threat.
        # It just passes data to the App (Mamba Model).
        is_threat = False 
        
        protocol = "TCP" if TCP in packet else "UDP" if UDP in packet else "ICMP"
        LATEST_DATA["network"] = f"[{protocol}] {src} -> {dst}"
        LATEST_DATA["source_ip"] = src 
        LATEST_DATA["packet_count"] += 1
        
        # We leave threat_match False here. The APP decides.
        LATEST_DATA["threat_match"] = False

def start_sniffing():
    print("Sniffer Started...")
    try:
        # Try real sniffing
        sniff(filter="ip", prn=packet_callback, store=0)
    except Exception as e:
        print(f"Sniffer Error: {e}")
        print("[INFO] Switching to MOCK TRAFFIC MODE (Demo)...")
        import random
        
        while True:
            # Generate fake IPs
            fake_src = f"192.168.1.{random.randint(1, 255)}"
            
            # Occasionally pick a BAD IP to demonstrate the alert system
            if len(BAD_IPS) > 0 and random.random() < 0.1:
                fake_dst = list(BAD_IPS)[random.randint(0, len(BAD_IPS)-1)]
            else:
                fake_dst = f"10.0.0.{random.randint(1, 255)}"

            is_threat = (fake_src in BAD_IPS) or (fake_dst in BAD_IPS)
            protocol = random.choice(["TCP", "UDP"])
            
            # Update global state
            LATEST_DATA["network"] = f"[{protocol}] {fake_src} -> {fake_dst}"
            LATEST_DATA["source_ip"] = fake_src # Just show activity
            LATEST_DATA["packet_count"] += 1
            
            if is_threat:
                LATEST_DATA["threat_match"] = True
                LATEST_DATA["source_ip"] = fake_dst if fake_dst in BAD_IPS else fake_src
                LATEST_DATA["threat_timestamp"] = time.time()
                print(f"[MOCK ALERT] TRAFFIC MATCH: {fake_dst} is in BAD_IPS!")
            else:
                if time.time() - LATEST_DATA["threat_timestamp"] > 5:
                    LATEST_DATA["threat_match"] = False
            
            time.sleep(random.uniform(0.5, 2.0))

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

# --- 5. ACTIVE DEFENSE (FIREWALL BLOCKING) ---
def block_ip(ip_address):
    """
    Executes a Windows Firewall rule to block the malicious IP.
    Requires Admin privileges to work fully, but we try anyway.
    """
    rule_name = f"BLOCK_MALICIOUS_{ip_address}"
    # Check if rule exists first (simple check to avoid duplicate error spam)
    check_cmd = f"Get-NetFirewallRule -DisplayName '{rule_name}'"
    
    # Command to create block rule
    block_cmd = (
        f"New-NetFirewallRule -DisplayName '{rule_name}' "
        f"-Direction Inbound -RemoteAddress {ip_address} "
        f"-Action Block -Protocol Any"
    )
    
    try:
        # Check if already blocked
        subprocess.run(["powershell", "-Command", check_cmd], capture_output=True, check=True)
        # If success, it exists, so do nothing
    except subprocess.CalledProcessError:
        # Rule doesn't exist, create it
        try:
            print(f"⚡ [ACTIVE DEFENSE] ENGAGING FIREWALL: BLOCKING {ip_address}...")
            subprocess.run(["powershell", "-Command", block_cmd], capture_output=True)
            LATEST_DATA["log"] = f"ACTIVE DEFENSE: BLOCKED {ip_address}"
        except Exception as e:
            print(f"[!] Block Failed: {e}")

# Start Threads
t1 = threading.Thread(target=start_sniffing, daemon=True)
t2 = threading.Thread(target=monitor_windows_logs, daemon=True)
t1.start()
t2.start()

