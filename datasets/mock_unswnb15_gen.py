
import pandas as pd
import numpy as np
import os

def generate_mock_unswnb15(root_dir="data/raw/UNSW-NB15"):
    os.makedirs(root_dir, exist_ok=True)
    
    # Standard UNSW-NB15 Columns
    columns = [
        'id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
        'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
        'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
        'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
        'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
        'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
        'attack_cat', 'label'
    ]
    
    # Generate Train (2000 samples)
    print("Generating mock training data...")
    df_train = pd.DataFrame(np.random.rand(2000, len(columns)), columns=columns)
    df_train['id'] = range(1, 2001)
    df_train['proto'] = np.random.choice(['tcp', 'udp', 'icmp'], 2000)
    df_train['service'] = np.random.choice(['http', 'ftp', 'ssh', 'dns', '-'], 2000)
    df_train['state'] = np.random.choice(['FIN', 'CON', 'INT', 'REQ'], 2000)
    df_train['attack_cat'] = np.random.choice(['Normal', 'Generic', 'Exploits', 'Fuzzers'], 2000)
    df_train['label'] = (df_train['attack_cat'] != 'Normal').astype(int)
    
    # Generate Test (500 samples)
    print("Generating mock testing data...")
    df_test = pd.DataFrame(np.random.rand(500, len(columns)), columns=columns)
    df_test['id'] = range(1, 501)
    df_test['proto'] = np.random.choice(['tcp', 'udp', 'icmp'], 500)
    df_test['service'] = np.random.choice(['http', 'ftp', 'ssh', 'dns', '-'], 500)
    df_test['state'] = np.random.choice(['FIN', 'CON', 'INT', 'REQ'], 500)
    df_test['attack_cat'] = np.random.choice(['Normal', 'Generic', 'Exploits', 'Fuzzers'], 500)
    df_test['label'] = (df_test['attack_cat'] != 'Normal').astype(int)
    
    train_path = os.path.join(root_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(root_dir, "UNSW_NB15_testing-set.csv")
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Created {train_path}")
    print(f"Created {test_path}")

if __name__ == "__main__":
    generate_mock_unswnb15()
