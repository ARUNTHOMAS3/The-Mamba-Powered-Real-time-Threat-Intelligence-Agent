
import json
import pandas as pd
import os

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def generate_latex_offline(data):
    if not data: return "% No Offline Data Available"
    
    results = data.get("results", [])
    if not results: return "% No Offline Results"
    
    df = pd.DataFrame(results)
    
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    # Ensure cols exist
    for c in cols:
        if c not in df.columns: df[c] = 0.0
        
    df = df[cols].copy()
    
    latex = df.to_latex(index=False, float_format="%.4f", caption="Offline Window-Based Classification Performance (CICIDS2017 Tabular)", label="tab:offline_results")
    return latex

def generate_latex_streaming(data):
    if not data: return "% No Streaming Data Available"
    
    if not data: return "% No Streaming Results"
    df = pd.DataFrame(data)
    
    cols = ["Model", "Latency_Mean_ms", "Throughput_Seq_req_s", "Streaming_F1", "FP_Rate", "Projected_FP_per_min_at_1kEPS"]
    # Ensure cols exist
    for c in cols:
        if c not in df.columns: df[c] = 0.0
        
    df = df[cols].copy()
    
    df.columns = ["Model", "Latency (ms)", "Throughput (req/s)", "Stream F1", "FP Rate", "Proj FP/min"]
    
    latex = df.to_latex(index=False, float_format="%.4f", caption="Causal Streaming Performance & Efficiency (CICIDS2017 Tabular)", label="tab:streaming_results")
    return latex

def main():
    print("Generating Tables...")
    
    offline_data = load_json("outputs/metrics_offline.json")
    streaming_data = load_json("outputs/metrics_streaming.json")
    
    tex_content = ""
    tex_content += r"\section{Experimental Results}" + "\n\n"
    
    tex_content += r"\subsection{Offline Detection Performance}" + "\n"
    tex_content += generate_latex_offline(offline_data) + "\n\n"
    
    tex_content += r"\subsection{Real-Time Streaming Efficiency}" + "\n"
    tex_content += generate_latex_streaming(streaming_data) + "\n"
    
    with open("outputs/comparison_tables.tex", "w") as f:
        f.write(tex_content)
        
    print("Tables generated at outputs/comparison_tables.tex")

if __name__ == "__main__":
    main()
