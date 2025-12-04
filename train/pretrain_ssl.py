# train/pretrain_ssl.py
import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultimodalDataset
from models.mamba_backbone import MambaBackbone
from utils.utils import load_config, mkdirp, set_seed

def mask_input(x, mask_prob=0.15):
    # x: tensor (b, seq, feat)
    mask = (torch.rand_like(x[:,:,0]) > mask_prob).unsqueeze(-1)
    return x * mask.float()

def main(config, max_epochs=2):
    cfg = load_config(config)
    set_seed(42)
    device = torch.device(cfg.get("device","cpu"))

    ds = MultimodalDataset(path="data/processed/synth_small.json", seq_len=cfg["data"]["seq_len"], create_synthetic=True, n_samples=1000)
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True)

    # We'll pretrain a backbone that maps concatenated modality -> seq encodings
    input_dim = 32 + 64 + 16
    backbone = MambaBackbone(d_input=input_dim, d_model=cfg["model"]["d_model"], d_state=cfg["model"]["d_state"]).to(device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()

    for epoch in range(max_epochs):
        backbone.train()
        total = 0
        for batch in dl:
            x_log = batch["x_log"].to(device)
            x_text = batch["x_text"].to(device)
            x_cve = batch["x_cve"].to(device)
            # concat features along last dim to feed backbone
            x = torch.cat([x_log, x_text, x_cve], dim=-1)
            x_mask = mask_input(x, mask_prob=0.15)
            enc = backbone(x_mask)
            # next-step prediction on enc-> reconstruct original next-step raw input vector
            pred_next = enc[:, :-1, :].mean(-1)  # simplified predictor (toy)
            target = x[:, 1:, :].mean(-1)
            loss = criterion(pred_next, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"[SSL] Epoch {epoch} loss {total/len(dl):.4f}")
        os.makedirs("outputs", exist_ok=True)
        torch.save(backbone.state_dict(), f"outputs/ssl_backbone_epoch{epoch}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_epochs", type=int, default=2)
    args = parser.parse_args()
    main(args.config, args.max_epochs)
