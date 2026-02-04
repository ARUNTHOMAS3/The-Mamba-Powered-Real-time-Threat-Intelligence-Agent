#!/usr/bin/env python
"""
Real-time streaming evaluation (causal, no shuffling)
- Detection latency
- Alerts/hour, FP/min
- Rolling precision/recall/F1
- Throughput (events/sec)
- Inference latency

Plug your model by editing load_model() and device.
"""

import time
import math
import random
from typing import Iterator, Dict, Any, List, Callable, Tuple
from collections import deque

import numpy as np
import torch


class ScoreNormalizer:
    """EMA min-max normalizer for streaming scores."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.min_ema = None
        self.max_ema = None

    def __call__(self, score: float) -> float:
        s = float(score)
        if self.min_ema is None:
            self.min_ema = s
            self.max_ema = s
        else:
            self.min_ema = (1 - self.alpha) * self.min_ema + self.alpha * s
            self.max_ema = (1 - self.alpha) * self.max_ema + self.alpha * s
        denom = self.max_ema - self.min_ema + 1e-6
        norm = (s - self.min_ema) / denom
        return float(max(0.0, min(1.0, norm)))

# -------------------------------------------------
# 0) USER: adjust these
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NORMALIZER DISABLED: Model outputs calibrated probabilities [0,1]
# NORMALIZER = ScoreNormalizer(alpha=0.05)


def load_model():
    from models.classifier import ThreatModel

    model = ThreatModel(
        input_dims=(32, 64, 16),
        d_model=128
    )

    ckpt = torch.load(
        "outputs/supervised_epoch5.pt",
        map_location=DEVICE
    )

    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()

    print("[OK] Loaded trained ThreatModel")
    return model



def user_model_fn(model, window_t: Dict[str, torch.Tensor]) -> float:
    """
    Streaming inference function.
    Uses trained classifier probability directly.
    """
    with torch.no_grad():
        out = model(
            x_log=window_t["x_log"].to(DEVICE),
            x_text=window_t["x_net"].to(DEVICE),   # map net → text
            x_cve=window_t["x_ti"].to(DEVICE),     # map ti → cve
        )

    # ThreatModel returns a dict
    # out["score"] is already sigmoid-ed
    score = out["score"]

    if score.ndim > 0:
        score = score.mean()

    return float(score.item())




# -------------------------------------------------
# 1) Event stream (time-ordered, SOC-like)
# -------------------------------------------------
def stream_events(normal_iter: Iterator[Dict[str, Any]],
                  attack_iter: Iterator[Dict[str, Any]],
                  attack_rate = 0.05) -> Iterator[Dict[str, Any]]:
    """
    Yields events in chronological order.
    Each event: {ts, x_log, x_net, x_ti, is_attack, attack_id or None}
    """
    for ts, normal in enumerate(normal_iter):
        # maybe inject attack before the normal event at this ts
        if random.random() < attack_rate:
            atk = next(attack_iter, None)
            if atk:
                ev_atk = dict(atk)
                ev_atk.update({"ts": ts, "is_attack": 1})
                yield ev_atk
        ev = dict(normal)
        ev.update({"ts": ts, "is_attack": 0})
        ev.setdefault("attack_id", None)
        yield ev


# -------------------------------------------------
# 2) Sliding buffer (causal)
# -------------------------------------------------
class SlidingBuffer:
    def __init__(self, T: int):
        self.T = T
        self.log = deque(maxlen=T)
        self.net = deque(maxlen=T)
        self.ti = deque(maxlen=T)

    def push(self, ev: Dict[str, Any]):
        self.log.append(ev["x_log"])
        self.net.append(ev["x_net"])
        self.ti.append(ev["x_ti"])

    def full(self) -> bool:
        return len(self.log) == self.T

    def window_np(self) -> Dict[str, np.ndarray]:
        return {
            "x_log": np.stack(self.log),
            "x_net": np.stack(self.net),
            "x_ti": np.stack(self.ti),
        }


# -------------------------------------------------
# 3) Live metrics (incremental)
# -------------------------------------------------
class LiveMetrics:
    def __init__(self,
                 base_threshold: float,
                 horizon_sec: int = 300,
                 k: float = 2.5,
                 consecutive_required: int = 2,
                 min_history: int = 30):
        self.th = base_threshold
        self.horizon = horizon_sec
        self.k = k
        self.consecutive_required = consecutive_required
        self.min_history = min_history
        self.records = deque()  # (ts, y, yhat)
        self.attacks = {}       # attack_id -> {start, det or None}
        self.alerts = []        # (ts, score, y, dyn_th)
        self.infer_times = []
        self.event_count = 0
        self.start_wall = time.time()
        self.score_history = deque()
        self.streak = 0

    def register_attack(self, attack_id, ts):
        self.attacks[attack_id] = {"start": ts, "det": None}

    def update(self, ts: float, y_true: int, score: float, attack_id=None):
        # Only learn background distribution from normal traffic
        if y_true == 0:
            self.score_history.append((ts, score))
        while self.score_history and ts - self.score_history[0][0] > self.horizon:
            self.score_history.popleft()

        scores = [s for _, s in self.score_history]
        if len(scores) >= self.min_history:
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            dyn_th = mean + self.k * std
        else:
            dyn_th = self.th

        if score > dyn_th:
            self.streak += 1
        else:
            self.streak = 0

        yhat = int(self.streak >= self.consecutive_required)
        self.records.append((ts, y_true, yhat))
        self.alerts.append((ts, score, y_true, dyn_th))
        if attack_id and y_true == 1 and attack_id in self.attacks:
            if self.attacks[attack_id]["det"] is None and yhat == 1:
                self.attacks[attack_id]["det"] = ts
        self.event_count += 1
        # trim horizon
        while self.records and ts - self.records[0][0] > self.horizon:
            self.records.popleft()

    def log_infer_time(self, dt: float):
        self.infer_times.append(dt)

    def rollup(self, now_ts: float) -> Dict[str, float]:
        if not self.records:
            return {}
        tp = fp = fn = tn = 0
        for _, y, yhat in self.records:
            if y == 1 and yhat == 1:
                tp += 1
            elif y == 0 and yhat == 1:
                fp += 1
            elif y == 1 and yhat == 0:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        duration_min = max(1e-6, self.horizon / 60)
        fpr_per_min = fp / duration_min
        alerts_per_hour = (tp + fp) / (duration_min / 60)

        lats = []
        for a in self.attacks.values():
            if a.get("det") is not None:
                lats.append(a["det"] - a["start"])
        avg_lat = sum(lats) / len(lats) if lats else math.nan

        wall_elapsed = time.time() - self.start_wall
        throughput = self.event_count / max(1e-6, wall_elapsed)
        infer_lat = float(np.mean(self.infer_times)) if self.infer_times else math.nan

        return {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "fpr_per_min": fpr_per_min,
            "alerts_per_hour": alerts_per_hour,
            "avg_detection_latency": avg_lat,
            "events_per_sec": throughput,
            "infer_latency_sec": infer_lat,
        }


# -------------------------------------------------
# 4) Main streaming loop
# -------------------------------------------------
def run_stream(event_stream: Iterator[Dict[str, Any]],
               model_fn: Callable[[Dict[str, torch.Tensor]], float],
               T: int = 128,
               infer_every: int = 1,
               threshold: float = 0.5,
               horizon_sec: int = 300,
               rollup_interval_events: int = 200,
               k: float = 2.5,
               consecutive_required: int = 3,
               min_history: int = 30) -> Dict[str, float]:
    buf = SlidingBuffer(T)
    metrics = LiveMetrics(threshold, horizon_sec, k, consecutive_required, min_history)
    last_rollup_ev = 0

    for ev_idx, ev in enumerate(event_stream, 1):
        buf.push(ev)
        if not buf.full():
            continue  # warmup
        if ev_idx % infer_every != 0:
            continue

        # Prepare tensors (batch=1)
        wnp = buf.window_np()
        window_t = {
            "x_log": torch.from_numpy(wnp["x_log"]).unsqueeze(0).float(),
            "x_net": torch.from_numpy(wnp["x_net"]).unsqueeze(0).float(),
            "x_ti": torch.from_numpy(wnp["x_ti"]).unsqueeze(0).float(),
        }

        t0 = time.time()
        score = model_fn(window_t)
        metrics.log_infer_time(time.time() - t0)

        # track attack start if labeled
        if ev.get("attack_id") and ev.get("is_attack") == 1:
            metrics.register_attack(ev["attack_id"], ev["ts"])

        metrics.update(ev["ts"], ev["is_attack"], score, ev.get("attack_id"))

        if ev_idx - last_rollup_ev >= rollup_interval_events:
            r = metrics.rollup(ev["ts"])
            if r:
                print(f"[ts={ev['ts']}] F1={r['f1']:.3f} Prec={r['precision']:.3f} Rec={r['recall']:.3f} "
                      f"FP/min={r['fpr_per_min']:.3f} Alerts/hr={r['alerts_per_hour']:.2f} "
                      f"AvgLat={r['avg_detection_latency']:.2f}s EPS={r['events_per_sec']:.1f} "
                      f"InferLat={r['infer_latency_sec']:.4f}s")
            last_rollup_ev = ev_idx

    return metrics.rollup(ev["ts"])


# -------------------------------------------------
# 5) Threshold sweep on validation stream
# -------------------------------------------------
def sweep_thresholds(k_values: List[float],
                     val_stream_factory: Callable[[], Iterator[Dict[str, Any]]],
                     model_fn: Callable[[Dict[str, torch.Tensor]], float],
                     T: int = 128,
                     infer_every: int = 1,
                     horizon_sec: int = 300,
                     base_threshold: float = 0.5,
                     consecutive_required: int = 3,
                     min_history: int = 30) -> Tuple[float, Dict[str, float]]:
    best = None
    for k_val in k_values:
        final = run_stream(
            val_stream_factory(),
            model_fn,
            T,
            infer_every,
            threshold=base_threshold,
            horizon_sec=horizon_sec,
            rollup_interval_events=500,
            k=k_val,
            consecutive_required=consecutive_required,
            min_history=min_history,
        )
        if not final:
            continue
        if best is None or final["f1"] > best[1]["f1"]:
            best = (k_val, final)
    return best


# -------------------------------------------------
# 6) Example synthetic generators (replace with your data loaders)
# -------------------------------------------------
def make_normal_iter(n=500, d_log=32, d_net=64, d_ti=16):
    for _ in range(n):
        yield {
            "x_log": np.random.randn(d_log).astype(np.float32),
            "x_net": np.random.randn(d_net).astype(np.float32),
            "x_ti": np.random.randn(d_ti).astype(np.float32),
        }


def make_attack_iter(n=50, burst_len=(10, 30), d_log=32, d_net=64, d_ti=16):
    for i in range(n):
        L = np.random.randint(burst_len[0], burst_len[1])
        base = np.random.randn(d_log + d_net + d_ti)

        for _ in range(L):
            yield {
                "x_log": (np.random.randn(d_log) + 2.5).astype(np.float32),
                "x_net": (np.random.randn(d_net) + 2.5).astype(np.float32),
                "x_ti":  (np.random.randn(d_ti) + 2.5).astype(np.float32),
                "attack_id": f"atk_{i}",
            }



# -------------------------------------------------
# 7) Main: threshold sweep + final eval
# -------------------------------------------------
if __name__ == "__main__":

    # 1️⃣ Load trained model
    model = load_model()

    # 2️⃣ QUICK SANITY CHECK (VERY IMPORTANT)
    dummy = {
        "x_log": torch.randn(1, 128, 32).to(DEVICE),
        "x_text": torch.randn(1, 128, 64).to(DEVICE),
        "x_cve":  torch.randn(1, 128, 16).to(DEVICE),
    }

    with torch.no_grad():
        out = model(**dummy)

    print("MODEL OUTPUT:", out)

    # 3️⃣ Bind model to inference function
    model_fn = lambda window_t: user_model_fn(model, window_t)

    # 4️⃣ Continue normal streaming evaluation

    def val_stream_factory():
        return stream_events(make_normal_iter(), make_attack_iter(), attack_rate = 0.05)

    # 1) Sweep adaptive k (std multiplier) on validation stream
    k_candidates = [2.0, 2.25, 2.5, 2.75, 3.0]
    best = sweep_thresholds(
    k_candidates,
    val_stream_factory,
    model_fn,
    consecutive_required=1,   # 🔥 detect immediately
    min_history=10            # 🔥 faster adaptive threshold
)

    if best is None:
        raise RuntimeError("No best threshold found")
    best_k, best_metrics = best
    print("\nBest k (std multiplier):", best_k)
    print("Best metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v}")

    # 2) Final evaluation stream with fixed k
    final_stream = stream_events(make_normal_iter(), make_attack_iter(), attack_rate = 0.05)
    final = run_stream(final_stream, model_fn, k=best_k)

    print("\n--------------------------------------------------")
    print("FINAL STREAMING RESULTS")
    print("--------------------------------------------------")
    for k, v in final.items():
        print(f"{k}: {v}")
