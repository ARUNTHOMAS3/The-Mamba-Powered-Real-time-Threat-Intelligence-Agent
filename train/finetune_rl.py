# train/finetune_rl.py
"""
Skeleton RL fine-tuning script. This uses stable-baselines3 PPO on a custom gym env.
The env should wrap the model and emit rewards based on whether a high TCI precedes a true attack.
This file outlines usage; implementing a high-fidelity env requires SOC-like simulator, which is domain work.
"""
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import numpy as np
import torch
from models.classifier import ThreatModel
from utils.utils import load_config, set_seed

class SimpleSOCEnv(gym.Env):
    """
    Minimal env: observation is concatenated vector for last step,
    action is threshold in [0,1] (we discretize for PPO).
    Reward: +1 if threshold>0.5 and next step has attack label; -1 for false positive.
    This is simplified for demonstration.
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.seq_len = dataset.seq_len
        # observation: flattened last-step features
        obs_dim = 32+64+16
        self.observation_space = spaces.Box(low=-10, high=10, shape=(obs_dim,), dtype=np.float32)
        # action: continuous 1-d threshold
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.idx = 0

    def reset(self):
        self.idx = 0
        item = self.dataset[self.idx]
        last = np.concatenate([item["x_log"][-1].numpy(), item["x_text"][-1].numpy(), item["x_cve"][-1].numpy()])
        return last

    def step(self, action):
        # evaluate reward against label of current sample
        item = self.dataset[self.idx]
        label = item["label"].item()
        thresh = float(action[0])
        reward = 1.0 if (thresh > 0.5 and label==1) else (-1.0 if (thresh > 0.5 and label==0) else 0.0)
        self.idx = (self.idx + 1) % len(self.dataset)
        done = (self.idx==0)
        next_item = self.dataset[self.idx]
        next_obs = np.concatenate([next_item["x_log"][-1].numpy(), next_item["x_text"][-1].numpy(), next_item["x_cve"][-1].numpy()])
        return next_obs.astype(np.float32), reward, done, {}

def main(config):
    cfg = load_config(config)
    set_seed(42)
    dataset = __import__("datasets.multimodal_dataset", fromlist=['MultimodalDataset']).multimodal_dataset.MultimodalDataset("data/processed/synth_small.json", seq_len=cfg["data"]["seq_len"], create_synthetic=True, n_samples=500)
    env = DummyVecEnv([lambda: SimpleSOCEnv(dataset)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000)
    model.save("outputs/ppo_threshold_model")

if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv)>1 else "configs/default.yaml"
    main(cfg_path)
