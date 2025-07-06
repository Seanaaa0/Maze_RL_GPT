
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import importlib.util
import os
from gymnasium.vector import SyncVectorEnv

# === 環境設定 ===
env_path = "C:/Users/seana/maze/env_partial/maze1_prim_pomdp.py"
spec = importlib.util.spec_from_file_location("maze1_prim_pomdp", env_path)
maze1_prim_pomdp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze1_prim_pomdp)

# 產生多個環境
NUM_ENVS = 1
MAX_STEPS_PER_EPISODE = 1000
WARMUP = 1000


def make_env():
    return lambda: maze1_prim_pomdp.Maze1PrimPOMDPEnv(render_mode=None)


envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs_shape = (1, 11, 11)

# === 超參數 ===
EPISODES = 3000
GAMMA = 0.99
LR = 1e-3
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_RATE = 200

epsilon = EPS_START
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === 工具函數 ===


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=device)


def preprocess(obs):
    if isinstance(obs, dict):
        view = obs["view"]
        return to_tensor(view).unsqueeze(0).unsqueeze(0)  # [1, 1, 11, 11]

    elif isinstance(obs, (list, tuple)) and isinstance(obs[0], dict):
        views = np.stack([o["view"] for o in obs])  # [B, 11, 11]
        return to_tensor(views).unsqueeze(1)        # [B, 1, 11, 11]

    else:
        raise ValueError(
            f"❌ preprocess received unsupported obs type: {type(obs)}")


# === QDN 模型 ===


class QDN(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * h * w, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        return self.fc(x)

# === Replay Buffer ===


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        for i in range(len(s)):
            if r[i] > 0 or done[i]:
                self.buffer.append((s[i], a[i], r[i], s_[i], done[i]))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, done = map(np.array, zip(*batch))
        return s, a, r, s_, done

    def __len__(self):
        return len(self.buffer)


# === 初始化 ===
n_actions = envs.single_action_space.n
policy_net = QDN(obs_shape, n_actions).to(device)
target_net = QDN(obs_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

raw_obs, _ = envs.reset(seed=None)
obs = raw_obs


for episode in range(1, EPISODES + 1):
    total_rewards = np.zeros(NUM_ENVS)
    dones = [False] * NUM_ENVS
    step_count = 0

    while not all(dones) and step_count < MAX_STEPS_PER_EPISODE:
        print(
            f"[DEBUG] Step: {step_count} | Eps: {epsilon:.3f} | Done: {sum(dones)}/{NUM_ENVS}", flush=True)
        obs_tensor = preprocess(obs)
        print("obs_tensor.shape =", obs_tensor.shape, flush=True)  # Debug
        with torch.no_grad():
            q_vals = policy_net(obs_tensor)
        greedy_actions = q_vals.argmax(dim=1).cpu().numpy()
        random_actions = np.random.randint(0, n_actions, size=NUM_ENVS)
        actions = np.where(np.random.rand(NUM_ENVS) < epsilon,
                           random_actions, greedy_actions)

        next_raw_obs, rewards, terminated, truncated, _ = envs.step(actions)
        next_obs = next_raw_obs if NUM_ENVS == 1 else list(next_raw_obs)
        done_flags = np.logical_or(terminated, truncated)

        memory.push([obs], actions, rewards, [next_obs], done_flags)

        obs = next_obs
        total_rewards += rewards
        dones = np.logical_or(dones, done_flags)
        step_count += 1

        if len(memory) >= max(BATCH_SIZE, WARMUP):
            s, a, r, s_, d = memory.sample(BATCH_SIZE)
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(1).to(device)
            a = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device)
            r = torch.tensor(r, dtype=torch.float32).to(device)
            s_ = torch.tensor(s_, dtype=torch.float32).unsqueeze(1).to(device)
            d = torch.tensor(d, dtype=torch.bool).to(device)

            q_vals = policy_net(s).gather(1, a).squeeze()
            with torch.no_grad():
                max_q = target_net(s_).max(1)[0]
                target = r + GAMMA * max_q * (~d)

            loss = F.mse_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(0.05, epsilon * 0.995)
    print(f"Ep {episode * NUM_ENVS} steps | AvgReward: {total_rewards.mean():.2f} | Eps: {epsilon:.3f} | Steps: {step_count}", flush=True)

envs.close()
