import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import importlib.util
from gymnasium.vector import SyncVectorEnv
import os
from datetime import datetime

# === 載入環境 ===
env_path = "C:/Users/seana/maze/env_partial/maze1_prim_pomdp.py"
spec = importlib.util.spec_from_file_location("maze1_prim_pomdp", env_path)
maze1_prim_pomdp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze1_prim_pomdp)


def make_env():
    return lambda: maze1_prim_pomdp.Maze1PrimPOMDPEnv(render_mode=None)


NUM_ENVS = 1
EPISODES = 1000
MAX_STEPS = 300

BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE = 10
OBS_HISTORY = 7
WARMUP = 100  # 確保有足夠經驗後才開始訓練

# === 建立環境 ===
envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs_shape = (1, 11, 11)
n_actions = envs.single_action_space.n

# === 模型 ===


class RNN_QNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(16 * h * w, 128, batch_first=True)
        self.fc = nn.Linear(128, n_actions)

    def forward(self, x_seq):
        if x_seq.dim() == 6:
            x_seq = x_seq.squeeze(2)  # 修正多出來的 C 維度
        elif x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(0)  # [T, C, H, W] → [1, T, C, H, W]
        elif x_seq.dim() != 5:
            raise ValueError(
                f"❌ Expected input with 4~6 dims, got {x_seq.shape}")

        B, T, C, H, W = x_seq.shape
        x_seq = x_seq.view(B * T, C, H, W)
        x_seq = self.conv(x_seq)
        x_seq = x_seq.view(B, T, -1)
        lstm_out, _ = self.lstm(x_seq)
        return self.fc(lstm_out[:, -1])


# === 工具 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_tensor(x): return torch.tensor(x, dtype=torch.float32, device=device)
def preprocess(obs): return to_tensor(obs["view"]).unsqueeze(0)

# === 記憶庫 ===


class SeqReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs_seq, action, reward, next_obs_seq, done):
        self.buffer.append((obs_seq, action, reward, next_obs_seq, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_seq, actions, rewards, next_obs_seq, dones = zip(*batch)
        return (
            torch.stack(obs_seq).to(device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_obs_seq).to(device),
            torch.tensor([bool(d) for d in dones],
                         dtype=torch.bool, device=device)

        )

    def __len__(self): return len(self.buffer)


# === 初始化 ===
policy_net = RNN_QNet(obs_shape, n_actions).to(device)
target_net = RNN_QNet(obs_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = SeqReplayBuffer(10000)

print(f"✅ Using device: {device}")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# === 新增 early stopping 超參數 ===
REWARD_THRESHOLD = 0.8
PATIENCE = 10
recent_rewards = deque(maxlen=PATIENCE)

# === 訓練迴圈 ===
for episode in range(1, EPISODES + 1):
    raw_obs, _ = envs.reset(seed=42 + episode)
    obs = raw_obs
    obs_seq = []

    # 探索式初始化觀測序列
    for i in range(OBS_HISTORY):
        obs_seq.append(preprocess(obs))
        action = i % n_actions  # 嘗試不同動作
        obs, _, terminated, truncated, _ = envs.step([action])
        if terminated[0] or truncated[0]:
            break  # 避免提前結束後繼續亂跑

    # 若不足 OBS_HISTORY，補足（必要時用最後一個畫面補齊）
    while len(obs_seq) < OBS_HISTORY:
        obs_seq.append(obs_seq[-1].clone())

    total_reward = 0
    step = 0

    while step < MAX_STEPS:
        seq_tensor = torch.stack(obs_seq).unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals = policy_net(seq_tensor)
            action = q_vals.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = envs.step([action])
        done = terminated[0] or truncated[0]

        next_obs_tensor = preprocess(next_obs)
        next_obs_seq = obs_seq[1:] + [next_obs_tensor]
        assert len(
            next_obs_seq) == OBS_HISTORY, f"❌ next_obs_seq 長度錯誤：{len(next_obs_seq)}"

        memory.push(torch.stack(obs_seq), action,
                    reward[0], torch.stack(next_obs_seq), done)

        obs_seq = next_obs_seq
        total_reward += reward[0]
        pos = envs.envs[0].current_pos
        print(
            f"[EP {episode} STEP {step}] Pos: {pos}, Action: {action}, Reward: {reward[0]:.4f}")

        step += 1

        if len(memory) >= max(BATCH_SIZE, WARMUP):
            s, a, r, s_, d = memory.sample(BATCH_SIZE)
            q_pred = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_next = target_net(s_).max(1)[0]
                target = r + GAMMA * q_next * (~d)
            loss = F.mse_loss(q_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    recent_rewards.append(total_reward)
    print(f"[EP {episode}] Total Reward: {total_reward:.2f} | Avg({PATIENCE}) = {np.mean(recent_rewards):.3f}")

    if len(recent_rewards) == PATIENCE and np.mean(recent_rewards) >= REWARD_THRESHOLD:
        print(
            f"✅ Early stopping triggered at episode {episode} (avg reward ≥ {REWARD_THRESHOLD})")
        break


# === 儲存模型 ===
os.makedirs("C:/Users/seana/maze/outputs/pth", exist_ok=True)
torch.save(policy_net.state_dict(),
           "C:/Users/seana/maze/outputs/pth/rnn_lstm_model2.pth")
print("\u2705 模型已儲存為 rnn_lstm_model2.pth")

envs.close()
