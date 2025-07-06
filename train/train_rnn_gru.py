import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import importlib.util
from gymnasium.vector import SyncVectorEnv

# === 環境設定 ===
env_path = "C:/Users/seana/maze/env_partial/maze1_prim_pomdp.py"
spec = importlib.util.spec_from_file_location("maze1_prim_pomdp", env_path)
maze1_prim_pomdp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze1_prim_pomdp)


def make_env():
    return lambda: maze1_prim_pomdp.Maze1PrimPOMDPEnv(render_mode=None)


NUM_ENVS = 1
envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs_shape = (1, 11, 11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === 超參數 ===
EPISODES = 2000
MAX_STEPS = 200
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
WARMUP = 500
SEQ_LEN = 7
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# === 工具 ===


def preprocess(obs):
    view = obs["view"]  # shape: (11, 11)
    # [1, 11, 11]
    return torch.tensor(view, dtype=torch.float32).unsqueeze(0).to(device)

# === 模型 ===


class GRUNet(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size=128):
        super().__init__()
        c, h, w = input_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.flat_dim = 16 * h * w
        self.gru = nn.GRU(input_size=self.flat_dim,
                          hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_actions)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.size()
        x_seq = x_seq.view(B * T, C, H, W)
        x_seq = self.cnn(x_seq)
        x_seq = x_seq.view(B, T, -1)
        out, _ = self.gru(x_seq)
        return self.fc(out[:, -1, :])  # 最後一個時間點輸出

# === Replay Buffer ===


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, seq, action, reward, next_seq, done):
        self.buffer.append((seq, action, reward, next_seq, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        seq, a, r, next_seq, done = zip(*batch)
        return (
            torch.stack(seq),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(r, dtype=torch.float32).to(device),
            torch.stack(next_seq),
            torch.tensor(done, dtype=torch.bool).to(device)
        )

    def __len__(self):
        return len(self.buffer)


# === 初始化 ===
n_actions = envs.single_action_space.n
policy_net = GRUNet(obs_shape, n_actions).to(device)
target_net = GRUNet(obs_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

epsilon = EPS_START

# === 主訓練迴圈 ===
for ep in range(1, EPISODES + 1):
    raw_obs, _ = envs.reset(seed=None)
    obs = raw_obs
    total_reward = 0
    step = 0

    history = deque(maxlen=SEQ_LEN)
    for _ in range(SEQ_LEN):
        history.append(preprocess(obs).unsqueeze(0))  # [1, 1, 11, 11]

    done = False
    while not done and step < MAX_STEPS:
        seq_tensor = torch.cat(list(history), dim=0).unsqueeze(
            0)  # [1, 7, 1, 11, 11]

        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            with torch.no_grad():
                q_vals = policy_net(seq_tensor)
                action = q_vals.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = envs.step([action])
        done = terminated[0] or truncated[0]
        next_history = history.copy()
        next_history.append(preprocess(next_obs).unsqueeze(0))

        memory.push(seq_tensor.squeeze(0), action,
                    reward[0], torch.cat(list(next_history), dim=0), done)

        obs = next_obs
        history = next_history
        total_reward += reward[0]
        step += 1

        # 訓練
        if len(memory) >= max(BATCH_SIZE, WARMUP):
            s, a, r, s_, d = memory.sample(BATCH_SIZE)
            q_vals = policy_net(s).gather(1, a).squeeze()
            with torch.no_grad():
                max_q = target_net(s_).max(1)[0]
                target = r + GAMMA * max_q * (~d)
            loss = F.mse_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    print(f"[EP {ep}] Total Reward: {total_reward:.2f} | Eps: {epsilon:.3f}")

envs.close()
