import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
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
SEQ_LEN = 7

# === 模型 ===


class RNN_QNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden_size=128):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.conv_output_dim = 16 * h * w
        self.lstm = nn.LSTM(self.conv_output_dim,
                            hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_actions)

    def forward(self, x_seq):  # x_seq: (B, T, C, H, W)
        B, T, C, H, W = x_seq.size()
        x = x_seq.view(B * T, C, H, W)        # (B*T, C, H, W)
        x = self.conv(x)                      # (B*T, F)
        x = x.view(B, T, -1)                  # (B, T, F)
        _, (h_n, _) = self.lstm(x)            # h_n: (1, B, H)
        return self.fc(h_n.squeeze(0))        # (B, n_actions)


# === 工具 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=device)


def preprocess(obs):
    view = obs["view"]
    return to_tensor(view).unsqueeze(0)  # shape: (1, 1, 11, 11)


# === 初始化 ===
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

n_actions = envs.single_action_space.n
policy_net = RNN_QNetwork(obs_shape, n_actions).to(device)

raw_obs, _ = envs.reset(seed=seed)
obs = raw_obs
obs_sequence = []

step_count = 0
MAX_STEPS = 100

# === 探索執行 ===
while step_count < MAX_STEPS:
    obs_tensor = preprocess(obs)
    obs_sequence.append(obs_tensor.squeeze(0))  # shape: (1, 11, 11)
    if len(obs_sequence) > SEQ_LEN:
        obs_sequence.pop(0)

    if len(obs_sequence) == SEQ_LEN:
        sequence = torch.stack(obs_sequence, dim=0)      # (T=7, 1, 11, 11)
        seq_tensor = sequence.unsqueeze(0).to(device)    # (1, 7, 1, 11, 11)
        with torch.no_grad():
            q_vals = policy_net(seq_tensor)
            action = q_vals.argmax(dim=1).item()
    else:
        action = np.random.randint(n_actions)  # 尚未滿足序列長度，先隨機行動

    obs, reward, terminated, truncated, _ = envs.step([action])
    done = terminated[0] or truncated[0]

    pos = envs.envs[0].current_pos
    print(
        f"[STEP {step_count}] Action: {action} | Pos: {pos} | Reward: {reward[0]:.4f}", flush=True)

    step_count += 1
    if done:
        print("✅ Reached goal or terminated")
        break


envs.close()
