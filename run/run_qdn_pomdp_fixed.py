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

# === 超參數 ===
MAX_STEPS = 100
seed = 42
EPS_START = 1.0
epsilon = EPS_START  # 不 decay 先探索

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === 工具 ===


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=device)


def preprocess(obs):
    view = obs["view"]
    return to_tensor(view).unsqueeze(0)  # shape: [1, 1, 11, 11]

# === 模型 ===


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


# === 初始化 ===
n_actions = envs.single_action_space.n
policy_net = QDN(obs_shape, n_actions).to(device)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

raw_obs, _ = envs.reset(seed=seed)
obs = raw_obs

# === 記憶軌跡 ===
visited_set = set()
step_count = 0

# === 執行一次（greedy） ===
while step_count < MAX_STEPS:
    obs_tensor = preprocess(obs)
    with torch.no_grad():
        q_vals = policy_net(obs_tensor)
        action = q_vals.argmax(dim=1).item()

    obs, reward, terminated, truncated, _ = envs.step([action])
    done = terminated[0] or truncated[0]

    # 印當前位置與是否走過
    pos = envs.envs[0].current_pos
    first_time = pos not in visited_set
    visited_set.add(pos)

    print(f"[STEP {step_count}] Action: {action} | Pos: {pos} | {'🆕' if first_time else '↩️ visited'} | Reward: {reward[0]:.4f}", flush=True)

    step_count += 1
    if done:
        print("✅ Reached goal or terminated")
        break

envs.close()
