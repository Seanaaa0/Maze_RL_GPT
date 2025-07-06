import torch
import numpy as np
import random
import importlib.util
from env_partial.qdn_gru import QDN_GRU
from gymnasium.vector import SyncVectorEnv

# === 環境載入 ===
env_path = "C:/Users/seana/maze/env_partial/maze1_prim_pomdp.py"
spec = importlib.util.spec_from_file_location("maze1_prim_pomdp", env_path)
maze1_prim_pomdp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze1_prim_pomdp)


def make_env():
    return lambda: maze1_prim_pomdp.Maze1PrimPOMDPEnv(render_mode=None)


# === 設定 ===
NUM_ENVS = 1
MAX_STEPS = 1000
HISTORY_LENGTH = 7
SEED = 42

envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs_shape = (1, 11, 11)
n_actions = envs.single_action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = QDN_GRU(obs_shape, n_actions).to(device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
obs, _ = envs.reset(seed=SEED)

# === 工具 ===


def preprocess(obs):
    view = obs["view"]
    # (1, 11, 11)
    return torch.tensor(view, dtype=torch.float32).unsqueeze(0).to(device)


history = []

# === 逐步執行 ===
for step in range(MAX_STEPS):
    obs_tensor = preprocess(obs).unsqueeze(0)  # (1, 1, 11, 11)
    history.append(obs_tensor)

    if len(history) < HISTORY_LENGTH:
        action = np.random.randint(n_actions)
    else:
        seq_tensor = torch.cat(
            history[-HISTORY_LENGTH:], dim=1)  # (1, T=7, 11, 11)
        seq_tensor = seq_tensor.unsqueeze(2)  # (1, 7, 1, 11, 11)
        with torch.no_grad():
            q_vals = policy_net(seq_tensor)
            action = q_vals.argmax(dim=1).item()

    obs, reward, terminated, truncated, _ = envs.step([action])
    pos = envs.envs[0].current_pos
    done = terminated[0] or truncated[0]

    print(
        f"[STEP {step}] Action: {action} | Pos: {pos} | Reward: {reward[0]:.4f}", flush=True)
    if done:
        print("✅ Reached goal or terminated.")
        break

envs.close()
