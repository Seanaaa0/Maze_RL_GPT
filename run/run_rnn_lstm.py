import torch
import torch.nn as nn
import numpy as np
import importlib.util
from gymnasium.vector import SyncVectorEnv

# === 載入環境 ===
env_path = "C:/Users/seana/maze/env_partial/maze1_prim_pomdp.py"
spec = importlib.util.spec_from_file_location("maze1_prim_pomdp", env_path)
maze1_prim_pomdp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze1_prim_pomdp)


def make_env():
    return lambda: maze1_prim_pomdp.Maze1PrimPOMDPEnv(render_mode=None)


NUM_ENVS = 1
OBS_HISTORY = 7
MAX_STEPS = 300

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
            x_seq = x_seq.squeeze(2)
        elif x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(0)
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


# === 載入模型 ===
model_path = "C:/Users/seana/maze/outputs/pth/rnn_lstm_model2.pth"
model = RNN_QNet(obs_shape, n_actions).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("✅ Model loaded and ready.")

# === 開始測試 ===
raw_obs, _ = envs.reset(seed=999)
obs = raw_obs
obs_seq = [preprocess(obs) for _ in range(OBS_HISTORY)]

for step in range(MAX_STEPS):
    seq_tensor = torch.stack(obs_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        q_vals = model(seq_tensor)
        action = q_vals.argmax(dim=1).item()

    obs, reward, terminated, truncated, _ = envs.step([action])
    pos = envs.envs[0].current_pos
    print(
        f"[STEP {step}] Pos: {pos} | Action: {action} | Reward: {reward[0]:.4f}", flush=True)

    obs_seq = obs_seq[1:] + [preprocess(obs)]
    if terminated[0] or truncated[0]:
        print("✅ Terminated (goal reached or max steps).")
        break

envs.close()
