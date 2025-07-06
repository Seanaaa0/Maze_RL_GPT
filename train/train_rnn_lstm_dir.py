import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# === 超參數 ===
NUM_ENVS = 1
OBS_HISTORY = 7
MAX_STEPS = 300
EPISODES = 50
GAMMA = 0.99
LR = 1e-3
EPSILON = 0.5

# === 建立環境與模型參數 ===
envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs_shape = (1, 11, 11)
n_actions = envs.single_action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 模型定義 ===


class RNN_QNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(16 * h * w + 4, 128, batch_first=True)
        self.fc = nn.Linear(128, n_actions)

    def forward(self, x_seq, dir_seq):
        B, T, C, H, W = x_seq.shape
        x_seq = x_seq.view(B * T, C, H, W)
        x_seq = self.conv(x_seq).view(B, T, -1)
        x_combined = torch.cat([x_seq, dir_seq], dim=-1)
        lstm_out, _ = self.lstm(x_combined)
        return self.fc(lstm_out[:, -1])

# === 資料處理 ===


def preprocess(obs):
    view = torch.tensor(obs["view"], dtype=torch.float32,
                        device=device)  # 不要 unsqueeze(0)
    view = view.unsqueeze(0)  # 加一個 channel 維度 -> [1, 11, 11]

    dir_idx = obs["dir"]
    return view, torch.tensor(dir_idx, dtype=torch.long, device=device)


# === 初始化 ===
model = RNN_QNet(obs_shape, n_actions).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
episode_rewards = []

# === 訓練迴圈 ===
for episode in range(1, EPISODES + 1):
    obs, _ = envs.reset(seed=42 + episode)
    view, direction = preprocess(obs)
    dir_onehot = F.one_hot(direction.unsqueeze(0), num_classes=4).float()

    obs_seq = [view.squeeze(0) for _ in range(OBS_HISTORY)]
    dir_seq = [dir_onehot.squeeze(0) for _ in range(OBS_HISTORY)]

    total_reward = 0
    step = 0

    while step < MAX_STEPS:
        seq_tensor = torch.stack(obs_seq).unsqueeze(0)  # [1, T, 1, 11, 11]
        dir_tensor = torch.stack(dir_seq).unsqueeze(0).squeeze(2)
        # print(
        #     f"seq_tensor: {seq_tensor.shape}, dir_tensor: {dir_tensor.shape}")

        with torch.no_grad():
            q_vals = model(seq_tensor, dir_tensor)
            if np.random.rand() < EPSILON:
                action = envs.single_action_space.sample()
            else:
                action = q_vals.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = envs.step([action])
        next_view, next_dir = preprocess(next_obs)
        next_dir_onehot = F.one_hot(
            next_dir.unsqueeze(0), num_classes=4).float()

        if reward[0] == 0:
            reward[0] = -0.1
        elif reward[0] == 1:
            reward[0] = 10.0

        total_reward += reward[0]

        obs_seq = obs_seq[1:] + [next_view.squeeze(0)]
        dir_seq = dir_seq[1:] + [next_dir_onehot.squeeze(0)]

        target = torch.tensor(reward[0], dtype=torch.float32, device=device)
        if not (terminated[0] or truncated[0]):
            with torch.no_grad():
                next_seq_tensor = torch.stack(obs_seq).unsqueeze(0)

                next_dir_tensor = torch.stack(
                    [d.squeeze(0) for d in dir_seq]).unsqueeze(0)  # ➜ [1, 7, 4]

                next_q_vals = model(next_seq_tensor, next_dir_tensor)
                target += GAMMA * next_q_vals.max()

        pred = model(seq_tensor, dir_tensor)[0, action]
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if terminated[0] or truncated[0]:
            break
        step += 1

    episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards[-10:])
    print(
        f"[EP {episode}] Total Reward: {total_reward:.2f} | Avg(10) = {avg_reward:.3f}", flush=True)

# === 儲存模型 ===
os.makedirs("C:/Users/seana/maze/outputs/pth", exist_ok=True)
torch.save(model.state_dict(),
           "C:/Users/seana/maze/outputs/pth/rnn_lstm_model2_dir.pth")
print("\u2705 Model saved: rnn_lstm_model2_dir.pth")
