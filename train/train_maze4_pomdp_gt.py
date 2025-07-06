import os
import torch
import numpy as np
import importlib.util
from gymnasium.vector import SyncVectorEnv

# === 載入環境 ===
env_path = "C:/Users/seana/maze/env_partial/maze4_pomdp_gt.py"
spec = importlib.util.spec_from_file_location("maze4_pomdp_gt", env_path)
maze4_pomdp_gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze4_pomdp_gt)


def make_env():
    return lambda: maze4_pomdp_gt.Maze4POMDPGTEnv(render_mode=None)


# === 設定參數 ===
NUM_ENVS = 1
MAX_STEPS = 200000
EPISODES = 100  # 可以設大一點，反正條件滿 3 次成功才停
EARLY_STOP_GOAL_REACHED = 3
SIZE = 200
SEED = 912

# === 初始化環境 ===
envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs, _ = envs.reset(seed=SEED)

# === 儲存 ground truth 地圖（可選）===
full_env = maze4_pomdp_gt.Maze4POMDPGTEnv()
full_env.reset(seed=SEED)
gt = {
    "seed": SEED,
    "size": SIZE,
    "start_pos": full_env.agent_pos,
    "goal_pos": full_env.goal_pos,
    "wall_map": full_env.grid.copy()
}
gt_path = f"C:/Users/seana/maze/outputs/multi_{SIZE}x{SIZE}_SEED{SEED}.npy"
np.save(gt_path, gt)
print(f"🗺 完整地圖已儲存：{gt_path}")

n_actions = envs.single_action_space.n
results = []
goal_reached_count = 0

# === 探索迴圈 ===
for episode in range(EPISODES):
    obs, _ = envs.reset(seed=SEED)
    internal_map = np.full((SIZE, SIZE), -1, dtype=np.int8)
    pos = tuple(obs["position"][0])
    internal_map[pos] = 2  # 初始位置
    trajectory = [pos]

    for step in range(MAX_STEPS):
        action = envs.single_action_space.sample()
        next_obs, reward, terminated, truncated, info = envs.step([action])
        next_pos = tuple(next_obs["position"][0])
        next_view = next_obs["view"][0]

        if pos == next_pos:
            dx, dy = maze4_pomdp_gt.MOVE[action]
            wall_pos = (pos[0] + dx, pos[1] + dy)
            if 0 <= wall_pos[0] < SIZE and 0 <= wall_pos[1] < SIZE:
                internal_map[wall_pos] = 0  # 標記為牆
        else:
            internal_map[next_pos] = 2  # 標記為走過
            pos = next_pos
            trajectory.append(next_pos)

        if terminated[0]:
            goal_reached_count += 1
            print(
                f"\U0001F3AF 第 {episode+1} 次成功抵達目標！（總成功次數：{goal_reached_count}）")
            break

        if truncated[0]:
            break

    print(f"[EP{episode+1}] 探索完成（步數: {step+1}）")
    print("✔️ 探索區格數：", np.sum(internal_map == 2))
    print("✔️ 已知牆數：", np.sum(internal_map == 0))

    results.append({
        "explored_map": (internal_map == 2).astype(np.uint8),
        "known_walls": (internal_map == 0).astype(np.uint8),
        "start_pos": (1, 1),
        "goal_pos": envs.envs[0].goal,
        "maze_id": f"maze4_ep{episode+1}",
        "trajectory": trajectory
    })

    # ✅ 只在「成功達到三次」當下那一輪之後才停止
    if goal_reached_count >= EARLY_STOP_GOAL_REACHED:
        print("✅ 已成功抵達 3 次目標，結束訓練")
        break

# === 儲存探索紀錄 ===
save_path = "C:/Users/seana/maze/outputs/mem/maze4_train_n3.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, results)
print(f"\U0001F4C1 成功儲存 {len(results)} 筆探索紀錄於 {save_path}")
