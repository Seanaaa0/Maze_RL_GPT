import json
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
EPISODES = 100
EARLY_STOP_GOAL_REACHED = 3
SIZE = 200
SEED = 999

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
os.makedirs(os.path.dirname(gt_path), exist_ok=True)
np.save(gt_path, gt)
print(f"\U0001F5FA\uFE0F 完整地圖已儲存：{gt_path}")

n_actions = envs.single_action_space.n
results = []
goal_reached_count = 0

# === 探索迴圈 ===
for episode in range(EPISODES):
    obs, _ = envs.reset(seed=SEED)
    internal_map = np.full((SIZE, SIZE), -1, dtype=np.int8)
    pos = tuple(obs["position"][0])
    internal_map[pos] = 2
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
                internal_map[wall_pos] = 0
        else:
            internal_map[next_pos] = 2
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

    if goal_reached_count >= EARLY_STOP_GOAL_REACHED:
        print("\u2705 已成功抵達 3 次目標，結束訓練")
        break

# === 儲存探索紀錄為 JSONL ===
save_json_path = "C:/Users/seana/maze/outputs/mem2/maze4_2.jsonl"
os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

with open(save_json_path, "w", encoding="utf-8") as f:
    for i, record in enumerate(results):
        json_record = {
            "episode": i + 1,
            "maze_id": str(record["maze_id"]),
            "start_pos": [int(v) for v in record["start_pos"]],
            "goal_pos": [int(v) for v in record["goal_pos"]],
            "explored_map": record["explored_map"].astype(int).tolist(),
            "known_walls": record["known_walls"].astype(int).tolist(),
            "trajectory": [[int(x), int(y)] for x, y in record["trajectory"]]
        }
        f.write(json.dumps(json_record) + "\n")

print(f"📄 成功儲存 {len(results)} 筆探索紀錄至 JSONL：{save_json_path}")
