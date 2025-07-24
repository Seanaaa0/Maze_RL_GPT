import json
import os
import numpy as np
import importlib.util
from gymnasium.vector import SyncVectorEnv

# === 載入 maze6_multi_goal 環境 ===
env_path = "C:/Users/seana/maze/env_partial/maze6_multigoals.py"
spec = importlib.util.spec_from_file_location("maze6_multi_goal", env_path)
maze6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze6)

SIZE = 75
SEED = 42598
MAX_STEPS = 50000
REQUIRED_SUCCESS = 2


def make_env():
    return lambda: maze6.Maze6MultiGoalEnv(render_mode=None, size=SIZE, num_goals=3)


envs = SyncVectorEnv([make_env()])
obs, _ = envs.reset(seed=SEED)

# === 儲存地圖資訊 ===
full_env = maze6.Maze6MultiGoalEnv(size=SIZE, num_goals=3)
full_env.reset(seed=SEED)

gt = {
    "seed": SEED,
    "size": SIZE,
    "start_pos": list(map(int, full_env.agent_pos)),
    "goal_list": [list(map(int, g)) for g in full_env.goal_list],
    "wall_map": full_env.grid.copy()
}

gt_path = f"C:/Users/seana/maze/outputs/mem_trap/gt_maze6_{SIZE}x{SIZE}_SEED{SEED}.npy"
os.makedirs(os.path.dirname(gt_path), exist_ok=True)
np.save(gt_path, gt)
print(f"\U0001F5FA\ufe0f 地圖儲存於：{gt_path}")

# === 探索直到成功次數達標 ===
results = []
success_count = 0
attempts = 0

while success_count < REQUIRED_SUCCESS:
    obs, _ = envs.reset(seed=SEED)
    internal_map = np.full((SIZE, SIZE), -1, dtype=np.int8)
    pos = tuple(obs["position"][0])
    internal_map[pos] = 2
    trajectory = [pos]
    collected_goals = []

    for step in range(MAX_STEPS):
        action = envs.single_action_space.sample()
        next_obs, reward, terminated, truncated, info = envs.step([action])
        next_pos = tuple(next_obs["position"][0])
        next_view = next_obs["view"][0]

        if pos == next_pos:
            dx, dy = maze6.MOVE[action]
            wall_pos = (pos[0] + dx, pos[1] + dy)
            if 0 <= wall_pos[0] < SIZE and 0 <= wall_pos[1] < SIZE:
                internal_map[wall_pos] = 0
        else:
            internal_map[next_pos] = 2
            pos = next_pos
            trajectory.append(next_pos)

        if terminated[0]:
            success_count += 1
            print(f"\U0001F3AF 成功收集全部 goals！（目前成功次數：{success_count}）")
            break

        if truncated[0]:
            break

    results.append({
        "explored_map": (internal_map == 2).astype(np.uint8).tolist(),
        "known_walls": (internal_map == 0).astype(np.uint8).tolist(),
        "start_pos": list(map(int, full_env.agent_pos)),
        "goal_list": [list(map(int, g)) for g in full_env.goal_list],
        "collected_goals": [list(map(int, g)) for g in full_env.collected_goals],
        "trajectory": [[int(x), int(y)] for x, y in trajectory],
        "maze_id": f"maze6_ep{attempts+1}"
    })
    attempts += 1

# === 儲存為 JSONL ===
save_path = "C:/Users/seana/maze/outputs/mem_trap/maze6_{SIZE}_2.jsonl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w", encoding="utf-8") as f:
    for r in results:
        json.dump(r, f)
        f.write("\n")

print(f"\U0001F4BE 已儲存探索紀錄，共 {len(results)} 筆：{save_path}")
