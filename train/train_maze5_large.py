import json
import os
import torch
import numpy as np
import importlib.util
from gymnasium.vector import SyncVectorEnv

# === 載入 maze5_trap 環境 ===
env_path = "C:/Users/seana/maze/env_partial/maze5_trap.py"
spec = importlib.util.spec_from_file_location("maze5_trap", env_path)
maze5_trap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze5_trap)


SIZE = 15
SEED = 659
MAX_STEPS = 500
REQUIRED_SUCCESS = 2


def make_env():
    return lambda: maze5_trap.Maze5TrapEnv(render_mode=None, size=SIZE)


# === 初始化環境 ===
envs = SyncVectorEnv([make_env()])
obs, _ = envs.reset(seed=SEED)

# === 儲存 ground truth 地圖 ===
full_env = maze5_trap.Maze5TrapEnv(size=SIZE)
full_env.reset(seed=SEED)
gt = {
    "seed": SEED,
    "size": SIZE,
    "start_pos": full_env.agent_pos,
    "goal_pos": full_env.goal_pos,
    "wall_map": full_env.grid.copy(),
    "trap_list": list(full_env.traps)
}
gt_path = f"C:/Users/seana/maze/outputs/trap_{SIZE}x{SIZE}_SEED{SEED}.npy"
os.makedirs(os.path.dirname(gt_path), exist_ok=True)
np.save(gt_path, gt)
print(f"\U0001F5FA\ufe0f 地圖儲存於：{gt_path}")

# === 探索三次直到至少兩次成功 ===
results = []
success_count = 0
attempts = 0
known_traps = set()

while success_count < REQUIRED_SUCCESS:
    obs, _ = envs.reset(seed=SEED)
    internal_map = np.full((SIZE, SIZE), -1, dtype=np.int8)
    pos = tuple(obs["position"][0])
    internal_map[pos] = 2
    trajectory = [pos]
    traps_seen = set()

    for step in range(MAX_STEPS):
        action = envs.single_action_space.sample()
        next_obs, reward, terminated, truncated, info = envs.step([action])
        next_pos = tuple(next_obs["position"][0])
        next_view = next_obs["view"][0]

        if pos == next_pos:
            dx, dy = maze5_trap.MOVE[action]
            wall_pos = (pos[0] + dx, pos[1] + dy)
            if 0 <= wall_pos[0] < SIZE and 0 <= wall_pos[1] < SIZE:
                internal_map[wall_pos] = 0
        else:
            internal_map[next_pos] = 2
            pos = next_pos
            trajectory.append(next_pos)

        if terminated[0]:
            if reward[0] == 1.0:
                success_count += 1
                print(f"\U0001F3AF 成功走到出口！（目前成功次數：{success_count}）")
            elif reward[0] == -1.0:
                traps_seen.add(pos)
                print(f"\u274C 踩到陷阱 {pos}，重新開始")
            break

        if truncated[0]:
            break

    results.append({
        "explored_map": (internal_map == 2).astype(np.uint8),
        "known_walls": (internal_map == 0).astype(np.uint8),
        "known_traps": [[x, y] for x, y in known_traps.union(traps_seen)],
        "start_pos": [1, 1],
        "goal_pos": list(full_env.goal_pos),
        "maze_id": f"maze5_ep{attempts+1}",
        "trajectory": trajectory
    })
    known_traps.update(traps_seen)
    attempts += 1

# === 儲存 jsonl ===
save_path = "C:/Users/seana/maze/outputs/mem_trap/maze5_1.jsonl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    for i, r in enumerate(results):
        json.dump({
            "episode": i + 1,
            "maze_id": r["maze_id"],
            "start_pos": r["start_pos"],
            "goal_pos": r["goal_pos"],
            "explored_map": r["explored_map"].tolist(),
            "known_walls": r["known_walls"].tolist(),
            "known_traps": r["known_traps"],
            "trajectory": [[int(x), int(y)] for x, y in r["trajectory"]]
        }, f)
        f.write("\n")

print(f"\U0001F4BE 已儲存探索紀錄，共 {len(results)} 筆：{save_path}")
