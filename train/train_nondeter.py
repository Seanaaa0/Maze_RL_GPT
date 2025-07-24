import json
import os
import numpy as np
import importlib.util

# === 載入環境 ===
env_path = "C:/Users/seana/maze/env_partial/maze1_nondeter.py"
spec = importlib.util.spec_from_file_location("maze1_nondeter", env_path)
maze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze)

SIZE = 15
SEED = 8874
MAX_STEPS = 1000
REPEAT_PER_DIR = 2

DIRECTIONS = {
    "lt": (0, 0),
    "rt": (0, SIZE - 1),
    "lb": (SIZE - 1, 0),
    "rb": (SIZE - 1, SIZE - 1)
}

print("🚀 開始訓練...")
results = []
success_count = {k: 0 for k in DIRECTIONS}

# === 建立 ground truth 地圖 ===
gt_env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.0)
gt_env.reset(seed=SEED)
goal_pos = gt_env.goal_pos
wall_map = gt_env.grid.copy()

# 儲存正確地圖
gt = {
    "size": SIZE,
    "goal": list(goal_pos),
    "wall_map": wall_map.copy()
}
filename = f"non_size{SIZE}_seed{SEED}.npy"
save_path = os.path.join(
    "C:/Users/seana\maze/outputs/nondeter/ground_truth/", filename)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, gt)
print("✅ ground truth 地圖儲存完成")

# === 執行探索任務 ===
for direction, start_pos in DIRECTIONS.items():
    attempts = 0
    while success_count[direction] < REPEAT_PER_DIR:
        # print(f"🌱 嘗試探索方向 {direction} 第 {attempts + 1} 次")

        env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.02)

        # 套用固定地圖與目標位置
        env.grid = wall_map.copy()
        env.goal_pos = goal_pos

        obs, _ = env.reset()
        env.agent_pos = np.array(start_pos)
        obs["position"] = np.array(start_pos)

        trajectory = [tuple(obs["position"])]
        intended = []
        actual = []

        for _ in range(MAX_STEPS):
            action = env.action_space.sample()
            intended.append(action)

            prev_pos = tuple(obs["position"])
            obs, reward, terminated, _, _ = env.step(action)
            current_pos = tuple(obs["position"])
            trajectory.append(current_pos)

            dx = current_pos[0] - prev_pos[0]
            dy = current_pos[1] - prev_pos[1]
            actual_action = next(
                (a for a, (adx, ady) in maze.MOVE.items() if (adx, ady) == (dx, dy)), action)
            actual.append(actual_action)

            if terminated:
                success_count[direction] += 1
                print(
                    f"\u2705 {direction} 成功次數 {success_count[direction]}/{REPEAT_PER_DIR}")
                break

        results.append({
            "start_direction": direction,
            "start_pos": [int(start_pos[0]), int(start_pos[1])],
            "goal_pos": [int(env.goal_pos[0]), int(env.goal_pos[1])],
            "trajectory": [[int(p[0]), int(p[1])] for p in trajectory],
            "intended_actions": [int(a) for a in intended],
            "actual_actions": [int(a) for a in actual],
            "success": bool(terminated),
            "seed": int(SEED + attempts)
        })

        attempts += 1

# === 儲存 JSONL ===
save_path = f"C:/Users/seana/maze/outputs/nondeter/nondeter_{SEED}.jsonl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    for r in results:
        json.dump(r, f)
        f.write("\n")
print(f"\U0001F4BE 共儲存 {len(results)} 筆訓練資料到 {save_path}")
