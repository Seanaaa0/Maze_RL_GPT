import json
import os
import numpy as np
import importlib.util
from collections import deque
import copy

# === 載入環境 ===
env_path = "C:/Users/seana/maze/env_partial/maze1_nondeter.py"
spec = importlib.util.spec_from_file_location("maze1_nondeter", env_path)
maze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze)

SIZE = 15
SEED = 500
MAX_STEPS = 150
REPEAT = 1

print("🚀 開始記憶型 BFS 探索...")

results = []

# === 建立 ground truth 環境 ===
gt_env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.1)
gt_env.reset(seed=SEED)
goal_pos = gt_env.goal_pos
wall_map = gt_env.grid.copy()

# === 儲存 ground truth 地圖 ===
gt = {
    "size": SIZE,
    "goal": list(goal_pos),
    "wall_map": wall_map.copy().tolist()
}
filename = f"non_size{SIZE}_seed{SEED}_gt.json"
save_path = os.path.join("C:/Users/seana/maze/outputs/non_gt/", filename)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(gt, f)
print("✅ ground truth 地圖儲存完成")

# === 開始探索任務 ===
success_count = 0
attempts = 0

while success_count < REPEAT:
    attempts += 1
    print(f"🚶 Attempt{attempts} 開始探索...")

    env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.1)
    env.grid = wall_map.copy()
    env.goal_pos = goal_pos
    _ = env.reset()
    env.agent_pos = np.array([0, 0])
    obs = env._get_obs()
    obs["position"] = np.array([0, 0])

    # -1: 未知, 0: 牆, 2: 探索過
    internal_map = np.full((SIZE, SIZE), -1, dtype=np.int8)
    internal_map[0, 0] = 2

    trajectory = [tuple(env.agent_pos)]
    intended = []

    for step_count in range(MAX_STEPS):
        selected = None
        x, y = env.agent_pos

        # 優先選擇沒探索過的方向
        for action in range(4):
            dx, dy = maze.MOVE[action]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                continue
            if env.grid[nx, ny] == 1:
                continue
            if internal_map[nx, ny] != 2:
                selected = action
                break

        # 若沒有，就從能走的隨便選一個
        if selected is None:
            for action in range(4):
                dx, dy = maze.MOVE[action]
                nx, ny = x + dx, y + dy
                if 0 <= nx < SIZE and 0 <= ny < SIZE and env.grid[nx, ny] == 0:
                    selected = action
                    break

        if selected is None:
            continue

        intended.append(selected)
        prev_pos = tuple(env.agent_pos)
        obs, reward, terminated, truncated, info = env.step(selected)
        current_pos = tuple(obs["position"])

        # print(
        #     f"[DEBUG] Step {step_count:>3} | 指令: {selected} | 位置: {prev_pos} → {current_pos} | 成功: {current_pos != prev_pos}")

        if current_pos != prev_pos:
            trajectory.append(current_pos)
            internal_map[current_pos] = 2
        else:
            dx, dy = maze.MOVE[selected]
            wall_pos = (prev_pos[0] + dx, prev_pos[1] + dy)
            if 0 <= wall_pos[0] < SIZE and 0 <= wall_pos[1] < SIZE:
                internal_map[wall_pos] = 0

        if terminated:
            print(f"✅ 成功抵達終點，共花費 {len(trajectory) - 1} 步")
            results.append({
                "start_pos": [0, 0],
                "goal_pos": [int(env.goal_pos[0]), int(env.goal_pos[1])],
                "trajectory": copy.deepcopy([[int(p[0]), int(p[1])] for p in trajectory]),
                "intended_actions": copy.deepcopy([int(a) for a in intended]),
                "success": True,
                "seed": int(SEED + success_count),
                "internal_map": copy.deepcopy(internal_map.tolist())
            })
            success_count += 1
            break
    else:
        print(f"❌ Attempt{attempts}探索失敗，重新嘗試...")

# === 儲存結果 ===
save_path = f"C:/Users/seana/maze/outputs/nondeter3/nondeter_mem_{SEED}.jsonl"
with open(save_path, "w", encoding="utf-8") as f:
    for r in results:
        json.dump(r, f)
        f.write("\n")

print(f"📄 記憶探索資料已儲存於 {save_path}")
