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
SEED = 1
MAX_STEPS = 150
REPEAT = 3
MAX_ATTEMPTS = 30  # 最多嘗試次數，避免卡死
success_count = 0
print("🚀 開始 BFS 探索...")

results = []

# === Ground truth 環境 ===
gt_env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.1)
gt_env.reset(seed=SEED)
goal_pos = gt_env.goal_pos
wall_map = gt_env.grid.copy()

# 儲存 ground truth 地圖
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

# === 探索任務 ===
attempts = 0
while success_count < REPEAT and attempts < MAX_ATTEMPTS:
    attempts += 1

    env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.1)
    env.grid = wall_map.copy()
    env.goal_pos = goal_pos
    _ = env.reset()
    env.agent_pos = np.array([0, 0])
    obs = env._get_obs()
    obs["position"] = np.array([0, 0])

    trajectory = [tuple(obs["position"])]
    intended = []

    for step_count in range(MAX_STEPS):
        selected = None

        for action in range(4):  # 上下左右 0 1 2 3
            dx, dy = maze.MOVE[action]
            x, y = env.agent_pos
            nx, ny = x + dx, y + dy
            if not (0 <= nx < SIZE and 0 <= ny < SIZE):
                continue
            if env.grid[nx, ny] == 1:
                continue
            selected = action
            break

        if selected is None:
            continue

        intended.append(selected)
        prev_pos = tuple(env.agent_pos)
        obs, reward, terminated, truncated, info = env.step(selected)
        current_pos = tuple(obs["position"])

        print(
            f"[DEBUG] Step {step_count:>3} | 指令: {selected} | 位置: {prev_pos} → {current_pos} | 成功: {current_pos != prev_pos}"
        )

        if current_pos != prev_pos:
            trajectory.append(current_pos)

        if terminated:
            print(f"✅ 成功抵達終點，共花費 {len(trajectory) - 1} 步")
            results.append({
                "start_pos": [0, 0],
                "goal_pos": [int(env.goal_pos[0]), int(env.goal_pos[1])],
                "trajectory": [[int(p[0]), int(p[1])] for p in trajectory],
                "intended_actions": [int(a) for a in intended],
                "success": True,
                "seed": int(SEED + success_count)
            })
            success_count += 1
            break
    else:
        print("❌ 本次探索失敗，重新嘗試...")

# 儲存結果
save_path = f"C:/Users/seana/maze/outputs/nondeter2/nondeter_bfs_{SEED}.jsonl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    for r in results:
        json.dump(r, f)
        f.write("\n")

print(f"📄 BFS 探索資料已儲存於 {save_path}，共儲存 {len(results)} 筆紀錄")
