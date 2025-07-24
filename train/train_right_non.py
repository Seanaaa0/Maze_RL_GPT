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
MAX_STEPS = 500
REPEAT = 3
success_count = 0

MOVE_ORDER = [0, 3, 1, 2]  # 優先方向：右手、前方、左手、後退


def rotate_dir(facing, turn):
    return {
        'R': (facing + 1) % 4,
        'L': (facing + 3) % 4,
        'B': (facing + 2) % 4,
        'F': facing
    }[turn]


def get_right_hand_priority(facing):
    return [rotate_dir(facing, t) for t in ['R', 'F', 'L', 'B']]


print("🚀 開始右手法則探索...")

results = []

# === 建立 ground truth 地圖 ===
gt_env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.1)
gt_env.reset(seed=SEED)
goal_pos = gt_env.goal_pos
wall_map = gt_env.grid.copy()

# 儲存正確地圖
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

# === 執行探索任務 ===
while success_count < REPEAT:
    env = maze.Maze1NonDeter(size=SIZE, noise_prob=0.1)
    env.grid = wall_map.copy()
    env.goal_pos = goal_pos
    env.agent_pos = np.array([0, 0])
    obs = env._get_obs()

    trajectory = [tuple(obs["position"])]
    intended = []
    facing = 1  # 初始面向下
    success = False

    for step_count in range(MAX_STEPS):
        selected = None

        for action in get_right_hand_priority(facing):
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
            f"[DEBUG] Step {step_count:>3} | 指令: {selected} | 位置: {prev_pos} → {current_pos} | 成功: {current_pos != prev_pos}")

        if current_pos != prev_pos:
            trajectory.append(current_pos)
            facing = selected

        if terminated:
            print(f"✅ 成功抵達終點，共花費 {len(trajectory) - 1} 步")
            success_count += 1
            results.append({
                "start_pos": [0, 0],
                "goal_pos": [int(env.goal_pos[0]), int(env.goal_pos[1])],
                "trajectory": [[int(p[0]), int(p[1])] for p in trajectory],
                "intended_actions": [int(a) for a in intended],
                "success": True,
                "seed": int(SEED + success_count)
            })
            success = True
            break

    # 若這輪探索未成功，不印任何東西，自動 retry

# === 儲存資料 ===
save_path = f"C:/Users/seana/maze/outputs/nondeter2/nondeter_right_{SEED}.jsonl"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w", encoding="utf-8") as f:
    for r in results:
        json.dump(r, f)
        f.write("\n")

print(f"📄 右手法探索資料已儲存於 {save_path}，共儲存 {len(results)} 筆紀錄")
