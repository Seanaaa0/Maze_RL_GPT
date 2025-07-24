import json
import os
import argparse
import numpy as np
import importlib.util

# === 載入環境 ===
ENV_PATH = "C:/Users/seana/maze/env_partial/maze_real.py"
spec = importlib.util.spec_from_file_location("maze_real", ENV_PATH)
maze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--size", type=int, default=15)
args = parser.parse_args()

SIZE = args.size
SEED = args.seed

env = maze.MazeReal(size=SIZE, seed=SEED)
obs = env.reset()

trajectory = [obs["position"].tolist()]
facing_record = [obs["facing"]]
view_record = [obs["view"]]
action_record = []

success = False
MAX_STEPS = 300

for step in range(MAX_STEPS):
    view = obs["view"]
    if len(view) >= 3:
        action = 3
    elif len(view) == 0:
        action = np.random.choice([0, 1])
    else:
        action = np.random.choice([0, 1, 3])

    obs, done = env.step(action)

    trajectory.append(obs["position"].tolist())
    facing_record.append(obs["facing"])
    view_record.append(obs["view"])
    action_record.append(action)

    if done:
        success = True
        print(f"✅ 到達目標！步數：{step+1}")
        break

if success:
    output = {
        "seed": int(SEED),
        "size": int(SIZE),
        "start_pos": [1, 1],
        "goals": [[int(x), int(y)] for (x, y) in obs["goals"]],
        "trajectory": [[int(x), int(y)] for (x, y) in trajectory],
        "facing": [int(d) for d in facing_record],
        "view": [[[int(x), int(y)] for (x, y) in v] for v in view_record],
        "actions": [int(a) for a in action_record],
        "success": True
    }

    out_dir = f"C:/Users/seana/maze/outputs/real/real_{SIZE}x{SIZE}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"real_mem_seed{SEED}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

    print(f"📦 成功紀錄已儲存於：{out_path}")
else:
    print("⚠️ 未到達任一目標，紀錄不儲存。")
