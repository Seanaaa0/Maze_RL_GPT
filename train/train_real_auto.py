import json
import os
import argparse
import numpy as np
import importlib.util
from tqdm import tqdm

# === 載入環境 ===
ENV_PATH = "C:/Users/seana/maze/env_partial/maze_real.py"
spec = importlib.util.spec_from_file_location("maze_real", ENV_PATH)
maze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze)

# === 解析參數 ===
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="起始 SEED（包含）")
parser.add_argument("--end", type=int, default=10, help="結束 SEED（不包含）")
parser.add_argument("--size", type=int, default=15, help="迷宮大小")
args = parser.parse_args()

SIZE = args.size
START = args.start
END = args.end

out_dir = f"C:/Users/seana/maze/outputs/real_auto/real_{SIZE}x{SIZE}/"
os.makedirs(out_dir, exist_ok=True)

# === 批次產生訓練資料（直到成功才儲存） ===
for SEED in tqdm(range(START, END), desc="🧠 Generating"):
    MAX_RETRIES = 10
    retry_count = 0
    success = False

    while not success and retry_count < MAX_RETRIES:
        retry_count += 1
        env = maze.MazeReal(size=SIZE, seed=SEED)
        obs = env.reset()

        trajectory = [obs["position"].tolist()]
        facing_record = [obs["facing"]]
        view_record = [obs["view"]]
        action_record = []

        MAX_STEPS = 500
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
                break

    # 儲存成功資料
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

    out_path = os.path.join(out_dir, f"real_mem_seed{SEED}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))
