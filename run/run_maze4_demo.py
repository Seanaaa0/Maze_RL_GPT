import importlib.util
import time
from pathlib import Path

# === 指定路徑 ===
env_file = Path("C:/Users/seana/maze/env/maze4_multi_path.py")

# === 動態載入 maze4_multi_path.py ===
spec = importlib.util.spec_from_file_location("maze4_multi_path", env_file)
maze4_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze4_module)
Maze4MultiPath = maze4_module.Maze4MultiPath

# === 建立並顯示環境 ===
if __name__ == "__main__":
    env = Maze4MultiPath(width=25, height=25, render_mode="human")
    obs = env.reset()

    print(f"✅ 起點: {env.agent_pos}")
    print(f"✅ 目標: {env.goal_pos}")

    env.render()
    input("🔍 按 Enter 鍵關閉視窗...")
    env.save_maze()
