import numpy as np
import matplotlib.pyplot as plt

# 👉 請把這裡換成你的檔案路徑
path = "C:/Users/seana/maze/outputs/non_size15_seed1496.npy"

# 正確讀取 dict 格式的 .npy
maze_data = np.load(path, allow_pickle=True).item()

# 拿出地圖與座標
maze = maze_data["wall_map"]
start = maze_data["start_pos"]
goal = maze_data["goal_pos"]

# 顯示
plt.figure(figsize=(8, 8))
plt.imshow(maze, cmap="gray_r")  # 0 = 通道（白），1 = 牆（黑）
plt.plot(start[1], start[0], "go")  # 綠色起點
plt.plot(goal[1], goal[0], "ro")    # 紅色終點
plt.title(f"Maze {maze.shape[0]}x{maze.shape[1]} | Seed {maze_data['seed']}")
plt.axis("off")
plt.show()
