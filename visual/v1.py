
import numpy as np
import matplotlib.pyplot as plt

# 👉 請把這裡換成你的檔案路徑
path = "C:/Users/seana/maze/outputs/105x105_SEED311.npy"

# 讀取迷宮
maze = np.load(path)

# 顯示
plt.figure(figsize=(8, 8))
plt.imshow(maze, cmap="gray_r")  # 0 = 通道（白），1 = 牆（黑）
start = (1, 1)
goal = (maze.shape[0] - 2, maze.shape[1] - 2)
plt.plot(start[1], start[0], "go")  # 綠色起點
plt.plot(goal[1], goal[0], "ro")    # 紅色終點
plt.title("Maze Visualization")
plt.axis("off")
plt.show()
