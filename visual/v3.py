import numpy as np
import matplotlib.pyplot as plt
import os
# === 讀取含陷阱的地圖資料 ===
folder = "C:/Users/seana/maze/outputs/"
for f in os.listdir(folder):
    if f.endswith(".npy") and f.startswith("trap_"):
        path = os.path.join(folder, f)
        break
data = np.load(path, allow_pickle=True).item()
maze = data["wall_map"]
start = data["start_pos"]
goal = data["goal_pos"]
traps = set(map(tuple, data["trap_list"]))

# === 顯示地圖 ===
H, W = maze.shape
img = np.zeros((H, W, 3))
for i in range(H):
    for j in range(W):
        if maze[i, j] == 1:
            img[i, j] = (0.0, 0.0, 0.0)     # 牆
        else:
            img[i, j] = (1.0, 1.0, 1.0)     # 通道

for tx, ty in traps:
    img[tx, ty] = (1.0, 1.0, 0.0)  # 陷阱（黃色）

img[start] = (0.0, 1.0, 0.0)   # 起點（綠）
img[goal] = (1.0, 0.0, 0.0)    # 終點（紅）

plt.figure(figsize=(6, 6))
plt.imshow(img, interpolation='nearest')
plt.title(f"Maze {H}x{W} with Traps | Seed {data['seed']}")
plt.xticks([]), plt.yticks([])
plt.show()
