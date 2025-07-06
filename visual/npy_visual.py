import numpy as np
import matplotlib.pyplot as plt

# 載入訓練資料
data_list = np.load(
    "C:/Users/seana/maze/outputs/mem/maze4_train_n3.npy", allow_pickle=True).tolist()

H, W = data_list[0]["explored_map"].shape
combined_explored = np.zeros((H, W), dtype=np.uint8)
combined_walls = np.zeros((H, W), dtype=np.uint8)

for data in data_list:
    combined_explored |= data["explored_map"]
    combined_walls |= data["known_walls"]

# 顏色設定
COLOR_UNEXPLORED = (0.0, 0.0, 0.0)
COLOR_WALL = (0.0, 0.0, 0.0)
COLOR_EXPLORED = (1.0, 1.0, 1.0)
COLOR_START = (0.1, 1.0, 0.1)
COLOR_GOAL = (1.0, 0.2, 0.2)

# 建立圖像
img = np.ones((H, W, 3)) * COLOR_UNEXPLORED
for i in range(H):
    for j in range(W):
        if combined_walls[i, j]:
            img[i, j] = COLOR_WALL
        elif combined_explored[i, j]:
            img[i, j] = COLOR_EXPLORED

start_pos = tuple(data_list[0]["start_pos"])
goal_pos = tuple(data_list[0]["goal_pos"])
img[start_pos] = COLOR_START
img[goal_pos] = COLOR_GOAL

# 顯示
plt.figure(figsize=(8, 8))
plt.imshow(img, interpolation="nearest")
plt.title("maze4_train_n3.npy (Explored & Walls)")
plt.xticks([])
plt.yticks([])
plt.show()
