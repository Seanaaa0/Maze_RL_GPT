import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# === 載入所有 EP 資料 ===
data_list = np.load(
    "C:/Users/seana/maze/outputs/mem/maze4_train_n2.npy", allow_pickle=True).tolist()

H, W = data_list[0]["explored_map"].shape
start_pos = tuple(data_list[0]["start_pos"])
goal_pos = tuple(data_list[0]["goal_pos"])

# ✅ 用所有 trajectory 建立 guaranteed 通道（走過的地方一定通）
combined_walls = np.ones((H, W), dtype=np.uint8)
for data in data_list:
    for x, y in data["trajectory"]:
        combined_walls[x, y] = 0  # 走過 = 通道

# （optional）額外畫出 exploration 區域（這裡先不顯示）
# combined_explored = np.zeros((H, W), dtype=np.uint8)
# for data in data_list:
#     combined_explored |= data["explored_map"]

# === BFS 尋找路徑 ===


def bfs(start, goal):
    visited = set()
    queue = deque([(start, [start])])
    while queue:
        cur, path = queue.popleft()
        if cur == goal:
            return path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < H and 0 <= ny < W:
                if combined_walls[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
    return []


# === 尋找路徑 ===
path = bfs(start_pos, goal_pos)
print(f"🧭 BFS path length: {len(path)}")
if not path:
    print("❌ 找不到路徑")
    exit()

# === 顏色定義 ===
COLOR_UNEXPLORED = (0.3, 0.3, 0.3)   # 深灰
COLOR_WALL = (0.0, 0.0, 0.0)         # 黑
COLOR_PATH = (0.2, 0.4, 0.9)         # 藍
COLOR_AGENT = (0.4, 1.0, 0.4)        # 綠
COLOR_GOAL = (1.0, 0.4, 0.3)         # 紅

# === 顯示動畫 ===
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

for step, (x, y) in enumerate(path):
    img = np.ones((H, W, 3)) * COLOR_UNEXPLORED
    for i in range(H):
        for j in range(W):
            if combined_walls[i, j]:
                img[i, j] = COLOR_WALL

    for prev in path[:step]:
        img[prev] = COLOR_PATH

    img[goal_pos] = COLOR_GOAL
    img[x, y] = COLOR_AGENT

    ax.clear()
    ax.imshow(img, interpolation='nearest', aspect='equal')
    ax.set_title(f"BFS (Trajectory Only) | Step {step+1}/{len(path)}")
    ax.set_xticks([]), ax.set_yticks([])
    plt.pause(0.1)

plt.ioff()
plt.show()
