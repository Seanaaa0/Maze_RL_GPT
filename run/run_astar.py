import numpy as np
import matplotlib.pyplot as plt
import heapq

# === 載入所有 EP 資料 ===
data_list = np.load(
    "C:/Users/seana/maze/outputs/mem/maze4_train_n2.npy", allow_pickle=True).tolist()

H, W = data_list[0]["explored_map"].shape
start_pos = tuple(data_list[0]["start_pos"])
goal_pos = tuple(data_list[0]["goal_pos"])

# ✅ 用所有 trajectory 建立 guaranteed 通道
combined_walls = np.ones((H, W), dtype=np.uint8)
for data in data_list:
    for x, y in data["trajectory"]:
        combined_walls[x, y] = 0  # 走過 = 通道

# === A* 演算法 ===


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # 曼哈頓距離


def astar(start, goal):
    heap = [(heuristic(start, goal), 0, start, [start])]
    visited = set()

    while heap:
        _, cost, cur, path = heapq.heappop(heap)
        if cur == goal:
            return path
        if cur in visited:
            continue
        visited.add(cur)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < H and 0 <= ny < W and combined_walls[nx, ny] == 0:
                heapq.heappush(heap, (
                    cost + 1 + heuristic((nx, ny), goal),
                    cost + 1,
                    (nx, ny),
                    path + [(nx, ny)]
                ))
    return []


# === 尋找路徑 ===
path = astar(start_pos, goal_pos)
print(f"🧭 A* path length: {len(path)}")
if not path:
    print("❌ 找不到路徑")
    exit()

# === 顏色定義 ===
COLOR_UNEXPLORED = (0.3, 0.3, 0.3)
COLOR_WALL = (0.0, 0.0, 0.0)
COLOR_PATH = (0.2, 0.4, 0.9)
COLOR_AGENT = (0.4, 1.0, 0.4)
COLOR_GOAL = (1.0, 0.4, 0.3)

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
    ax.set_title(f"A* (Trajectory Only) | Step {step+1}/{len(path)}")
    ax.set_xticks([]), ax.set_yticks([])
    plt.pause(0.1)

plt.ioff()
plt.show()
