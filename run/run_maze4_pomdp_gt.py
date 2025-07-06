import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# === 讀取全部訓練資料 ===
load_path = "C:/Users/seana/maze/outputs/mem/maze4_train_6.npy"
data_list = np.load(load_path, allow_pickle=True)

# === 檢查是否為多筆 dict 組成的 list ===
if isinstance(data_list, np.ndarray) and isinstance(data_list[0], dict):
    data_list = list(data_list)
else:
    raise ValueError("載入的檔案格式錯誤：應為 list of dicts")

# === 地圖大小取第一筆 ===
H, W = data_list[0]['explored_map'].shape
combined_explored = np.zeros((H, W), dtype=np.uint8)
combined_walls = np.zeros((H, W), dtype=np.uint8)
goal_pos = tuple(data_list[0]['goal_pos'])
start_pos = tuple(data_list[0]['start_pos'])

# === 合併所有探索結果 ===
for data in data_list:
    combined_explored |= data["explored_map"]
    combined_walls |= data["known_walls"]

# === 找最短路徑（BFS） ===


def get_neighbors(pos):
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W and combined_walls[nx, ny] == 0:
            yield (nx, ny)


def bfs_path(start, goal):
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        cur, path = queue.popleft()
        if cur == goal:
            return path
        for nei in get_neighbors(cur):
            if nei not in visited:
                visited.add(nei)
                queue.append((nei, path + [nei]))
    return []


path = bfs_path(start_pos, goal_pos)

# === 畫圖參數 ===
COLOR_UNEXPLORED = (0.05, 0.05, 0.05)
COLOR_EXPLORED = (0.6, 0.6, 0.6)
COLOR_WALL = (0.3, 0.3, 0.3)
COLOR_AGENT = (0.2, 0.9, 0.2)
COLOR_GOAL = (0.9, 0.2, 0.2)
COLOR_PATH = (0.2, 0.2, 0.8)

render_map = np.ones((H, W, 3)) * COLOR_UNEXPLORED
for i in range(H):
    for j in range(W):
        if combined_walls[i, j]:
            render_map[i, j] = COLOR_WALL
        elif combined_explored[i, j]:
            render_map[i, j] = COLOR_EXPLORED
render_map[goal_pos] = COLOR_GOAL

# === 顯示動畫或除錯地圖 ===
if not path:
    print("❌ 找不到從 start_pos 到 goal_pos 的路徑！")
    debug_map = render_map.copy()
    debug_map[start_pos] = COLOR_AGENT
    plt.imshow(debug_map)
    plt.title("探索地圖 (無法連通)")
    plt.axis('off')
    plt.show()
    exit()

# === 顯示動畫 ===
plt.ion()
fig, ax = plt.subplots(figsize=(10, 10))
for step, (x, y) in enumerate(path):
    img = render_map.copy()
    img[x, y] = COLOR_AGENT
    if step > 0:
        px, py = path[step - 1]
        img[px, py] = COLOR_PATH

    ax.clear()
    ax.imshow(img)
    ax.set_title(f"Step {step+1}/{len(path)} | maze4_epAll")
    ax.axis('off')
    plt.pause(0.3)

plt.ioff()
plt.show()
