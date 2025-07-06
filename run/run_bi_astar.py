import numpy as np
import matplotlib.pyplot as plt
import heapq

# === 載入探索資料（partial information）===
data_path = "C:/Users/seana/maze/outputs/mem/maze4_train_n1.npy"
data_list = np.load(data_path, allow_pickle=True).tolist()

H, W = data_list[0]["explored_map"].shape
start_pos = tuple(data_list[0]["start_pos"])
goal_pos = tuple(data_list[0]["goal_pos"])

# 初始預設全為牆，只要任何一集曾探索過且不是牆，就標為通道
combined_walls = np.ones((H, W), dtype=np.uint8)
for data in data_list:
    mask = (data["explored_map"] == 1) & (data["known_walls"] == 0)
    combined_walls[mask] = 0  # 永遠信任通道

# 探索過的格子，用來畫圖
combined_explored = np.zeros((H, W), dtype=np.uint8)
for data in data_list:
    combined_explored |= data["explored_map"]

# === Bi-A* 演算法 ===


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos):
    x, y = pos
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W and combined_walls[nx, ny] == 0:
            yield (nx, ny)


def bidirectional_astar(start, goal):
    f_start = [(heuristic(start, goal), 0, start, [start])]
    f_goal = [(heuristic(goal, start), 0, goal, [goal])]
    visited_start = {start: [start]}
    visited_goal = {goal: [goal]}

    while f_start and f_goal:
        _, cost_s, cur_s, path_s = heapq.heappop(f_start)
        for nei in get_neighbors(cur_s):
            if nei in visited_goal:
                return path_s + visited_goal[nei][::-1][1:]
            if nei not in visited_start:
                visited_start[nei] = path_s + [nei]
                heapq.heappush(
                    f_start, (cost_s + 1 + heuristic(nei, goal), cost_s + 1, nei, path_s + [nei]))

        _, cost_g, cur_g, path_g = heapq.heappop(f_goal)
        for nei in get_neighbors(cur_g):
            if nei in visited_start:
                return visited_start[nei] + path_g[::-1][1:]
            if nei not in visited_goal:
                visited_goal[nei] = path_g + [nei]
                heapq.heappush(
                    f_goal, (cost_g + 1 + heuristic(nei, start), cost_g + 1, nei, path_g + [nei]))
    return []


# === 尋找路徑 ===
path = bidirectional_astar(start_pos, goal_pos)
print(f"🧭 Bi-A* (Partial) path length: {len(path)}")
if not path:
    print("❌ 找不到路徑")
    exit()

# === 顏色定義 ===
COLOR_UNEXPLORED = (0.0, 0.0, 0.0)
COLOR_WALL = (0.2, 0.2, 0.2)
COLOR_EXPLORED = (0.65, 0.65, 0.65)
COLOR_PATH = (0.2, 0.4, 0.9)
COLOR_AGENT = (0.4, 1.0, 0.4)
COLOR_GOAL = (1.0, 0.4, 0.3)

# === 顯示動畫 ===
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

for step, (x, y) in enumerate(path):
    img = np.ones((H, W, 3)) * COLOR_UNEXPLORED

    # 牆 & 探索
    for i in range(H):
        for j in range(W):
            if combined_walls[i, j]:
                img[i, j] = COLOR_WALL
            elif combined_explored[i, j]:
                img[i, j] = COLOR_EXPLORED

    # 路徑（不穿牆）
    for prev in path[:step]:
        if combined_walls[prev] == 0:
            img[prev] = COLOR_PATH

    # 終點與 agent
    img[goal_pos] = COLOR_GOAL
    if combined_walls[x, y] == 0:
        img[x, y] = COLOR_AGENT
    else:
        print(f"⚠️ Agent 在牆上 ({x},{y})")

    ax.clear()
    ax.imshow(img, interpolation='nearest', aspect='equal')
    ax.set_title(f"Bi-A* (Partial) | Step {step+1}/{len(path)}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.pause(0.2)

plt.ioff()
plt.show()
