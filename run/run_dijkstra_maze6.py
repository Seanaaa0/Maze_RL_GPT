import json
import numpy as np
import matplotlib.pyplot as plt
import heapq

# === 載入多筆探索記憶 ===
jsonl_path = "C:/Users/seana/maze/outputs/mem_trap/maze6_1.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    episodes = [json.loads(line) for line in f]

H, W = len(episodes[0]["explored_map"]), len(episodes[0]["explored_map"][0])
combined_walls = np.ones((H, W), dtype=np.uint8)
for ep in episodes:
    for x, y in ep["trajectory"]:
        combined_walls[x][y] = 0

start = tuple(episodes[0]["start_pos"])
goals = [tuple(g) for g in episodes[0]["goal_list"]]

# === Dijkstra 演算法 ===


def dijkstra(start, goals):
    heap = [(0, start, [])]
    visited = set()
    found = []
    goal_set = set(goals)

    while heap and goal_set:
        cost, cur, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        path = path + [cur]

        if cur in goal_set:
            found.append((cur, path))
            goal_set.remove(cur)
            if not goal_set:
                break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0 <= nx < H and 0 <= ny < W and combined_walls[nx, ny] == 0:
                heapq.heappush(heap, (cost + 1, (nx, ny), path))

    return [p for _, p in found]


paths = dijkstra(start, goals)
print("\U0001F9ED 總找到路徑數：", len(paths))

# === 合併完整路徑（按順序串接） ===
full_path = []
for p in paths:
    if full_path:
        p = p[1:]  # 移除重複點
    full_path.extend(p)

# === 顯示動畫 ===
COLOR_WALL = (0.0, 0.0, 0.0)
COLOR_PATH = (0.2, 0.4, 0.9)
COLOR_AGENT = (0.0, 1.0, 0.4)
COLOR_GOAL = (1.0, 0.6, 0.2)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

for step, (x, y) in enumerate(full_path):
    img = np.ones((H, W, 3)) * 0.8
    for i in range(H):
        for j in range(W):
            if combined_walls[i, j]:
                img[i, j] = COLOR_WALL

    for gx, gy in goals:
        img[gx, gy] = COLOR_GOAL
    for px, py in full_path[:step]:
        img[px, py] = COLOR_PATH
    img[x, y] = COLOR_AGENT

    ax.clear()
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f"Dijkstra Multi-goal | Step {step+1}/{len(full_path)}")
    ax.set_xticks([]), ax.set_yticks([])
    plt.pause(0.1)

plt.ioff()
plt.show()
