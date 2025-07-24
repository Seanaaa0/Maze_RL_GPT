import json
import numpy as np
import matplotlib.pyplot as plt
import heapq
import itertools

# === 載入探索記憶 ===
jsonl_path = "C:/Users/seana/maze/outputs/mem_trap/maze6_1.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    episodes = [json.loads(line) for line in f]

H, W = len(episodes[0]["explored_map"]), len(episodes[0]["explored_map"][0])
combined = np.ones((H, W), dtype=np.uint8)
for ep in episodes:
    for x, y in ep["trajectory"]:
        combined[x, y] = 0

start = tuple(episodes[0]["start_pos"])
goals = [tuple(g) for g in episodes[0]["goal_list"]]

# === Dijkstra 計算最短路徑 ===


def dijkstra(source, target):
    heap = [(0, source, [])]
    visited = set()
    while heap:
        cost, cur, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        path = path + [cur]
        if cur == target:
            return path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0 <= nx < H and 0 <= ny < W and combined[nx, ny] == 0:
                heapq.heappush(heap, (cost + 1, (nx, ny), path))
    return []


# === 建立所有 pair 間的最短路徑 ===
all_points = [start] + goals
paths = {}
for a, b in itertools.permutations(all_points, 2):
    paths[(a, b)] = dijkstra(a, b)

# === 嘗試所有 goal 排列，找最短完整路徑 ===
best_path = None
min_length = float("inf")
for perm in itertools.permutations(goals):
    full = []
    current = start
    for g in perm:
        segment = paths[(current, g)]
        if full:
            segment = segment[1:]  # 避免重複點
        full.extend(segment)
        current = g
    if len(full) < min_length:
        min_length = len(full)
        best_path = full

print("\U0001F9ED 最短收集順序總長度：", min_length)

# === 顯示動畫 ===
COLOR_WALL = (0.0, 0.0, 0.0)
COLOR_PATH = (0.2, 0.4, 0.9)
COLOR_AGENT = (0.0, 1.0, 0.4)
COLOR_GOAL = (1.0, 0.6, 0.2)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

for step, (x, y) in enumerate(best_path):
    img = np.ones((H, W, 3)) * 0.8
    for i in range(H):
        for j in range(W):
            if combined[i, j]:
                img[i, j] = COLOR_WALL
    for gx, gy in goals:
        img[gx, gy] = COLOR_GOAL
    for px, py in best_path[:step]:
        img[px, py] = COLOR_PATH
    img[x, y] = COLOR_AGENT

    ax.clear()
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f"TSP Route | Step {step+1}/{len(best_path)}")
    ax.set_xticks([]), ax.set_yticks([])
    plt.pause(0.1)

plt.ioff()
plt.show()
