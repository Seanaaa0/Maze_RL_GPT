import json
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq

# === 載入所有 episodes 並合併 trajectory 為通道 ===
jsonl_path = "C:/Users/seana/maze/outputs/mem2/maze4_1.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    data_list = [json.loads(line) for line in f]

H, W = len(data_list[0]["explored_map"]), len(data_list[0]["explored_map"][0])
combined_walls = np.ones((H, W), dtype=np.uint8)
for ep in data_list:
    for x, y in ep["trajectory"]:
        combined_walls[x][y] = 0

# 取第 1 集的 start / goal 當作路徑起點終點
start_pos = tuple(data_list[0]["start_pos"])
goal_pos = tuple(data_list[0]["goal_pos"])

# === 判斷是否有視線（可走直線）===


def line_of_sight(a, b):
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        if not (0 <= x < H and 0 <= y < W) or combined_walls[x, y] == 1:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True

# === Theta* ===


def theta_star(start, goal):
    heap = [(0 + heuristic(start, goal), 0, start, start, [start])]
    visited = {}
    while heap:
        f, g, cur, parent, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited[cur] = path
        if cur == goal:
            return path
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == dy == 0:
                    continue
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < H and 0 <= ny < W):
                    continue
                if combined_walls[nx, ny] == 1:
                    continue
                next_pos = (nx, ny)
                if line_of_sight(parent, next_pos):
                    heapq.heappush(heap, (
                        g + dist(parent, next_pos) + heuristic(next_pos, goal),
                        g + dist(parent, next_pos),
                        next_pos,
                        parent,
                        path + [next_pos]
                    ))
                else:
                    heapq.heappush(heap, (
                        g + dist(cur, next_pos) + heuristic(next_pos, goal),
                        g + dist(cur, next_pos),
                        next_pos,
                        cur,
                        path + [next_pos]
                    ))
    return []


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# === 執行路徑規劃 ===
path = theta_star(start_pos, goal_pos)
print(f"\U0001F9ED Theta* path length: {len(path)}")
if not path:
    print("❌ 找不到路徑")
    exit()

# === 顯示動畫 ===
COLOR_UNEXPLORED = (0.3, 0.3, 0.3)
COLOR_WALL = (0.0, 0.0, 0.0)
COLOR_PATH = (0.2, 0.4, 0.9)
COLOR_AGENT = (0.4, 1.0, 0.4)
COLOR_GOAL = (1.0, 0.4, 0.3)

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
    ax.set_title(f"Theta* (Merged) | Step {step+1}/{len(path)}")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.pause(0.1)

plt.ioff()
plt.show()
