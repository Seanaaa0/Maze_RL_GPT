import json
import numpy as np
import matplotlib.pyplot as plt
import heapq
import itertools

# === 載入探索記憶（含陷阱與多目標） ===
jsonl_path = "C:/Users/seana/maze/outputs/mem_trap/maze6_multi_1.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    episodes = [json.loads(line) for line in f]

H, W = len(episodes[0]["explored_map"]), len(episodes[0]["explored_map"][0])
combined = np.ones((H, W), dtype=np.uint8)
trap_set = set()
for ep in episodes:
    for x, y in ep["trajectory"]:
        combined[x, y] = 0
    for tx, ty in ep.get("known_traps", []):
        trap_set.add((tx, ty))

start = tuple(episodes[0]["start_pos"])
goals = [tuple(g) for g in episodes[0]["goal_list"]]

# === 視線檢查 ===


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
        if not (0 <= x < H and 0 <= y < W) or combined[x, y] == 1:
            return False
        if (x, y) in trap_set:
            return False
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True

# === Theta* 規劃（避開陷阱） ===


def theta_star(source, target):
    heap = [(0, 0, source, source, [source])]
    visited = set()
    while heap:
        f, g, cur, parent, path = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == target:
            return path
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cur[0] + dx, cur[1] + dy
                if 0 <= nx < H and 0 <= ny < W and combined[nx, ny] == 0 and (nx, ny) not in trap_set:
                    next_pos = (nx, ny)
                    if line_of_sight(parent, next_pos):
                        heapq.heappush(heap, (g + np.hypot(*(np.subtract(next_pos, parent))) + np.hypot(*(np.subtract(
                            target, next_pos))), g + np.hypot(*(np.subtract(next_pos, parent))), next_pos, parent, path + [next_pos]))
                    else:
                        heapq.heappush(heap, (g + np.hypot(*(np.subtract(next_pos, cur))) + np.hypot(*(np.subtract(
                            target, next_pos))), g + np.hypot(*(np.subtract(next_pos, cur))), next_pos, cur, path + [next_pos]))
    return []


# === 建立所有合法 pair 間的路徑 ===
all_points = [start] + goals
paths = {}
for a, b in itertools.permutations(all_points, 2):
    path = theta_star(a, b)
    if path:
        paths[(a, b)] = path

# === 嘗試所有 goal 排列，選擇避開陷阱的最短組合 ===
best_path = None
min_length = float("inf")
for perm in itertools.permutations(goals):
    full = []
    current = start
    valid = True
    for g in perm:
        key = (current, g)
        if key not in paths:
            valid = False
            break
        segment = paths[key]
        if full:
            segment = segment[1:]
        full.extend(segment)
        current = g
    if valid and len(full) < min_length:
        min_length = len(full)
        best_path = full

print("\U0001F9ED 最短合法路徑長度（避開陷阱）：", min_length)

# === 顯示動畫 ===
COLOR_WALL = (0.0, 0.0, 0.0)
COLOR_PATH = (0.2, 0.4, 0.9)
COLOR_AGENT = (0.0, 1.0, 0.4)
COLOR_GOAL = (1.0, 0.6, 0.2)
COLOR_TRAP = (1.0, 1.0, 0.0)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

for step, (x, y) in enumerate(best_path):
    img = np.ones((H, W, 3)) * 0.8
    for i in range(H):
        for j in range(W):
            if combined[i, j]:
                img[i, j] = COLOR_WALL
    for tx, ty in trap_set:
        img[tx, ty] = COLOR_TRAP
    for gx, gy in goals:
        img[gx, gy] = COLOR_GOAL
    for px, py in best_path[:step]:
        img[px, py] = COLOR_PATH
    img[x, y] = COLOR_AGENT

    ax.clear()
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f"TSP Theta* (No Trap) | Step {step+1}/{len(best_path)}")
    ax.set_xticks([]), ax.set_yticks([])
    plt.pause(0.1)

plt.ioff()
plt.show()
