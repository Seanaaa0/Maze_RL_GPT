import json
import numpy as np
import matplotlib.pyplot as plt
import collections

# === 載入所有記憶 ===
jsonl_path = "C:/Users/seana/maze/outputs/mem_trap/maze5_5.jsonl"
with open(jsonl_path, "r", encoding="utf-8") as f:
    episodes = [json.loads(line) for line in f]

H, W = len(episodes[0]["explored_map"]), len(episodes[0]["explored_map"][0])
combined = np.ones((H, W), dtype=np.uint8)
trap_set = set()

for ep in episodes:
    for x, y in ep["trajectory"]:
        combined[x, y] = 0
    for tx, ty in ep["known_traps"]:
        trap_set.add((tx, ty))

start_pos = tuple(episodes[0]["start_pos"])
goal_pos = tuple(episodes[0]["goal_pos"])

# === BFS ===


def bfs(start, goal):
    visited = set()
    queue = collections.deque([(start, [start])])
    while queue:
        cur, path = queue.popleft()
        if cur == goal:
            return path
        if cur in visited:
            continue
        visited.add(cur)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if (0 <= nx < H and 0 <= ny < W and
                    combined[nx, ny] == 0 and (nx, ny) not in trap_set):
                queue.append(((nx, ny), path + [(nx, ny)]))
    return []


path = bfs(start_pos, goal_pos)
print(f"\U0001F9ED Combined BFS path length: {len(path)}")
if not path:
    print("❌ 找不到避開陷阱的通道")
    exit()

# === 顯示動畫 ===
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

for step, (x, y) in enumerate(path):
    img = np.ones((H, W, 3)) * 0.7
    for i in range(H):
        for j in range(W):
            if combined[i, j] == 1:
                img[i, j] = (0.0, 0.0, 0.0)  # 牆
    for tx, ty in trap_set:
        img[tx, ty] = (1.0, 1.0, 0.0)      # 陷阱
    for px, py in path[:step]:
        img[px, py] = (0.4, 0.6, 1.0)      # 已走過
    img[goal_pos] = (1.0, 0.3, 0.3)        # goal
    img[x, y] = (0.0, 1.0, 0.4)            # agent

    ax.clear()
    ax.imshow(img, interpolation='nearest')
    ax.set_title(f"BFS (Combined) | Step {step+1}/{len(path)}")
    ax.set_xticks([]), ax.set_yticks([])
    plt.pause(0.1)

plt.ioff()
plt.show()
