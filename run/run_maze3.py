import importlib.util
import heapq
import sys
import time
import os
import numpy as np
from datetime import datetime
import pygame

# ✅ 載入 maze3.py
env_path = "C:/Users/seana/maze/env/maze3.py"
spec = importlib.util.spec_from_file_location("maze3", env_path)
maze3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze3)
maze3.register_maze3_env()

# ✅ 初始化環境
env = maze3.Maze3Env(render_mode="human")
state, _ = env.reset()

# ✅ 建立狀態圖（合法行為轉移）
graph = {}
for s in range(1, env.rows * env.cols + 1):
    y, x = env._state_to_coord(s)
    if env.maze[y][x] == 1:
        continue
    graph[s] = []
    for i, a in enumerate(env.actions):
        key = f"{s}_{a}"
        ns = env.transition.get(key, s)
        if ns != s:
            graph[s].append((ns, i))

# ✅ 使用 Dijkstra 尋找最短路徑


def dijkstra(start, goal):
    dist = {s: float("inf") for s in graph}
    prev = {}
    action_taken = {}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        for v, a in graph.get(u, []):
            if dist[u] + 1 < dist[v]:
                dist[v] = dist[u] + 1
                prev[v] = u
                action_taken[v] = a
                heapq.heappush(pq, (dist[v], v))

    if goal not in prev:
        raise ValueError("🚫 無法從起點走到終點，請重新產生迷宮")

    path = []
    s = goal
    while s != start:
        path.append(action_taken[s])
        s = prev[s]
    path.reverse()
    return path


# ✅ 執行 agent
env.render()
path = dijkstra(env.current_state, env.goal_state)
print("📌 最短路徑動作序列:", [env.actions[a] for a in path])

state_seq = [env.current_state]
total_reward = 0

for a_index in path:
    prev_state = env.current_state
    next_state, reward, terminated, truncated, _ = env.step(a_index)
    total_reward += reward
    state_seq.append(next_state)

    print(
        f"➡️ 狀態 {prev_state} → 動作 {env.actions[a_index]} → {next_state} / 獎勵: {reward:.3f}")
    env.render()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

    if terminated:
        print("🎉 成功抵達終點！")
        break

env.close()

# ✅ 儲存 agent 走過的路徑為 .npy
os.makedirs("C:/Users/seana/maze/outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"C:/Users/seana/maze/outputs/path_maze3_{timestamp}.npy"
np.save(save_path, np.array(state_seq))
print(f"✅ agent 路徑已儲存至: {save_path}")
