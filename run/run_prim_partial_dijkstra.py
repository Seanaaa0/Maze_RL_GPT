import importlib.util
import heapq
import sys
import time
import os
import numpy as np
from datetime import datetime
import pygame

# ✅ 載入部分可見迷宮環境
env_path = "C:/Users/seana/maze/env_partial/maze1_prim_partial.py"
spec = importlib.util.spec_from_file_location("maze1_prim_partial", env_path)
maze1_prim_partial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze1_prim_partial)
env = maze1_prim_partial.Maze1PrimPartialEnv(render_mode="human")

# ✅ 初始化環境
state, _ = env.reset()

# ✅ 建立狀態轉移圖（修正 version）
graph = {}
for y in range(env.rows):
    for x in range(env.cols):
        if env.maze[y][x] == 1:
            continue
        s = env._coord_to_state((y, x))
        graph[s] = []
        for i, a in enumerate(env.actions):
            key = f"{s}_{a}"
            ns = env.transition.get(key, s)
            if ns != s:
                graph[s].append((ns, i))

# ✅ Dijkstra 最短路徑


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
        # 額外可視化失敗
        print("🚫 無法從起點走到終點，請重新產生迷宮")
        env.render()
        time.sleep(3)
        raise ValueError("Unreachable")

    path = []
    s = goal
    while s != start:
        path.append(action_taken[s])
        s = prev[s]
    path.reverse()
    return path


# ✅ 執行最短路徑
try:
    path = dijkstra(env.current_state, env.goal_state)
except ValueError:
    env.close()
    sys.exit()

print("📌 最短動作序列:", [env.actions[a] for a in path])

state_seq = [env.current_state]
for a_index in path:
    prev_state = env.current_state
    next_state, reward, terminated, truncated, _ = env.step(a_index)
    state_seq.append(next_state)

    print(
        f"➡️ 狀態 {prev_state} → 動作 {env.actions[a_index]} → {next_state} / 獎勵: {reward:.3f}")
    env.render()

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()

    if terminated:
        print("🎉 成功抵達終點！")
        break

env.close()

# ✅ 儲存 agent 走過的路徑
os.makedirs("C:/Users/seana/maze/outputs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"C:/Users/seana/maze/outputs/prim_partial1_{timestamp}.npy"
np.save(save_path, np.array(state_seq))
print(f"✅ agent 路徑已儲存至: {save_path}")
