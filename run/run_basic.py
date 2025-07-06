import gymnasium as gym
import importlib.util
import sys
import numpy as np
import heapq

# ✅ 載入 maze_basic.py
env_path = "C:/Users/seana/maze/env/maze_basic.py"
spec = importlib.util.spec_from_file_location("maze_basic", env_path)
maze_basic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maze_basic)
maze_basic.register_maze_basic_env()


# ✅ 初始化環境
env = maze_basic.MazeBasicEnv(render_mode="human")
env.reset()

# ✅ 建立圖：以狀態為節點，合法動作為邊
graph = {}
for s in range(1, env.rows * env.cols + 1):
    if s in env.wall_states:
        continue
    graph[s] = []
    for i, a in enumerate(env.actions):
        key = f"{s}_{a}"
        ns = env.transition.get(key, s)
        if ns != s:
            graph[s].append((ns, i))  # 鄰居狀態與動作索引

# ✅ 使用 Dijkstra 找最短路徑


# ✅ 使用 Dijkstra 找最短路徑
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
        for v, a in graph[u]:
            if dist[u] + 1 < dist[v]:
                dist[v] = dist[u] + 1
                prev[v] = u
                action_taken[v] = a
                heapq.heappush(pq, (dist[v], v))

    # ✅ 檢查是否能到達終點
    if goal not in prev:
        raise ValueError(f"🚫 找不到路徑可從 {start} 抵達終點 {goal}，請重新產生迷宮")

    # 回推路徑（動作序列）
    path = []
    s = goal
    while s != start:
        path.append(action_taken[s])
        s = prev[s]
    path.reverse()
    return path


# ✅ 計算路徑並執行
path = dijkstra(env.start_state, env.goal_state)
print("🔄 最短動作序列:", [env.actions[a] for a in path])

state, _ = env.reset()
for a_index in path:
    next_state, reward, terminated, truncated, _ = env.step(a_index)
    print(
        f"狀態 {state} -> 動作 {env.actions[a_index]} -> 狀態 {next_state}, 獎勵 {reward}")
    state = next_state
    if terminated:
        print("🎉 成功到達終點！")
        break

env.close()
