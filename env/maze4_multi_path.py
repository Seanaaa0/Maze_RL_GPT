# Maze 4.0 - 隨機挖空 + Prim 主路徑 + 多重解

import numpy as np
import pygame
import random
import time
from datetime import datetime
import os


class Maze4MultiPath:
    def __init__(self, width=25, height=25, render_mode="human"):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.cell_size = min(1000 // self.width, 1000 // self.height)
        self.map = np.ones((height, width), dtype=np.int8)  # 1: wall, 0: path
        self.agent_pos = (0, 0)
        self.goal_pos = None
        self._generate_maze()
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width * self.cell_size, self.height * self.cell_size))
            pygame.display.set_caption("Maze 4.0 - Multi-path")

    def _generate_maze(self):
        # Step 1: 隨機挖空 40% 地圖
        total_cells = self.width * self.height
        target_path_cells = int(total_cells * 0.4)
        path_cells = set()
        path_cells.add((0, 0))  # 起點
        while len(path_cells) < target_path_cells:
            x, y = random.randint(
                1, self.width - 2), random.randint(1, self.height - 2)
            if (x, y) not in path_cells:
                path_cells.add((x, y))
        for (x, y) in path_cells:
            self.map[y][x] = 0

        # Step 2: 選一個離起點遠的目標點
        candidates = [(x, y) for (x, y) in path_cells if abs(
            x) + abs(y) >= int(self.height * 3 / 5)]
        if not candidates:
            raise ValueError("❌ 無法找到足夠遠的目標點。請降低條件或增加地圖大小。")
        self.goal_pos = random.choice(candidates)

        # Step 3: 用 Prim 強制建立主通路（保證有解）
        start = (0, 0)
        goal = self.goal_pos
        visited = set()
        frontier = [(start, [])]  # (座標, 路徑)

        while frontier:
            frontier.sort(key=lambda x: random.random())  # 類似 Prim 的隨機選擇
            (cx, cy), path = frontier.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            path = path + [(cx, cy)]
            if (cx, cy) == goal:
                for (x, y) in path:
                    self.map[y][x] = 0
                break
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in visited:
                    frontier.append(((nx, ny), path))

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos

    def render(self):
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.screen.fill((0, 0, 0))
        for y in range(self.height):
            for x in range(self.width):
                color = (255, 255, 255) if self.map[y][x] == 0 else (0, 0, 0)
                pygame.draw.rect(self.screen, color, (x * self.cell_size,
                                 y * self.cell_size, self.cell_size, self.cell_size))
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        pygame.draw.rect(self.screen, (0, 255, 0), (gx * self.cell_size,
                         gy * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (ax * self.cell_size,
                         ay * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()
        time.sleep(0.5)

    def step(self, action):
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height and self.map[ny][nx] == 0:
            self.agent_pos = (nx, ny)
        done = self.agent_pos == self.goal_pos
        reward = 1 if done else -0.01
        return self.agent_pos, reward, done, {}

    def save_maze(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"C:/Users/seana/maze/outputs/maze4_{timestamp}.npy"
        np.save(path, self.map)
        print(f"✅ Maze saved to: {path}")
