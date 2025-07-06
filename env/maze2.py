import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import os
import time


class Maze2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None):
        super().__init__()
        self.rows = 35
        self.cols = 35
        self.render_mode = render_mode

        self.actions = ['n', 'e', 's', 'w']
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        self.start = (0, 0)
        self.goal = self._random_goal(min_distance=5)
        self.map = np.ones((self.rows, self.cols), dtype=np.int32)
        self.visited = set()
        self.transition = {}
        self.current_state = self._coord_to_state(self.start)
        self.goal_state = self._coord_to_state(self.goal)

        self._generate_main_path()
        self._generate_full_maze()
        self._build_transitions()

        self.cell_size = min(1000 // self.cols, 1000 // self.rows)
        self.screen = None

        self._save_map()

    def _coord_to_state(self, pos):
        y, x = pos
        return y * self.cols + x + 1

    def _state_to_coord(self, s):
        s -= 1
        return divmod(s, self.cols)

    def _random_goal(self, min_distance):
        candidates = [(y, x) for y in range(self.rows)
                      for x in range(self.cols)
                      if abs(y - self.start[0]) + abs(x - self.start[1]) >= min_distance]
        return random.choice(candidates)

    def _neighbors(self, y, x):
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.rows and 0 <= nx < self.cols:
                yield ny, nx

    def _generate_main_path(self):
        stack = [self.start]
        self.visited.add(self.start)
        self.map[self.start] = 0
        reached_goal = False

        while stack:
            y, x = stack[-1]
            if (y, x) == self.goal:
                reached_goal = True
                break

            neighbors = [(ny, nx) for ny, nx in self._neighbors(y, x)
                         if (ny, nx) not in self.visited]
            if neighbors:
                ny, nx = random.choice(neighbors)
                self.map[ny][nx] = 0
                self.visited.add((ny, nx))
                stack.append((ny, nx))
            else:
                stack.pop()

        if not reached_goal:
            raise RuntimeError("Failed to generate DFS path to goal")

    def _generate_full_maze(self):
        frontier = list(self.visited)
        max_cells = int(self.rows * self.cols * 0.4)  # 最多 40% 通路

        while frontier and len(self.visited) < max_cells:
            y, x = random.choice(frontier)
            neighbors = [(ny, nx) for ny, nx in self._neighbors(y, x)
                         if (ny, nx) not in self.visited]

            if neighbors:
                ny, nx = random.choice(neighbors)

                # 計算中間格子位置（保留牆壁結構）
                wall_y = (y + ny) // 2
                wall_x = (x + nx) // 2

                self.map[wall_y][wall_x] = 0  # 打通中間牆
                self.map[ny][nx] = 0          # 打通目標格
                self.visited.add((ny, nx))
                frontier.append((ny, nx))
            else:
                frontier.remove((y, x))

    def _build_transitions(self):
        for y in range(self.rows):
            for x in range(self.cols):
                if self.map[y][x] == 1:
                    continue
                s = self._coord_to_state((y, x))
                for i, (dy, dx) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols and self.map[ny][nx] == 0:
                        ns = self._coord_to_state((ny, nx))
                        self.transition[f"{s}_{self.actions[i]}"] = ns
                    else:
                        self.transition[f"{s}_{self.actions[i]}"] = s

    def _save_map(self):
        os.makedirs("C:/Users/seana/maze/outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"C:/Users/seana/maze/outputs/maze_{timestamp}.npy"
        np.save(path, self.map)
        print(f"✅ 迷宫儲存至: {path}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self._coord_to_state(self.start)
        return self.current_state, {}

    def step(self, action_index):
        a = self.actions[action_index]
        key = f"{self.current_state}_{a}"
        next_state = self.transition.get(key, self.current_state)

        reward = 1.0 if next_state == self.goal_state else -0.01
        terminated = next_state == self.goal_state
        self.current_state = next_state
        return next_state, reward, terminated, False, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            width = self.cols * self.cell_size
            height = self.rows * self.cell_size
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Maze2")

        self.screen.fill((255, 255, 255))

        for y in range(self.rows):
            for x in range(self.cols):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                   self.cell_size, self.cell_size)
                if self.map[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # 牆壁：黑色
                elif (y, x) == self.start:
                    pygame.draw.rect(self.screen, (255, 165, 0), rect)  # 起點：橘色
                elif (y, x) == self.goal:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # 終點：綠色
                else:
                    pygame.draw.rect(
                        self.screen, (255, 255, 255), rect)  # 通路：白色

        cy, cx = self._state_to_coord(self.current_state)
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            (cx * self.cell_size + self.cell_size // 2,
             cy * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )

        pygame.display.flip()
        time.sleep(0.5)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
