import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
import os
import time
import heapq


class Maze3Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None):
        super().__init__()
        self.rows = 105
        self.cols = 105
        self.render_mode = render_mode

        self.actions = ['n', 'e', 's', 'w']
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        self.start = (0, 0)
        self.maze = np.ones((self.rows, self.cols),
                            dtype=np.int32)  # 1 = wall, 0 = path
        self.visited = set()
        self.transition = {}

        self._generate_main_path()
        # 其他分支功能先關閉
        # num_fake_goals = max(1, self.rows // 6)
        # self._generate_branches(num_branches=num_fake_goals, max_length=2)

        self.current_state = self._coord_to_state(self.start)
        self.goal_state = self._find_furthest_state()
        self.goal = self._state_to_coord(self.goal_state)

        self._build_transitions()

        self.cell_size = min(1000 // self.cols, 1000 // self.rows)
        self.screen = None

        self._save_maze()

    def _coord_to_state(self, pos):
        y, x = pos
        return y * self.cols + x + 1

    def _state_to_coord(self, s):
        s -= 1
        return divmod(s, self.cols)

    def _neighbors(self, y, x):
        for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.rows and 0 <= nx < self.cols:
                yield ny, nx

    def _generate_main_path(self):
        stack = [self.start]
        self.visited.add(self.start)
        self.maze[self.start] = 0

        while stack:
            y, x = stack[-1]
            neighbors = [(ny, nx) for ny, nx in self._neighbors(y, x)
                         if (ny, nx) not in self.visited]
            if neighbors:
                ny, nx = random.choice(neighbors)
                self.maze[ny][nx] = 0
                self.visited.add((ny, nx))
                stack.append((ny, nx))
            else:
                stack.pop()

    def _generate_branches(self, num_branches=5, max_length=5):
        path_cells = list(self.visited)
        for _ in range(num_branches):
            base = random.choice(path_cells)
            queue = [base]
            seen = {base}
            length = 0

            while queue and length < max_length:
                y, x = queue.pop(0)
                neighbors = [(ny, nx) for ny, nx in self._neighbors(y, x)
                             if (ny, nx) not in self.visited and (ny, nx) not in seen]
                if neighbors:
                    ny, nx = random.choice(neighbors)
                    self.maze[ny][nx] = 0
                    self.visited.add((ny, nx))
                    queue.append((ny, nx))
                    seen.add((ny, nx))
                    length += 1

    def _find_furthest_state(self):
        start = self._coord_to_state(self.start)
        dist = {start: 0}
        pq = [(0, start)]
        max_state = start

        while pq:
            d, u = heapq.heappop(pq)
            y, x = self._state_to_coord(u)
            for ny, nx in self._neighbors(y, x):
                if self.maze[ny][nx] == 0:
                    v = self._coord_to_state((ny, nx))
                    if v not in dist:
                        dist[v] = d + 1
                        heapq.heappush(pq, (d + 1, v))
                        if dist[v] > dist[max_state]:
                            max_state = v

        return max_state

    def _build_transitions(self):
        for y in range(self.rows):
            for x in range(self.cols):
                if self.maze[y][x] == 1:
                    continue
                s = self._coord_to_state((y, x))
                for i, (dy, dx) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols and self.maze[ny][nx] == 0:
                        ns = self._coord_to_state((ny, nx))
                        self.transition[f"{s}_{self.actions[i]}"] = ns
                    else:
                        self.transition[f"{s}_{self.actions[i]}"] = s

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
            pygame.display.set_caption("Maze3")

        self.screen.fill((255, 255, 255))

        for y in range(self.rows):
            for x in range(self.cols):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                   self.cell_size, self.cell_size)
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # wall
                elif (y, x) == self.start:
                    pygame.draw.rect(self.screen, (255, 165, 0), rect)  # start
                elif (y, x) == self.goal:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)  # goal
                else:
                    pygame.draw.rect(
                        self.screen, (255, 255, 255), rect)  # path

        cy, cx = self._state_to_coord(self.current_state)
        pygame.draw.circle(self.screen, (0, 0, 255),
                           (cx * self.cell_size + self.cell_size // 2,
                            cy * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        pygame.display.flip()
        time.sleep(0.5)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _save_maze(self):
        os.makedirs("C:/Users/seana/maze/outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"C:/Users/seana/maze/outputs/maze_{timestamp}.npy"
        np.save(path, self.maze)
        print(f"✅ 迷宮儲存至: {path}")


def register_maze3_env():
    from gymnasium.envs.registration import register
    try:
        register(
            id="Maze3-v0",
            entry_point="maze3:Maze3Env",
        )
    except gym.error.Error:
        pass
