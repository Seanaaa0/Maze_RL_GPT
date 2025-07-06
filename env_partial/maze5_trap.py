import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import time
import random

MOVE = {
    0: (-1, 0),  # 上
    1: (1, 0),   # 下
    2: (0, -1),  # 左
    3: (0, 1),   # 右
}


class Maze5TrapEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None, size=15):
        self.size = size
        self.grid = np.ones((self.size, self.size), dtype=np.int32)
        self.traps = set()
        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.int32),
            "view": spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode
        self.window_size = 600
        self.cell_size = self.window_size // self.size
        self.window = None
        self.clock = None

        self.agent_pos = None
        self.goal_pos = None
        self.visited_map = np.zeros((self.size, self.size), dtype=np.int32)

        self._generate_maze()

    def _generate_maze(self):
        self.grid = np.ones((self.size, self.size), dtype=np.int32)
        self.traps = set()
        if hasattr(self, "manual_seed"):
            random.seed(self.manual_seed)

        def neighbors(x, y):
            dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            result = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    result.append((nx, ny))
            random.shuffle(result)
            return result

        # === Step 1: Growing Tree 建立主迷宮 ===
        start_x, start_y = 1, 1
        self.agent_pos = (start_x, start_y)
        self.grid[start_x, start_y] = 0
        stack = [(start_x, start_y)]

        while stack:
            x, y = stack[-1]
            nbs = [n for n in neighbors(x, y) if self.grid[n] == 1]
            if nbs:
                nx, ny = random.choice(nbs)
                self.grid[(x + nx)//2, (y + ny)//2] = 0
                self.grid[nx, ny] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        # Step 2: 隨機設陷阱（不能在通路上）
        path_set = set(zip(*np.where(self.grid == 0)))
        wall_set = set(zip(*np.where(self.grid == 1)))
        wall_candidates = list(wall_set - path_set)
        num_traps = max(1, self.size // 5)
        self.traps = set(random.sample(
            wall_candidates, min(num_traps, len(wall_candidates))))

        # Step 3: 打開額外 10% 牆壁
        all_walls = list(wall_set)
        random.shuffle(all_walls)
        for i in range(int(len(all_walls) * 0.1)):
            wx, wy = all_walls[i]
            self.grid[wx, wy] = 0

        # Step 4: 指定 goal 位於通道上且離起點有距離
        while True:
            gx, gy = random.randint(
                0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[gx, gy] == 0 and (gx, gy) != self.agent_pos and abs(gx - start_x) + abs(gy - start_y) > 6:
                self.goal_pos = (gx, gy)
                break

    def _get_obs(self):
        x, y = self.agent_pos
        self.visited_map[x, y] = 1
        return {
            "position": np.array([x, y], dtype=np.int32),
            "view": np.array([[self.grid[x, y]]], dtype=np.int32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.manual_seed = seed
        self._generate_maze()
        self.visited_map = np.zeros((self.size, self.size), dtype=np.int32)
        return self._get_obs(), {}

    def step(self, action):
        x, y = self.agent_pos
        dx, dy = MOVE[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx, ny] == 0:
            self.agent_pos = (nx, ny)

        terminated = False
        reward = 0.0

        if self.agent_pos == self.goal_pos:
            terminated = True
            reward = 1.0
        elif self.agent_pos in self.traps:
            terminated = True
            reward = -1.0

        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) in self.traps:
                    color = (255, 255, 0)
                elif self.grid[i, j] == 0:
                    color = (255, 255, 255)
                else:
                    color = (30, 30, 30)
                if self.visited_map[i, j]:
                    color = (100, 100, 100)
                pygame.draw.rect(
                    self.window,
                    color,
                    pygame.Rect(j * self.cell_size, i *
                                self.cell_size, self.cell_size, self.cell_size)
                )

        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        pygame.draw.rect(self.window, (0, 255, 0), pygame.Rect(
            gy * self.cell_size, gx * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.window, (255, 0, 0), pygame.Rect(
            ay * self.cell_size, ax * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
