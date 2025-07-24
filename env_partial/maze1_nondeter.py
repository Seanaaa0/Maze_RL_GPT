
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import random

MOVE = {
    0: (-1, 0),  # 上
    1: (1, 0),   # 下
    2: (0, -1),  # 左
    3: (0, 1),   # 右
}

REVERSE_ACTION = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}


class Maze1NonDeter(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None, size=15, noise_prob=0.1):
        self.size = size
        self.noise_prob = noise_prob
        self.grid = np.ones((self.size, self.size), dtype=np.int32)

        self.start_pos = None
        self.goal_pos = None

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
        self._generate_maze()

    def _generate_maze(self):
        assert self.size % 2 == 1, "❌ maze size 必須為奇數，否則迷宮可能卡住！"
        self.grid = np.ones((self.size, self.size), dtype=np.int32)

        def neighbors(x, y):
            dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            random.shuffle(dirs)
            result = []
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    result.append((nx, ny))
            return result

        sx, sy = self.size // 2, self.size // 2
        # 確保起點在奇數格
        if sx % 2 == 0:
            sx -= 1
        if sy % 2 == 0:
            sy -= 1

        self.grid[sx, sy] = 0
        stack = [(sx, sy)]
        step_count = 0
        MAX_STEPS = 10000

        while stack:
            x, y = stack[-1]
            nbs = [n for n in neighbors(x, y) if self.grid[n] == 1]
            if nbs:
                nx, ny = random.choice(nbs)
                self.grid[(x + nx) // 2, (y + ny) // 2] = 0
                self.grid[nx, ny] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

            step_count += 1
            if step_count > MAX_STEPS:
                print("⚠️ generate_maze 超出最大步數，強制退出！")
                break

        # 額外挖牆：打通 10% 牆壁
        wall_cells = [(i, j) for i in range(self.size)
                      for j in range(self.size) if self.grid[i, j] == 1]
        dig_count = int(len(wall_cells) * 0.1)
        for cell in random.sample(wall_cells, dig_count):
            self.grid[cell] = 0

    def _get_obs(self):
        x, y = self.agent_pos
        return {
            "position": np.array([x, y], dtype=np.int32),
            "view": np.array([[self.grid[x, y]]], dtype=np.int32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._generate_maze()

        # 隨機挑選通道格子作為起點與終點
        path_cells = [(i, j) for i in range(self.size)
                      for j in range(self.size) if self.grid[i, j] == 0]
        self.goal_pos = random.choice(path_cells)
        self.agent_pos = self.start_pos if self.start_pos else random.choice(
            path_cells)
        return self._get_obs(), {}

    def step(self, action):
        if random.random() < self.noise_prob:
            action = REVERSE_ACTION[action]

        dx, dy = MOVE[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx, ny] == 0:
            self.agent_pos = (nx, ny)

        terminated = tuple(self.agent_pos) == tuple(self.goal_pos)
        reward = 1.0 if terminated else 0.0

        return self._get_obs(), reward, terminated, False, {}

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
                color = (30, 30, 30) if self.grid[i, j] == 1 else (
                    255, 255, 255)
                pygame.draw.rect(
                    self.window, color,
                    pygame.Rect(j * self.cell_size, i *
                                self.cell_size, self.cell_size, self.cell_size)
                )

        gx, gy = self.goal_pos
        pygame.draw.rect(
            self.window, (255, 165, 0),
            pygame.Rect(gy * self.cell_size, gx * self.cell_size,
                        self.cell_size, self.cell_size)
        )

        ax, ay = self.agent_pos
        pygame.draw.rect(
            self.window, (255, 0, 0),
            pygame.Rect(ay * self.cell_size, ax * self.cell_size,
                        self.cell_size, self.cell_size)
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
