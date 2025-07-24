import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import time
import random

MOVE = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


class Maze6MultiGoalEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None, size=55, num_goals=3, num_traps=2):
        self.size = size
        self.num_goals = num_goals
        self.num_traps = num_traps
        self.grid = np.ones((self.size, self.size), dtype=np.int32)
        self.goal_list = []
        self.collected_goals = []
        self.traps = set()

        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.int32),
            "view": spaces.Box(low=0, high=1, shape=(1, 1), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(4)

        self.render_mode = render_mode
        self.window_size = 800
        self.cell_size = self.window_size // self.size
        self.window = None
        self.clock = None

        self.agent_pos = None
        self._generate_maze()

    def _generate_maze(self):
        self.grid = np.ones((self.size, self.size), dtype=np.int32)
        self.goal_list = []
        self.collected_goals = []
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

        wall_candidates = [(i, j) for i in range(1, self.size - 1)
                           for j in range(1, self.size - 1) if self.grid[i, j] == 1]
        random.shuffle(wall_candidates)
        for i in range(int(len(wall_candidates) * 0.15)):
            self.grid[wall_candidates[i]] = 0

        # 安排 traps（由牆體中轉換）
        trap_candidates = [(i, j)
                           for (i, j) in wall_candidates if self.grid[i, j] == 1]
        selected_traps = random.sample(trap_candidates, k=min(
            self.num_traps, len(trap_candidates)))
        for tx, ty in selected_traps:
            self.grid[tx, ty] = 0
            self.traps.add((tx, ty))

        # 安排 goals（通道中選取）
        path_cells = [(i, j) for i in range(self.size)
                      for j in range(self.size)
                      if self.grid[i, j] == 0 and (i, j) != self.agent_pos and (i, j) not in self.traps]
        self.goal_list = random.sample(path_cells, self.num_goals)

    def _get_obs(self):
        x, y = self.agent_pos
        return {
            "position": np.array([x, y], dtype=np.int32),
            "view": np.array([[self.grid[x, y]]], dtype=np.int32)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.manual_seed = seed
        self._generate_maze()
        return self._get_obs(), {}

    def step(self, action):
        x, y = self.agent_pos
        dx, dy = MOVE[action]
        nx, ny = x + dx, y + dy

        if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx, ny] == 0:
            self.agent_pos = (nx, ny)

        reward = 0.0
        done = False

        if self.agent_pos in self.traps:
            self.agent_pos = (1, 1)  # 重生

        elif self.agent_pos in self.goal_list:
            self.collected_goals.append(self.agent_pos)
            self.goal_list.remove(self.agent_pos)

        if not self.goal_list:
            reward = 1.0
            done = True

        return self._get_obs(), reward, done, False, {}

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
                elif self.grid[i, j] == 1:
                    color = (30, 30, 30)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(
                    self.window, color,
                    pygame.Rect(j * self.cell_size, i *
                                self.cell_size, self.cell_size, self.cell_size)
                )

        for gx, gy in self.goal_list:
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
