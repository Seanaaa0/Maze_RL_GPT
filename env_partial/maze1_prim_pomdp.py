import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random
import os
from datetime import datetime


class Maze1PrimPOMDPEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None):
        super().__init__()
        self.rows = 25
        self.cols = 25
        self.render_mode = render_mode

        self.actions = ['n', 'e', 's', 'w']
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({
            "view": spaces.Box(low=-1, high=1, shape=(11, 11), dtype=np.int32),
            "dir": spaces.Discrete(4)
        })

        self.maze = np.ones((self.rows, self.cols), dtype=np.int32)
        self.start = (1, 1)
        self._generate_prims_maze()
        self.goal = self._set_central_goal()

        if not self._is_reachable(self.start, self.goal):
            raise ValueError("❌ 起點無法走到終點，請重新產生")

        self.current_pos = self.start
        self.last_action = 0  # 初始方向向北
        self.cell_size = min(1000 // self.cols, 1000 // self.rows)
        self.screen = None
        self._save_maze()

    def _generate_prims_maze(self):
        frontier = []
        self.maze[self.start] = 0
        for ny, nx in self._neighbors(*self.start, step=2):
            frontier.append(((ny, nx), self.start))

        while frontier:
            idx = random.randint(0, len(frontier) - 1)
            (y, x), from_cell = frontier.pop(idx)
            if self.maze[y][x] == 1:
                fy, fx = from_cell
                wall_y, wall_x = (y + fy) // 2, (x + fx) // 2
                self.maze[y][x] = 0
                self.maze[wall_y][wall_x] = 0
                for ny, nx in self._neighbors(y, x, step=2):
                    if self.maze[ny][nx] == 1:
                        frontier.append(((ny, nx), (y, x)))
        self._ensure_connectivity()

    def _set_central_goal(self):
        cy, cx = self.rows // 2, self.cols // 2
        if self.maze[cy, cx] == 0 and self._is_reachable(self.start, (cy, cx)):
            return (cy, cx)
        for radius in range(1, max(self.rows, self.cols)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols:
                        if self.maze[ny][nx] == 0 and self._is_reachable(self.start, (ny, nx)):
                            return (ny, nx)
        return (self.rows - 2, self.cols - 2)

    def _ensure_connectivity(self):
        from collections import deque
        reachable = set()
        queue = deque([self.start])
        reachable.add(self.start)
        while queue:
            y, x = queue.popleft()
            for ny, nx in self._neighbors(y, x):
                if self.maze[ny][nx] == 0 and (ny, nx) not in reachable:
                    reachable.add((ny, nx))
                    queue.append((ny, nx))
        for y in range(self.rows):
            for x in range(self.cols):
                if self.maze[y][x] == 0 and (y, x) not in reachable:
                    for ny, nx in self._neighbors(y, x):
                        if self.maze[ny][nx] == 0 and (y, x) in reachable:
                            wall_y, wall_x = (y + ny) // 2, (x + nx) // 2
                            self.maze[wall_y][wall_x] = 0
                            reachable.add((y, x))
                            break

    def _is_reachable(self, start, goal):
        from collections import deque
        visited = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            y, x = queue.popleft()
            if (y, x) == goal:
                return True
            for ny, nx in self._neighbors(y, x):
                if self.maze[ny][nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        return False

    def _neighbors(self, y, x, step=1):
        for dy, dx in [(-step, 0), (step, 0), (0, -step), (0, step)]:
            ny, nx = y + dy, x + dx
            if 0 < ny < self.rows and 0 < nx < self.cols:
                yield ny, nx

    def _get_observation(self):
        view = np.full((11, 11), -1, dtype=np.int32)
        cy, cx = self.current_pos
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.rows and 0 <= nx < self.cols:
                    view[dy + 5, dx + 5] = self.maze[ny][nx]
        return {"view": view, "dir": self.last_action}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_prims_maze()
        self.agent_pos = (1, 1)

        self.start = (1, 1)
        self.goal = self._set_central_goal()
        self.current_pos = self.start
        self.visited = set([self.current_pos])
        self.steps = 0
        self.last_action = 0
        return self._get_observation(), {}

    def compute_reward(self, pos, next_pos, goal):
        if next_pos == goal:
            return 1.0
        prev_dist = self._distance(pos, goal)
        next_dist = self._distance(next_pos, goal)
        delta = prev_dist - next_dist
        step_penalty = -0.02
        progress_reward = delta * 0.1
        revisit_penalty = -0.05 if next_pos in self.visited else 0
        return step_penalty + progress_reward + revisit_penalty

    def _distance(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def step(self, action):
        x, y = self.current_pos

        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dir_priority = [0, 2, 1, 3]

        dx, dy = directions[action]
        nx, ny = x + dx, y + dy

        if not (0 <= ny < self.rows and 0 <= nx < self.cols) or self.maze[ny][nx] == 1:
            for alt_action in dir_priority:
                dx, dy = directions[alt_action]
                alt_nx, alt_ny = x + dx, y + dy
                if 0 <= alt_ny < self.rows and 0 <= alt_nx < self.cols and self.maze[alt_ny][alt_nx] == 0:
                    action = alt_action
                    nx, ny = alt_nx, alt_ny
                    break

        self.current_pos = (nx, ny)
        self.visited.add(self.current_pos)
        self.steps += 1
        self.last_action = action

        terminated = self.current_pos == self.goal
        truncated = self.steps >= 300
        reward = self.compute_reward((x, y), self.current_pos, self.goal)

        return self._get_observation(), reward, bool(terminated), bool(truncated), {}

    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.cols * self.cell_size, self.rows * self.cell_size))
            pygame.display.set_caption("Maze1PrimPOMDP")

        self.screen.fill((255, 255, 255))
        for y in range(self.rows):
            for x in range(self.cols):
                rect = pygame.Rect(x * self.cell_size, y *
                                   self.cell_size, self.cell_size, self.cell_size)
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                elif (y, x) == self.start:
                    pygame.draw.rect(self.screen, (255, 165, 0), rect)
                elif (y, x) == self.goal:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)

        cy, cx = self.current_pos
        pygame.draw.circle(self.screen, (0, 0, 255),
                           (cx * self.cell_size + self.cell_size // 2,
                            cy * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)
        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _save_maze(self):
        os.makedirs("C:/Users/seana/maze/outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"C:/Users/seana/maze/outputs/prim_pomdp_{timestamp}.npy"
        np.save(path, self.maze)
        print(f"✅ 迷宮儲存至: {path}")
