import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class MazeBasicEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.rows = 10
        self.cols = 10
        self.render_mode = render_mode

        self.actions = ['n', 'e', 's', 'w']
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        self.wall_states = set()
        self.goal_state = self.rows * self.cols
        self.start_state = 1
        self.current_state = self.start_state
        self.transition = {}

        self._generate_maze()
        self._generate_transitions()

        self.screen = None
        self.cell_size = 50

    def _generate_maze(self):
        visited = set()
        maze = [[1] * self.cols for _ in range(self.rows)]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        stack = [(0, 0)]
        maze[0][0] = 0
        visited.add((0, 0))

        def neighbors(y, x):
            for dy, dx in dirs:
                ny, nx = y + dy * 2, x + dx * 2
                if 0 <= ny < self.rows and 0 <= nx < self.cols and (ny, nx) not in visited:
                    yield (dy, dx, ny, nx)

        while stack:
            y, x = stack[-1]
            options = list(neighbors(y, x))
            if options:
                dy, dx, ny, nx = random.choice(options)
                maze[y + dy][x + dx] = 0
                maze[ny][nx] = 0
                visited.add((ny, nx))
                stack.append((ny, nx))
            else:
                stack.pop()

        for y in range(self.rows):
            for x in range(self.cols):
                if maze[y][x] == 1:
                    s = y * self.cols + x + 1
                    self.wall_states.add(s)

    def _generate_transitions(self):
        for s in range(1, self.rows * self.cols + 1):
            if s in self.wall_states:
                continue
            y, x = divmod(s - 1, self.cols)
            for i, a in enumerate(self.actions):
                dy, dx = [(-1, 0), (0, 1), (1, 0), (0, -1)][i]
                ny, nx = y + dy, x + dx
                ns = ny * self.cols + nx + 1
                if 0 <= ny < self.rows and 0 <= nx < self.cols and ns not in self.wall_states:
                    self.transition[f"{s}_{a}"] = ns
                else:
                    self.transition[f"{s}_{a}"] = s

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
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
            width, height = self.cols * self.cell_size, self.rows * self.cell_size
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Maze Basic")

        self.screen.fill((255, 255, 255))

        for y in range(self.rows):
            for x in range(self.cols):
                s = y * self.cols + x + 1
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if s in self.wall_states:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                elif s == self.goal_state:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)
                elif s == self.current_state:
                    pygame.draw.circle(self.screen, (0, 0, 255), rect.center, self.cell_size // 3)

        pygame.display.flip()
        pygame.time.delay(100)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None


def register_maze_basic_env():
    from gymnasium.envs.registration import register
    try:
        register(
            id='MazeBasic-v0',
            entry_point='maze_basic:MazeBasicEnv',
        )
    except gym.error.Error:
        pass
