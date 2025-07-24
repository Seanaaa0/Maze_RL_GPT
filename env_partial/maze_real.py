import numpy as np
import random
import argparse
import time

MOVE = {
    0: (-1, 0),  # 上
    1: (0, 1),   # 右
    2: (1, 0),   # 下
    3: (0, -1),  # 左
}

TURN_LEFT = {
    0: 3,
    1: 0,
    2: 1,
    3: 2,
}

TURN_RIGHT = {
    0: 1,
    1: 2,
    2: 3,
    3: 0,
}


class MazeReal:
    def __init__(self, size=15, seed=None):
        self.size = size
        self.grid = np.ones((size, size), dtype=np.int8)  # 1 = wall, 0 = free
        self.agent_pos = np.array([0, 0])
        self.agent_dir = 1  # 初始面向右
        self.goal_positions = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._generate_maze()
        self._place_goals()

    def _generate_maze(self):
        self.grid[1:-1, 1:-1] = 0

    def _place_goals(self):
        free_positions = list(zip(*np.where(self.grid == 0)))
        self.goal_positions = random.sample(free_positions, 3)

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.agent_pos = np.array([1, 1])
        self.agent_dir = 1
        return self._get_obs()

    def step(self, action):
        if action == 0:
            self.agent_dir = TURN_LEFT[self.agent_dir]
        elif action == 1:
            self.agent_dir = TURN_RIGHT[self.agent_dir]
        elif action == 3:
            dx, dy = MOVE[self.agent_dir]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.grid[nx, ny] == 0:
                self.agent_pos = np.array([nx, ny])

        obs = self._get_obs()
        done = tuple(self.agent_pos) in self.goal_positions
        return obs, done

    def _get_obs(self):
        view = []
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        dx, dy = MOVE[self.agent_dir]
        nx, ny = x + dx, y + dy

        while True:
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                break
            if self.grid[nx, ny] != 0:
                break
            view.append([int(nx), int(ny)])
            nx += dx
            ny += dy

        return {
            "position": self.agent_pos.copy(),
            "facing": self.agent_dir,
            "view": view,
            "goals": self.goal_positions,
        }
