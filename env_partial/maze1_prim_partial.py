import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random
import os
import time
from datetime import datetime


class Maze1PrimPartialEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None):
        super().__init__()
        self.rows = 25
        self.cols = 25
        self.render_mode = render_mode

        self.actions = ['n', 'e', 's', 'w']
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        self.maze = np.ones((self.rows, self.cols), dtype=np.int32)
        self.transition = {}

        self.start = (1, 1)
        self._generate_prims_maze()
        self.goal = self._find_furthest()

        if not self._is_reachable(self.start, self.goal):
            raise ValueError("❌ 起點無法走到終點，請重新生成")

        self.current_state = self._coord_to_state(self.start)
        self.goal_state = self._coord_to_state(self.goal)
        self._build_transitions()

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

    def _ensure_connectivity(self):
        # """確保起點可以連到所有通道格"""
        reachable = set()
        from collections import deque
        queue = deque([self.start])
        reachable.add(self.start)
        while queue:
            y, x = queue.popleft()
            for ny, nx in self._neighbors(y, x):
                if self.maze[ny][nx] == 0 and (ny, nx) not in reachable:
                    reachable.add((ny, nx))
                    queue.append((ny, nx))

        for x in range(self.cols):
            if self.maze[y][x] == 0 and (y, x) not in reachable:
                # 強制打通一格接起來
                for ny, nx in self._neighbors(y, x):
                    if self.maze[ny][nx] == 0 and (ny, nx) in reachable:
                        wall_y, wall_x = (y + ny) // 2, (x + nx) // 2
                        self.maze[wall_y][wall_x] = 0  # 打通牆壁
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

    def _coord_to_state(self, pos):
        y, x = pos
        return y * self.cols + x + 1

    def _state_to_coord(self, s):
        s -= 1
        return divmod(s, self.cols)

    def _find_furthest(self):
        from collections import deque
        visited = set()
        queue = deque([(self.start, 0)])
        visited.add(self.start)
        furthest = self.start
        max_dist = 0
        while queue:
            (y, x), d = queue.popleft()
            if d > max_dist:
                furthest, max_dist = (y, x), d
            for ny, nx in self._neighbors(y, x):
                if self.maze[ny][nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append(((ny, nx), d + 1))
        return furthest

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
        return next_state, reward, terminated, False, {"visible": self._visible_area(*self._state_to_coord(next_state))}

    def _visible_area(self, y, x, radius=3):
        visible = set()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, radius + 1):
                ny, nx = y + dy * dist, x + dx * dist
                if 0 <= ny < self.rows and 0 <= nx < self.cols:
                    if self.maze[ny][nx] == 1:
                        break
                    visible.add((ny, nx))
                else:
                    break
        return visible

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.cols * self.cell_size, self.rows * self.cell_size))
            pygame.display.set_caption("Maze1PrimPartial")

        self.screen.fill((255, 255, 255))

        cy, cx = self._state_to_coord(self.current_state)
        visible = self._visible_area(cy, cx)

        for y in range(self.rows):
            for x in range(self.cols):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                   self.cell_size, self.cell_size)
                if (y, x) in visible:
                    pygame.draw.rect(self.screen, (255, 255, 0), rect)
                elif self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                elif (y, x) == self.start:
                    pygame.draw.rect(self.screen, (255, 165, 0), rect)
                elif (y, x) == self.goal:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)

        pygame.draw.circle(self.screen, (0, 0, 255),
                           (cx * self.cell_size + self.cell_size // 2,
                            cy * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)
        pygame.display.flip()
        time.sleep(0.2)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None

    def _save_maze(self):
        os.makedirs("C:/Users/seana/maze/outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"C:/Users/seana/maze/outputs/prim_partial_{timestamp}.npy"
        np.save(path, self.maze)
        print(f"✅ 迷宮儲存至: {path}")


def register_maze_prim_partial_env():
    from gymnasium.envs.registration import register
    try:
        register(
            id="Maze1PrimPartial-v0",
            entry_point="maze1_prim_partial:Maze1PrimPartialEnv",
        )
    except gym.error.Error:
        pass
