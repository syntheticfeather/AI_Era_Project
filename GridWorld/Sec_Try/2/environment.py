# environment.py
import pygame
import numpy as np
from pygame.locals import *
import torch

# 颜色定义
COLORS = {
    "background": (255, 255, 255),
    "agent": (0, 128, 255),
    "goal": (255, 0, 0),
    "obstacle": (0, 0, 0),
    "available_obstacle": (255, 165, 0),
    "grid_line": (200, 200, 200),
    "reached_goal": (0, 255, 0),
}


class Environment:
    def __init__(self, grid_size=10, obstacle_density=0.2, render_mode="human"):
        self.grid_size = grid_size
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size
        self.obstacle_density = obstacle_density
        self.render_mode = render_mode
        # Initialize grid states
        self.obstacles = np.random.rand(grid_size, grid_size) < obstacle_density
        self.agent_pos = self._random_free_position()
        self.target_pos = self._random_free_position()

        # Pygame init
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Grid World")
        self.clock = pygame.time.Clock()  # 添加时钟以控制帧率

    def _random_free_position(self):
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if not self.obstacles[pos[0], pos[1]]:
                return pos

    def reset(self):
        self.agent_pos = self._random_free_position()
        self.target_pos = self._random_free_position()
        return self.get_state()

    def get_state(self):
        # Create 3 channel grid: obstacles, agent, target
        state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        state[0] = self.obstacles.astype(np.float32)
        state[1, self.agent_pos[0], self.agent_pos[1]] = 1.0
        state[2, self.target_pos[0], self.target_pos[1]] = 1.0
        return torch.tensor(state)

    def step(self, action):
        self.render()

        reward = 0.0
        done = False
        # 0:up, 1:down, 2:left, 3:right
        x, y = self.agent_pos
        new_x, new_y = x, y
        old_dis = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])
        if action == 0:
            new_x = x - 1
        elif action == 1:
            new_x = x + 1
        elif action == 2:
            new_y = y - 1
        elif action == 3:
            new_y = y + 1
        # 触界要收到惩罚
        if not 0 <= new_x < self.grid_size:
            reward -= -5
            new_x = x
        if not 0 <= new_y < self.grid_size:
            reward -= -5
            new_y = y

        x, y = new_x, new_y
        new_dis = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])

        reward += old_dis - new_dis  # 计算奖励
        reward -= 5  # step penalty

        if (x, y) == self.target_pos:
            reward = 100.0
            print("Reached goal!")
            done = True
        elif self.obstacles[x, y]:
            reward = -10.0
            done = True

        self.agent_pos = (x, y)
        return self.get_state(), reward, done

    def render(self, fps=10000):
        if self.render_mode != "human":
            return

        self.screen.fill((255, 255, 255))  # 修正括号错误

        # 绘制网格线
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.window_size, y))

        # 绘制障碍物（修正坐标转换）
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacles[x][y]:
                    rect = (
                        y * self.cell_size + 1,  # 列转x坐标
                        x * self.cell_size + 1,  # 行转y坐标
                        self.cell_size - 2,
                        self.cell_size - 2
                    )
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)

        # 绘制目标（修正坐标顺序）
        tx, ty = self.target_pos
        pygame.draw.circle(self.screen, (0, 255, 0),
                           (int((ty + 0.5) * self.cell_size),  # 列坐标
                            int((tx + 0.5) * self.cell_size)),  # 行坐标
                           int(self.cell_size / 3))

        # 绘制agent（修正坐标顺序）
        ax, ay = self.agent_pos
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (int((ay + 0.5) * self.cell_size),  # 列坐标
                            int((ax + 0.5) * self.cell_size)),  # 行坐标
                           int(self.cell_size / 3))
        pygame.display.flip()
        self.clock.tick(fps)
