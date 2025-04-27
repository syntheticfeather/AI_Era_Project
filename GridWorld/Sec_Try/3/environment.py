# environment.py
import pygame
import numpy as np
from pygame.locals import *
import torch

fps = 10000
# 改为传周围4 * 4状态
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

        self.rate = 0

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
        """返回4x4局部观察，包含三个通道：
        通道0: 障碍物 (1表示障碍物)
        通道1: agent位置 (1表示当前位置)
        通道2: 目标位置 (1表示目标位置)
        """
        ax, ay = self.agent_pos
        state = np.zeros((3, 4, 4), dtype=np.float32)

        # 障碍物通道
        for dx in range(-1, 3):
            for dy in range(-1, 3):
                x = ax + dx
                y = ay + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    state[0, dx + 1, dy + 1] = self.obstacles[x, y]
                else:
                    state[0, dx + 1, dy + 1] = 1.0  # 边界视为障碍物

        # Agent位置 (中心点)
        state[1, 1, 1] = 1.0

        # 目标位置
        tx_rel = self.target_pos[0] - (ax - 1)
        ty_rel = self.target_pos[1] - (ay - 1)
        if tx_rel < 0:
            tx_rel = 0
        elif tx_rel >= 4:
            tx_rel = 3
        if ty_rel < 0:
            ty_rel = 0
        elif ty_rel >= 4:
            ty_rel = 3  # 给个大致方向
        state[2, tx_rel, ty_rel] = 1.0

        return torch.tensor(state)

    def step(self, action):
        self.render(fps=fps)
        x, y = self.agent_pos
        # 保存旧位置用于奖励计算
        prev_pos = (x, y)
        # 执行动作
        if action == 0:
            x = x - 1
        elif action == 1:
            x = x + 1
        elif action == 2:
            y = y - 1
        elif action == 3:
            y = y + 1

        # # 执行动作
        # if action == 0:
        #     x = max(0, x - 1)
        # elif action == 1:
        #     x = min(self.grid_size - 1, x + 1)
        # elif action == 2:
        #     y = max(0, y - 1)
        # elif action == 3:
        #     y = min(self.grid_size - 1, y + 1)

        done = False
        reward = -1  # 基础步长惩罚
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            reward = -10.0
            done = True
        # 计算新位置的奖励
        elif (x, y) == self.target_pos:
            reward = 10.0
            print("Reached goal!")
            self.rate += 1
            done = True
        elif self.obstacles[x, y]:
            reward = -10.0
            done = True
        else:
            # 添加距离奖励
            old_dist = abs(prev_pos[0] - self.target_pos[0]) + abs(prev_pos[1] - self.target_pos[1])
            new_dist = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])
            reward += 2 * (old_dist - new_dist)  # 鼓励接近目标

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
