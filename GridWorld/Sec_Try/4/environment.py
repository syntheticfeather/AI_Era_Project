# environment.py
import pygame
import numpy as np
import os
from pygame.locals import *
import torch

size = 5
fps = 10000
COLORS = {
    "background": (255, 255, 255),
    "agent": (0, 128, 255),
    "goal": (255, 0, 0),
    "obstacle": (0, 0, 0),
    "available_obstacle": (255, 165, 0),
    "grid_line": (200, 200, 200),
    "reached_goal": (0, 255, 0),
    "path": (144, 238, 144),  # 新增路径颜色
}


class Environment:
    def __init__(self, grid_size=10, obstacle_density=0.2, render_mode="human", obstacles=None):
        self.grid_size = grid_size
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size
        self.obstacle_density = obstacle_density
        self.render_mode = render_mode
        self.obstacles = None
        self.visited_positions = []  # 新增路径记录

        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.generate_connected_obstacles()

        self.agent_pos = self._random_free_position()
        self.target_pos = self._random_free_position()

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Grid World")
        self.clock = pygame.time.Clock()
        self.rate = 0

        # 新增路径记忆参数
        self.path_memory_enabled = True  # 是否启用路径记忆
        self.visited_penalty = -0.3  # 重复访问惩罚系数
        self.visited_decay = 0.9  # 位置记忆衰减系数
        self.position_memory = {}  # 带衰减的位置记忆字典

    @staticmethod
    def save_map(obstacles, filename='saved_map.npy'):
        """保存地图到文件"""
        np.save(filename, obstacles)

    @staticmethod
    def load_map(filename='saved_map.npy', grid_size=10):
        """从文件加载地图"""
        try:
            if os.path.exists(filename):
                obstacles = np.load(filename)
                if obstacles.shape == (grid_size, grid_size):
                    return obstacles
                print(f"Loaded map size {obstacles.shape} doesn't match {grid_size}, using new map")
            return None
        except Exception as e:
            print(f"Error loading map: {str(e)}")
            return None

    def generate_connected_obstacles(self):
        """生成连通障碍物布局"""
        while True:
            temp_obstacles = np.random.rand(self.grid_size, self.grid_size) < self.obstacle_density
            if np.sum(~temp_obstacles) == 0:
                continue
            if self._check_connectivity(temp_obstacles):
                self.obstacles = temp_obstacles
                break

    def _check_connectivity(self, obstacles):
        """检查连通性"""
        free_cells = list(zip(*np.where(~obstacles)))
        if not free_cells:
            return False

        start = free_cells[0]
        visited = set()
        queue = [start]

        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if not obstacles[nx, ny] and (nx, ny) not in visited:
                        queue.append((nx, ny))

        return len(visited) == len(free_cells)

    def _random_free_position(self):
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if not self.obstacles[pos[0], pos[1]]:
                return pos

    def reset(self):
        self.agent_pos = self._random_free_position()
        self.target_pos = self._random_free_position()
        self.visited_positions = [self.agent_pos]  # 重置路径记录
        self.position_memory = {}  # 重置记忆
        return self.get_state()

    def get_state(self):
        """返回5x5局部观察"""
        half_size = size // 2
        ax, ay = self.agent_pos
        state = np.zeros((6, size, size), dtype=np.float32)

        # 障碍物通道
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                x = ax + dx
                y = ay + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    state[0, dx + half_size, dy + half_size] = self.obstacles[x, y]
                else:
                    state[0, dx + half_size, dy + half_size] = 1.0

        # Agent位置
        state[1, half_size, half_size] = 1.0

        # 目标位置指示
        tx_rel = self.target_pos[0] - (ax - half_size)
        ty_rel = self.target_pos[1] - (ay - half_size)
        if 0 <= tx_rel < size and 0 <= ty_rel < size:
            state[2, tx_rel, ty_rel] = 1.0

        # 新增全局方向通道（归一化到[-1,1]）
        dx = (self.target_pos[0] - ax) / self.grid_size
        dy = (self.target_pos[1] - ay) / self.grid_size
        state[3, :, :] = dx  # 第四通道填充dx
        state[4, :, :] = dy  # 第五通道填充dy

        # 新增路径记忆通道（第5通道）
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                x = ax + dx
                y = ay + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # 使用衰减后的记忆值
                    state[5, dx + half_size, dy + half_size] = self.position_memory.get((x, y), 0.0)
                else:
                    state[5, dx + half_size, dy + half_size] = 0.0

        return torch.tensor(state)

    def step(self, action):
        self.render(fps=fps)
        x, y = self.agent_pos
        prev_pos = (x, y)

        # 执行动作
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        done = False
        reward = -2

        # 边界检查
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            reward = -5
            done = True
        elif (x, y) == self.target_pos:
            reward = 10
            self.rate += 1
            print("reach goal")
            done = True
        elif self.obstacles[x, y]:
            reward = -5
            done = True
        else:
            old_dist = abs(prev_pos[0] - self.target_pos[0]) + abs(prev_pos[1] - self.target_pos[1])
            new_dist = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])
            reward += 4 * (old_dist - new_dist)

        if not done and self.path_memory_enabled:
            self.agent_pos = (x, y)
            if self.agent_pos not in self.visited_positions:  # 记录新位置
                self.visited_positions.append(self.agent_pos)
            # 衰减已有记忆
            for pos in self.position_memory:
                self.position_memory[pos] *= self.visited_decay
            # 更新当前位置记忆
            visit_count = self.position_memory.get(self.agent_pos, 0) + 1
            self.position_memory[self.agent_pos] = min(visit_count, 1.0)  # 归一化

            # 添加动态惩罚
            if visit_count > 1:
                reward += self.visited_penalty * visit_count
            else:
                reward += 1

        return self.get_state(), reward, done

    def render(self, fps=10000):
        if self.render_mode != "human":
            return

        self.screen.fill(COLORS["background"])

        # 绘制路径轨迹（新增部分）
        for pos in self.visited_positions:
            x, y = pos
            rect = (
                y * self.cell_size + 1,
                x * self.cell_size + 1,
                self.cell_size - 2,
                self.cell_size - 2
            )
            pygame.draw.rect(self.screen, COLORS["path"], rect)

        # 绘制网格线
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, COLORS["grid_line"], (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, COLORS["grid_line"], (0, y), (self.window_size, y))

        # 绘制障碍物
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacles[x][y]:
                    rect = (
                        y * self.cell_size + 1,
                        x * self.cell_size + 1,
                        self.cell_size - 2,
                        self.cell_size - 2
                    )
                    pygame.draw.rect(self.screen, COLORS["obstacle"], rect)

        # 绘制目标
        tx, ty = self.target_pos
        pygame.draw.circle(self.screen, COLORS["goal"],
                           (int((ty + 0.5) * self.cell_size),
                            int((tx + 0.5) * self.cell_size)),
                           int(self.cell_size / 3))

        # 绘制Agent
        ax, ay = self.agent_pos
        pygame.draw.circle(self.screen, COLORS["agent"],
                           (int((ay + 0.5) * self.cell_size),
                            int((ax + 0.5) * self.cell_size)),
                           int(self.cell_size / 3))

        pygame.display.flip()
        self.clock.tick(fps)


if __name__ == "__main__":
    env = Environment(grid_size=10)
    print(env.obstacles.shape)
    print(env.agent_pos)
