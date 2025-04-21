import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import pygame
from sklearn.metrics.pairwise import manhattan_distances

# 测试DQN算法，模拟随机位置随机目标点。
# 网格参数
GRID_SIZE = 10  # 10x10网格
CELL_SIZE = 50  # 每个格子像素大小
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

# 颜色定义
COLORS = {
    "background": (255, 255, 255),
    "agent": (0, 128, 255),
    "goal": (255, 0, 0),
    "obstacle": (0, 0, 0),
    "grid_line": (200, 200, 200),
}
# 环境参数
ACTIONS = ['left', 'right', 'up', 'down']

# DQN参数
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
input_size = 2  # x,y坐标
output_size = len(ACTIONS)


class GridWorld:
    def __init__(self):
        self.target_pos = []
        self.agent_pos = []
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

        # 初始化地图：0=空地，1=障碍，2=目标
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        # 固定障碍物布局（例如手动定义或种子固定）
        np.random.seed(42)  # 固定随机种子
        self._place_obstacles(num_obstacles=15)

        self.grid_size = GRID_SIZE
        self.reset()

    def reset(self):
        # 随机生成智能体和目标位置
        self.agent_pos = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
        self.target_pos = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
        while 1:
            self.agent_pos = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
            self.target_pos = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
            if self.grid[self.agent_pos[0]][
                self.agent_pos[1]] == 0 and self.grid[self.target_pos[0]][self.target_pos[1]] == 0:
                break
        return self.get_state()

    def get_state(self):
        # 将位置转换为状态索引
        return self.agent_pos

    def step(self, action, render=False, tick=30):
        x, y = self.agent_pos
        # 动作映射：0=下，1=上，2=左，3=右
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_x = max(0, min(GRID_SIZE - 1, x + dx))
        new_y = max(0, min(GRID_SIZE - 1, y + dy))
        manha1 = self._manhattan_distance_(self.agent_pos, self.target_pos)
        manha2 = self._manhattan_distance_(self.target_pos, [new_x, new_y])
        reward = manha2 - manha1
        done = False
        # 奖励函数不行
        if [new_x, new_y] == self.target_pos:
            print("reach_target")
            self.agent_pos = [new_x, new_y]
            reward += 100  # 到达目标
            done = True
            # 超出边界或者碰到障碍物
        elif self.grid[new_x][new_y] == 1 or new_x < 0 or new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE:
            reward += -10  # 碰撞障碍
        else:
            self.agent_pos = [new_x, new_y]
            reward += -0.1  # 移动惩罚
        if render:
            self.render(tick=tick)
        return self.get_state(), reward, done

    # 曼哈顿距离计算公式
    def _manhattan_distance_(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _place_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            x, y = np.random.randint(0, GRID_SIZE, 2)
            self.grid[x][y] = 1

    def render(self, tick):
        self.screen.fill(COLORS["background"])
        # 绘制网格线
        for i in range(GRID_SIZE):
            pygame.draw.line(self.screen, COLORS["grid_line"], (0, i * CELL_SIZE), (SCREEN_SIZE, i * CELL_SIZE))
            pygame.draw.line(self.screen, COLORS["grid_line"], (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_SIZE))
        # 绘制障碍和目标
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[x][y] == 1:
                    # 绘制障碍
                    pygame.draw.rect(self.screen, COLORS["obstacle"],
                                     (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif [x, y] == self.target_pos:
                    # 绘制目标
                    pygame.draw.circle(self.screen, COLORS["goal"],
                                       (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
        # 绘制Agent
        pygame.draw.circle(self.screen, COLORS["agent"],
                           (self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                            self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 3)
        pygame.display.flip()
        self.clock.tick(tick)  # 控制渲染帧率


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train():
    epsilon = EPSILON_START
    # 训练循环
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        done = False
        max_steps = 1000
        while not done and max_steps > 0:
            # epsilon-greedy策略
            max_steps -= 1
            if random.random() < epsilon:
                action = random.randint(0, output_size - 1)
            else:
                state_tensor = torch.FloatTensor(state)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()

            # 执行动作
            next_state, reward, done = env.step(action, tick=10000)
            if max_steps == 0:
                reward = -100  # 超时惩罚
                done = True
            total_reward += reward

            # 存储经验
            memory.push(state, action, reward, next_state, done)
            state = next_state

            # 经验回放
            if len(memory) >= BATCH_SIZE:
                # 采样批次
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                # 转换为张量
                states_tensor = torch.FloatTensor(np.array(states))
                actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(np.array(next_states))
                dones_tensor = torch.BoolTensor(dones)

                # 计算目标Q值
                with torch.no_grad():
                    target_q = target_net(next_states_tensor).max(1)[0]
                    target_q[dones_tensor] = 0.0
                    target = rewards_tensor + GAMMA * target_q

                # 计算当前Q值
                current_q = policy_net(states_tensor).gather(1, actions_tensor)

                # 计算损失
                loss = nn.MSELoss()(current_q.squeeze(), target)

                # 优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 更新目标网络
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 衰减epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    torch.save({
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon
    }, 'dqn_model.pth')
    print("Model saved to dqn_model.pth")


def run_policy():
    # 测试训练好的模型
    env = GridWorld()
    state = env.reset()
    done = False
    max_steps = 1000
    while not done and max_steps > 0:
        print(f"Agent position: {state}")
        max_steps -= 1
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action = torch.argmax(policy_net(state_tensor)).item()
        state, reward, done = env.step(action, render=True, tick=1000)


if __name__ == '__main__':
    # 初始化环境、网络和优化器
    env = GridWorld()

    policy_net = DQN(input_size, output_size)
    target_net = DQN(input_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    if os.path.exists('dqn_model.pth'):
        checkpoint = torch.load('dqn_model.pth')
        # 加载核心参数
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 恢复训练进度
        epsilon = checkpoint['epsilon']  # 当前探索率
        # 恢复网络模式
        policy_net.train()
        target_net.eval()
    train()
    # run_policy()
