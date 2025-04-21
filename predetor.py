import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib as plt
import random


class EnhancedGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.obstacles = [(1, 0), (2, 3), (3, 1), (0, 3)]
        self.reset()

    def reset(self):
        while True:
            self.agent_pos = (random.randint(0, self.size - 1),
                              random.randint(0, self.size - 1))
            self.goal = (random.randint(0, self.size - 1),
                         random.randint(0, self.size - 1))
            if (self.goal not in self.obstacles and
                    self.agent_pos != self.goal):
                break
        return self._get_state()

    def _get_state(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal

        # 基础状态
        state = [
            ax / (self.size - 1),
            ay / (self.size - 1),
            (gx - ax) / self.size,
            (gy - ay) / self.size
        ]

        # 障碍物感知
        obstacle_sensor = [
            int((ax - 1, ay) in self.obstacles),
            int((ax + 1, ay) in self.obstacles),
            int((ax, ay - 1) in self.obstacles),
            int((ax, ay + 1) in self.obstacles)
        ]
        return state + obstacle_sensor

    def step(self, action):
        old_pos = self.agent_pos
        ax, ay = old_pos
        old_dist = abs(self.goal[0] - ax) + abs(self.goal[1] - ay)

        # 执行动作
        if action == 0: ax = max(0, ax - 1)
        if action == 1: ax = min(self.size - 1, ax + 1)
        if action == 2: ay = max(0, ay - 1)
        if action == 3: ay = min(self.size - 1, ay + 1)

        new_pos = (ax, ay)
        reward = 0
        done = False
        obstacle_sensor = self._get_state()[-4:]  # 获取障碍物感知状态

        # 奖励计算
        if new_pos == self.goal:
            reward = 20
            done = True
        elif new_pos in self.obstacles:
            reward = -10
            self.agent_pos = old_pos  # 保持原位
        else:
            new_dist = abs(self.goal[0] - ax) + abs(self.goal[1] - ay)
            distance_reward = (old_dist - new_dist) * 1.0
            step_penalty = -0.2
            obstacle_penalty = -0.5 if any(obstacle_sensor) else 0
            reward = distance_reward + step_penalty + obstacle_penalty
            self.agent_pos = new_pos

        return self._get_state(), reward, done, {}

    def render(self):
        grid = [['·' for _ in range(self.size)] for _ in range(self.size)]
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        gx, gy = self.goal
        grid[gx][gy] = 'G'
        ax, ay = self.agent_pos
        grid[ax][ay] = 'A'
        for row in grid:
            print(' '.join(row))
        print('-----')


class EnhancedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 修改超参数
EPISODES = 3000
BATCH_SIZE = 128
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.998
TARGET_UPDATE = 15
LR = 0.0005

# 初始化组件
env = EnhancedGridWorld()
state_dim = 8  # 更新后的状态维度
action_dim = 4

policy_net = EnhancedDQN(state_dim, action_dim)
target_net = EnhancedDQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(100000)
epsilon = EPS_START
success_history = []

# 训练循环（添加成功率记录）
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    episode_success = 0

    while True:
        # ε-greedy选择动作
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action = policy_net(state_tensor).argmax().item()

        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # 经验回放
        if len(buffer) >= BATCH_SIZE:
            batch = list(zip(*buffer.sample(BATCH_SIZE)))
            states = torch.FloatTensor(batch[0])
            actions = torch.LongTensor(batch[1])
            rewards = torch.FloatTensor(batch[2])
            next_states = torch.FloatTensor(batch[3])
            dones = torch.FloatTensor(batch[4])

            # 计算目标Q值（Double DQN）
            with torch.no_grad():
                next_actions = policy_net(next_states).argmax(1)
                next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards + (1 - dones) * GAMMA * next_q.squeeze()

            # 计算当前Q值
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))

            # 优化步骤
            loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        if done:
            episode_success = 1 if reward > 0 else 0
            success_history.append(episode_success)
            break

    # 更新目标网络
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 探索率衰减
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # 训练监控
    if episode % 100 == 0:
        success_rate = np.mean(success_history[-100:]) if success_history else 0
        print(f"Ep {episode} | Reward: {total_reward:.1f} | Success: {success_rate * 100:.1f}% | Eps: {epsilon:.3f}")


# 测试智能体（添加路径可视化）
def visualize_episode():
    state = env.reset()
    path = [env.agent_pos]
    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = policy_net(state_tensor).argmax().item()
        next_state, _, done, _ = env.step(action)
        path.append(env.agent_pos)
        if done:
            break
        state = next_state

    # 可视化路径
    grid = np.zeros((env.size, env.size))
    for x, y in path:
        grid[x][y] += 1
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.title("Agent Navigation Path")
    plt.show()


visualize_episode()
