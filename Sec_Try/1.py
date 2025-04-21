import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


# 自定义网格世界环境
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size - 1, size - 1)  # 右下角为目标
        self.obstacles = [(1, 1), (2, 3), (3, 2)]  # 障碍物位置
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)  # 起点在左上角
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        # 动作映射：0=上, 1=下, 2=左, 3=右
        if action == 0: x = max(0, x - 1)
        if action == 1: x = min(self.size - 1, x + 1)
        if action == 2: y = max(0, y - 1)
        if action == 3: y = min(self.size - 1, y + 1)

        new_pos = (x, y)
        done = False
        reward = 0

        # 碰撞检测
        if new_pos in self.obstacles:
            reward = -1
            done = False  # 撞到障碍物不终止
        elif new_pos == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1  # 每一步的小惩罚

        self.agent_pos = new_pos
        return new_pos, reward, done, {}

    def render(self):
        grid = [['·' for _ in range(self.size)] for _ in range(self.size)]
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        for row in grid:
            print(' '.join(row))
        print('-----')


# DQN网络
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.fc(x)


# 经验回放
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 超参数
EPISODES = 150
BATCH_SIZE = 32
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
LR = 0.001

# 初始化环境与网络
env = GridWorld()
state_size = 2  # (x,y)坐标
action_size = 4  # 上下左右

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer(1000)
epsilon = EPS_START

# 训练循环
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0

    while True:
        # 将状态转换为张量
        state_tensor = torch.FloatTensor(state)

        # ε-greedy策略
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        # 执行动作
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # 存储经验
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        # 经验回放
        if len(buffer) >= BATCH_SIZE:
            transitions = buffer.sample(BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

            # 转换为张量
            state_tensor = torch.FloatTensor(batch_state)
            action_tensor = torch.LongTensor(batch_action).unsqueeze(1)
            reward_tensor = torch.FloatTensor(batch_reward)
            next_state_tensor = torch.FloatTensor(batch_next_state)
            done_tensor = torch.FloatTensor(batch_done)

            # 计算Q值
            current_q = policy_net(state_tensor).gather(1, action_tensor)
            next_q = target_net(next_state_tensor).max(1)[0].detach()
            target_q = reward_tensor + (1 - done_tensor) * GAMMA * next_q

            # 计算损失
            loss = nn.MSELoss()(current_q.squeeze(), target_q)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # 定期更新目标网络
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 探索率衰减
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    print(f"Episode {episode + 1}, Total Reward: {total_reward:.1f}")

# 测试训练结果
state = env.reset()
env.render()
while True:
    state_tensor = torch.FloatTensor(state)
    with torch.no_grad():
        action = policy_net(state_tensor).argmax().item()

    next_state, reward, done, _ = env.step(action)
    env.render()

    if done:
        print("Reached goal!")
        break
    state = next_state
