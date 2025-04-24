import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DynamicGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.obstacles = [(1, 1), (2, 3), (3, 2)]  # 固定障碍物
        self.goal_pos_list = [(self.size - 2, self.size - 2), (0, 0)]
        self.reset()

    def reset(self):
        # 生成智能体和目标位置（确保不重复且不在障碍）
        while True:
            self.agent_pos = (random.randint(0, self.size - 1),
                              random.randint(0, self.size - 1))
            self.goal_pos = self.goal_pos_list[random.randint(0, len(self.goal_pos_list) - 1)]
            if (self.goal_pos not in self.obstacles and
                    self.agent_pos != self.goal_pos):
                break
        return self._get_state()

    def _get_state(self):
        # 修改后的状态包含智能体坐标和目标的绝对坐标
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        return [ax / (self.size - 1), ay / (self.size - 1),  # 归一化坐标
                gx / (self.size - 1), gy / (self.size - 1)]  # 格式：[ax_norm, ay_norm, gx_norm, gy_norm]

    def step(self, action):
        ax, ay = self.agent_pos
        # 改进的移动逻辑：优先保持方向移动
        if action == 0:
            ax = max(0, ax - 1)  # 上
        elif action == 1:
            ax = min(self.size - 1, ax + 1)  # 下
        elif action == 2:
            ay = max(0, ay - 1)  # 左
        elif action == 3:
            ay = min(self.size - 1, ay + 1)  # 右

        new_pos = (ax, ay)
        reward = 0
        done = False

        # 改进的奖励函数
        if new_pos == self.goal_pos:
            reward = 20  # 增加目标奖励
            done = True
        elif new_pos in self.obstacles:
            reward = -10  # 增加障碍惩罚
        else:
            # 基于曼哈顿距离的渐进奖励
            current_dist = abs(self.goal_pos[0] - ax) + abs(self.goal_pos[1] - ay)
            reward = current_dist  # 距离奖励
            # 添加方向奖励（鼓励朝向目标移动）
            dx = self.goal_pos[0] - ax
            dy = self.goal_pos[1] - ay

        self.agent_pos = new_pos
        return self._get_state(), reward, done, {}

    def render(self):
        grid = [['·' for _ in range(self.size)] for _ in range(self.size)]
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        gx, gy = self.goal_pos
        grid[gx][gy] = 'G'
        ax, ay = self.agent_pos
        grid[ax][ay] = 'A'
        for row in grid:
            print(' '.join(row))
        print('-----')


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 增强网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # 增加隐藏层尺寸
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# 调整超参数
EPISODES = 2000  # 增加训练轮数
BATCH_SIZE = 256  # 增加批量大小
GAMMA = 0.99  # 增加折扣因子
EPS_START = 0.95  # 减小初始ε值
EPS_END = 0.01
EPS_DECAY = 0.9999  # 增加ε衰减速度
TARGET_UPDATE = 100  # 调整目标网络更新频率

env = DynamicGridWorld(size=5)
state_dim = 4  # 新状态维度
action_dim = 4

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)  # 调整学习率
buffer = ReplayBuffer(200000)
epsilon = EPS_START

# 训练循环改进
for episode in range(EPISODES):
    state = env.reset()
    episode_reward = 0
    step_count = 0

    while True:
        # ε-greedy with linear decay
        epsilon = max(EPS_END, EPS_START * (EPS_DECAY ** episode))

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action = policy_net(state_tensor).argmax().item()

        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        step_count += 1

        # 优先经验回放
        if len(buffer) >= BATCH_SIZE:
            batch = list(zip(*buffer.sample(BATCH_SIZE)))

            states = torch.FloatTensor(batch[0])
            actions = torch.LongTensor(batch[1])
            rewards = torch.FloatTensor(batch[2])
            next_states = torch.FloatTensor(batch[3])
            dones = torch.BoolTensor(batch[4])

            # Double DQN 更新
            with torch.no_grad():
                next_actions = policy_net(next_states).argmax(1)
                next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards + (~dones) * GAMMA * next_q_values.squeeze()

            current_q = policy_net(states).gather(1, actions.unsqueeze(1))

            # 梯度裁剪和优化
            loss = nn.functional.mse_loss(current_q.squeeze(), target_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

        if done or step_count >= 50:  # 防止无限循环
            break

    # 延迟目标网络更新
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 定期保存和输出
    if episode % 500 == 0:
        print(f"Ep {episode:04d} | Reward: {episode_reward:6.1f} | Steps: {step_count:2d} | Eps: {epsilon:.3f}")
        # 保存模型
        torch.save({
            'policy_net': policy_net.state_dict(),
            'target_net': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epsilon': epsilon,
        }, 'dqn_model.pth')


# 增强测试函数
def test_agent(episodes=20):
    policy_net.eval()
    success = 0
    steps_list = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 30:  # 限制最大步数
            env.render()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action = policy_net(state_tensor).argmax().item()

            state, _, done, _ = env.step(action)
            steps += 1
            if done:
                success += 1
        steps_list.append(steps)

    avg_steps = np.mean(steps_list)
    print(f"成功率: {success / episodes * 100:.1f}% | 平均步数: {avg_steps:.1f}")
    policy_net.train()


test_agent()
