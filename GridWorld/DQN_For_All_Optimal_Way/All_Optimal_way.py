import torch
import pygame
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.cluster import KMeans

# 全局常量配置
GRID_SIZE = 40
SCREEN_SIZE = 800
CELL_SIZE = SCREEN_SIZE // GRID_SIZE  # 每个格子像素大小
NUM_GOALS = 3  # 默认目标点数（当auto_k=False时生效）

COLORS = {
    "background": (255, 255, 255),
    "grid_line": (200, 200, 200),
    "obstacle": (0, 0, 0),
    "available_obstacle": (255, 165, 0),
    "goal": (255, 0, 0),
    "agent": (0, 0, 255)
}

# 超参数配置
BATCH_SIZE = 64
BUFFER_SIZE = 10000
GAMMA = 0.99
LR = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 100

# 固定目标点坐标 (根据你的需求修改)
FIXED_GOALS = [
    [9, 18],  # 目标点0
    [28, 29],  # 目标点1
    [12, 6],  # 目标点2
    [30, 10],  # 目标点3
    [10, 32]  # 目标点4
]


class DQN(nn.Module):
    """双分支DQN网络"""

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()

        # 视觉分支处理局部视野
        self.visual_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 坐标分支处理位置信息
        self.coord_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64))

        # 合并分支
        self.fc = nn.Sequential(
            nn.Linear(32 * (5 // 2) * (5 // 2) + 64, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, visual, coord):
        vis_out = self.visual_net(visual)
        coord_out = self.coord_net(coord)
        combined = torch.cat([vis_out, coord_out], dim=1)
        return self.fc(combined)


class ReplayBuffer:
    """经验回放缓冲区"""

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


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化网络
        self.policy_net = DQN(input_dim=5, output_dim=4).to(self.device)
        self.target_net = DQN(input_dim=5, output_dim=4).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.steps_done = 0

    def preprocess_state(self, state):
        """预处理状态为网络输入"""
        # 视觉部分: 5x5局部视野 (1x5x5 tensor)
        visual = torch.FloatTensor(state['local_view']).view(1, 5, 5).unsqueeze(0).to(self.device)

        # 坐标部分: [当前x, 当前y, 目标x, 目标y]
        coord = torch.FloatTensor([
            state['agent_pos'][0] / GRID_SIZE,
            state['agent_pos'][1] / GRID_SIZE,
            state['target_pos'][0] / GRID_SIZE,
            state['target_pos'][1] / GRID_SIZE
        ]).unsqueeze(0).to(self.device)

        return (visual, coord)

    def select_action(self, state, training=True):
        """epsilon-greedy动作选择"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                visual, coord = self.preprocess_state(state)
                q_values = self.policy_net(visual, coord)
                return q_values.argmax().item()

    def update_model(self):
        """更新策略网络"""
        if len(self.memory) < BATCH_SIZE:
            return

        # 从缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # 预处理批量数据
        batch_visual = torch.cat([self.preprocess_state(s)[0] for s in states])
        batch_coord = torch.cat([self.preprocess_state(s)[1] for s in states])
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        current_q = self.policy_net(batch_visual, batch_coord).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_visual = torch.cat([self.preprocess_state(s)[0] for s in next_states])
            next_coord = torch.cat([self.preprocess_state(s)[1] for s in next_states])
            next_q = self.target_net(next_visual, next_coord).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

        return loss.item()

    def update_target(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


class GridEnv:
    def __init__(self, n_clusters=5, auto_k=False, max_k=10):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

        # 初始化网格
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.used_obstacles = []
        self.placed_goals = []

        # 聚类参数
        self.n_clusters = n_clusters
        self.auto_k = auto_k
        self.max_k = max_k

        # 放置障碍物
        self._place_obstacles()
        # 自动聚类放置目标
        self._place_goals_with_clustering()

        # Agent初始位置
        self.agent_pos = [0, 0]

    def _place_obstacles(self):
        # 固定种子生成可重复地形
        seed = 50
        rng = np.random.RandomState(seed)  # 创建独立随机数生成器

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if i == 0 or i == GRID_SIZE - 1 or j == 0 or j == GRID_SIZE - 1:
                    self.grid[i][j] = 0
                else:
                    np.random.seed(seed)
                    if np.random.rand() < 0.2:
                        self.grid[i][j] = 1  # 随机障碍物
                        seed += 1
                    else:
                        self.grid[i][j] = 0  # 空地
                        seed += 1

    def _get_passable_points(self):
        """获取所有可通行点坐标"""
        return np.array([[x, y]
                         for x in range(GRID_SIZE)
                         for y in range(GRID_SIZE)
                         if self.grid[x][y] == 0
                         ])

    def _calculate_weights(self, points):
        """计算每个点的权重（周围障碍物密度）"""
        weights = []
        for (x, y) in points:
            obstacle_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        obstacle_count += (self.grid[nx][ny] == 1)
            weights.append(obstacle_count)
        return np.array(weights)

    def _find_optimal_k(self, points, weights):
        """肘部法则自动选择最优K值"""
        distortions = []
        max_possible_k = min(len(points), self.max_k)

        for k in range(1, max_possible_k + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(points, sample_weight=weights)
            distortions.append(kmeans.inertia_)

        # 曲率变化检测
        deltas = np.diff(distortions)
        if len(deltas) < 2:
            return 1
        curvature = np.diff(deltas)
        optimal_k = np.argmax(curvature) + 2  # 拐点位置补偿
        return min(optimal_k, self.max_k)

    def _place_goals_with_clustering(self):
        """通过聚类放置目标点"""
        points = self._get_passable_points()
        if len(points) == 0:
            raise ValueError("No passable points for clustering!")

        weights = self._calculate_weights(points)

        # 自动选择K值
        if self.auto_k:
            self.n_clusters = self._find_optimal_k(points, weights)

        # 安全限制
        self.n_clusters = min(self.n_clusters, len(points))
        if self.n_clusters <= 0:
            self.n_clusters = 1

        # 执行加权K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(points, sample_weight=weights)
        centroids = kmeans.cluster_centers_.astype(int)
        # 放置目标点
        self.placed_goals = []
        for (x, y) in centroids:
            x = np.clip(x, 0, GRID_SIZE - 1)
            y = np.clip(y, 0, GRID_SIZE - 1)
            if self.grid[x][y] == 0 and (x, y) not in self.placed_goals:
                self.grid[x][y] = 2
                self.placed_goals.append((x, y))
            elif self.grid[x][y] == 1:
                flag = False
                # 在周围八个格子中随机选择一个空地作为红点
                for dx in [-1, 0, 1]:
                    if flag: break
                    for dy in [-1, 0, 1]:
                        if flag: break
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[nx][ny] == 0:
                            self.grid[nx][ny] = 2
                            self.placed_goals.append((nx, ny))
                            flag = True

    def reset(self, new_k=None):
        """重置环境并可选新K值"""
        if new_k is not None:
            self.n_clusters = new_k

        # 清除旧目标
        for x, y in self.placed_goals:
            self.grid[x][y] = 0
        self.placed_goals = []

        # 重新聚类
        self._place_goals_with_clustering()
        self.agent_pos = [0, 0]
        return self._get_state()

    # --------------------------
    # 以下方法与原始代码保持一致
    # --------------------------
    def step(self, action):
        # ...（保持原有逻辑不变）
        x, y = self.agent_pos
        done = False

        if action < 4:  # 移动动作
            dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_x = max(0, min(GRID_SIZE - 1, x + dx))
            new_y = max(0, min(GRID_SIZE - 1, y + dy))

            # 碰撞检测
            if self.grid[new_x][new_y] == 1:  # 黑色障碍
                reward = -10
            elif self.grid[new_x][new_y] == 3:  # 橙色障碍
                self.agent_pos = [new_x, new_y]
                reward = -4
            else:
                self.agent_pos = [new_x, new_y]
                reward = -1  # 基础移动惩罚
        else:  # 动作4：放置目标
            if self.grid[x][y] == 0 and (x, y) not in self.placed_goals and len(self.placed_goals) < self.n_clusters:
                self.grid[x][y] = 2
                self.placed_goals.append((x, y))
                reward = self.count(x, y)
            else:
                reward = self.count(x, y)
                reward -= 200  # 非法放置惩罚
        return self._get_state(), reward, done

    def count(self, x, y):
        # ...（保持原有逻辑不变）
        radius = 3
        obstacle_count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in self.used_obstacles:
                    obstacle_count += (self.grid[nx][ny] == 1)
                    self.used_obstacles.append((nx, ny))
        return obstacle_count * 5

    def _get_state(self):
        return self.agent_pos[0], self.agent_pos[1], len(self.placed_goals)

    def render(self):
        # ...（保持原有渲染逻辑不变）
        self.screen.fill(COLORS["background"])
        # 绘制网格线
        for i in range(GRID_SIZE):
            pygame.draw.line(self.screen, COLORS["grid_line"], (0, i * CELL_SIZE), (SCREEN_SIZE, i * CELL_SIZE))
            pygame.draw.line(self.screen, COLORS["grid_line"], (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_SIZE))
        # 绘制障碍和目标
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[x][y] == 1:
                    pygame.draw.rect(self.screen, COLORS["obstacle"],
                                     (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif self.grid[x][y] == 3:
                    pygame.draw.rect(self.screen, COLORS["available_obstacle"],
                                     (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for x, y in self.placed_goals:
            pygame.draw.circle(self.screen, COLORS["goal"],
                               (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2),
                               CELL_SIZE // 3)
        # 绘制Agent
        pygame.draw.circle(self.screen, COLORS["agent"],
                           (self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                            self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 3)
        pygame.display.flip()
        self.clock.tick(30)


class EnhancedGridEnv(GridEnv):
    """增强的网格环境"""

    def __init__(self):
        super().__init__()
        self.fixed_goals = FIXED_GOALS
        self.prev_distance = 0
        self.current_target = 0  # 初始默认值
        self.start_idx = 0
        self.reset()

    def reset(self, start_idx=None, target_idx=None):
        """重置环境并指定起始点和目标点"""
        super().reset()
        # 确保属性存在
        if not hasattr(self, 'current_target'):
            self.current_target = 0

        # 随机选择起始和目标点
        if start_idx is None or target_idx is None:
            indices = random.sample(range(len(self.fixed_goals)), 2)
            start_idx, target_idx = indices[0], indices[1]

        # 设置起始位置
        start_pos = self.fixed_goals[start_idx]
        self.agent_pos = list(start_pos)

        # 设置目标位置
        self.current_target = target_idx
        target_pos = self.fixed_goals[target_idx]

        # 清除旧目标点
        for (x, y) in self.placed_goals:
            self.grid[x][y] = 0
        self.placed_goals = [tuple(target_pos)]
        self.grid[target_pos[0]][target_pos[1]] = 2

        # 初始化距离跟踪
        self.prev_distance = self._calculate_distance()

        return self._get_state()

    def _calculate_distance(self):
        """计算当前与目标的曼哈顿距离"""
        ax, ay = self.agent_pos
        tx, ty = self.fixed_goals[self.current_target]
        return abs(ax - tx) + abs(ay - ty)

    def _get_state(self):
        """获取增强状态信息"""
        # 局部5x5视野
        local_view = []
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                nx = self.agent_pos[0] + dx
                ny = self.agent_pos[1] + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    local_view.append(1 if self.grid[nx][ny] == 1 else 0)
                else:
                    local_view.append(1)  # 边界视为障碍

        return {
            'agent_pos': self.agent_pos.copy(),
            'target_pos': self.fixed_goals[self.current_target],
            'local_view': np.array(local_view)
        }

    def step(self, action):
        """修改后的step函数"""
        x, y = self.agent_pos
        done = False
        reward = -0.1  # 基础移动惩罚

        # 移动动作
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_x = np.clip(x + dx, 0, GRID_SIZE - 1)
        new_y = np.clip(y + dy, 0, GRID_SIZE - 1)

        # 碰撞检测
        if self.grid[new_x][new_y] == 1:
            reward -= 10
        else:
            self.agent_pos = [new_x, new_y]

            # 到达目标检测
            if (new_x, new_y) == tuple(self.fixed_goals[self.current_target]):
                reward += 100
                done = True

            # 距离奖励
            new_distance = self._calculate_distance()
            reward += (self.prev_distance - new_distance) * 0.5
            self.prev_distance = new_distance

        next_state = self._get_state()
        return next_state, reward, done


# 训练流程
def train():
    env = EnhancedGridEnv()
    agent = DQNAgent(env)

    episode_rewards = []
    loss_history = []

    for episode in range(2000):
        # 随机选择起点和目标
        print(f"Episode {episode}")
        start_idx, target_idx = random.sample(range(len(FIXED_GOALS)), 2)
        state = env.reset(start_idx=start_idx, target_idx=target_idx)
        total_reward = 0
        done = False

        while not done:
            # 选择并执行动作
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)

            # 更新模型
            loss = agent.update_model()
            if loss is not None:
                loss_history.append(loss)

            state = next_state
            total_reward += reward

            # 更新目标网络
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target()
            agent.steps_done += 1

        episode_rewards.append(total_reward)

        # 每100回合输出进度
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # 保存模型
    torch.save(agent.policy_net.state_dict(), "path_planning_dqn.pth")


if __name__ == "__main__":
    train()
