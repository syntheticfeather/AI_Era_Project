import pygame
import numpy as np
import os

# 本agent将训练一个Bus Station ?
# 目标点的奖励通过周围城市数量(黑块数量)而定， 要在特定的点选取，作为目标点。

string = "q_table5.npy"

# 网格参数
GRID_SIZE = 20  # 20x20网格
CELL_SIZE = 25  # 每个格子像素大小
# 在初始化时根据目标数量动态计算状态码范围
STATE_CODE_SIZE = 4  # 已放置公交站的可能数量：0,1,2,3
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

# 颜色定义
COLORS = {
    "background": (255, 255, 255),
    "agent": (0, 128, 255),
    "goal": (255, 0, 0),
    "obstacle": (0, 0, 0),
    "available_obstacle": (255, 165, 0),
    "grid_line": (200, 200, 200),
    "station": (0, 255, 0),  # 新增公交站颜色
    "Covered_Block": (0, 0, 128)
}


class GridEnv:
    def __init__(self, random_obstacles=False):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()
        # 初始化地图：0=空地，1=障碍，2=车站
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self._place_obstacles()
        self.agent_pos = [0, 0]
        self.stations = []  # 存储已放置的公交站
        self.covered_blocks = set()  # 重置覆盖记录

    def reset(self):
        """重置环境到初始状态，返回初始观察（状态）"""
        self.agent_pos = [0, 0]  # 重置Agent到起点
        self.stations = []  # 重置公交站列表
        self.covered_blocks = set()  # 重置覆盖记录
        return self._get_state()  # 返回初始状态

    def _place_obstacles(self):
        # 设计一个复杂的迷宫式障碍物布局
        # 横向障碍
        for x in range(4, 18):
            self.grid[x][5] = 1  # 横向障碍
        for x in range(0, 10):
            self.grid[x][15] = 1  # 横向障碍
        for x in range(11, 14):
            self.grid[x][15] = 1  # 横向障碍
        for x in range(6, 13):
            self.grid[x][9] = 1  # 横向障碍
        for x in range(6, 20):
            self.grid[x][7] = 1  # 横向障碍
        for x in range(0, 14):
            self.grid[x][18] = 1  # 横向障碍

        # # 纵向障碍
        for y in range(3, 13):
            self.grid[1][y] = 1  # 纵向障碍
        for y in range(13, 20):
            self.grid[15][y] = 1  # 纵向障碍
        for y in range(9, 12):
            self.grid[10][y] = 1  # 纵向障碍
        #
        # # 小型障碍块
        for x in range(4, 6):
            for y in range(0, 4):
                self.grid[x][y] = 1
        for x in range(4, 6):
            for y in range(11, 15):
                self.grid[x][y] = 1
        for x in range(7, 19, 4):
            for y in range(1, 5):
                self.grid[x][y] = 1
        for x in range(9, 19, 4):
            for y in range(0, 4):
                self.grid[x][y] = 1
        for x in range(17, 20):
            for y in range(9, 18):
                self.grid[x][y] = 1
        # 可通过的障碍物
        self.grid[6][15] = 3  # 可通过的橙色障碍物
        self.grid[7][4] = 3  # 可通过的橙色障碍物
        self.grid[19][7] = 3  # 可通过的橙色障碍物
        self.grid[0][18] = 3  # 可通过的橙色障碍物
        for i in range(9, 18):
            self.grid[19][i] = 3  # 随机生成障碍物
        for x in range(3, 5):
            for y in range(7, 10):
                self.grid[x][y] = 3

    def count_nearby_blocks(self, x, y, radius=2):
        """统计未被其他车站覆盖的有效黑块"""
        n = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                # 检查坐标有效性和是否已被覆盖
                if (0 <= nx < GRID_SIZE and
                        0 <= ny < GRID_SIZE and
                        self.grid[nx][ny] == 1 and
                        (nx, ny) not in self.covered_blocks):
                    n += 1
                    self.covered_blocks.add((nx, ny))  # 记录已覆盖的区块
        # 记录这些新覆盖的区块（但先不修改原数据）
        return n

    def calculate_spread_score(self):
        """计算车站分布的均匀性得分"""
        if len(self.stations) < 2:
            return 0
        min_dist = float('inf')
        for i in range(len(self.stations)):
            for j in range(i + 1, len(self.stations)):
                dx = abs(self.stations[i][0] - self.stations[j][0])
                dy = abs(self.stations[i][1] - self.stations[j][1])
                min_dist = min(min_dist, dx + dy)
        return min_dist  # 使用最小车站间距作为分布得分

    def step(self, action):
        x, y = self.agent_pos
        done = False
        if action < 4:  # 移动动作
            # 动作映射：0=下，1=上，2=左，3=右
            dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_x = max(0, min(GRID_SIZE - 1, x + dx))
            new_y = max(0, min(GRID_SIZE - 1, y + dy))
            # 超出边界或者碰到黑色障碍物
            if (self.grid[new_x][new_y] == 1 or new_x < 0 or
                    new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE):
                reward = -10  # 碰到障碍物或超出边界
            # 检查碰到橙色障碍物
            elif self.grid[new_x][new_y] == 3:
                self.agent_pos = [new_x, new_y]
                reward = -5  # 碰到可通过障碍物
            else:
                self.agent_pos = [new_x, new_y]
                reward = -1  # 移动一次扣一次步数
        else:
            if len(self.stations) >= 3:
                reward = -10
            else:  # 设置公交站
                # 检查当前位置是否有效
                if (x, y) in self.stations:
                    reward = -10
                else:
                    # 计算奖励
                    self.grid[x][y] = 2  # 标记为公交站
                    # 标记此处获得的奖励
                    reward = self.count_nearby_blocks(x, y) * 10
                    distance_penalty = 0
                    for (sx, sy) in self.stations:
                        dist = abs(x - sx) + abs(y - sy)  # 曼哈顿距离
                        if dist < 5:
                            distance_penalty += (5 - dist) * 5
                    self.stations.append((x, y))
                    done = (len(self.stations) == 3)
                    if (len(self.stations) == 3):
                        print("Congratulations! You have placed all three stations!")
        # 在step函数中添加：
        if done and len(self.stations) == 3:
            # 最终奖励：总覆盖率 + 分布均匀性
            total_coverage = len(self.covered_blocks)
            spread_score = self.calculate_spread_score()
            reward += total_coverage * 5 + spread_score * 2

        self.render()  # 实时显示移动过程
        return self._get_state(), reward, done

    def _get_state(self):
        return self.agent_pos[0], self.agent_pos[1], len(self.stations)

    def render(self):
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
                elif self.grid[x][y] == 1 and self.grid[x][y] in self.covered_blocks:
                    # 绘制障碍
                    pygame.draw.rect(self.screen, COLORS["Covered_Block"],
                                     (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif self.grid[x][y] == 3:
                    # 绘制可通过障碍
                    pygame.draw.rect(self.screen, COLORS["available_obstacle"],
                                     (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                # elif (x, y) in self.stations and not self.goals_done[self.goals.index((x, y))]:
                #     # 绘制目标
                #     pygame.draw.circle(self.screen, COLORS["goal"],
                #                        (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
                elif (x, y) in self.stations:
                    # 绘制已完成目标
                    pygame.draw.circle(self.screen, COLORS["station"],
                                       (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
        for (x, y) in self.covered_blocks:
            surf = pygame.Surface((CELL_SIZE, CELL_SIZE))
            surf.set_alpha(100)
            surf.fill((0, 255, 0))
            self.screen.blit(surf, (y * CELL_SIZE, x * CELL_SIZE))
        # 绘制Agent
        pygame.draw.circle(self.screen, COLORS["agent"],
                           (self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                            self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 3)
        pygame.display.flip()
        self.clock.tick(tt)  # 控制渲染帧率


class QLearningAgent:
    def __init__(self):
        if not os.path.exists(string):
            # 没有Q表时初始化全零
            self.q_table = np.ones((GRID_SIZE, GRID_SIZE, STATE_CODE_SIZE, 5))  # 形状 (10,10,8,4)
        else:
            # 加载时直接读取
            self.q_table = np.load(string)
            print("Q table loaded!")
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1

    def choose_action(self, state):
        x, y, num_stations = state
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[x, y, num_stations])

    def learn(self, state, action, reward, next_state):
        x, y, num = state
        next_x, next_y, next_num = next_state
        predict = self.q_table[x, y, num, action]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y, next_num])
        self.q_table[x, y, num, action] += self.alpha * (target - predict)


def train():
    # 训练循环
    env = GridEnv()
    agent = QLearningAgent()
    for episode in range(1000):
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state
        print(f"Episode: {episode}, Total Reward: {total_reward}")
    np.save(string, agent.q_table)


tt = 100  # 控制训练速度
run_tt = 10  # 控制查看效果速度

if __name__ == '__main__':
    train()
