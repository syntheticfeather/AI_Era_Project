from itertools import count

import pygame
import numpy as np
import os

# 本agent将训练牺牲一定的reward，以便更快到达目标点


string = "q_table4.npy"

# 网格参数
GRID_SIZE = 21  # 20x20网格
CELL_SIZE = 25  # 每个格子像素大小
# 在初始化时根据目标数量动态计算状态码范围
NUM_GOALS = 3
STATE_CODE_SIZE = 2 ** NUM_GOALS
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

# 颜色定义
COLORS = {
    "background": (255, 255, 255),
    "agent": (0, 128, 255),
    "goal": (255, 0, 0),
    "obstacle": (0, 0, 0),
    "available_obstacle": (255, 165, 0),
    "grid_line": (200, 200, 200),
}


class GridEnv:
    def __init__(self, random_obstacles=False):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

        # 初始化地图：0=空地，1=障碍，2=目标, 3为可通过的空地
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.used_obstacles = []  # 已使用的障碍物坐标
        self.placed_goals = []  # 新增：存储已放置的目标

        # 训练时固定障碍物
        self._place_obstacles()

        # Agent初始位置左上角
        self.agent_pos = [0, 0]

    def reset(self):
        """重置时清除已放置目标"""
        self.agent_pos = [0, 0]
        # 清除已放置目标
        for x, y in self.placed_goals:
            self.grid[x][y] = 0
        self.placed_goals = []
        self.used_obstacles = []
        return self._get_state()

    def _place_obstacles(self):
        for x in range(1, 5):
            for y in range(1, 5):
                self.grid[x][y] = 1
        for x in range(1, 5):
            for y in range(6, 10):
                self.grid[x][y] = 1
        for x in range(1, 5):
            for y in range(11, 15):
                self.grid[x][y] = 1
        for x in range(1, 5):
            for y in range(16, 20):
                self.grid[x][y] = 1
        for x in range(6, 8):
            for y in range(1, 20):
                self.grid[x][y] = 1
        for x in range(9, 20):
            for y in range(1, 10):
                self.grid[x][y] = 1
        for x in range(9, 20):
            for y in range(11, 20):
                self.grid[x][y] = 1

    def step(self, action):
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
            if self.grid[x][y] == 0 and (x, y) not in self.placed_goals and len(self.placed_goals) < NUM_GOALS:
                self.grid[x][y] = 2
                self.placed_goals.append((x, y))
                reward = self.count(x, y)
            else:
                reward = self.count(x, y)
                reward -= 200  # 非法放置惩罚
        return self._get_state(), reward, done

    def count(self, x, y):
        # 计算周围8格障碍数量
        radius = 3
        obstacle_count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in self.used_obstacles:
                    obstacle_count += (self.grid[nx][ny] == 1)
                    self.used_obstacles.append((nx, ny))
        print("obstacle_count:", obstacle_count)
        return obstacle_count * 5  # 奖励等于周围障碍数

    def _get_state(self):
        """简化状态为坐标+周围障碍物数量"""
        return self.agent_pos[0], self.agent_pos[1], len(self.placed_goals)

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
                elif self.grid[x][y] == 3:
                    # 绘制可通过障碍
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
        self.clock.tick(tt)  # 控制渲染帧率


def train(endless_train=False):
    # 超参数
    alpha = 0.1  # 学习率
    gamma = 0.95  # 折扣因子
    env = GridEnv()
    num_episodes = 20
    run_done = False
    while not run_done:
        epsilon = 1  # 探索率
        for _ in range(num_episodes):
            ############################### 查看你想要的东西
            print("Episode:", _)
            ############################### 查看你想要的东西
            state = env.reset()
            done = False
            Max_steps = 100  # 最大步数
            reward_sum = 0
            while Max_steps > 0 and not done:
                x, y, placed_num = state
                # ε-greedy选择动作
                if np.random.rand() < epsilon:
                    # print("Random action")
                    action = np.random.randint(5)
                else:
                    action = np.argmax(Q[x][y][placed_num])

                next_state, reward, done = env.step(action)
                reward_sum += reward
                env.render()  # 每一步动作
                # 之后调用render方法
                Q[x][y][placed_num][action] += alpha * (
                        reward + gamma * np.max(Q[next_state[0]][next_state[1]][next_state[2]]) -
                        Q[x][y][placed_num][action])
                state = next_state
                Max_steps -= 1
                ############################### 查看你想要的东西
                # if code != old_code:
                #     # 状态改变，更新Q表
                #     print("State change:", old_code, "->", code)
                #     old_code = code
                # print("code:", code, " || ", "done:", done)
                # reward_sum += reward
                ############################### 查看你想要的东西
            # 衰减epsilon（可选）
            epsilon *= 0.999
            ############################### 查看你想要的东西
            print(f"step: {1000 - Max_steps}", "reward:", reward_sum)
        if not endless_train:
            run_done = True  # 训练结束
        else:
            run_done = run_policy()  # 现在每一百次训练，都运行一次策略
        np.save(string, Q)  # 保存为二进制文件
        ############################### 查看你想要的东西
    ############################### 查看你想要的东西
    print("Training completed!")
    ############################### 查看你想要的东西


# 使用训练后的策略运行
def run_policy():
    env = GridEnv()
    state = env.reset()
    # 打印agent所在位置
    done = False
    step = 0
    reward_sum = 0
    while not done and step < 300:
        action = np.argmax(Q[state[0], state[1], state[2]])
        state, _, done = env.step(action)
        ############################### 查看你想要的东西
        # print(f"[{state[0]},{state[1]}]", end=",")
        step += 1
        reward_sum += 0  # 这里不给奖励，只看看效果
        ############################### 查看你想要的东西
        env.render()  # 实时显示移动过程
        env.clock.tick(run_tt)  # 控制速度
    ############################### 查看你想要的东西
    ############################### 查看你想要的东西
    return done


def reset_Q(Q):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_GOALS + 1, 5))  # 形状 (10,10,8,4)
    np.save(string, Q)  # 保存为二进制文件
    print("Q table reset!")


tt = 100  # 控制训练速度
run_tt = 100  # 控制查看效果速度

if __name__ == "__main__":
    # 初始化Q表：状态为(x,y)，动作为0-3
    if not os.path.exists(string):
        # 没有Q表时初始化全零
        Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_GOALS + 1, 5))  # 形状 (10,10,8,4)
    else:
        # 加载时直接读取
        Q = np.load(string)
        print("Q table loaded!")
    # reset_Q(Q)
    train(endless_train=False)  # 训练/
    # run_policy()  # # 看看效果
