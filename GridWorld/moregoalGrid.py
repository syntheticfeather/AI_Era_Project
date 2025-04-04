import pygame
import numpy as np
import os

# 本agent将训练到达多个目标点
string = "q_table2.npy"

# 网格参数
GRID_SIZE = 10  # 10x10网格
CELL_SIZE = 50  # 每个格子像素大小
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
    "grid_line": (200, 200, 200),
}


class GridEnv:
    def __init__(self, random_obstacles=False):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

        # 初始化地图：0=空地，1=障碍，2=目标
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

        # 定义多个目标点（示例为三个角落）
        self.goals = [(GRID_SIZE - 1, GRID_SIZE - 1), (0, GRID_SIZE - 1), (GRID_SIZE - 1, 0)]
        self.goals_done = [False] * len(self.goals)  # 跟踪目标完成状态
        for goal in self.goals:
            self.grid[goal[0], goal[1]] = 2  # 标记目标点

        if random_obstacles:
            # 训练时固定障碍物
            self._place_obstacles(num_obstacles=15)
        else:
            # 固定障碍物布局（例如手动定义或种子固定）
            np.random.seed(42)  # 固定随机种子
            self._place_obstacles(num_obstacles=15)

        # Agent初始位置左上角
        self.agent_pos = [0, 0]

    def reset(self):
        """重置环境到初始状态，返回初始观察（状态）"""
        self.agent_pos = [0, 0]  # 重置Agent到起点
        self.goals_done = [False, False, False]
        # 如果障碍物需要随机生成，可以在此处重新生成：
        # self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        # self._place_obstacles(num_obstacles=15)
        return self._get_state()  # 返回初始状态

    def _place_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            while True:
                x, y = np.random.randint(0, GRID_SIZE, 2)
                # 确保障碍物不在目标点或起点
                if self.grid[x][y] == 0 and (x, y) not in self.goals and (x, y) != (0, 0):
                    self.grid[x][y] = 1
                    break

    def step(self, action):
        x, y = self.agent_pos
        # 动作映射：0=下，1=上，2=左，3=右
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_x = max(0, min(GRID_SIZE - 1, x + dx))
        new_y = max(0, min(GRID_SIZE - 1, y + dy))

        done = False
        if (new_x, new_y) in self.goals:
            # print("Goal reached!")
            self.agent_pos = [new_x, new_y]
            if not self.goals_done[self.goals.index((new_x, new_y))]:
                reward = 100  # 到达目标
                self.goals_done[self.goals.index((new_x, new_y))] = True
            else:
                reward = -10  # 重复到达目标,设为空地
        # 超出边界或者碰到障碍物
        elif self.grid[new_x][new_y] == 1 or new_x < 0 or new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE:
            reward = -10  # 碰撞障碍
        # 检查是否所有目标完成
        elif all(self.goals_done):
            reward = 500  # 最终完成奖励
            done = True
        else:
            self.agent_pos = [new_x, new_y]
            reward = -0.1  # 移动惩罚
        self.render()  # 实时显示移动过程
        return self._get_state(), reward, done

    def _get_state(self):
        # 将目标完成状态转换为二进制编码（例如 [True, False, True] → 101=5）
        state_code = sum(d << i for i, d in enumerate(self.goals_done))
        return self.agent_pos[0], self.agent_pos[1], state_code

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
                if (x, y) in self.goals and not self.goals_done[self.goals.index((x, y))]:
                    # 绘制目标
                    pygame.draw.circle(self.screen, COLORS["goal"],
                                       (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
                elif (x, y) in self.goals and self.goals_done[self.goals.index((x, y))]:
                    # 绘制已完成目标
                    pygame.draw.circle(self.screen, (0, 255, 0),
                                       (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
        # 绘制Agent
        pygame.draw.circle(self.screen, COLORS["agent"],
                           (self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                            self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 3)
        pygame.display.flip()
        self.clock.tick(1000)  # 控制渲染帧率


def train():
    # 超参数
    alpha = 0.1  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon = 0.5  # 探索率
    env = GridEnv()
    num_episodes = 100

    for _ in range(num_episodes):
        ############################### 查看你想要的东西
        print("Episode:", _)
        ############################### 查看你想要的东西
        state = env.reset()
        done = False
        Max_steps = 1000  # 最大步数
        reward_sum = 0
        old_code = state[2]
        while Max_steps > 0 and not done:
            x, y, code = state
            # ε-greedy选择动作
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[x][y][code])
            next_state, reward, done = env.step(action)
            env.render()  # 每一步动作
            # 之后调用render方法
            Q[x][y][code][action] += alpha * (
                    reward + gamma * (0 if done else np.max(Q[next_state[0]][next_state[1]][next_state[2]]))
                    - Q[x][y][code][action]
            )
            state = next_state
            Max_steps -= 1
            ############################### 查看你想要的东西
            if code != old_code:
                # 状态改变，更新Q表
                print("State change:", old_code, "->", code)
                old_code = code
            print("code:", code, " || ", "done:", done)
            ############################### 查看你想要的东西
            reward_sum += reward
        # 衰减epsilon（可选）
        epsilon *= 0.95
        ############################### 查看你想要的东西
        print(f"step: {1000 - Max_steps}", "reward:", reward_sum)
        ############################### 查看你想要的东西
    np.save(string, Q)  # 保存为二进制文件
    ############################### 查看你想要的东西
    print("Training completed!")
    ############################### 查看你想要的东西


# 使用训练后的策略运行
def run_policy():
    env = GridEnv()
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = np.argmax(Q[state[0], state[1], state[2]])
        state, _, done = env.step(action)
        ############################### 查看你想要的东西
        print("Step:", step, "Action:", action, "State:", state, "done:", done)
        ############################### 查看你想要的东西
        env.render()  # 实时显示移动过程
        env.clock.tick(10)  # 控制速度
        step += 1
    ############################### 查看你想要的东西
    print("Total steps:", step)
    ############################### 查看你想要的东西


def reset_Q(Q):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, STATE_CODE_SIZE, 4))  # 形状 (10,10,8,4)
    np.save(string, Q)  # 保存为二进制文件
    # print(Q)


if __name__ == "__main__":
    # 初始化Q表：状态为(x,y)，动作为0-3
    if not os.path.exists(string):
        # 没有Q表时初始化全零
        Q = np.zeros((GRID_SIZE, GRID_SIZE, STATE_CODE_SIZE, 4))  # 形状 (10,10,8,4)
    else:
        # 加载时直接读取
        Q = np.load(string)
    # print(Q)
    # train() # 训练
    run_policy()  # 看看效果
    # reset_Q(Q)
