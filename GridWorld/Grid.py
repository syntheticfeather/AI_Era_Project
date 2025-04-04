import pygame
import numpy as np
import os

# 本agent将训练最简单的最优路线寻找。

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


class GridEnv:
    def __init__(self, random_obstacles=False):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

        # 初始化地图：0=空地，1=障碍，2=目标
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        if random_obstacles:
            # 训练时固定障碍物
            self._place_obstacles(num_obstacles=15)
        else:
            # 固定障碍物布局（例如手动定义或种子固定）
            np.random.seed(42)  # 固定随机种子
            self._place_obstacles(num_obstacles=15)

        # 目标位置右下角
        self.goal = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.grid[self.goal] = 2
        # Agent初始位置左上角
        self.agent_pos = [0, 0]

    def reset(self):
        """重置环境到初始状态，返回初始观察（状态）"""
        self.agent_pos = [0, 0]  # 重置Agent到起点
        # 如果障碍物需要随机生成，可以在此处重新生成：
        # self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        # self._place_obstacles(num_obstacles=15)
        return self._get_state()  # 返回初始状态  

    def _place_obstacles(self, num_obstacles):
        for _ in range(num_obstacles):
            x, y = np.random.randint(0, GRID_SIZE, 2)
            self.grid[x][y] = 1

    def step(self, action):
        x, y = self.agent_pos
        # 动作映射：0=下，1=上，2=左，3=右
        dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_x = max(0, min(GRID_SIZE - 1, x + dx))
        new_y = max(0, min(GRID_SIZE - 1, y + dy))

        done = False
        if [new_x, new_y] == list(self.goal):
            self.agent_pos = [new_x, new_y]
            reward = 100  # 到达目标
            done = True
            # 超出边界或者碰到障碍物
        elif self.grid[new_x][new_y] == 1 or new_x < 0 or new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE:
            reward = -10  # 碰撞障碍
        else:
            self.agent_pos = [new_x, new_y]
            reward = -0.1  # 移动惩罚
        self.render()  # 实时显示移动过程
        return self._get_state(), reward, done

    def _get_state(self):
        return tuple(self.agent_pos)  # 状态为坐标(x,y)

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
                elif (x, y) == self.goal:
                    # 绘制目标
                    pygame.draw.circle(self.screen, COLORS["goal"],
                                       (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
        # 绘制Agent
        pygame.draw.circle(self.screen, COLORS["agent"],
                           (self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                            self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 3)
        pygame.display.flip()
        self.clock.tick(100)  # 控制渲染帧率


def train():
    # 超参数
    alpha = 0.1  # 学习率
    gamma = 0.99  # 折扣因子
    epsilon = 0.2  # 探索率
    env = GridEnv()
    num_episodes = 100

    for _ in range(num_episodes):
        print("Episode:", _)
        state = env.reset()
        done = False
        Max_steps = 1000  # 最大步数
        while not done and Max_steps > 0:
            # ε-greedy选择动作
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state[0], state[1]])
            next_state, reward, done = env.step(action)
            # print("Action:", action, end=" || ")
            # print("Reward:", reward)
            env.render()  # 每一步动作之后调用render方法
            Q[state[0], state[1], action] += alpha * (
                    reward + gamma * (0 if done else np.max(Q[next_state[0], next_state[1]]))
                    - Q[state[0], state[1], action]
            )
            state = next_state
            Max_steps -= 1
        # 衰减epsilon（可选）
        epsilon *= 0.995
        print(f"step: {1000 - Max_steps}")
    # 假设Q表是形状为 (GRID_SIZE, GRID_SIZE, 4) 的NumPy数组
    # print("Q-table:", Q)
    np.save("q_table.npy", Q)  # 保存为二进制文件
    print("Training completed!")


# 使用训练后的策略运行
def run_policy():
    env = GridEnv()
    state = env.reset()
    done = False
    step = 0
    while not done:
        action = np.argmax(Q[state[0], state[1]])
        state, _, done = env.step(action)
        print("Step:", step, "Action:", action, "State:", state, "done:", done)
        env.render()  # 实时显示移动过程
        env.clock.tick(10)  # 控制速度
        step += 1
    print("Total steps:", step)


if __name__ == "__main__":
    # 初始化Q表：状态为(x,y)，动作为0-3
    if not os.path.exists("q_table.npy"):
        # 没有Q表时初始化全零
        Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    else:
        # 加载时直接读取
        Q = np.load("q_table.npy")
    # train()
    run_policy()
