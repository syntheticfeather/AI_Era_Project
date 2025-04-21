import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        # 随机生成起始点和目标点
        self.agent_pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        self.goal_pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        while self.agent_pos == self.goal_pos:
            self.goal_pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
        return self._get_state()

    def _get_state(self):
        # 状态表示为一个二维数组，其中1表示智能体位置，2表示目标位置
        state = np.zeros((self.width, self.height))
        state[self.agent_pos] = 1
        state[self.goal_pos] = 2
        return state

    def step(self, action):
        # 动作空间：0=上，1=下，2=左，3=右
        x, y = self.agent_pos
        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and y < self.height - 1:
            y += 1
        elif action == 2 and x > 0:
            x -= 1
        elif action == 3 and x < self.width - 1:
            x += 1

        self.agent_pos = (x, y)
        new_state = self._get_state()

        # 奖励函数
        if self.agent_pos == self.goal_pos:
            reward = 10  # 到达目标点
            done = True
        else:
            reward = -1  # 每一步的惩罚
            done = False

        return new_state, reward, done

    def render(self):
        # 可视化网格世界
        state = self._get_state()
        plt.imshow(state, cmap='viridis')
        plt.show()


class Replay_Buffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, state_shape, num_actions, learning_rate=0.001):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.state_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)

    def update_target_network(self, target_network):
        target_network.model.set_weights(self.model.get_weights())


def train_dqn(env, dqn, target_dqn, replay_buffer, num_episodes=1000, batch_size=32, gamma=0.99, epsilon=1.0,
              epsilon_decay=0.995, min_epsilon=0.01):
    for episode in range(num_episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        total_reward = 0

        while not done:
            # ε-贪心策略选择动作
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, env.num_actions - 1)
            else:
                q_values = dqn.predict(state)
                action = np.argmax(q_values[0])

            # 执行动作
            next_state, reward, done = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            total_reward += reward

            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)

            # 经验回放
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                targets = dqn.predict(states)
                next_q_values = target_dqn.predict(next_states)
                for i in range(batch_size):
                    if dones[i]:
                        targets[i, actions[i]] = rewards[i]
                    else:
                        targets[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
                dqn.train(states, targets)

            # 更新状态
            state = next_state

        # 更新目标网络
        if episode % 10 == 0:
            dqn.update_target_network(target_dqn)

        # 衰减探索率
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")


# 主程序
if __name__ == "__main__":
    # 初始化环境
    env = GridWorld(width=5, height=5)
    env.num_actions = 4  # 上、下、左、右

    # 初始化DQN和目标网络
    state_shape = (env.width, env.height)
    dqn = DQN(state_shape, env.num_actions)
    target_dqn = DQN(state_shape, env.num_actions)

    # 初始化经验回放缓存
    replay_buffer = Replay_Buffer(capacity=10000)

    # 训练DQN
    train_dqn(env, dqn, target_dqn, replay_buffer, num_episodes=1000)

    # 测试智能体
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    done = False
    while not done:
        q_values = dqn.predict(state)
        action = np.argmax(q_values[0])
        next_state, _, done = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        state = next_state
        env.render()
