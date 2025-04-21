import random
import time

from brain import DQN
from Record import ReplayBuffer
from Environmnet import Env
import torch


class Agent(DQN):
    def __init__(self, input_size, hidden_size, output_size, epsilon_delay, min_epsilon, gamma,
                 device, learning_rate, update_step=5):
        super().__init__(input_size, hidden_size, output_size, epsilon_delay, min_epsilon, gamma,
                         device, learning_rate, update_step)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        action_values = self.get_q_value(state)
        if random.uniform(0, 1) < self.epsilon or torch.all(action_values == action_values[0][0]):
            action = random.choice(range(4))
        else:
            action = action_values.max(1)[1].item()
        return action

    def test(self):
        self.epsilon = 0.1
        self.q_net.eval()


def train():
    agent = Agent(input_size, hidden_size, output_size, epsilon_delay, min_epsilon, gamma, device, learning_rate, update_steps)
    env = Env()
    replay_buffer = ReplayBuffer(capacity)
    reward_positive_count = 0
    for episode in range(episodes):
        state = env.reset()
        is_terminate = True

        while is_terminate:
            action = agent.choose_action(state)
            next_state, action, reward, done, info = env.step(action)

            if info["end"]:
                replay_buffer.add_important(state, action, reward, next_state, done)
                is_terminate = False
                print(f"第{episode}次学习, 总共花费{env.step_counter}步 奖励为{env.reward}")
                if env.reward > 0:
                    reward_positive_count += 1
                else:
                    reward_positive_count -= 1
                    reward_positive_count = max(reward_positive_count, 0)
            else:
                replay_buffer.add_common(state, action, reward, next_state, done)

            env.render()
            state = next_state

            if len(replay_buffer) > min_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                # 构造训练集
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'done': d
                }
                agent.update(transition_dict)

        if reward_positive_count > 20:
            break

    env.close()
    return agent


def test(agent: Agent):
    env = Env()
    agent.test()
    state = env.reset()
    while True:
        action = agent.choose_action(state)
        next_state, action, reward, done, info = env.step(action)
        state = next_state
        env.render()
        time.sleep(0.2)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    capacity = 2000
    input_size = 2
    hidden_size = 128
    output_size = 4
    epsilon_delay = 0.995
    min_epsilon = 0.2

    gamma = 0.9
    learning_rate = 0.1
    episodes = 500

    update_steps = 100
    min_size = 256
    batch_size = 128
    agent = train()
    test(agent)

