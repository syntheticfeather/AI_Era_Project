# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dqn_model import DQN
from replay_buffer import ReplayBuffer

BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
TARGET_UPDATE = 10


def train_model(env, callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(10000)
    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = np.random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state.unsqueeze(0).to(device)).argmax().view(1)
        else:
            return torch.tensor([np.random.randint(0, 3)], device=device)

    def optimize_model():
        if len(replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        states = states.to(device)
        next_states = next_states.to(device)

        q_values = policy_net(states).gather(1, actions.unsqueeze(1))

        next_q_values = target_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values

        loss = F.mse_loss(q_values.squeeze(), expected_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    num_episodes = 1000
    for i_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = select_action(state)
            next_state, reward, done = env.step(action.item())
            total_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)

            optimize_model()

            state = next_state

            if done:
                break

        if callback and callback(i_episode, total_reward):
            print("提前终止训练")
            break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {i_episode}, Total reward: {total_reward}")

    torch.save(policy_net.state_dict(), "dqn_model.pth")
