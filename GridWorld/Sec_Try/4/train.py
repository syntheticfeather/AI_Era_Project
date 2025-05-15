# train.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dqn_model import DQN
from replay_buffer import ReplayBuffer
import os
import math
import random

# 新增配置参数
MODEL_PATH = "dqn_model.pth"
RESUME_TRAINING = True  # 是否继续上次训练
TOTAL_EPISODES = 200000  # 总训练回合数（包括之前训练的）

BATCH_SIZE = 128
GAMMA = 0.95
LR = 3e-4
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 20000
TARGET_UPDATE = 200
NOISE_SCALE = 0.3  # 新增动作噪声

TT = 0


def train_model(env, callback=None):
    global TT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(10000)

    # 加载已有模型和训练状态
    start_episode = 0
    steps_done = 0

    if RESUME_TRAINING and os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        policy_net.load_state_dict(checkpoint['model'])
        target_net.load_state_dict(checkpoint['model'])  # 使用相同模型初始化target网络
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_episode = checkpoint['episode']
        steps_done = checkpoint['steps_done']
        print(f"从 {MODEL_PATH} 加载模型，继续训练从第 {start_episode} 回合开始")

    target_net.eval()

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # 添加动作噪声
                q_values = policy_net(state.unsqueeze(0).to(device))
                noise = torch.randn_like(q_values) * NOISE_SCALE
                return (q_values + noise).argmax().view(1)
        else:
            return torch.tensor([[random.randint(0, 3)]], device=device)

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

    # 训练循环
    for i_episode in range(start_episode, TOTAL_EPISODES):
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
        print(f"回合 {i_episode} 奖励 {total_reward}")

        if callback and callback(i_episode, total_reward):
            print("Training stopped early")
            TT = i_episode
            break

        # 定期更新target网络和保存模型
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

            # 保存完整训练状态（每100回合）
            if i_episode % 50 == 0:
                torch.save({
                    'episode': i_episode,
                    'model': policy_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'steps_done': steps_done
                }, MODEL_PATH)
                print(env.rate / (i_episode + 1))

    # 最终保存模型
    torch.save({
        'episode': TT,
        'model': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'steps_done': steps_done
    }, MODEL_PATH)
