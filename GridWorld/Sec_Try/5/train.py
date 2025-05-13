# train.py (支持模型续训版本)
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from environment import Environment
from dqn_model import DQN
from replay_buffer import ReplayBuffer

# 新增配置参数
MODEL_PATH = "dqn_model.pth"
RESUME_TRAINING = True  # 是否继续上次训练
TOTAL_EPISODES = 20000  # 总训练回合数（包括之前训练的）

BATCH_SIZE = 64
GAMMA = 0.999
LR = 1e-4
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 10000
TARGET_UPDATE = 10

# 初始化环境和设备
env = Environment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和优化器
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
    global steps_done
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


# 训练循环
for i_episode in range(start_episode, TOTAL_EPISODES):
    state = env.reset()
    total_reward = 0
    if i_episode >= TOTAL_EPISODES / 10 and i_episode % 500 == 0:
        env.obstacles = np.random.rand(env.grid_size, env.grid_size) < env.obstacle_density
        # 每100回合更换环境布局（修改这里）
    if i_episode % 100 == 0 and i_episode != start_episode and i_episode > 2000:
        env.generate_connected_obstacles()  # 生成新连通地图
        print(f"第 {i_episode} 回合更换环境布局")
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
    # 定期更新target网络和保存模型
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # 保存完整训练状态（每100回合）
    if (i_episode) % 100 == 0:
        torch.save({
            'episode': i_episode,
            'model': policy_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'steps_done': steps_done
        }, MODEL_PATH)
        print(env.rate / (i_episode + 1))

# 最终保存模型
torch.save({
    'episode': TOTAL_EPISODES,
    'model': policy_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'steps_done': steps_done
}, MODEL_PATH)
