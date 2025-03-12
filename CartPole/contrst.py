import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist


# 策略网络定义
class CartPolePolicy(nn.Module):
    def __init__(self):
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 输入层（4个状态特征）到隐藏层（128神经元）
        self.fc2 = nn.Linear(128, 2)  # 隐藏层到输出层（2个动作）
        self.dropout = nn.Dropout(p=0.6)  # Dropout层（训练时随机丢弃60%神经元）

    def forward(self, x):
        x = self.fc1(x)  # 全连接层
        x = F.relu(x)  # ReLU激活
        x = self.dropout(x)  # 应用Dropout（仅在训练模式生效）
        x = self.fc2(x)  # 输出层
        return F.softmax(x, dim=1)  # 转换为动作概率分布


# 策略梯度损失计算
def compute_Policy_Loss(n, log_p):
    r = []
    for i in range(n, 0, -1):  # 生成倒序回报序列（最新步奖励最高）
        r.append(i * 1.0)  # 假设每个步骤的奖励都是1
    r = torch.tensor(r)  # 转换为张量
    r = (r - r.mean()) / (r.std() + 1e-8)  # 标准化回报（加epsilon防止除零）
    loss = 0
    for pi, ri in zip(log_p, r):  # 遍历每个时间步
        loss += -pi * ri  # 策略梯度损失公式：-log_prob * reward
    return loss


# 训练函数
def train():
    np.bool8 = np.bool_  # numpy版本兼容性处理

    # 环境初始化
    env = gym.make('CartPole-v1')  # 创建CartPole环境
    state, _ = env.reset(seed=543)  # 重置环境并获取初始状态
    torch.manual_seed(543)  # 设置PyTorch随机种子

    # 模型和优化器
    policy = CartPolePolicy()  # 实例化策略网络
    optimizer = optim.Adam(policy.parameters(), lr=0.01)  # Adam优化器
    policy.train()  # 设置为训练模式（启用Dropout）

    # 训练参数
    max_episode = 1000  # 最大训练轮次
    max_action = 10000  # 单回合最大动作数
    max_steps = 5000  # 成功标准（超过此步数停止训练）

    # 训练循环
    for episode in range(max_episode):
        state, _ = env.reset()  # 重置环境
        step = 0  # 当前回合步数计数器
        log_p = []  # 存储动作的log概率

        # 单回合交互
        for _ in range(max_action):
            # 状态转张量（添加batch维度）
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)

            # 获取动作概率
            action_probs = policy(state_tensor)  # 前向传播
            m = dist.Categorical(action_probs)  # 创建分类分布
            action = m.sample()  # 采样动作
            log_prob = m.log_prob(action)  # 获取log概率
            log_p.append(log_prob)  # 记录log概率

            # 执行动作
            state, _, done, _, _ = env.step(action.item())
            step += 1  # 步数递增

            # 环境终止判断
            if done:
                break

        # 提前终止条件
        if step > max_steps:
            print("达到最大步数，提前终止训练")
            break

        # 空episode处理
        if len(log_p) == 0:
            continue

        # 参数更新
        optimizer.zero_grad()  # 梯度清零
        loss = compute_Policy_Loss(step, log_p)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 训练日志
        if episode % 10 == 0:
            print(f"回合: {episode}, 步数: {step}, 损失: {loss.item()}")

    # 保存模型
    torch.save(policy.state_dict(), "CartPolePolicy.pth")
    env.close()


# 测试函数
def test_policy():
    np.bool8 = np.bool_  # numpy版本兼容性处理

    # 创建可视化环境
    env = gym.make('CartPole-v1', render_mode='human')  # 启用图形界面
    policy = CartPolePolicy()

    # 加载训练好的模型
    policy.load_state_dict(torch.load("CartPolePolicy.pth"))
    policy.eval()  # 设置为评估模式（禁用Dropout）

    # 运行测试回合
    state, _ = env.reset()
    total_reward = 0

    with torch.no_grad():  # 禁用梯度计算
        for _ in range(500):  # CartPole-v1的最大步数限制
            # 状态转张量
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)

            # 获取动作概率
            action_probs = policy(state_tensor)

            # 选择最大概率动作（确定性策略）
            action = torch.argmax(action_probs).item()

            # 执行动作
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # 环境终止判断
            if done:
                break

    print(f"总奖励: {total_reward}")
    env.close()


# 主程序
if __name__ == '__main__':
    # train()   # 取消注释以进行训练
    test_policy()  # 运行测试（需先训练生成模型文件）
