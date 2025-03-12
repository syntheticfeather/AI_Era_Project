import gym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist


# 运行前需要先安装python
# 安装requirement.txt中的依赖包 -> pip install -r requirement.txt
# 如果命令行运行python，则需要在命令行下进入到该文件所在的目录（cd 文件所在目录）下，然后运行python CartPole.py
# 如果vscode 或者 pycharm 直接配置环境运行。
class CartPolePolicy(nn.Module):
    def __init__(self):  # 对象初始化
        super(CartPolePolicy, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # 四个输入层， 中间128个隐藏层
        self.fc2 = nn.Linear(128, 2)  # 128个隐藏层， 输出两个值， 对应左右移动
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):  # 前向传播
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)  # 输出softmax值， 对应左右移动的概率


def Compute_Policy_Loss(n, Log_p):
    r = list()

    for i in range(n, 0, -1):
        r.append(i * 1.0)
    r = torch.tensor(r)
    if len(r) > 1:  # 确保有多个奖励来计算标准差
        r = (r - r.mean()) / r.std()
    else:
        r = r - r.mean()  # 如果只有一个奖励，避免除以零
    loss = 0
    for pi, ri in zip(Log_p, r):
        loss += -pi * ri
    return loss


if __name__ == '__main__':
    np.bool8 = np.bool_  # 类型别名替换，版本冲突问题

    env = gym.make('CartPole-v1')
    state, _ = env.reset(seed=543)
    torch.manual_seed(543)  # 设置随机种子

    policy = CartPolePolicy()  # 实例化策略网络
    optimizer = optim.Adam(policy.parameters(), lr=0.01)  # 优化器
    # policy.load_state_dict(torch.load("CartPolePolicy.pth")) # 加载模型参数
    policy.train()  # 测试模式

    StartTime = time.time()  # 记录开始时间
    max_episode = 1000  # 最大回合数
    max_action = 10000  # 最大动作次数
    max_steps = 5000  # 完成训练的步数

    for episode in range(max_episode):
        state, _ = env.reset()  # 重置环境
        step = 0
        log_p = list()

        for i in range(max_action):

            state = torch.from_numpy(state).float().unsqueeze(0)  # 转换为张量

            action_probs = policy(state)  # 得到动作概率

            m = dist.Categorical(action_probs)  # 实例化Categorical分布
            action = m.sample()  # 采样动作

            state, _, done, _, _ = env.step(action.item())  # 执行动作， 得到下一个状态， 奖励， 是否结束， 信息

            if done:  # 结束
                break
            log_p.append(m.log_prob(action))  # 记录动作的log概率
            step += 1  # 步数+1

        if step > max_steps:  # 超过最大步数
            print("Reach max steps, can play more")
            print("Episode:", episode, "Step:", step, "Loss:", loss.item())
            break
        optimizer.zero_grad()  # 清空梯度
        loss = Compute_Policy_Loss(step, log_p)  # 计算策略损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        if episode % 10 == 0:  # 每10轮保存一次模型
            print("Episode:", episode, "Step:", step, "Loss:", loss.item())

    torch.save(policy.state_dict(), "CartPolePolicy.pth")
    env.close()  # 关闭环境
