from cgi import print_form

s = "abcdefg"
s[0] = "A"
print(s)

import gym
import numpy as np


class SimpleAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        print("observation = ", observation)
        try:
            position, velocity = observation[0]
        except:
            position, velocity = observation  # 版本差异

        print(f"position = {position}, velocity = {velocity}")
        lb = min([-0.09 * (position + 0.25) ** 2 + 0.03])
        ub = np.max([-0.07 * (position + 0.38) ** 2 + 0.07])
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass


def play(env, agent, render=False, train=False):
    episode_reward = 0.  # 记录回合总奖励，初始值为0
    observation = env.reset(seed=3)  # 重置游戏环境，开始新回合
    while True:  # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()  # 显示图形界面
        action = agent.decide(observation)
        next_observation, reward, terminated, truncated, _ = env.step(action)  # 执行动作
        print(next_observation)
        episode_reward += reward  # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward, terminated, truncated)  # 学习
        if terminated or truncated:  # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward  # 返回回合总奖励


if __name__ == '__main__':
    np.bool8 = np.bool  # 定义bool8类型 版本差异
    env = gym.make('MountainCar-v0', render_mode="human")

    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))
    print('观测范围 = {} ~ {}'.format(env.observation_space.low,
                                      env.observation_space.high))
    print('动作数 = {}'.format(env.action_space.n))

    agent = SimpleAgent(env)

    env.reset(seed=3)  # 设置随机种子，让结果可复现

    episode_reward = play(env, agent, render=True)
    print('回合奖励 = {}'.format(episode_reward))
    env.close()  # 关闭图形界面
