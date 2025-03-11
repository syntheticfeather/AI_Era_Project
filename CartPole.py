import gym
import numpy as np

# 运行前需要先安装python
# 安装requirement.txt中的依赖包 -> pip install -r requirement.txt
# 如果命令行运行python，则需要在命令行下进入到该文件所在的目录（cd 文件所在目录）下，然后运行python CartPole.py
# 如果vscode 或者 pycharm 直接配置环境运行。

def run_episode(env):
    env.reset() # 重置环境
    for _ in range(50):# 模拟次数，可调
        action = env.action_space.sample()  # 从行为库种随机抽取一个动作，0 向左 或者 1 向右
        observation, reward, terminated, truncated, info = env.step(action) # 也可直接改为 0 或 1 
        # 执行动作，得到环境反馈，返回的元组拆包
        # observation：当前状态（小车位置、速度、杆角度、角速度等）
        # reward：立即奖励（CartPole每存活一步固定+1）
        # terminated：是否达到终止条件（如杆子倾斜超过阈值）
        # truncated：是否因步数限制被中断（新版gym特性）
        # info：调试用的附加信息（通常为空字典）
        print(observation, "||",reward,"||", terminated,"||", truncated,"||", info)  # 打印环境反馈
        #-------------------





        #-------------------
        if terminated or truncated: # 如果环境终止，则重置环境
            env.reset()
    env.close()

def close_env(env):
    env.close()


if __name__ == '__main__':
    np.bool8 = np.bool_  # 类型别名替换
    env = gym.make('CartPole-v1', render_mode='human')  # 启动gym中的环境，环境名为v1

    # close_env(env)# 如果想关闭python的运行窗口，就调用这个函数，然后注释下面那个函数
    run_episode(env)# cartpole运行调用的函数