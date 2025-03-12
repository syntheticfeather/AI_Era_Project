import pygame
import gym
import time
import random
import numpy as np

if __name__ == '__main__':
    np.bool8 = np.bool_  # 类型别名替换
    pygame.init() # 初始化pygame    
    
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()

    print(state) # 小车位置， 小车速度， 杆子的角度， 杆子的尖端速度

    time.sleep(1)


    StartTime = time.time() # 记录开始时间
    max_steps = 1000 # 最大步数
    for i in range(max_steps):
        time.sleep(0.5) # 控制每一步的间隔时间

        keys = pygame.key.get_pressed() # 获取按键信息

        action = 0
        if keys[pygame.K_LEFT]: # 按左箭头方向键
            action = 0
        elif keys[pygame.K_RIGHT]: # 按右箭头方向键
            action = 1

        state, _, done, _, _ = env.step(action) # 执行动作， 得到下一个状态， 奖励， 是否结束， 信息

        print("State:", state,"Action:", action) # 打印状态和动作

        if done: # 结束
            EndTime = time.time() # 记录结束时间
            print("Episode finished after {} timesteps".format(i+1)) # 打印步数
            print("total time:", EndTime - StartTime) # 打印总时间
            break


    env.close() # 关闭环境