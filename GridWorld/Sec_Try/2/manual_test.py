from environment import Environment
import numpy as np
import pygame

env = Environment()
env.obstacles = np.zeros((10, 10), dtype=bool)  # 清空障碍物
env.agent_pos = (5, 5)
env.target_pos = (2, 2)

while True:
    env.render()
    pygame.time.delay(1000)
