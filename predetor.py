import numpy as np
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')  # 设置matplotlib的风格

SIZE = 10  # 游戏区域的大小
EPISODES = 30000  # 局数
SHOW_EVERY = 3000  # 定义每隔多少局展示一次图像

FOOD_REWARD = 25  # agent获得食物的奖励
ENEMY_PENALITY = 300  # 遇上对手的惩罚
MOVE_PENALITY = 1  # 每移动一步的惩罚

epsilon = 0.6
EPS_DECAY = 0.9998
DISCOUNT = 0.95
LEARNING_RATE = 0.1

q_table = None
# 设定三个部分的颜色分别是蓝、绿、红
d = {1: (255, 0, 0),  # blue
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3


# 智能体的类，有其 位置信息 和 动作函数
class Cube:
    def __init__(self):  # 随机初始化位置坐标
        self.x = np.random.randint(0, SIZE - 1)
        self.y = np.random.randint(0, SIZE - 1)

    def __str__(self):
        return f'{self.x},{self.y}'

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self, choise):
        if choise == 0:
            self.move(x=1, y=1)
        elif choise == 1:
            self.move(x=-1, y=1)
        elif choise == 2:
            self.move(x=1, y=-1)
        elif choise == 3:
            self.move(x=-1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        if self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        if self.y > SIZE - 1:
            self.y = SIZE - 1


# 初始化Q表格
if q_table is None:  # 如果没有实现提供，就随机初始化一个Q表格
    q_table = {}
    for x1 in range(-SIZE + 1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.randint(-5, 0) for i in range(4)]
else:  # 提供了，就使用提供的Q表格
    with open(q_table, 'rb') as f:
        q_table = pickle.load(f)
