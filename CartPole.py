import gym
import pygame
import numpy as np
import pygame

# 初始化 Pygame
pygame.init()

# 设置窗口大小
width, height = 800, 600
screen = pygame.display.set_mode((width, height))  # 开启窗口
pygame.display.set_caption('CartPole Game')

cart_width, cart_height = 50, 20  # 设置cart与pole 的大小参数
pole_length, pole_width = 150, 6
cart_x = width // 2
cart_y = height - cart_height - 10
pole_angle = 0

running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 检测键盘事件
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        # 向左移动小车
        cart_x -= 5
        pole_angle -= 0.1  # 向左移动时，杆的角度变小（假设为负方向）
    if keys[pygame.K_RIGHT]:
        # 向右移动小车
        cart_x += 5
        pole_angle += 0.1  # 向右移动时，杆的角度变大（假设为正方向）
        
    # 检查杆的角度是否超过限制
    if abs(pole_angle) > np.radians(12):
        running = False
    # 清空屏幕
    screen.fill((255, 255, 255))

    # 画小车
    pygame.draw.rect(screen, (0, 0, 0),
                     (cart_x - cart_width // 2, cart_y, cart_width, cart_height))

    # 画杆
    pole_end_x = cart_x + pole_length * np.sin(pole_angle)
    pole_end_y = cart_y - pole_length * np.cos(pole_angle)
    pygame.draw.line(screen, (0, 0, 0), (cart_x, cart_y),
                     (pole_end_x, pole_end_y), pole_width)

    # 更新显示
    pygame.display.flip()

    # 控制帧率
    clock.tick(30)

pygame.quit()
