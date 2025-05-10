# test.py
import torch
import pygame
from pygame.locals import *
import sys
from environment import Environment
from dqn_model import DQN


class Tester:
    def __init__(self, model_path="dqn_model.pth"):
        # 初始化环境
        self.env = Environment(grid_size=10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = DQN().to(self.device)
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model'])
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            sys.exit(1)

        self.model.eval()

        # 测试参数
        self.test_episodes = 20  # 测试回合数
        self.render_delay = 50  # 渲染延迟(ms)

    def draw_local_obs(self, state):
        """绘制局部观察窗口"""
        obs_size = 100  # 观察窗口大小
        panel = pygame.Surface((obs_size, obs_size))
        pygame.transform.scale(panel, (obs_size, obs_size))

        # 绘制4x4网格
        cell = obs_size // 4
        for i in range(4):
            for j in range(4):
                color = (200, 200, 200)
                if state[0, i, j] == 1:
                    color = (0, 0, 0)
                pygame.draw.rect(panel, color, (j * cell, i * cell, cell - 1, cell - 1))

                if state[1, i, j] == 1:
                    pygame.draw.circle(panel, (255, 0, 0),
                                       (j * cell + cell // 2, i * cell + cell // 2), cell // 3)
                if state[2, i, j] == 1:
                    pygame.draw.circle(panel, (0, 255, 0),
                                       (j * cell + cell // 2, i * cell + cell // 2), cell // 3)

        # 显示在窗口右上角
        self.env.screen.blit(panel, (self.env.window_size - obs_size - 10, 10))

    def draw_info_panel(self, episode, total_reward, steps):
        """绘制信息面板"""
        font = pygame.font.SysFont('arial', 18)
        info = [
            f"Episode: {episode + 1}/{self.test_episodes}",
            f"Total Reward: {total_reward:.1f}",
            f"Steps: {steps}"
        ]
        y_offset = 10
        for text in info:
            surface = font.render(text, True, (0, 0, 0))
            self.env.screen.blit(surface, (self.env.window_size + 10, y_offset))
            y_offset += 30

    def run(self):
        """执行测试"""
        success_count = 0
        total_steps = 0

        for episode in range(self.test_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 100:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            pygame.quit()
                            sys.exit()

                # 扩展窗口显示信息
                self.env.screen.fill((255, 255, 255),
                                     (self.env.window_size, 0, 200, self.env.window_size))
                self.draw_info_panel(episode, total_reward, steps)

                pygame.display.flip()

                # 模型预测
                with torch.no_grad():
                    action = self.model(state.unsqueeze(0).to(self.device)).argmax().item()

                # 执行动作
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                state = next_state

                pygame.time.delay(self.render_delay)  # 控制渲染速度

                if done:
                    if reward > 0:  # 成功到达目标
                        success_count += 1
                        print(f"Episode {episode + 1}: 成功! 奖励: {total_reward:.1f} 步数: {steps}")
                    else:  # 碰撞障碍物
                        print(f"Episode {episode + 1}: 失败! 奖励: {total_reward:.1f} 步数: {steps}")
                    total_steps += steps

        # 显示最终统计
        print("\n=== 测试结果 ===")
        print(f"成功率: {success_count / self.test_episodes * 100:.1f}%")
        print(f"平均步数: {total_steps / self.test_episodes:.1f}")

        pygame.quit()


if __name__ == "__main__":
    tester = Tester()
    tester.run()
