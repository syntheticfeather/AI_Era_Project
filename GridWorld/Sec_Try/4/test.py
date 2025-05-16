# test.py
import torch
import pygame
from pygame.locals import *
import sys
from environment import Environment
from dqn_model import DQN


class Tester:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN().to(self.device)
        try:
            checkpoint = torch.load("dqn_model.pth")
            self.model.load_state_dict(checkpoint['model'])
            print("Model loaded successfully")
        except:
            print("No trained model found, using random policy")

        self.model.eval()
        self.render_delay = 50

    def run(self, stop_callback=None, status_callback=None):
        success_count = 0
        total_steps = 0
        test_episodes = 10

        for episode in range(test_episodes):
            if stop_callback and stop_callback():
                break

            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and not (stop_callback and stop_callback()):
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()

                self.env.render()
                pygame.time.delay(self.render_delay)

                with torch.no_grad():
                    action = self.model(state.unsqueeze(0).to(self.device)).argmax().item()

                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1
                state = next_state

                if status_callback:
                    status_callback(episode, steps, self.env.agent_pos)

                if done:
                    if reward > 0:
                        success_count += 1
                    total_steps += steps
                    break

        final_status = (f"Success Rate: {success_count / test_episodes * 100:.1f}%\n"
                        f"Average Steps: {total_steps / test_episodes:.1f}")
        if status_callback:
            status_callback(final_status)
        print(final_status)
        return final_status


class InstallTester:
    def __init__(self, env, start_pos, target_pos):
        self.env = env
        self.start_pos = start_pos
        self.targets = target_pos
        self.env.reset(keep_pos=True, clear_visited=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型（与Tester保持一致）
        self.model = DQN().to(self.device)
        try:
            checkpoint = torch.load("dqn_model.pth")
            self.model.load_state_dict(checkpoint['model'])
            print("Model loaded successfully")
        except:
            print("No trained model found, using random policy")

        self.model.eval()
        self.render_delay = 50

        # 安装测试专用属性
        self.current_target_idx = 0
        self.success_count = 0
        self.total_steps = 0

    def run(self, stop_callback=None, status_callback=None):
        self.env.agent_pos = self.start_pos  # 设置固定起始位置

        while self.current_target_idx < len(self.targets) and not (stop_callback and stop_callback()):
            # 设置当前目标
            target = self.targets[self.current_target_idx]
            self.env.target_pos = target

            # 重置环境保持位置
            state = self.env.reset(keep_pos=True, clear_visited=False)
            done = False
            steps = 0

            # 单目标测试循环
            while not done and not (stop_callback and stop_callback()):
                # 处理Pygame事件
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        pygame.quit()
                        sys.exit()

                # 渲染环境
                self.env.render()
                pygame.time.delay(self.render_delay)

                # 模型推理
                with torch.no_grad():
                    action = self.model(state.unsqueeze(0).to(self.device)).argmax().item()

                # 执行动作
                next_state, reward, done = self.env.step(action)
                steps += 1
                state = next_state
                if steps > 1000:  # 超时退出
                    print("time out")
                    break
                # 更新状态回调
                if status_callback:
                    status_callback(steps, self.env.agent_pos)

                # 到达目标或超时
                if done:
                    self.total_steps += steps
                    if reward > 0:
                        self.success_count += 1
                    self.current_target_idx += 1  # 移动到下一个目标
                    break

        # 生成最终报告
        final_status = (f"Install Test Complete\n"
                        f"Reached {self.success_count}/{len(self.targets)} targets\n"
                        f"Average Steps: {self.total_steps / max(len(self.targets), 1):.1f}")
        if status_callback:
            status_callback(final_status)
        print(final_status)
        return final_status


def test_model(env):
    tester = Tester(env)
    tester.run()
