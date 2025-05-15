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


def test_model(env):
    tester = Tester(env)
    tester.run()
