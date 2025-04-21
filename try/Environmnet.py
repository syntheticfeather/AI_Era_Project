import tkinter as tk
import random
import numpy as np
import copy
import time

maze_array = np.array(
    [
        [3, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ]
)

barriers = np.transpose(np.where(maze_array == 1)).tolist()
target = np.transpose(np.where(maze_array == 2)).tolist()
start_pos = np.transpose(np.where(maze_array == 3)).tolist()[0]
target_pos = target[0]

print(barriers, target, start_pos, sep="\n")

row, column = maze_array.shape
row_sep = 50
column_sep = 50

color_map = {0: 'white', 1: 'black', 2: "yellow", 3: "red"}


class Env(tk.Tk):
    def __init__(self):
        super().__init__()
        self.maze = copy.copy(maze_array)
        self.widget_dict = {}
        self.step_counter = 0
        self.env_info = {"end": False, "action": None, "target": False}
        self.state = copy.copy(start_pos)
        self.reward = 0

        self.focus_pos()
        self.create_widget()

    def focus_pos(self):
        # 设置窗口大小
        window_width = (column + 1) * column_sep
        window_height = (row + 1) * row_sep
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width - window_width) / 2)
        y = int((screen_height - window_height) / 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_widget(self):
        state_show_label = tk.Label(self, text="", fg="black")
        state_show_label.pack()
        self.widget_dict[2] = state_show_label

        width = column * column_sep
        height = row * row_sep
        board = tk.Canvas(self, width=width, height=height, bg='white')
        board.pack()
        self.widget_dict[1] = board
        self.generate_maze()

    def reset(self):
        self.state = copy.copy(start_pos)
        self.maze = copy.deepcopy(maze_array)
        self.env_info = {"end": False, "action": None}
        self.step_counter = 0
        self.reward = 0

        self.render()

        return copy.copy(self.state)

    def step(self, action):
        "action: 0:left 1:right 2:up 3:down"
        reward = (-(self.state[0] - target_pos[0])**2 - (self.state[1] - target_pos[1])**2) / 5
        self.step_counter += 1
        done = False
        self.maze[tuple(self.state)] = 0
        self.env_info["action"] = action

        if action == 0:
            if self.state[1] == 0:
                reward -= 100
                done = True
            else:
                self.state[1] -= 1
        elif action == 1:
            if self.state[1] == column - 1:
                reward -= 100
                done = True
            else:
                self.state[1] += 1
        elif action == 2:
            if self.state[0] == 0:
                reward -= 100
                done = True
            else:
                self.state[0] -= 1
        elif action == 3:
            if self.state[0] == row - 1:
                reward -= 100
                done = True
            else:
                self.state[0] += 1
        else:
            raise Exception("action range in 0 to 3")

        if self.state in target:
            self.env_info["end"] = True
            done = True
            reward += 500

        if self.state in barriers:
            self.env_info["end"] = True
            done = True
            reward -= 300

        # role movement
        self.maze[tuple(self.state)] = 3
        # remove calculation
        self.reward += reward
        # step over
        if self.step_counter > 2000:
            self.env_info["end"] = True

        "返回observation action reward done env_info"
        return copy.copy(self.state), action, reward, done, copy.deepcopy(self.env_info)

    def render(self):
        # 用于渲染环境
        self.generate_maze()
        if self.env_info["end"]:
            font_color = 'red'
        else:
            font_color = 'black'
        self.widget_dict[2].config(text=f"steps: {self.step_counter}"
                                        f" action: {self.env_info['action']}", fg=font_color)
        self.update()

        if self.env_info["end"]:
            time.sleep(0.2)
            self.reset()

    def close(self):
        self.destroy()

    def generate_maze(self):
        board = self.widget_dict[1]
        board.delete("all")
        for column_idx, maze_row in enumerate(self.maze):
            for row_idx, item in enumerate(maze_row):
                start_pos = (row_idx * row_sep, column_idx * column_sep)
                end_pos = ((row_idx + 1) * row_sep, (column_idx + 1) * column_sep)
                if item == 2:
                    board.create_rectangle(start_pos, end_pos, fill='white')
                    board.create_oval(start_pos, end_pos, fill=color_map[item])
                    continue
                else:
                    board.create_rectangle(start_pos, end_pos, fill=color_map[item])

    def random_walk(self, steps=10):
        for i in range(steps):
            action = random.choice(range(4))
            self.step(action)
            self.render()
            time.sleep(0.2)


if __name__ == '__main__':
    env = Env()
    env.random_walk(steps=200)

