# frontend.py
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from environment import Environment
from train import train_model
from test import test_model


class TrainingVisualizer:
    def __init__(self, master):
        self.master = master
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Training Progress")
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Average Reward")
        self.line, = self.ax.plot([], [])
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.episodes = []
        self.rewards = []

    def update_plot(self, episode, reward):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.line.set_data(self.episodes, self.rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


class GridWorldGUI:
    def __init__(self, master, grid_size=10):
        self.master = master
        self.grid_size = grid_size
        self.cell_size = 40
        self.obstacles = np.zeros((grid_size, grid_size), dtype=bool)

        # Training control
        self.training = False
        self.should_stop = False
        self.testing = False
        self.lock = threading.Lock()

        # UI initialization
        self.setup_ui()
        self.env = Environment(grid_size=grid_size)
        self.env.obstacles = self.obstacles
        self.visualizer = TrainingVisualizer(self.master)

    def setup_ui(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Control panel
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Buttons
        ttk.Button(control_frame, text="Init Obstacles", command=self.reset_obstacles).pack(pady=5)
        ttk.Button(control_frame, text="Start Training", command=self.start_training).pack(pady=5)
        self.stop_btn = ttk.Button(control_frame, text="Stop Training",
                                   command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        ttk.Button(control_frame, text="Run Test", command=self.start_testing).pack(pady=5)
        ttk.Button(control_frame, text="Clear Obstacles", command=self.clear_obstacles).pack(pady=5)

        # Status display
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack(pady=10)

        # Grid canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame,
                                width=self.grid_size * self.cell_size,
                                height=self.grid_size * self.cell_size)
        self.canvas.pack()
        self.draw_grid()

        # Event bindings
        self.canvas.bind("<Button-1>", self.toggle_obstacle)
        self.canvas.bind("<B1-Motion>", self.toggle_obstacle)

    def draw_grid(self):
        self.canvas.delete("all")
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                if self.obstacles[x, y]:
                    self.canvas.create_rectangle(y0, x0, y1, x1, fill="black")
                else:
                    self.canvas.create_rectangle(y0, x0, y1, x1, fill="white")
                self.canvas.create_rectangle(y0, x0, y1, x1, outline="gray")

    def toggle_obstacle(self, event):
        if not self.training and not self.testing:
            x = event.y // self.cell_size
            y = event.x // self.cell_size
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.obstacles[x, y] = not self.obstacles[x, y]
                self.env.obstacles[x, y] = self.obstacles[x, y]
                self.draw_grid()

    def reset_obstacles(self):
        self.obstacles = np.random.rand(self.grid_size, self.grid_size) < 0.2
        self.env.obstacles = self.obstacles.copy()
        self.draw_grid()

    def clear_obstacles(self):
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.env.obstacles = self.obstacles.copy()
        self.draw_grid()

    def start_training(self):
        if not self.training:
            self.training = True
            self.should_stop = False
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Training...")
            self.visualizer.episodes.clear()
            self.visualizer.rewards.clear()

            threading.Thread(target=self.run_training, daemon=True).start()

    def stop_training(self):
        with self.lock:
            self.should_stop = True
            self.status_var.set("Stopping...")

    def run_training(self):
        def training_callback(episode, total_reward):
            self.master.after(0, lambda:
            self.visualizer.update_plot(episode, total_reward))
            with self.lock:
                return self.should_stop

        try:
            train_model(self.env, callback=training_callback)
        finally:
            self.master.after(0, self.on_training_end)

    def on_training_end(self):
        self.training = False
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Training Completed")

    def start_testing(self):
        if not self.testing:
            self.testing = True
            self.status_var.set("Testing...")
            threading.Thread(target=self.run_testing, daemon=True).start()

    def run_testing(self):
        try:
            test_model(self.env)
        finally:
            self.master.after(0, lambda: self.status_var.set("Test Completed"))
            self.testing = False


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Grid World Trainer")
    app = GridWorldGUI(root)
    root.mainloop()
