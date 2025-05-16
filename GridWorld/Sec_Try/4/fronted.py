import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import matplotlib
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from environment import Environment
from train import train_model
import test


class TrainingVisualizer:
    def __init__(self, master, show_visualizer=True):  # 新增显示控制参数
        self.master = master
        self.show = show_visualizer  # 保存显示状态

        # 初始化图形组件
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Training Progress")
        self.ax.set_ylabel("Average Reward")
        self.line, = self.ax.plot([], [])

        # 创建画布但暂不显示
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.widget = self.canvas.get_tk_widget()

        # 根据显示参数决定初始状态
        if self.show:
            self.widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 数据存储
        self.episodes = []
        self.rewards = []

    def set_visibility(self, visible):
        """设置可视化显示状态的方法"""
        self.show = visible
        if self.show:
            self.widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        else:
            self.widget.pack_forget()  # 移除画布显示

    def update_plot(self, episode, reward):
        """更新图表（无论是否显示都保持更新）"""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.line.set_data(self.episodes, self.rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        if self.show:  # 仅在显示状态下刷新
            self.canvas.draw()


class GridWorldGUI:
    def __init__(self, master, grid_size=10, show=True):
        self.master = master
        self.grid_size = grid_size
        self.cell_size = 80
        self.obstacles = np.zeros((grid_size, grid_size), dtype=bool)

        # 加载地图数据
        self.obstacles = self.load_or_create_map()

        # 初始化环境时传入障碍物
        self.env = Environment(
            grid_size=grid_size,
            obstacles=self.obstacles,
            render_mode="human"
        )
        self.install_mode = True
        self.custom_start = None
        self.custom_targets = []
        # 颜色配置
        self.colors = {
            "obstacle": "#2c3e50",
            "empty": "#ecf0f1",
            "grid_line": "#bdc3c7"
        }
        # 新增属性存储自定义起点和目标点
        self.custom_start = None
        self.custom_targets = []
        # 训练控制
        self.training = False
        self.should_stop = False
        self.testing = False
        self.lock = threading.Lock()
        # 初始化界面
        self.setup_ui()
        self.env = Environment(grid_size=grid_size)
        self.env.obstacles = self.obstacles
        self.visualizer = TrainingVisualizer(self.right_panel, show_visualizer=show)
        # 立即绘制加载的地图
        self.draw_loaded_map()

    def load_or_create_map(self):
        """加载或创建新地图"""
        try:
            if os.path.exists('saved_map.npy'):
                obstacles = np.load('saved_map.npy')
                if obstacles.shape == (self.grid_size, self.grid_size):
                    return obstacles
        except Exception as e:
            ...
        # 创建新地图
        return np.zeros((self.grid_size, self.grid_size), dtype=bool)

    def draw_loaded_map(self):
        """绘制已加载的地图"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                fill_color = self.colors["obstacle"] if self.obstacles[x, y] else self.colors["empty"]
                self.canvas.itemconfig(self.grid_rects[x, y], fill=fill_color)
        self.canvas.update()

    def save_current_map(self):
        """保存当前地图状态"""
        Environment.save_map(self.obstacles)

    def setup_ui(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        control_frame = ttk.Frame(main_frame, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 中间网格画布
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame,
                                width=self.grid_size * self.cell_size,
                                height=self.grid_size * self.cell_size)
        self.canvas.pack()
        # 初始化网格显示（基于加载的地图数据）
        self.grid_rects = np.empty((self.grid_size, self.grid_size), dtype=object)
        self.init_grid()

        # 右侧训练进度面板
        self.right_panel = ttk.Frame(main_frame, width=300)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 按钮
        ttk.Button(control_frame, text="Init Obstacles", command=self.reset_obstacles).pack(pady=5)
        ttk.Button(control_frame, text="Start Training", command=self.start_training).pack(pady=5)
        self.stop_train_btn = ttk.Button(control_frame, text="Stop Training",
                                         command=self.stop_training, state=tk.DISABLED)
        self.stop_train_btn.pack(pady=5)
        ttk.Button(control_frame, text="Run Test", command=self.start_testing).pack(pady=5)
        self.stop_test_btn = ttk.Button(control_frame, text="Stop Test",
                                        command=self.stop_testing, state=tk.DISABLED)
        self.stop_test_btn.pack(pady=5)
        ttk.Button(control_frame, text="Clear Map", command=self.clear_map).pack(pady=5)
        # 新增Install按钮
        ttk.Button(control_frame, text="Install", command=self.start_install_test).pack(pady=5)
        # 状态显示
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var).pack(pady=10)

        # 修改事件绑定
        self.canvas.bind("<Button-1>", self.toggle_cell)
        self.canvas.bind("<Button-2>", self.set_start_pos)  # 中键设置起点
        self.canvas.bind("<Button-3>", self.add_target_pos)  # 右键添加目标点
        self.canvas.bind("<B1-Motion>", self.toggle_cell)

    # 新增方法
    def set_start_pos(self, event):
        """中键设置起始位置"""
        # 转换坐标
        x = event.y // self.cell_size
        y = event.x // self.cell_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # 清除旧起点
            if self.custom_start:
                old_x, old_y = self.custom_start
                self.canvas.itemconfig(self.grid_rects[old_x, old_y],
                                       fill=self.colors["empty"])

            # 设置新起点
            self.custom_start = (x, y)
            self.canvas.itemconfig(self.grid_rects[x, y], fill="#FFA500")  # 橙色表示起点

    def add_target_pos(self, event):
        """右键添加目标点"""
        # 转换坐标
        x = event.y // self.cell_size
        y = event.x // self.cell_size
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # 如果是障碍物则清除
            if (x, y) in self.custom_targets:
                # 移除目标点
                self.custom_targets.remove((x, y))
                self.canvas.itemconfig(self.grid_rects[x, y], fill=self.colors["empty"])
            else:
                # 添加目标点（最多5个）
                if (x, y) not in self.custom_targets and len(self.custom_targets) < 5:
                    self.custom_targets.append((x, y))
                    self.canvas.itemconfig(self.grid_rects[x, y], fill="#4CAF50")  # 绿色

    def init_grid(self):
        """初始化网格并存储矩形引用"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                fill_color = self.colors["obstacle"] if self.obstacles[x, y] else self.colors["empty"]
                # 创建矩形并存储引用
                rect = self.canvas.create_rectangle(
                    y0, x0, y1, x1,
                    fill=self.colors["empty"],
                    outline=self.colors["grid_line"]
                )
                self.grid_rects[x, y] = rect

    def start_install_test(self):
        """执行安装测试"""
        if not self.custom_start or len(self.custom_targets) == 0:
            self.status_var.set("Please set start and targets first!")
            return

        if not self.testing:
            self.testing = True
            self.should_stop_testing = False
            self.stop_test_btn.config(state=tk.NORMAL)
            self.status_var.set("Install Testing...")

            # 传递自定义配置给测试器
            threading.Thread(target=self.run_install_test, daemon=True).start()

    def run_install_test(self):
        try:
            # 自定义测试器
            tester = test.InstallTester(
                self.env,
                start_pos=self.custom_start,
                target_pos=self.custom_targets
            )
            tester.run(
                stop_callback=lambda: self.should_stop_testing,
                status_callback=lambda *args: self.update_test_status(*args)
            )
        finally:
            self.master.after(0, self.on_testing_end)

    def toggle_cell(self, event):
        """切换单元格状态（障碍物/空地）"""
        if not self.training and not self.testing:
            # 转换坐标
            x = event.y // self.cell_size
            y = event.x // self.cell_size

            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                # 切换状态
                self.obstacles[x, y] = not self.obstacles[x, y]
                self.env.obstacles[x, y] = self.obstacles[x, y]

                # 更新颜色
                fill_color = self.colors["obstacle"] if self.obstacles[x, y] else self.colors["empty"]
                self.canvas.itemconfig(self.grid_rects[x, y], fill=fill_color)

    def reset_obstacles(self):
        """随机生成障碍物"""
        self.obstacles = np.random.rand(self.grid_size, self.grid_size) < 0.2
        self.env.obstacles = self.obstacles.copy()
        self.update_grid_colors()

    def clear_map(self):
        """清除所有障碍物"""
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.env.obstacles = self.obstacles.copy()
        self.update_grid_colors()

    def update_grid_colors(self):
        """根据障碍物矩阵更新所有单元格颜色"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                fill_color = self.colors["obstacle"] if self.obstacles[x, y] else self.colors["empty"]
                self.canvas.itemconfig(self.grid_rects[x, y], fill=fill_color)

    def start_testing(self):
        if not self.testing:
            self.testing = True
            self.should_stop_testing = False
            self.stop_test_btn.config(state=tk.NORMAL)
            self.status_var.set("Random Testing...")
            threading.Thread(target=self.run_testing, daemon=True).start()

    def stop_testing(self):
        with self.lock:
            self.should_stop_testing = True
            self.status_var.set("Stopping Test...")

    def run_testing(self):
        try:
            tester = test.Tester(self.env)
            tester.run(
                stop_callback=lambda: self.should_stop_testing,
                status_callback=lambda *args: self.update_test_status(*args)
            )
        finally:
            self.master.after(0, self.on_testing_end)

    def update_test_status(self, episode=None, steps=None, pos=None, final=None):
        if final:
            self.status_var.set(final)
        elif episode is not None and steps is not None and pos is not None:
            status = f"Test Episode: {episode + 1}\nSteps: {steps}\nPosition: {pos}"
            self.status_var.set(status)

    def on_testing_end(self):
        self.testing = False
        self.stop_test_btn.config(state=tk.DISABLED)
        self.status_var.set("Test Completed")

    def start_training(self):
        if not self.training:
            self.save_current_map()
            self.env.obstacles = self.obstacles.copy()
            self.training = True
            self.should_stop = False
            self.stop_train_btn.config(state=tk.NORMAL)
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
        self.stop_train_btn.config(state=tk.DISABLED)
        self.status_var.set("Training Completed")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Grid World Trainer")
    app = GridWorldGUI(root)
    root.mainloop()
