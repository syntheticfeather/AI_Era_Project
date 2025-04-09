import numpy as np
import pygame
from sklearn.cluster import KMeans

# 全局常量配置
GRID_SIZE = 40
SCREEN_SIZE = 800
CELL_SIZE = SCREEN_SIZE // GRID_SIZE  # 每个格子像素大小
NUM_GOALS = 3  # 默认目标点数（当auto_k=False时生效）

COLORS = {
    "background": (255, 255, 255),
    "grid_line": (200, 200, 200),
    "obstacle": (0, 0, 0),
    "available_obstacle": (255, 165, 0),
    "goal": (255, 0, 0),
    "agent": (0, 0, 255)
}


class GridEnv:
    def __init__(self, n_clusters=5, auto_k=False, max_k=10):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

        # 初始化网格
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.used_obstacles = []
        self.placed_goals = []

        # 聚类参数
        self.n_clusters = n_clusters
        self.auto_k = auto_k
        self.max_k = max_k

        # 放置障碍物
        self._place_obstacles()
        # 自动聚类放置目标
        self._place_goals_with_clustering()

        # Agent初始位置
        self.agent_pos = [0, 0]

    def _place_obstacles(self):
        # 固定种子生成可重复地形
        seed = 50
        rng = np.random.RandomState(seed)  # 创建独立随机数生成器

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if i == 0 or i == GRID_SIZE - 1 or j == 0 or j == GRID_SIZE - 1:
                    self.grid[i][j] = 0
                else:
                    np.random.seed(seed)
                    if np.random.rand() < 0.2:
                        self.grid[i][j] = 1  # 随机障碍物
                        seed += 1
                    else:
                        self.grid[i][j] = 0  # 空地
                        seed += 1

    def _get_passable_points(self):
        """获取所有可通行点坐标"""
        return np.array([[x, y]
                         for x in range(GRID_SIZE)
                         for y in range(GRID_SIZE)
                         if self.grid[x][y] == 0
                         ])

    def _calculate_weights(self, points):
        """计算每个点的权重（周围障碍物密度）"""
        weights = []
        for (x, y) in points:
            obstacle_count = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        obstacle_count += (self.grid[nx][ny] == 1)
            weights.append(obstacle_count)
        return np.array(weights)

    def _find_optimal_k(self, points, weights):
        """肘部法则自动选择最优K值"""
        distortions = []
        max_possible_k = min(len(points), self.max_k)

        for k in range(1, max_possible_k + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(points, sample_weight=weights)
            distortions.append(kmeans.inertia_)

        # 曲率变化检测
        deltas = np.diff(distortions)
        if len(deltas) < 2:
            return 1
        curvature = np.diff(deltas)
        optimal_k = np.argmax(curvature) + 2  # 拐点位置补偿
        return min(optimal_k, self.max_k)

    def _place_goals_with_clustering(self):
        """通过聚类放置目标点"""
        points = self._get_passable_points()
        if len(points) == 0:
            raise ValueError("No passable points for clustering!")

        weights = self._calculate_weights(points)

        # 自动选择K值
        if self.auto_k:
            self.n_clusters = self._find_optimal_k(points, weights)

        # 安全限制
        self.n_clusters = min(self.n_clusters, len(points))
        if self.n_clusters <= 0:
            self.n_clusters = 1

        # 执行加权K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(points, sample_weight=weights)
        centroids = kmeans.cluster_centers_.astype(int)
        print("Cluster centroids:", centroids)
        # 放置目标点
        self.placed_goals = []
        for (x, y) in centroids:
            x = np.clip(x, 0, GRID_SIZE - 1)
            y = np.clip(y, 0, GRID_SIZE - 1)
            if self.grid[x][y] == 0 and (x, y) not in self.placed_goals:
                self.grid[x][y] = 2
                self.placed_goals.append((x, y))
            elif self.grid[x][y] == 1:
                flag = False
                # 在周围八个格子中随机选择一个空地作为红点
                for dx in [-1, 0, 1]:
                    if flag: break
                    for dy in [-1, 0, 1]:
                        if flag: break
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[nx][ny] == 0:
                            self.grid[nx][ny] = 2
                            self.placed_goals.append((nx, ny))
                            flag = True

    def reset(self, new_k=None):
        """重置环境并可选新K值"""
        if new_k is not None:
            self.n_clusters = new_k

        # 清除旧目标
        for x, y in self.placed_goals:
            self.grid[x][y] = 0
        self.placed_goals = []

        # 重新聚类
        self._place_goals_with_clustering()
        self.agent_pos = [0, 0]
        return self._get_state()

    # --------------------------
    # 以下方法与原始代码保持一致
    # --------------------------
    def step(self, action):
        # ...（保持原有逻辑不变）
        x, y = self.agent_pos
        done = False

        if action < 4:  # 移动动作
            dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_x = max(0, min(GRID_SIZE - 1, x + dx))
            new_y = max(0, min(GRID_SIZE - 1, y + dy))

            # 碰撞检测
            if self.grid[new_x][new_y] == 1:  # 黑色障碍
                reward = -10
            elif self.grid[new_x][new_y] == 3:  # 橙色障碍
                self.agent_pos = [new_x, new_y]
                reward = -4
            else:
                self.agent_pos = [new_x, new_y]
                reward = -1  # 基础移动惩罚
        else:  # 动作4：放置目标
            if self.grid[x][y] == 0 and (x, y) not in self.placed_goals and len(self.placed_goals) < self.n_clusters:
                self.grid[x][y] = 2
                self.placed_goals.append((x, y))
                reward = self.count(x, y)
            else:
                reward = self.count(x, y)
                reward -= 200  # 非法放置惩罚
        return self._get_state(), reward, done

    def count(self, x, y):
        # ...（保持原有逻辑不变）
        radius = 3
        obstacle_count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in self.used_obstacles:
                    obstacle_count += (self.grid[nx][ny] == 1)
                    self.used_obstacles.append((nx, ny))
        return obstacle_count * 5

    def _get_state(self):
        return self.agent_pos[0], self.agent_pos[1], len(self.placed_goals)

    def render(self):
        # ...（保持原有渲染逻辑不变）
        self.screen.fill(COLORS["background"])
        # 绘制网格线
        for i in range(GRID_SIZE):
            pygame.draw.line(self.screen, COLORS["grid_line"], (0, i * CELL_SIZE), (SCREEN_SIZE, i * CELL_SIZE))
            pygame.draw.line(self.screen, COLORS["grid_line"], (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_SIZE))
        # 绘制障碍和目标
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[x][y] == 1:
                    pygame.draw.rect(self.screen, COLORS["obstacle"],
                                     (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif self.grid[x][y] == 3:
                    pygame.draw.rect(self.screen, COLORS["available_obstacle"],
                                     (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for x, y in self.placed_goals:
            pygame.draw.circle(self.screen, COLORS["goal"],
                               (y * CELL_SIZE + CELL_SIZE // 2, x * CELL_SIZE + CELL_SIZE // 2),
                               CELL_SIZE // 3)
        # 绘制Agent
        pygame.draw.circle(self.screen, COLORS["agent"],
                           (self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                            self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2),
                           CELL_SIZE // 3)
        pygame.display.flip()
        self.clock.tick(30)


# 使用示例
if __name__ == "__main__":
    # 创建环境（自动选择K值）
    env = GridEnv(n_clusters=5, auto_k=False, max_k=8)

    # 可视化初始状态
    env.render()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.wait(100)

    pygame.quit()
