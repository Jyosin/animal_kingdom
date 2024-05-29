import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset

# 设置参数
duration = 30
fps = 20
total_frames = duration * fps
t = np.linspace(0, duration, total_frames)

# 定义路径
triangle_path_x = 100 * np.cos(2 * np.pi * 0.5 * t) + 150
triangle_path_y = 100 * np.sin(2 * np.pi * 0.5 * t) + 150
square_path_x = 100 * np.cos(2 * np.pi * 1.0 * t) + 300
square_path_y = 100 * np.sin(2 * np.pi * 1.0 * t) + 150
circle_path_x = 100 * np.cos(2 * np.pi * 0.2 * t) + 450
circle_path_y = 100 * np.sin(2 * np.pi * 0.2 * t) + 150

# 创建动画和数据集
data = []
labels = []

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 600)
ax.set_ylim(0, 300)
triangle, = ax.plot([], [], 'v-', markersize=10, color='orange', label="Triangle Path")
square, = ax.plot([], [], 's-', markersize=10, color='blue', label="Square Path")
circle, = ax.plot([], [], 'o-', markersize=10, color='red', label="Circle Path")

def init():
    triangle.set_data([], [])
    square.set_data([], [])
    circle.set_data([], [])
    return triangle, square, circle

def animate(i):
    triangle.set_data(triangle_path_x[i], triangle_path_y[i])
    square.set_data(square_path_x[i], square_path_y[i])
    circle.set_data(circle_path_x[i], circle_path_y[i])
    data.append(np.hstack([triangle_path_x[i], triangle_path_y[i], square_path_x[i], square_path_y[i], circle_path_x[i], circle_path_y[i]]))
    labels.extend([0, 1, 2])  # 假设标签 0=三角, 1=方形, 2=圆形
    return triangle, square, circle

ani = animation.FuncAnimation(fig, animate, frames=total_frames, init_func=init, blit=True)
plt.legend()
plt.title("Animated Paths of Shapes")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()

# 将数据保存为torch tensor
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)


torch.save(data, 'data/shape_paths_data.pt')
torch.save(labels, 'data/shape_labels.pt')
