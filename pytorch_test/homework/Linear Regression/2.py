import numpy as np
import matplotlib.pyplot as plt

# 生成一个随机的曲线，作为示例
x = np.linspace(0, 10, 60000)
y = np.sin(x) + np.random.normal(0, 0.1, 60000)

# 定义移动平均函数
def moving_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# 选择窗口大小
window_size = 1000

# 对曲线进行平滑处理
smoothed_y = moving_average(y, window_size)

# 绘制原始曲线和平滑后的曲线
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Original Curve', alpha=0.5)
plt.plot(x[window_size - 1:], smoothed_y, 'r', label=f'Moving Average (Window Size={window_size})')
plt.title('Smoothing of a Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
