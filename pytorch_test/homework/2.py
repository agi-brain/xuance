import torch
import torch.nn as nn
import numpy as np

# 定义简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, kernel1, kernel2):
        super(SimpleCNN, self).__init__()
        # 注意：这里我们需要根据输入数据的通道数来设置in_channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel1.shape, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel2.shape, bias=False)

        # 将手动设定的卷积核参数转换为PyTorch的Tensor
        self.conv1.weight = nn.Parameter(torch.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0))
        self.conv2.weight = nn.Parameter(torch.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# 输入数据
INPUT = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
], dtype=np.float32)   # 添加批次和通道维度

# 确保输入数据的维度是(batch_size, channels, height, width)
INPUT = torch.from_numpy(INPUT).unsqueeze(0).unsqueeze(0)  # 批次和通道维度

# 设置卷积核参数
kernel1 = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

kernel2 = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

# 实例化模型
model = SimpleCNN(kernel1, kernel2)

# 输出结果
output = model(INPUT).squeeze(0).squeeze(0)
reference=torch.from_numpy(np.array([
    [0,0],
    [0,0]],dtype=np.float32
))
print("输出结果:\n", output.detach().numpy())

# 计算损失
criterion = nn.MSELoss()
loss = criterion(output, reference)

print("Loss:", loss.detach().numpy())

# 反向传播计算梯度
loss.backward()

# 打印各层的梯度
print("Gradient of conv1 weights:\n", model.conv1.weight.grad)
print("Gradient of conv2 weights:\n", model.conv2.weight.grad)