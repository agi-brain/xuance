import torch
import torch.nn as nn

class model(nn.Module):#创建一个简单的神经网络，输入大小为input_size，输出大小为output_size
    def __init__(self, input_size, output_size):
        super(model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

torch.manual_seed(42)#设置随机种子以便复现结果

input_size = 5
output_size = 3
model = model(input_size, output_size)
criterion = nn.CrossEntropyLoss()#使用交叉熵损失函数
optim= torch.optim.Adam(model.parameters(), lr=0.01)#使用Adam优化器

input_data = torch.randn(10, input_size)#创建一个10*input_size的随机输入
print("input_data",input_data)
output = model(input_data)

target = torch.randint(0, output_size, (10,))#生成了形状为 (10,) 的虚拟目标数据，其中的值在 [0, output_size) 范围内随机选择

loss = criterion(output, target)

optim.zero_grad()# 清除先前的梯度
loss.backward()# 计算梯度
optim.step()# 使用优化器更新权重和偏置

# Print results
print("Forward pass output:")
print(output)
print("Categorical Cross-Entropy Loss:")
print(loss.item())
print("Gradients of the Loss with respect to Weights and Biases:")
for param in model.parameters():
    print(param.grad)
