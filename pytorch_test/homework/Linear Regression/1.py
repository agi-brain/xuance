import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
# 定义一个简单的线性回归模型，用于分类任务
class LinearRegressionClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearRegressionClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x

# 超参数设置
input_size = 28*28  # 例如，28x28 的手写数字图像展平后的尺寸
num_classes = 10  # MNIST 数据集的类别数
num_epochs = 10
batch_size = 64
learning_rate = 0.01
download=False # 如果数据已经下载就是False
Train=True
# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=Train, download=download, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=Train, download=download, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
model = LinearRegressionClassifier(input_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
if Train:
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # 将图像展平为一维向量，原本是四维数据[batch_size, channels, height, width]
            images = images.view(-1, input_size)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    #保存模型
    torch.save(model.state_dict(),'linear.pkl')
# 测试模型
model.load_state_dict(torch.load('linear.pkl'))
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = images.view(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')