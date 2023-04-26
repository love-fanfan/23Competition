import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
from resnet import ResNet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
from torch.optim.lr_scheduler import StepLR


from dataload import transform_test, transform_train

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型、损失函数和优化器
# model = ResNet18().to(device)
model = torchvision.models.resnet50(pretrained=True)
model.to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = 0.0001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.8) 

# 加载训练集和测试集
train_dataset = ImageFolder('./train/eh', transform=transform_train)
test_dataset = ImageFolder('./test/eh', transform=transform_test)

# 设置batchsize和加载数据
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
epochs = 150
best_acc = 0  # 记录最佳精度

for epoch in range(epochs):
    # 训练模型
    
    for param_group in optimizer.param_groups:
        print("Current learning rate: ", param_group['lr'])
    model.train()
    train_loss = 0
    train_acc = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs.max(),' ',inputs.min(), " ", inputs.mean())
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        train_acc += predicted.eq(labels).sum().item()
        
    # scheduler = lr_scheduler.StepLR(optimizer, step_size= 10, gamma=0.8) #gamma=0.1
    train_loss /= len(train_loader)
    train_acc /= total

    scheduler.step()
    # 测试模型
    model.eval()
    test_loss = 0
    test_acc = 0
    total = 0
    class_correct = list(0. for i in range(10)) # 每类正确的数量
    class_total = list(0. for i in range(10)) # 每类总数
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            test_acc += predicted.eq(labels).sum().item()

            # 统计每个类别的正确和总样本数
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    test_loss /= len(test_loader)
    test_acc /= total

    # 输出训练过程
    print('epoch: %d | train loss: %.4f | train acc: %.4f | test loss: %.4f | test acc: %.4f' % (
        epoch + 1, train_loss, train_acc, test_loss, test_acc))
    # 输出每个类别的准确率
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            test_dataset.classes[i], 100 * class_correct[i] / class_total[i]))

    # 更新最佳精度
    if test_acc > best_acc:
        best_acc = test_acc 
        torch.save(model.state_dict(), 'best_model.pth')
