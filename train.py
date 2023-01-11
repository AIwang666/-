

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import ResNet18
import  torch.nn.functional as F
from    torchvision import datasets, transforms
import os
import numpy as np
from visdom import Visdom
# 定义是否使用GPU
device = torch.device("cuda")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch  Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 150#135 135相对来说比较大可以考虑跑到60左右就可以了所以给一个50尝试一下   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128      #批处理尺寸(batch_size)
LR = 0.0001        #学习率



# # 准备数据集并预处理
# transform_train = transforms.Compose([
#     transforms.RandomCrop(224, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
# ])

# transform_test = transforms.Compose([
    
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#训练数据集
# def loadtraindata():
#     path = r"C:\Users\王一非\Desktop\智慧渔业\数据集\Fish_Dataset\训练集"  
#     #path = r"C:\Users\王一非\Desktop\深度学习训练模型代码\chest_xray\train"                                       # 路径
#     trainset = torchvision.datasets.ImageFolder(path,
#                                                 transform=transforms.Compose([
#                                                     transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
#                                                     transforms.RandomHorizontalFlip(),
#                                                     transforms.RandomVerticalFlip(),
#                                                     transforms.RandomCrop(224, 224),
#                                                     transforms.CenterCrop(224),
#                                                     transforms.ToTensor()])
#                                                 )

#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
#                                               shuffle=True, num_workers=2)
#     return trainloader
# #测试数据集
# def loadtestdata():
#     path = r"C:\Users\王一非\Desktop\智慧渔业\数据集\Fish_Dataset\验证集"
#     #path = r"C:\Users\王一非\Desktop\深度学习训练模型代码\chest_xray\test"
#     testset = torchvision.datasets.ImageFolder(path,
#                                                 transform=transforms.Compose([
#                                                     transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
#                                                     transforms.ToTensor()])
#                                                 )
#     testloader = torch.utils.data.DataLoader(testset, batch_size=32,
#                                              shuffle=True, num_workers=2)
#     return testloader
#训练数据集
def loadtraindata():
    path = r"C:\Users\王一非\Desktop\智慧渔业\数据集\Fish_Dataset\训练集"  
    #path = r"C:\Users\王一非\Desktop\智慧渔业\数据集\Fish_Dataset\灰度\训练集"                                       # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.RandomCrop(224, 224),

                                                    transforms.RandomHorizontalFlip(),
                                                    #transforms.RandomVerticalFlip(),
                                                   # transforms.RandomRotation(15),
                                                    #transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.0195, 0.0207, 0.0228), (0.0167, 0.0141, 0.0139)),
                                                    ])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    return trainloader
#测试数据集
def loadtestdata():
    path = r"C:\Users\王一非\Desktop\智慧渔业\数据集\Fish_Dataset\验证集"
    #path = r"C:\Users\王一非\Desktop\智慧渔业\数据集\Fish_Dataset\灰度\验证集"
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.0195, 0.0207, 0.0228), (0.0167, 0.0141, 0.0139)),
                                                    ])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True, num_workers=2)
    return testloader
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) #训练数据集
trainloader=loadtraindata()
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
testloader=loadtestdata()
# Cifar-10的标签
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#classes = ('胃癌', '胃溃疡', '胃糜烂', '胃息肉', '正常')
#classes =('NORMAL','PNEUMONIA')
classes = ('Black Sea Sprat', 'Red Mullet', 'Red Sea Bream', 'Trout')


# # 注意此处初始化visdom类
# viz = Visdom()
# # 绘制起点
# viz.line([0.], [0.], win="tloss", opts=dict(title='tloss'))
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
# viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',legend=['loss', 'acc.%']))
# 模型定义-ResNet
net = ResNet18().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

global_step = 0
# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
               # net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()


                    global_step += 1
                    # viz.line([loss.item()], [global_step], win='train_loss', update='append')

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('1')
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                    print('2')

                # 每训练完一个epoch测试一下准确率
                print('3')
                print("Waiting Test!")
                print('4')
                with torch.no_grad():
                    correct = 0
                    total = 0
                    test_loss = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                        test_loss += criterion(outputs, labels).item()
                    print(test_loss)
                    test_loss /= len(testloader.dataset)
                    print(test_loss)
                #     # 绘制epoch以及对应的测试集损失loss
                #    # viz.line([float(test_loss)], [epoch], win="train loss", update='append')
                #     viz.line([float(test_loss)], [global_step], win="tloss", update='append')
                #     viz.line([[test_loss, float(100.* correct / len(testloader.dataset))]],[global_step], win='test', update='append')
                #     print(test_loss)
                #     #viz.line([test_loss], [epoch], win="test loss", update='append')
                   
                   
                   
                   
                   
                   
                   
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                   # f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%,Tloss= %.03f" % (epoch + 1, acc,test_loss))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    print("999")
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
""" torch.save(model_object, 'model.pkl')
model = torch.load('model.pkl') """