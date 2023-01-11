import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = torchvision.models.vgg19(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False),
            nn.Linear(128, 4, bias=False)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # method 2 kaiming
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.features(x)  # 前向传播的时候先经过卷积层和池化层
        x = x.view(-1, 512)
        x = self.classifier(x)  # 再将features（得到网络输出的特征层）的结果拼接到分类器上
        return x


def vgg19():

    return VGG()