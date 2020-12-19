import torch
import torch.nn as nn

class ALEXNET_MNIST(nn.Module):

    def __init__(self, num_classes=10, feature_size=128):
        super(ALEXNET_MNIST, self).__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, self.feature_size, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.feature_size)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv2 = nn.Conv2d(64, self.feature_size, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(self.feature_size)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.linear1 = nn.Linear(self.feature_size * 7 * 7, 384)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(384, 192)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool2(self.relu4(self.bn4(self.conv4(x))))
        # print(x.size())
        x = x.flatten(1)
        x = self.relu3(self.linear1(x))
        x = self.relu4(self.linear2(x))
        x = self.linear(x)

#        x = self.classifier(x)
        return x

class ALEXNET_CUSTOM(nn.Module):

    def __init__(self, num_classes=10, feature_size=512):
        super(ALEXNET_CUSTOM, self).__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, self.feature_size, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.feature_size)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv2 = nn.Conv2d(64, self.feature_size, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(self.feature_size)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        # self.linear1 = nn.Linear(self.feature_size * 56 * 56, 4096)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.linear2 = nn.Linear(4096, 4096)
        # self.relu4 = nn.ReLU(inplace=True)
        self.linear = nn.Linear(self.feature_size * 14 * 14, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.maxpool4(self.relu4(self.bn4(self.conv4(x))))
        # print(x.size())
        x = x.flatten(1)
        # x = self.relu3(self.linear1(x))
        # x = self.relu4(self.linear2(x))
        x = self.linear(x)

#        x = self.classifier(x)
        return x

def alexnet_mnist(pretrained=False, progress=True, **kwargs):
    model = ALEXNET_MNIST(**kwargs)
    return model

def alexnet_custom(pretrained=False, progress=True, **kwargs):
    model = ALEXNET_CUSTOM(**kwargs)
    return model