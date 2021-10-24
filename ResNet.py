import torch
import torch.nn as nn

# resNet
# BasicBlock结构用于ResNet34及以下的网络，BotteNeck结构用于ResNet50及以上的网络。
# BasicBloc: conv3-relu-conv3-(+x)-relu
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.downsample = downsample # 降通道channels 以便于作残差
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        x += residual
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.downsample = downsample # 降通道channels 以便于作残差
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace= True)

    def forward(self,x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        x += residual
        x = self.relu(x)

        return x


if __name__ == '__main__':
    pass