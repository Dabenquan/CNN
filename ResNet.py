from typing import ForwardRef
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
 
 
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
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

class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes):
        super().__init__()
        self.in_channels = 64
        # blocks_num 控制block层数目,数组类型
        # 第一层 卷积层 + 最大池化层 conv7_64
        self.conv1 = nn.Conv2d(in_channels=3,out_channels= self.in_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # 残差网络层
        self.layer1 = self._makeLayer(block,64,blocks_num[0])
        self.layer2 = self._makeLayer(block,128,blocks_num[1],stride=2)
        self.layer3 = self._makeLayer(block,256,blocks_num[2],stride=2)
        self.layer4 = self._makeLayer(block,512,blocks_num[3],stride=2)

        # 平均池化层
        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 不知道
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _makeLayer(self,block,channels,block_num,stride = 1):
        downsample = None
        if stride != 1 or self.in_channels != channels*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels,channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                                        nn.BatchNorm2d(channels*block.expansion))
        layer = []
        layer.append(block(self.in_channels,channels,downsample = downsample,stride = stride ))
        self.in_channels = channels*block.expansion # channels 变
        for i in range(1,block_num):
            layer.append(block(self.in_channels,channels))
        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 构建不同层的resNet网络
def resnet18(pretrained = False,num_classes = 1000):
    model = ResNet(block=BasicBlock,blocks_num=[2,2,2,2],num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained = False,num_classes = 1000):
    model = ResNet(block=BasicBlock,blocks_num=[3,4,6,3],num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def resnet50(pretrained = False,num_classes = 1000):
    model = ResNet(block=Bottleneck,blocks_num=[3,4,6,3],num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained = False,num_classes = 1000):
    model = ResNet(block=Bottleneck,blocks_num=[3,4,23,3],num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def resnet152(pretrained = False,num_classes = 1000):
    model = ResNet(block=Bottleneck,blocks_num=[3,8,36,3],num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



if __name__ == '__main__':
    x = torch.rand(size=(8,3,224,224))
    # print(x.shape)
    res18 = resnet18(pretrained=False,num_classes=100)
    out = res18(x)
    print(out.size())

    