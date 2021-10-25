import torch 
import torch.nn as nn
from torchvision import models
from ResNet import BasicBlock,Bottleneck

def double_conv(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
    )
# BasicBlock
# Bottleneck

class ResUNet(nn.Module):
    def __init__(self,in_channel,num_classes,block,blocks_num):
        super().__init__()
        self.in_channels = 64 # 注意与in_channel 区别
        # blocks_num 控制block层数目,数组类型
        # 第一层 卷积层 + 最大池化层 conv7_64
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels= self.in_channels,kernel_size=7,stride=2,padding=3,bias=False),
                                nn.BatchNorm2d(self.in_channels),
                                nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # 残差网络层
        self.layer1 = self._makeLayer(block,64,blocks_num[0])
        self.layer2 = self._makeLayer(block,128,blocks_num[1],stride=2)
        self.layer3 = self._makeLayer(block,256,blocks_num[2],stride=2)
        self.layer4 = self._makeLayer(block,512,blocks_num[3],stride=2)

        # 去除 平均池化层，使用上采样替换，构成U-net 'U'型结构
        # # 平均池化层
        # self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        # # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 上采样
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        
        self.deconv_up3 = double_conv(256 + 512, 256)
        self.deconv_up2 = double_conv(128 + 256, 128)
        self.deconv_up1 = double_conv(128 + 64, 64)

        self.deconv_last = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                                        nn.Conv2d(64, num_classes,1)
        )


        # # 不知道
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    
    def _makeLayer(self,block,channels,block_num,stride = 1):
        downsample = None
        if stride != 1 or self.in_channels != channels*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels,channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                                        nn.BatchNorm2d(channels*block.expansion))
        layer = []
        layer.append(block(self.in_channels,channels,downsample = downsample,stride = stride )) # layer2-4 第一个block降
        self.in_channels = channels*block.expansion # channels 变
        for i in range(1,block_num):
            layer.append(block(self.in_channels,channels))
        return nn.Sequential(*layer)

    def forward(self,x):
        conv1 = self.conv1(x)
        temp = self.maxpool(conv1)

        conv2 = self.layer1(temp)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        bottle = self.layer4(conv4)

        x = self.upsample(bottle)
        x = torch.cat([x,conv4],dim=1)

        x = self.deconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x,conv3],dim=1)

        x = self.deconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x,conv2],dim=1)
        
        x = self.deconv_up1(x)
        x = self.upsample(x)
        x = torch.cat([x,conv1],dim=1)

        out = self.deconv_last(x)

        return out

    # Class ResUNet下的一个函数，用于导入预训练参数，函数：load_pretrained_weights()
    def load_pretrained_weights(self):
        # 导入自己模型的参数
        model_dict = self.state_dict()
        # 导入resnet34的参数，自动下载
        resnet34_weights = models.resnet34(True).state_dict()
        count_res = 0
        count_my = 0

        reskeys = list(resnet34_weights.keys())
        mykeys = list(model_dict.keys())
        print(self)   # 自己网络的结构
        print(models.resnet34()) # resnet34 的结构

        corresp_map = []
        while(True):   # 后缀相同的放入list
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]

            if "fc" in reskey:
                break
            while reskey.split(".")[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]

            corresp_map.append([reskey,mykey])
            count_res += 1
            count_my += 1

        for k_res, k_my in corresp_map:
            model_dict[k_my] = resnet34_weights[k_res]

        try:
            self.load_state_dict(model_dict)
            print("Loaded resnet34 weights in myNet !")
        except:
            print("Error resnet34 weights in myNet !")
            raise
# Class ResUNet下的一个函数，用于导入预训练参数，函数：load_pretrained_weights()
# 构建不同层的resUNet网络
def ResUNet18(in_channel,num_classes = 1000):
    model = ResUNet(in_channel,num_classes=num_classes,block=BasicBlock,blocks_num=[2,2,2,2])
    return model

def ResUNet34(in_channel,num_classes = 1000,pretrained = False):
    model = ResUNet(in_channel,num_classes=num_classes,block=BasicBlock,blocks_num=[3,4,6,3])
    if pretrained:
        model.load_pretrained_weights()
    return model

def ResUNet50(in_channel,num_classes = 1000):
    model = ResUNet(in_channel,num_classes=num_classes,block=Bottleneck,blocks_num=[3,4,6,3])
    return model

def ResUNet101(in_channel,num_classes = 1000):
    model = ResUNet(in_channel,num_classes=num_classes,block=Bottleneck,blocks_num=[3,4,23,3])
    return model

def ResUNet152(in_channel,num_classes = 1000):
    model = ResUNet(in_channel,num_classes=num_classes,block=Bottleneck,blocks_num=[3,8,36,3])
    return model

if __name__ == '__main__':
    resunet34 = ResUNet34(in_channel=3,num_classes=1)
    x = torch.rand(size=(8,3,512,512))
    out = resunet34(x)
    print(out.size())

