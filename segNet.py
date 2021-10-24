import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
# segNet
# encoder: (即vgg16)2*conv2_64,2*conv2_128,3*conv3_256,3*conv3_512,3*conv3_512
# decoder:对称 3*conv3_512,3*conv3_512,3*conv3_256,2*conv2_128,2*conv2_64
class EnMultiConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_layers = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        if num_layers == 3:
            self.multi_conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        elif num_layers == 2:
            self.multi_conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self,x):
        x = self.multi_conv(x)
        return x
class DeMultiConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_layers = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        if num_layers == 3:
            self.multi_conv = nn.Sequential(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        elif num_layers == 2:
            self.multi_conv = nn.Sequential(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self,x):
        x = self.multi_conv(x)
        return x

class SegNet(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # encoder
        self.block1 = EnMultiConv(in_channels= in_channels,out_channels= 64,num_layers=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

        self.block2 = EnMultiConv(in_channels= 64,out_channels= 128,num_layers=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

        self.block3 = EnMultiConv(in_channels= 128,out_channels= 256,num_layers=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

        self.block4 = EnMultiConv(in_channels= 256,out_channels= 512,num_layers=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)

        self.block5 = EnMultiConv(in_channels= 512,out_channels= 512,num_layers=3)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        # decoder
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2,stride = 2)
        self.deconv1 = DeMultiConv(in_channels = 512,out_channels= 512,num_layers=3)

        self.unpool2 = nn.MaxUnpool2d(kernel_size= 2, stride= 2)
        self.deconv2 = DeMultiConv(in_channels = 512,out_channels= 256,num_layers= 3)

        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2,stride = 2)
        self.deconv3 = DeMultiConv(in_channels = 256,out_channels= 128,num_layers=3)

        self.unpool4 = nn.MaxUnpool2d(kernel_size= 2, stride= 2)
        self.deconv4 = DeMultiConv(in_channels = 128,out_channels= 64,num_layers= 2)

        self.unpool5 = nn.MaxUnpool2d(kernel_size = 2,stride = 2)
        self.deconv5 = DeMultiConv(in_channels = 64,out_channels= num_classes,num_layers=2)
        # self.softmax = nn.Softmax2d()
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.block1(x)
        x,indices1 = self.pool1(x)
        x = self.block2(x)
        x,indices2 = self.pool2(x)
        x = self.block3(x)
        x,indices3 = self.pool3(x)
        x = self.block4(x)
        x,indices4 = self.pool4(x)
        x = self.block5(x)
        x,indices5 = self.pool5(x)

        x = self.unpool1(x,indices5)
        x = self.deconv1(x)
        x = self.unpool2(x,indices4)
        x = self.deconv2(x)
        x = self.unpool3(x,indices3)
        x = self.deconv3(x)
        x = self.unpool4(x,indices2)
        x = self.deconv4(x)
        x = self.unpool5(x,indices1)
        x = self.deconv5(x)
        # return x
        return self.softmax(x) 


class SegNet_Basic(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # encoder
        self.enconv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enconv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(80),
                                   nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enconv3 = nn.Sequential(nn.Conv2d(in_channels=80, out_channels=96, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.enconv4 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        # enconv4 - deconv 的存在使得unpool channel 不一致，另外，有说decoder不需要relu
        # decoder
        self.deconv = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU())

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=80, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(80),
                                   nn.ReLU())

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv3 = nn.Sequential(nn.Conv2d(in_channels=80, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        self.deconv4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=num_classes,kernel_size=3,stride=1,padding=1),
                                    nn.BatchNorm2d(12),
                                    nn.ReLU())
        self.softmax = nn.Softmax2d()
    def forward(self,x):
        x = self.enconv1(x)
        x,indices1 = self.pool1(x)
        x = self.enconv2(x)
        x,indices2 = self.pool2(x)
        x,indices3 = self.pool3(x)
        x = self.enconv4(x)

        x = self.deconv(x)
        x = self.unpool1(x,indices3)
        x = self.deconv1(x)
        x = self.unpool2(x,indices2)
        x = self.deconv2(x)
        x = self.unpool3(x,indices1)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.softmax(x)

        return x


        


if __name__ =="__main__":
    # pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
    # unpool = nn.MaxUnpool2d(kernel_size=2,stride=2)
    # input = torch.tensor([[[[ 1.,  2,  3,  4],
    #                         [ 5,  6,  7,  8],
    #                         [ 9, 10, 11, 12],
    #                         [13, 14, 15, 16]]]])
    # out, indices = pool(input)
    # print(out)
    # out = unpool(out,indices)
    # print(out)
    # ---------------------------------------------------------------------------------------------- #
    input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    print('size: ',input.size(),'\nOriginal: ',input)
    m = nn.Upsample(scale_factor=2, mode='nearest')
    input1 = m(input)
    print('size: ',input1.size(),'\n nearest: ',input1)

    m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
    input2 = m(input)
    print('size: ',input2.size(),'\nbilinear: ',input2)

    m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    input3 = m(input)
    print('size: ',input3.size(),'\nbilinear and  align_corners=True: ',input3)
    # Try scaling the same data in a larger tensor
    input_3x3 = torch.zeros(3, 3).view(1, 1, 3, 3)
    input_3x3[:, :, :2, :2].copy_(input)
    print('size: ',input_3x3.size(),'\nOriginal: ',input_3x3)

    m = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)  # align_corners=False
    # Notice that values in top left corner are the same with the small input (except at boundary)
    input4 = m(input_3x3)
    print('size: ',input4.size(),'\nbilinear: ',input4)
    m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    # Notice that values in top left corner are now changed
    input5 = m(input_3x3)
    print('size: ',input5.size(),'\nbilinear and  align_corners=True: ',input5)

    x = torch.rand(size=(8,3,320,320))
    seg = SegNet(in_channels= 3,num_classes= 21)
    out = seg(x)
    print(out.size())

