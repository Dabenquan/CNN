import torch
import torch.nn as nn

class FCN8s(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        # 前半部分是VGG16的特征提取网络，舍弃全连接层
        # 定义网络

        # block-1 2*conv3-64 maxPool  -- conv1 + pool1
        self.block1 = nn.Sequential(nn.Conv2d(in_channels = 3,out_channels = 64, padding = 1, kernel_size = 3,stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 64,out_channels = 64, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        # block-2 2*conv3-128 maxPool  -- conv2 + pool2
        self.block2 = nn.Sequential(nn.Conv2d(in_channels = 64,out_channels = 128, padding = 1, kernel_size = 3,stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 128,out_channels = 128, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2))

        # block-3 3*conv3-256 maxPool     -- conv3 + pool3
        self.block3 = nn.Sequential(nn.Conv2d(in_channels = 128,out_channels = 256, padding = 1, kernel_size = 3,stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 256,out_channels = 256, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 256,out_channels = 256, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2))        
        
        # block-4 3*conv3-512 maxPool     -- conv4 + pool4
        self.block4 = nn.Sequential(nn.Conv2d(in_channels = 256,out_channels = 512, padding = 1, kernel_size = 3,stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2))    
        
        # block-5 3*conv3-512 maxPool    -- conv5 + pool5
        self.block5 = nn.Sequential(nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3,stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2))  
        
        # conv6-7 
        self.conv6 = nn.Conv2d(in_channels = 512,out_channels = 512, padding = 0, kernel_size = 1,stride = 1,dilation = 1)
        self.conv7 = nn.Conv2d(in_channels = 512,out_channels = 512, padding = 0, kernel_size = 1,stride = 1,dilation = 1)
        self.relu = nn.ReLU(inplace = True)
        
        # 上采样
        self.deconv1 = nn.ConvTranspose2d(512,512,kernel_size = 3, stride = 2, padding = 1,output_padding = 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512,256,kernel_size = 3, stride = 2, padding = 1,output_padding = 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size = 3, stride = 2, padding = 1,output_padding = 1),
                                    nn.BatchNorm2d(128))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size = 3, stride = 2, padding = 1,output_padding = 1),
                                    nn.BatchNorm2d(64))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size = 3, stride = 2, padding = 1,output_padding = 1),
                                    nn.BatchNorm2d(32))
        
        # 分类
        self.classifier = nn.Conv2d(32,num_classes,kernel_size = 1)
        
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x3 = self.block3(x) # 做融合
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        
        score = self.relu(self.conv6(x5))
        score = self.relu(self.conv7(score))
        
        # 上采样 + 跳跃连接
        score = self.relu(self.deconv1(score)) # ??
        score = self.bn1(score + x4)
        
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        
        score = self.deconv3(score)
        score = self.deconv4(score)
        score = self.deconv5(score)
        
        score = self.classifier(score)
        return score
        
if __name__=='__main__':
    x = torch.rand(size= (8,3,224,224))
    fcn = FCN8s(num_classes = 100)
    out = fcn(x)
    print(out.size())