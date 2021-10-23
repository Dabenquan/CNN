import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes = num_classes
        # 定义网络
        net = []
        # block-1 2*conv3-64 maxPool
        net.append(nn.Conv2d(in_channels = 3,out_channels = 64, padding = 1, kernel_size = 3,stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 64,out_channels = 64, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        # block-2 2*conv3-128 maxPool
        net.append(nn.Conv2d(in_channels = 64,out_channels = 128, padding = 1, kernel_size = 3,stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 128,out_channels = 128, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

        # block-3 3*conv3-256 maxPool
        net.append(nn.Conv2d(in_channels = 128,out_channels = 256, padding = 1, kernel_size = 3,stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 256,out_channels = 256, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 256,out_channels = 256, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size = 2, stride = 2))        
        
        # block-4 3*conv3-512 maxPool
        net.append(nn.Conv2d(in_channels = 256,out_channels = 512, padding = 1, kernel_size = 3,stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size = 2, stride = 2))    
        
        # block-5 3*conv3-512 maxPool
        net.append(nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3,stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels = 512,out_channels = 512, padding = 1, kernel_size = 3, stride = 1))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size = 2, stride = 2))  
        
        # 将网络添加至类属性中
        self.feature = nn.Sequential(*net)
        
        # 定义分类器,全连接层
        classifier = []
        # 2*FC-4096,1*FC-num_classes
        classifier.append(nn.Linear(in_features = 512*7*7,out_features = 4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features = 4096,out_features = 4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))    
        
        classifier.append(nn.Linear(in_features = 4096,out_features = self.num_classes))
        
        # 将分类器添加至类属性中
        self.classifier = nn.Sequential(*classifier)
        
        # 前向
    def forward(self,x):
        feature = self.feature(x) # 输入张量
        feature = feature.view(x.size(0),-1)# reshape x 变成[batch_size,channels*width*height]
        result = self.classifier(feature)
            
        return result
        
if __name__=='__main__':
    x = torch.rand(size=(8,3,224,224))
    print(x.shape)
    vgg = VGG16(num_classes = 1000)
    out = vgg(x)
    print(out.size())