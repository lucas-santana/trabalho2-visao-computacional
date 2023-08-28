from torch import nn
# classe herda de nn.Module
class AlexNet(nn.Module):
    def __init__(self, num_classes, gray_scale=False):
        super(AlexNet, self).__init__()

        self.gray_scale = gray_scale
        self.num_classes = num_classes
        
        if self.gray_scale:
            in_channels = 1
        else:
            in_channels = 3

        self.layer1 = nn.Sequential(
            nn.Conv2d(3*in_channels, 96*in_channels, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(96*in_channels, 256*in_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(256*in_channels, 384*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384*in_channels),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(384*in_channels, 384*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384*in_channels),
            nn.ReLU())
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(384*in_channels, 256*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216*in_channels, 4096*in_channels),
            nn.ReLU())
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096*in_channels, 4096*in_channels),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096*in_channels, num_classes))
        
    def forward(self, x):
        print("SHAPE ", x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out