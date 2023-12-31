
from torch import nn
from torchvision import datasets, transforms

#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes, gray_scale=False):
        super(LeNet5, self).__init__()
        
        self.gray_scale = gray_scale
        self.num_classes = num_classes
        
        if self.gray_scale:
            in_channels = 1
        else:
            in_channels = 3
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1*in_channels, 6*in_channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(6*in_channels, 16*in_channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc = nn.Linear(400*in_channels, 120*in_channels)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120*in_channels, 84*in_channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84*in_channels, self.num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out