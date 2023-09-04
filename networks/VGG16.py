from torch import nn

class VGG16(nn.Module):
    def __init__(self, num_classes, gray_scale=False):
        super(VGG16, self).__init__()
        
        self.gray_scale = gray_scale
        self.num_classes = num_classes
        
        if self.gray_scale:
            in_channels = 1
        else:
            in_channels = 3
        
        
        #------------------------------------BLOCO 1------------------------------------
        # conv3-64
        self.layer1 = nn.Sequential(
            nn.Conv2d(1*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64*in_channels),
            nn.ReLU())
        
        # conv3-64
        self.layer2 = nn.Sequential(
            nn.Conv2d(64*in_channels, 64*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64*in_channels),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        #------------------------------------BLOCO 2------------------------------------
        # conv3-128
        self.layer3 = nn.Sequential(
            nn.Conv2d(64*in_channels, 128*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128*in_channels),
            nn.ReLU())
        
        # conv3-128
        self.layer4 = nn.Sequential(
            nn.Conv2d(128*in_channels, 128*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        #------------------------------------BLOCO 3------------------------------------
        # conv3-256 
        self.layer5 = nn.Sequential(
            nn.Conv2d(128*in_channels, 256*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256*in_channels),
            nn.ReLU())
        
        # conv3-256 
        self.layer6 = nn.Sequential(
            nn.Conv2d(256*in_channels, 256*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256*in_channels),
            nn.ReLU())
        # conv3-256 
        self.layer7 = nn.Sequential(
            nn.Conv2d(256*in_channels, 256*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        #------------------------------------BLOCO 4------------------------------------
        # conv3-512
        self.layer8 = nn.Sequential(
            nn.Conv2d(256*in_channels, 512*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512*in_channels),
            nn.ReLU())
        
        # conv3-512
        self.layer9 = nn.Sequential(
            nn.Conv2d(512*in_channels, 512*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512*in_channels),
            nn.ReLU())
        
        # conv3-512
        self.layer10 = nn.Sequential(
            nn.Conv2d(512*in_channels, 512*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        #------------------------------------BLOCO 5------------------------------------
        # conv3-512
        self.layer11 = nn.Sequential(
            nn.Conv2d(512*in_channels, 512*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512*in_channels),
            nn.ReLU())
        
        # conv3-512
        self.layer12 = nn.Sequential(
            nn.Conv2d(512*in_channels, 512*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512*in_channels),
            nn.ReLU())
        
        # conv3-512
        self.layer13 = nn.Sequential(
            nn.Conv2d(512*in_channels, 512*in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512*in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        
        # FC-4096
        self.fc = nn.Sequential(
            nn.Linear(7*7*512*in_channels, 4096*in_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        # FC-4096
        self.fc1 = nn.Sequential(
            nn.Linear(4096*in_channels, 4096*in_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        # FC-10
        self.fc2= nn.Sequential(
            nn.Linear(4096*in_channels, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out