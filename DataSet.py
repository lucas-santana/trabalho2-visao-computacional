from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DatasetTypeEnum import DataSetType
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class DataSet():
    def __init__(self, dataset_type, batch_size, input_size):
        
        self.dataset_type = dataset_type
        
        
        
        """
            - The output of torchvision datasets are PILImage images of range [0, 1]. 
            - We transform them to Tensors of normalized range [-1, 1].
        """
        transform = transforms.Compose([
                                        transforms.Resize((input_size, input_size)),
                                        transforms.ToTensor()])

        if self.dataset_type == DataSetType.FASHIONMNIST.value:
            """
                The Fashion-MNIST dataset contains 60,000 training images (and 10,000 test images) of fashion and clothing items, 
                taken from 10 classes. 
                Each image is a standardized 28x28x1 size in grayscale (784 total pixels).
            """
            self.gray_scale = True
            self.dataset_name = 'FASHIONMNIST'
            

            self.training_data = datasets.FashionMNIST(
                        root="data",
                        train=True,
                        download=True,
                        transform = transform
            )
            
            self.test_data = datasets.FashionMNIST(
                    root="data",
                    train=False,
                    download=True,
                    transform = transform
            )
        
        elif self.dataset_type == DataSetType.CIFAR10.value:
            """
                The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
            """
            self.gray_scale = False # 32x32x3
            self.dataset_name = 'CIFAR10'
            
            self.training_data = datasets.CIFAR10(
                        root="data",
                        train=True,
                        download=True,
                        transform = transform
            )
            
            self.test_data = datasets.CIFAR10(
                    root="data",
                    train=False,
                    download=True,
                    transform = transform
            )
        else:
            raise Exception("Dataset not supported!")
        
        self.classes = self.training_data.classes
        self.num_classes = len(self.classes)
        
        train_size = int(0.8 * len(self.training_data))
        valid_size = len(self.training_data) - train_size

        self.training_data, self.valid_data = random_split(self.training_data, [train_size, valid_size])
        
        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        
        for images, _ in self.train_dataloader:
            print('images.shape:', images.shape)
            f = plt.figure(figsize=(16,8))
            plt.axis('off')
            plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
            f.savefig('results/image_grid.png')
            break