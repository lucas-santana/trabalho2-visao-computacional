import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DatasetTypeEnum import DataSetType
from torch.utils.data import random_split

from Fer2013 import Fer2013

class DataSet():
    def __init__(self, dataset_type, batch_size, input_size, num_workers=0):
        print("TAMANHO DO BATCH: ", batch_size)
        print("NUM WORKERS: ", num_workers)
        
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
            self.gray_scale = True # 32x32x3
            self.dataset_name = 'CIFAR10'
            
            # transform = transforms.Compose([
            #                             transforms.ToTensor(),
            #                             # transforms.Normalize(
            #                             #     (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            #                             # )
            #                         ])
            
            transform = transforms.Compose([
                                transforms.Grayscale(num_output_channels=1),
                                transforms.Resize((input_size, input_size)),
                                transforms.ToTensor()])
            
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

        elif self.dataset_type == DataSetType.FER2013.value:
            """
                The data consists of 48x48 pixel grayscale images of faces.
                classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
                The training set consists of 28,709 examples. 
                The public test consists of 3,589 examples.
            """
            self.gray_scale = True
            self.dataset_name = 'FER2013'
            
            transform = transforms.Compose([
                                        transforms.Resize((input_size, input_size)),
                                        transforms.ToTensor()])
            
            filepath = './data/fer2013/fer2013.csv'
            
            self.training_data = Fer2013('./data/fer2013/fer2013.csv', split= "TRAIN", transform= transform)
            self.valid_data = Fer2013(filepath, split= "PUBLIC_TEST", transform= transform)
            self.test_data = Fer2013(filepath, split= "PRIVATE_TEST", transform= transform)
            
            
        
        if self.dataset_type == DataSetType.FER2013.value:
            self.classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral')
            self.num_classes = len(self.classes)
            
            self.train_dataloader = DataLoader(self.training_data, batch_size= batch_size, shuffle= True, num_workers= num_workers)
            self.valid_dataloader = DataLoader(self.valid_data, batch_size= batch_size, num_workers= num_workers)
            self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        else:
            self.classes = self.training_data.classes
            self.num_classes = len(self.classes)
                
            train_size = int(0.8 * len(self.training_data))
            valid_size = len(self.training_data) - train_size

            self.training_data, self.valid_data = random_split(self.training_data, [train_size, valid_size])
            
            self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
            self.valid_dataloader = DataLoader(self.valid_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
            self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
        
