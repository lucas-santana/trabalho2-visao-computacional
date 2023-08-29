from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DatasetTypeEnum import DataSetType
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
            self.num_classes = 10

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
            self.num_classes = 10
            
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
        
        self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)