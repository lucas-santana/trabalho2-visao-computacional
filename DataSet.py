from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DataSet():
    def __init__(self, dataset_type, batch_size):
        self.dataset_type = dataset_type
        
        if self.dataset_type == 'FashionMNIST':
            
            self.training_data = datasets.FashionMNIST(
                        root="data",
                        train=True,
                        download=True,
                        transform = transforms.Compose([
                                                          transforms.Resize((32,32)),
                                                          transforms.ToTensor()
                                                        ]),
            )
            
            self.test_data = datasets.FashionMNIST(
                    root="data",
                    train=False,
                    download=True,
                    transform = transforms.Compose([
                                                transforms.Resize((32,32)),
                                                transforms.ToTensor()
                                            ])
            )
            
            self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size)
            self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)
        
        elif self.dataset_type == 'CIFAR10':
            
            self.training_data = datasets.CIFAR10(
                        root="data",
                        train=True,
                        download=True,
                        transform = transforms.Compose([
                                                          transforms.Resize((32,32)),
                                                          transforms.ToTensor()
                                                        ]),
            )
            
            self.test_data = datasets.CIFAR10(
                    root="data",
                    train=False,
                    download=True,
                    transform = transforms.Compose([
                                                transforms.Resize((32,32)),
                                                transforms.ToTensor()
                                            ])
            )
            
            self.train_dataloader = DataLoader(self.training_data, batch_size=batch_size)
            self.test_dataloader = DataLoader(self.test_data, batch_size=batch_size)
        
        else:
            raise Exception("Dataset not supported!")
        