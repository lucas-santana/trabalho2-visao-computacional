import argparse
import os
import torch
import numpy as np

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms

from Hyperparameters import batch_size, learning_rate, weight_decay, momentum

from DataSet import DataSet

from NetworksEnum import Networks
from DatasetTypeEnum import DataSetType

from NeuralNetwork import NeuralNetwork
from networks.LeNet5 import LeNet5
from networks.AlexNet import AlexNet
from networks.VGG16 import VGG16

from plot import plot_loss, plot_acc, plot_confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

writer = SummaryWriter()

torch.manual_seed(7)
"""
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb
    https://github.com/gradient-ai/LeNet5-Tutorial
    https://colab.research.google.com/drive/1J7ViHL4eF_Ib6QAc_9yW82je0iyf8Hca?usp=sharing#scrollTo=iAxXEEyNcdw8
    https://nvsyashwanth.github.io/machinelearningmaster/cifar-10/
"""
    
def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    running_loss = 0.
    correct = 0
    total = 0
    
    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)
    
    # Obtém o número de lotes (iterações)   
    num_batches = len(dataloader)
        
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # Itera sobre os lotes
    for step, (X, y) in enumerate(dataloader):
        
        # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
        X, y = X.to(device), y.to(device)
        
        # Compute prediction and loss
        # forward pass        
        pred = model(X) # Faz a predição para os valores atuais dos parâmetros
        loss = loss_fn(pred, y)  # Estima o valor da função de perda
    
        # Backpropagation
        optimizer.zero_grad() # Limpa os gradientes
        loss.backward() # Estima os gradientes
        optimizer.step() # Atualiza os pesos da rede

        running_loss += loss.item()
                 
        _ , predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        if step % 100 == 0:
            
            loss, current = loss.item(), step * len(X)
            
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
               
    train_loss = running_loss/num_batches
    accu = 100. * correct/total
    
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss, accu))
    return train_loss, accu

def validation_loop(dataloader, model, loss_fn, epoch):
    valid_loss = 0.0
    correct = 0
    total = 0

    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)

    # Obtém o número de lotes (iterações)
    num_batches = len(dataloader)
    
    model.eval()
    with torch.no_grad():
        
        for X, y in dataloader:
            
            X, y = X.to(device), y.to(device)

            output = model(X)

            valid_loss += loss_fn(output, y).item()

            _ , predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    valid_loss = valid_loss/num_batches
    accu = 100.*(correct/size)

    print('Valid Loss: %.3f | Accuracy: %.3f'%(valid_loss, accu)) 

    return valid_loss, accu      

def test_loop(dataloader, model, loss_fn, epoch):

    running_loss = 0
    correct = 0
    total = 0

   # Obtém o tamanho do dataset
    size = len(dataloader.dataset)

    # Obtém o número de lotes (iterações)
    num_batches = len(dataloader)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices   
    model.eval() # Indica que o modelo está em processo de teste 
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    # Desabilita o cálculo do gradiente
    with torch.no_grad():
        # Itera sobre o conjunto de teste
        count = 0
        for X, y in dataloader:
            count += 1
            # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
            X, y = X.to(device), y.to(device)
            # Realiza a predição
            pred = model(X)

            # Calcula a perda
            running_loss += loss_fn(pred, y).item()

            _ , predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    # test_loss divide pelo n de batches ou tam do dataset
    
    test_loss = running_loss/num_batches
    accu = 100.*(correct/size)

    # LOG: mostra a acurácia e a perda
    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss, accu)) 

    return test_loss, accu

def build_data(network, dataset, batch_size):
    if network == Networks.LENET5.value:
        data = DataSet(dataset, batch_size=batch_size, input_size=32)
    elif network == Networks.ALEXNET.value:
        data = DataSet(dataset, batch_size=batch_size, input_size=227)
    elif network == Networks.VGG16.value:
        data = DataSet(dataset, batch_size=batch_size, input_size=224)
    else:
        raise Exception("Network not supported: ", network)
    
    return data

def model_train(network, dataset, batch_size, learning_rate, num_epochs):
    
    data = build_data(network, dataset, batch_size=batch_size)
    
    if network == Networks.LENET5.value:
        model = LeNet5(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    elif network == Networks.ALEXNET.value:
        model = AlexNet(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    elif network == Networks.VGG16.value:
        model = VGG16(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    else:
        raise Exception("Network not supported: ", network)
    
    print(f"Model structure: {model}\n\n")

    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    
    val_losses = []
    val_accuracies = []
    
    
    test_losses = []
    test_accuracies = []

    valid_loss_min = np.Inf
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        train_loss, train_acc = train_loop(data.train_dataloader, model, loss_fn, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # test_loss, test_acc = test_loop(data.test_dataloader, model, loss_fn, epoch)
        # test_losses.append(test_loss)
        # test_accuracies.append(test_acc)
        
        valid_loss, valid_acc = validation_loop(data.valid_dataloader, model, loss_fn, epoch)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        
        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased from : {valid_loss_min} ----> {valid_loss} ----> Saving Model.......")
            torch.save(model.state_dict(), "models/{}_{}.pth".format(data.dataset_name, type(model).__name__))
            valid_loss_min = valid_loss
             
    plot_acc(train_accuracies, val_accuracies)
    plot_loss(train_losses, val_losses)
    plot_confusion_matrix(model, data.test_dataloader )

    return model

def main():
    parser = argparse.ArgumentParser(description='Training on datasets using networks from scratch')
    
    parser.add_argument('-n',  '--network', type=str, choices=['LENET5', 'ALEXNET', 'VGG16'],  required=True, help='LENET5, ALEXNET or VGG16')
    parser.add_argument('-d',  '--dataset', type=str, choices=['FASHIONMNIST', 'CIFAR10'], required=True, help='FASHIONMNIST or CIFAR10')
    parser.add_argument('-e',  '--epochs', type=int, default=5, required=False, help='Number of epochs. Default = 5')
    
    args = parser.parse_args()
    
    dataset = DataSetType[args.dataset]
    network = Networks[args.network]
    num_epochs = args.epochs
    
    print("DataSet: ", dataset.name)
    print("Network: ", network.name)
    print("Épocas: ", args.epochs)
    
    model = model_train(network=network.value, dataset=dataset.value, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs)
    writer.flush()
    writer.close()
    # torch.save(model.state_dict(), "models/{}_{}.pth".format(dataset.name, network.name))

if __name__ == "__main__":
    main()