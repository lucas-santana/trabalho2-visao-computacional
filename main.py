import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hyperparameters import batch_size, learning_rate, num_epochs, weight_decay, momentum

from DataSet import DataSet

from NetworksEnum import Networks
from DatasetTypeEnum import DataSetType

from NeuralNetwork import NeuralNetwork
from models.LeNet5 import LeNet5
from models.AlexNet import AlexNet
from models.VGG16 import VGG16


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

"""
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb
    https://github.com/gradient-ai/LeNet5-Tutorial
"""
    
def train(dataloader, model, loss_fn, optimizer):
    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)
    # Indica que o modelo está em processo de treinamento
    model.train()

    # Itera sobre os lotes
    for batch, (X, y) in enumerate(dataloader):
        # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
        X, y = X.to(device), y.to(device)

        # Faz a predição para os valores atuais dos parâmetros
        pred = model(X)

        # Estima o valor da função de perda
        loss = loss_fn(pred, y)

        # Backpropagation

        # Limpa os gradientes
        optimizer.zero_grad()

        # Estima os gradientes
        loss.backward()

        # Atualiza os pesos da rede
        optimizer.step()

        # LOG: A cada 100 lotes (iterações) mostra a perda
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    # Obtém o tamanho do dataset
    size = len(dataloader.dataset)

    # Obtém o número de lotes (iterações)
    num_batches = len(dataloader)

    # Indica que o modelo está em processo de teste    
    model.eval()

    # Inicializa a perda de teste e a quantidade de acertos com 0
    test_loss, correct = 0, 0

    # Desabilita o cálculo do gradiente
    with torch.no_grad():
        # Itera sobre o conjunto de teste
        for X, y in dataloader:
            # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
            X, y = X.to(device), y.to(device)
            # Realiza a predição
            pred = model(X)

            # Calcula a perda
            test_loss += loss_fn(pred, y).item()
            # Verifica se a predição foi correta
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Determina a perda média e a proporção de acertos
    test_loss /= num_batches
    correct /= size
    # LOG: mostra a acurácia e a perda
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = weight_decay, momentum = momentum)
    
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(data.train_dataloader, model, loss_fn, optimizer)
        test(data.test_dataloader, model, loss_fn)
    print("Done!")
    
    
    return model

def main():
    dataset = DataSetType.FASHIONMNIST
    network = Networks.LENET5
    
    print("DataSet: ", dataset.name)
    print("Network: ", network.name)
    
    model = model_train(network=network.value, dataset=dataset.value, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs)
    torch.save(model.state_dict(), "{}_{}_model_weights.pth".format(dataset.name, network.name))

if __name__ == "__main__":
    main()