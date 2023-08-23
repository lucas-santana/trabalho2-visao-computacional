import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hyperparameters import batch_size, num_classes, learning_rate, num_epochs

from NeuralNetwork import NeuralNetwork
from LeNet5 import LeNet5
from DataSet import DataSet


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

def train_lenet5(dataset, batch_size, num_classes, learning_rate, num_epochs):
    
    if dataset == "FashionMNIST":
        data = DataSet('FashionMNIST', batch_size=batch_size)
    
        train_loader = data.train_dataloader
        test_loader = data.test_dataloader
    
        model = LeNet5(num_classes=num_classes, grayscale=True).to(device)

    elif dataset == "CIFAR10":
        data = DataSet('CIFAR10', batch_size=batch_size)
    
        train_loader = data.train_dataloader
        test_loader = data.test_dataloader
    
        model = LeNet5(num_classes=num_classes, grayscale=False).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)
    print("Done!")

def main():    
    train_lenet5(dataset = "CIFAR10", batch_size=batch_size, learning_rate=learning_rate, num_classes=num_classes, num_epochs=num_epochs)

if __name__ == "__main__":
    main()