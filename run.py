import gc
import torch
import numpy as np
import pandas as pd
import time
import logging
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

from pprint import pformat
from tqdm import tqdm

from torch import nn
# torch.set_flush_denormal(True)


from torchvision import datasets, transforms

from Hyperparameters import batch_size, learning_rate, weight_decay, momentum

from DataSet import DataSet

from NetworksEnum import Networks
from DatasetTypeEnum import DataSetType

from networks.LeNet5 import LeNet5
from networks.AlexNet import AlexNet
from networks.VGG16 import VGG16

from plot import plot_loss, plot_acc, plot_confusion_matrix, plot_samples

from util import make_experiment_folder, parse_exp_json, get_acc_data, get_loss_data

"""
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb
    https://github.com/gradient-ai/LeNet5-Tutorial
    https://colab.research.google.com/drive/1J7ViHL4eF_Ib6QAc_9yW82je0iyf8Hca?usp=sharing#scrollTo=iAxXEEyNcdw8
    https://nvsyashwanth.github.io/machinelearningmaster/cifar-10/
    https://www.youtube.com/watch?v=gbrHEsbTdF0
    https://www.youtube.com/watch?v=doT7koXt9vw
    https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    logging.info(f"Treinando época {epoch}")
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
    tic = time.perf_counter()
    for step, (X, y) in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()
        
        
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
        

        # loss é um tensor de 1 valor, por isso o item()
        running_loss += loss.item()

        total += y.size(0)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        if step % 100 == 0:
            
            loss, current = loss.item(), step * len(X)
            
            # print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        
        
        
        del pred, loss, X, y
        gc.collect()
    toc = time.perf_counter()
        
        
    # print(f"epoch {epoch} took {toc-tic:.2f} seconds")
               
    train_loss = running_loss
    accu = 100. * correct/total
    
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss, accu))
    return train_loss, accu

def validation_loop(dataloader, model, loss_fn, epoch):
    logging.info(f"Validando época {epoch}")
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

            
            total += y.size(0)
            correct += (output.argmax(1) == y).type(torch.float).sum().item()
    
    valid_loss = valid_loss
    accu = 100.*(correct/size)

    print('Valid Loss: %.3f | Accuracy: %.3f'%(valid_loss, accu)) 

    return valid_loss, accu      

def test_loop(dataloader, model, loss_fn, epoch=None):
    logging.info(f"Testando época {epoch}")
    
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
        for X, y in dataloader:
            # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
            X, y = X.to(device), y.to(device)
            # Realiza a predição
            pred = model(X)

            # Calcula a perda
            running_loss += loss_fn(pred, y).item()

            total += y.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # test_loss divide pelo n de batches ou tam do dataset
    
    test_loss = running_loss
    accu = 100.*(correct/size)

    # LOG: mostra a acurácia e a perda
    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss, accu)) 

    return test_loss, accu

def build_data(network, dataset, batch_size, num_workers=0):
    if network == Networks.LENET5.value:
        logging.info(f"Construindo dataset para a rede LENET5")
        data = DataSet(dataset, batch_size=batch_size, input_size=32, num_workers=num_workers)
    elif network == Networks.ALEXNET.value:
        logging.info(f"Construindo dataset para a rede ALEXNET")
        data = DataSet(dataset, batch_size=batch_size, input_size=227, num_workers=num_workers)
    elif network == Networks.VGG16.value:
        logging.info(f"Construindo dataset para a rede VGG16")
        data = DataSet(dataset, batch_size=batch_size, input_size=224, num_workers=num_workers)
    else:
        raise Exception("Network not supported: ", network)
    
    return data

def model_train(experiment_id):
    
    parameters = parse_exp_json(experiment_id)
    
    logging.info("Parametros")
    logging.info(pformat(parameters))
    
    network = Networks[parameters['network']].value
    dataset = DataSetType[parameters['dataset']].value
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_epochs = parameters['epochs']
    num_workers = parameters['num_workers']
    
    
    data = build_data(network, dataset, batch_size = batch_size, num_workers=num_workers)
    
    plot_samples(experiment_id, data.train_dataloader)
    
    if network == Networks.LENET5.value:
        logging.info(f"Construindo modelo para a rede LENET5")
        model = LeNet5(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    elif network == Networks.ALEXNET.value:
        logging.info(f"Construindo modelo para a rede ALEXNET")
        model = AlexNet(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    elif network == Networks.VGG16.value:
        logging.info(f"Construindo modelo para a rede VGG16")
        model = VGG16(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    else:
        raise Exception("Network not supported: ", network)
    
    print(f"Model structure: {model}\n\n")
    logging.info(pformat(model))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    
    val_losses = []
    val_accuracies = []
    
    
    test_losses = []
    test_accuracies = []

    valid_loss_min = np.Inf
    best_model = model
    
    logging.info("Iniciando treinamento")
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}\n-------------------------------")
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        train_loss, train_acc = train_loop(data.train_dataloader, model, loss_fn, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = test_loop(data.test_dataloader, model, loss_fn, epoch)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        valid_loss, valid_acc = validation_loop(data.valid_dataloader, model, loss_fn, epoch)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        
        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased from : {valid_loss_min} ----> {valid_loss} ----> Saving Model.......")
            valid_loss_min = valid_loss
            best_model = model
    
    # torch.save(best_model.state_dict(), f"results/experiment_{experiment_id}/model/{data.dataset_name}_{type(best_model).__name__}.pth")         
    torch.save(best_model.state_dict(), f"results/experiment_{experiment_id}/model/model.pth")         
    
    # test_loss, test_acc = test_loop(data.test_dataloader, best_model, loss_fn, epoch)
     
    # dataframes das acurácias
    tab_acc = {"train_acc": train_accuracies,
               "val_acc": val_accuracies,
               "test_acc": test_accuracies}
    
    tab_loss = {"train_loss": train_losses,
               "val_loss": val_losses,
               "test_loss": test_losses}
    
    logging.info("Métricas")
    logging.info("ACC")
    logging.info(pformat(tab_acc))
    
    logging.info("LOSS")
    logging.info(pformat(tab_loss))
    
    df_acc = pd.DataFrame(tab_acc)
    df_acc.to_csv(f'results/experiment_{experiment_id}/acc.csv', index=False, float_format='%.2f')
    
    df_loss = pd.DataFrame(tab_loss)
    df_loss.to_csv(f'results/experiment_{experiment_id}/loss.csv', index=False, float_format='%.2f')

    plot_acc(experiment_id, train_accuracies, test_accuracies)
    plot_loss(experiment_id, train_losses, test_losses)
    plot_confusion_matrix(experiment_id, best_model, data, data.test_dataloader )

    return model

def model_eval(experiment_id):
    logging.info("Rodando evaluation")
    parameters = parse_exp_json(experiment_id)
    
    network = Networks[parameters['network']].value
    dataset = DataSetType[parameters['dataset']].value
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_epochs = parameters['epochs']
    
    data = build_data(network, dataset, batch_size = batch_size)
    
    plot_samples(experiment_id, data.train_dataloader)
    
    if network == Networks.LENET5.value:
        model = LeNet5(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    elif network == Networks.ALEXNET.value:
        model = AlexNet(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    elif network == Networks.VGG16.value:
        model = VGG16(num_classes=data.num_classes, gray_scale=data.gray_scale).to(device)
    else:
        raise Exception("Network not supported: ", network)
    
    model.load_state_dict(torch.load(f"results/experiment_{experiment_id}/model/model.pth"))
    model.eval()

    # carregar arquivo csv
    
    train_accuracies, _, test_accuracies = get_acc_data(experiment_id)
    train_losses, _, test_losses = get_loss_data(experiment_id)
    
    plot_acc(experiment_id, train_accuracies, test_accuracies)
    plot_loss(experiment_id, train_losses, test_losses)
    plot_confusion_matrix(experiment_id, model, data, data.test_dataloader )
    