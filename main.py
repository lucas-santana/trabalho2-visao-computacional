"""
    https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb
    https://github.com/gradient-ai/LeNet5-Tutorial
    https://colab.research.google.com/drive/1J7ViHL4eF_Ib6QAc_9yW82je0iyf8Hca?usp=sharing#scrollTo=iAxXEEyNcdw8
    https://nvsyashwanth.github.io/machinelearningmaster/cifar-10/
    https://www.youtube.com/watch?v=gbrHEsbTdF0
    https://www.youtube.com/watch?v=doT7koXt9vw
    https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
    https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
"""
# -------------------- Bibliotecas Externas --------------------
import argparse
import torch
import logging
import gc
import numpy as np
import pandas as pd
import time
import logging
import gc

from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torchvision import datasets, transforms

from pprint import pformat
from tqdm import tqdm

#------------------------------------------------------------
from DatasetTypeEnum import DataSetType
from NetworksEnum import Networks
from util import make_experiment_folder, parse_exp_json, get_acc_data, get_loss_data, save_acc_result,check_exp_exist, check_model_exist

from Hyperparameters import batch_size, learning_rate, weight_decay, momentum

from DataSet import DataSet

from NetworksEnum import Networks
from DatasetTypeEnum import DataSetType

from networks.LeNet5 import LeNet5
from networks.AlexNet import AlexNet
from networks.VGG16 import VGG16
from networks.VGG11 import VGG11

from plot import plot_loss, plot_acc, plot_confusion_matrix, plot_samples, save_plots
#------------------------------------------------------------

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# torch.set_flush_denormal(True)
torch.manual_seed(7)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer, epoch):
    running_loss = 0.0
    correct = 0
    
    # Obtém o tamanho do dataset
    dataset_size = len(dataloader.dataset)
    
    # Obtém o número de lotes (iterações)   
    num_batches = len(dataloader)
        
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    # Itera sobre os lotes
    tic = time.perf_counter()
    scaler = GradScaler()
    
    for step, (X, y) in enumerate(tqdm(dataloader)):
        torch.cuda.empty_cache()
        
        
        # transforma as entradas no formato do dispositivo utilizado (CPU ou GPU)
        X, y = X.to(device), y.to(device)
        
        # Backpropagation
        
        # Runs the forward pass with autocasting.
        with autocast():
            # Compute prediction and loss
            # forward pass        
            pred = model(X) # Faz a predição para os valores atuais dos parâmetros
            loss = loss_fn(pred, y)  # Estima o valor da função de perda
        
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()# Estima os gradientes
        
        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer) # Atualiza os pesos da rede
        
        # Updates the scale for next iteration.
        scaler.update()
        optimizer.zero_grad() # Limpa os gradientes
        
        # loss é um tensor de 1 valor, por isso o item()
        # loss.item() contains the loss of the entire mini-batch,  
        # It’s because the loss given loss functions is divided by the number of elements 
        # i.e. the reduction parameter is mean by default(divided by the batch size). 
        # That’s why loss.item() is multiplied by the batch size, given by inputs.size(0), while calculating running_loss
        
        running_loss += loss.item() * X.size(0)

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        del pred, loss, X, y
        gc.collect()
    toc = time.perf_counter()
        
    # É mais correto multiplicar pelo tamaho do batch e depois dividir pelo numer de amostras do que acumular e dividir pelo num_batches, 
    # levando em conta que nem todos batches tem o mesmo tamanho
    train_loss = running_loss / dataset_size
    accu = 100. * correct/dataset_size
    
    print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss, accu))
    return train_loss, accu

def validation_loop(dataloader, model, loss_fn, epoch):
    valid_loss = 0.0
    correct = 0

    # Obtém o tamanho do dataset
    dataset_size = len(dataloader.dataset)

    # Obtém o número de lotes (iterações)
    num_batches = len(dataloader)
    
    model.eval()
    with torch.no_grad():
        

        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            output = model(X)

            valid_loss += loss_fn(output, y).item()* X.size(0)

            correct += (output.argmax(1) == y).type(torch.float).sum().item()
    
    valid_loss = valid_loss / dataset_size
    accu = 100.*(correct/dataset_size)

    print('Valid Loss: %.3f | Accuracy: %.3f'%(valid_loss, accu)) 

    return valid_loss, accu      

def test_loop(dataloader, model, loss_fn, epoch=None):    
    running_loss = 0
    correct = 0

   # Obtém o tamanho do dataset
    dataset_size = len(dataloader.dataset)

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
            running_loss += loss_fn(pred, y).item()* X.size(0)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # test_loss divide pelo n de batches ou tam do dataset
    
    test_loss = running_loss / dataset_size
    accu = 100.*(correct/dataset_size)

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
    elif network == Networks.VGG11.value:
        logging.info(f"Construindo dataset para a rede VGG11")
        data = DataSet(dataset, batch_size=batch_size, input_size=224, num_workers=num_workers)
    else:
        raise Exception("Network not supported: ", network)
    
    return data

def get_model(network, num_classes, gray_scale):
    if network == Networks.LENET5.value:
        model = LeNet5(num_classes=num_classes, gray_scale=gray_scale).to(device)
    elif network == Networks.ALEXNET.value:
        model = AlexNet(num_classes=num_classes, gray_scale=gray_scale).to(device)
    elif network == Networks.VGG16.value:
        model = VGG16(num_classes=num_classes, gray_scale=gray_scale).to(device)
    elif network == Networks.VGG11.value:
        model = VGG11(num_classes=num_classes, gray_scale=gray_scale).to(device)
    else:
        raise Exception("Network not supported: ", network)
    
    return model

def model_train(experiment_id):
    """
        
    """
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
    
    model = get_model(network, data.num_classes, data.gray_scale)
    
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

    # Salvar as méticas de acordo com o modelo com a menor loss
    valid_loss_min = np.Inf
    best_model_test_acc = 0
    best_model_val_acc = 0
    
    logging.info("Iniciando treinamento")
    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}\n-------------------------------")
        print(f"Epoch {epoch+1}/{num_epochs}\n-------------------------------")
        
        train_loss, train_acc = train_loop(data.train_dataloader, model, loss_fn, optimizer, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss, test_acc = test_loop(data.test_dataloader, model, loss_fn, epoch)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        valid_loss, valid_acc = validation_loop(data.valid_dataloader, model, loss_fn, epoch)
        val_losses.append(valid_loss)
        val_accuracies.append(valid_acc)
        
        """
            Pode acontecer da acurácia ir aumentando com as épocas mas a loss só aumenta, caracterizando overfiting
            Por isso salvo o modelo, com a menor loss
        """
        if valid_loss <= valid_loss_min:
            logging.info(f"Validation loss decreased from : {valid_loss_min} ----> {valid_loss} ----> Saving Model.......")
            logging.info(f"Validation acc:  {valid_acc}")
            logging.info(f"Best Test acc from {best_model_test_acc} ----> {test_acc}")
            
            valid_loss_min = valid_loss
            best_model_test_acc = test_acc
            best_model_val_acc = valid_acc
            # torch.save(model.state_dict(), f"results/experiment_{experiment_id}/model/model.pth")
        
        logging.info(f"Época {epoch+1}/{num_epochs}")
        logging.info(f"loss: {train_loss} - accuracy: {train_acc} - val_loss: {valid_loss} - val_accuracy: {valid_acc}")
        logging.info(f"[Test] ---> accuracy: {test_acc} - loss: {test_loss}")
        logging.info(f"test acc from best model : {best_model_test_acc}")
        
        print(f"test acc from best model : {best_model_test_acc}")
        
        #----------- Plot parcial --------------
        if (epoch + 1) % 5 == 0:
            plot_acc(experiment_id, train_accuracies, val_accuracies, filename="partial_acc.pdf")
            plot_loss(experiment_id, train_losses, val_losses, filename="partial_loss.pdf")
            save_plots(experiment_id, train_accuracies, val_accuracies, train_losses, val_losses, filename_acc="partial_train_val_acc.pdf", filename_loss="partial_train_val_loss.pdf")
    
    end_time = time.perf_counter()
    train_time = end_time - start_time

    epochs_values = [i+1 for i in range(0, num_epochs)]
    tab_acc = {
                "epoch": epochs_values,
                "train_acc": train_accuracies,
               "val_acc": val_accuracies,
               "test_acc": test_accuracies
            }
    
    tab_loss = {
                "epoch": epochs_values,
                "train_loss": train_losses,
               "val_loss": val_losses,
               "test_loss": test_losses
            }
    
    logging.info(f"Tempo treinamento:  {train_time:.2f} seconds")
    logging.info(f"Menor loss: {valid_loss_min}")
    logging.info(f"Acurácia de teste do melhor modelo: {best_model_test_acc}")
    logging.info("Métricas")
    logging.info("ACC")
    logging.info(pformat(tab_acc))
    logging.info("LOSS")
    logging.info(pformat(tab_loss))
    
    df_acc = pd.DataFrame(tab_acc)
    df_acc.to_csv(f'results/experiment_{experiment_id}/hist_acc.csv', index=False, float_format='%.2f')
    
    df_loss = pd.DataFrame(tab_loss)
    df_loss.to_csv(f'results/experiment_{experiment_id}/hist_loss.csv', index=False, float_format='%.2f')

    plot_acc(experiment_id, train_accuracies, val_accuracies)
    plot_loss(experiment_id, train_losses, val_losses)
    save_plots(experiment_id, train_accuracies, val_accuracies, train_losses, val_losses)
    
    save_pred(experiment_id, model, data.test_dataloader)
    
    y_pred, y_true = get_pred(experiment_id)
    
    plot_confusion_matrix(experiment_id, data, y_pred, y_true)
    
    save_acc_result(experiment_id, best_model_test_acc, best_model_val_acc, train_time)
    
    # model_eval(experiment_id, train_time=train_time)


def model_eval(experiment_id, train_time = -1):
    """
        Precisa que modelo esteja salvo para realizar o teste
    """
    logging.info("Rodando evaluation")
    parameters = parse_exp_json(experiment_id)
    
    network = Networks[parameters['network']].value
    dataset = DataSetType[parameters['dataset']].value
    batch_size = parameters['batch_size']
    learning_rate = parameters['learning_rate']
    num_epochs = parameters['epochs']
    num_workers = parameters['num_workers']
    
    data = build_data(network, dataset, batch_size = batch_size, num_workers=num_workers)
        
    model = get_model(network, data.num_classes, data.gray_scale)
    
    if check_model_exist(experiment_id):
        model.load_state_dict(torch.load(f"results/experiment_{experiment_id}/model/model.pth"))
        # Calcular a acuracia de teste para o melhor modelo
        loss_fn = nn.CrossEntropyLoss()
        test_loss, test_acc = test_loop(data.test_dataloader, model, loss_fn, 0)
        val_loss, val_acc = validation_loop(data.valid_dataloader, model, loss_fn, 0)
        save_acc_result(experiment_id, test_acc, val_acc, train_time)
    
    
    # carregar arquivo csv para plotar grafico acurácias
    train_accuracies, val_accuracies, test_accuracies = get_acc_data(experiment_id)
    train_losses, val_losses, test_losses = get_loss_data(experiment_id)
    
    plot_acc(experiment_id, train_accuracies, val_accuracies)
    plot_loss(experiment_id, train_losses, val_losses)
    
    
    y_pred, y_true = get_pred(experiment_id)
    plot_confusion_matrix(experiment_id, data, y_pred, y_true)
    
def get_pred(exp_id):
    """
        ler arquivo predicoes predictions.csv
    """
    predictions_filename = f'results/experiment_{exp_id}/predictions.csv'
    data = pd.read_csv(predictions_filename)
    
    return data['target'], data['prediction']

def save_pred(experiment_id, model, dataloader):
    """
        Salvar predições no arquivo predictions.csv
    """
    y_pred = []
    y_true = []

    model.eval()
    # iterate over test data
    for inputs, labels in dataloader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    tab_pred = {"target": y_true,
               "prediction": y_pred
    }

    df_pred = pd.DataFrame(tab_pred)
    df_pred.to_csv(f'results/experiment_{experiment_id}/predictions.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Training on datasets using networks from scratch')
    
    parser.add_argument('-e',  '--experiment_id', type=int, default="0",  required=True, help='Id do experimento a ser executado')
    parser.add_argument('-t',  '--train', type=int, default="1",  required=True, help='Treinar(1) ou Avaliar(0)')
    args = parser.parse_args()
    
    logging.basicConfig(
        filename=f"logs/exp_{args.experiment_id}.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if check_exp_exist(args.experiment_id):
        if args.train == 1:
            logging.info(f"--------------------- Iniciando Novo Treinamento {args.experiment_id} ---------------------")
            make_experiment_folder(args.experiment_id)
            model_train(args.experiment_id)
        elif args.train == 0:
            logging.info(f"--------------------- Iniciando Teste {args.experiment_id} ---------------------")
            model_eval(args.experiment_id)
    else:
        logging.error(f"Erro! Experimento {args.experiment_id} não existe")


if __name__ == "__main__":
    main()