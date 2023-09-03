
import matplotlib.pyplot as plt
import torch

import seaborn as sn
import pandas as pd
import numpy as np
import pandas as pd

from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix

def plot_acc(experiment_id, train_accu, eval_accu, title="Train vs Test Accuracy"):
    f = plt.figure() # gera uma figura do gr ́afico (antes de desenh ́a-lo)

    plt.plot(train_accu,'-o')
    plt.plot(eval_accu,'-o')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend(['Treino','Teste'])
    plt.title(title)
    
    f.savefig(f'results/experiment_{experiment_id}/acc.pdf')
    
def plot_loss(experiment_id, train_losses, eval_losses, title="Train vs Test Losses"):
    f = plt.figure() # gera uma figura do gr ́afico (antes de desenh ́a-lo)

    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(['Treino','Teste'])
    plt.title(title)
    
    f.savefig(f'results/experiment_{experiment_id}/loss.pdf')
    
def plot_confusion_matrix(experiment_id, data, y_pred, y_true):
    
    # constant for classes
    classes = data.classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    # Create pandas dataframe
    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
    
    df_cm.to_csv(f'results/experiment_{experiment_id}/df_cm.csv')
    
    plt.figure(figsize = (12, 7))
    sn.heatmap(df_cm, annot=True, cbar=None, cmap="OrRd",fmt="d")
    
    plt.title("Matriz de confusão"), plt.tight_layout()

    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    
    plt.savefig(f'results/experiment_{experiment_id}/cm.pdf')

def plot_samples(experiment_id, dataloader):
    for images, _ in dataloader:
        print('images.shape:', images.shape)
        f = plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        f.savefig(f'results/experiment_{experiment_id}/image_grid.pdf')
        break

def save_plots(experiment_id, train_acc, valid_acc, train_loss, valid_loss):
    """
        Salvar graficos de loss e acurácia
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='Acurácia de Treino'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='Acurácia de validação'
    )
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(f'results/experiment_{experiment_id}/train_val_acc.pdf')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='Loss de Treino'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='Perda de Validação'
    )
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    
    plt.savefig(f'results/experiment_{experiment_id}/train_val_loss.pdf')