
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
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Test'])
    plt.title(title)

    # plt.show()
    
    f.savefig(f'results/experiment_{experiment_id}/acc.pdf')
    
def plot_loss(experiment_id, train_losses, eval_losses, title="Train vs Test Losses"):
    f = plt.figure() # gera uma figura do gr ́afico (antes de desenh ́a-lo)

    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Test'])
    plt.title(title)

    # plt.show()
    
    f.savefig(f'results/experiment_{experiment_id}/loss.pdf')
    
def plot_confusion_matrix(experiment_id, model, data, dataloader):
    y_pred = []
    y_true = []

    model.eval()
    # iterate over test data
    for inputs, labels in dataloader:
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    tab_pred = {"target": y_true,
               "prediction": y_pred
            }

    df_pred = pd.DataFrame(tab_pred)
    df_pred.to_csv(f'results/experiment_{experiment_id}/predictions.csv', index=False)

    # constant for classes
    classes = data.classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    # Create pandas dataframe
    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
    
    df_cm.to_csv(f'results/experiment_{experiment_id}/df_cm.csv')
    
    plt.figure(figsize = (12, 7))
    sn.heatmap(df_cm, annot=True, cbar=None, cmap="OrRd",fmt="d")
    
    plt.title("Confusion Matrix"), plt.tight_layout()

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