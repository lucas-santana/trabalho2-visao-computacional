
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

def plot_acc(train_accu, eval_accu):
    f = plt.figure() # gera uma figura do gr ́afico (antes de desenh ́a-lo)

    plt.plot(train_accu,'-o')
    plt.plot(eval_accu,'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Test'])
    plt.title('Train vs Test Accuracy')

    # plt.show()
    
    f.savefig('results/acc.pdf')
    
def plot_loss(train_losses, eval_losses):
    f = plt.figure() # gera uma figura do gr ́afico (antes de desenh ́a-lo)

    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Test'])
    plt.title('Train vs Test Losses')

    # plt.show()
    
    f.savefig('results/loss.pdf')
    
def plot_confusion_matrix(model, dataloader):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in dataloader:
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    
    # Create pandas dataframe
    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
    
    df_cm.to_csv('results/df_cm.csv')
    
    plt.figure(figsize = (12, 7))
    sn.heatmap(df_cm, annot=True, cbar=None, cmap="OrRd",fmt="d")
    
    plt.title("Confusion Matrix"), plt.tight_layout()

    plt.ylabel("True Class"), 
    plt.xlabel("Predicted Class")
    
    plt.savefig('results/cm.pdf')