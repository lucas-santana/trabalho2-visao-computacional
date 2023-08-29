
import matplotlib.pyplot as plt

def plot_acc(accuracies):
    plt.plot(accuracies)
    plt.title('Accuracy Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
def plot_loss(losses):
    plt.plot(losses)
    plt.title('Loss Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
# def plot_confusion_matrix():
    
    