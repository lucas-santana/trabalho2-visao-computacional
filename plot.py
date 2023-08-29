
import matplotlib.pyplot as plt

def plot_acc(train_accu, eval_accu):
    plt.plot(train_accu,'-o')
    plt.plot(eval_accu,'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')

    plt.show()
    
def plot_loss(train_losses, eval_losses):
    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')

    plt.show()
    
# def plot_confusion_matrix():
    
    