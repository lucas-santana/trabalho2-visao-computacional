
import matplotlib.pyplot as plt

def plot_acc(train_accu, eval_accu):
    f = plt.figure() # gera uma figura do gr ́afico (antes de desenh ́a-lo)

    plt.plot(train_accu,'-o')
    plt.plot(eval_accu,'-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Accuracy')

    plt.show()
    f.savefig('results/acc.pdf')
    
def plot_loss(train_losses, eval_losses):
    f = plt.figure() # gera uma figura do gr ́afico (antes de desenh ́a-lo)

    plt.plot(train_losses,'-o')
    plt.plot(eval_losses,'-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train','Valid'])
    plt.title('Train vs Valid Losses')

    plt.show()
    f.savefig('results/loss.pdf')
    
# def plot_confusion_matrix():
    
    