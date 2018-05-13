"""
Define some useful plot functions
"""

import matplotlib.pyplot as plt

def plot_learning_curves(loss, accuracy):
    """
    Plot learning curves
    Args:
        loss (dictionary): Dictionnary that contains loss (on the form: {'train': [loss_epoch1, loss_epoch2, ...], 'test': [loss_epoch1, loss_epoch2, ...]})
        accuracy (dictionary): Dictionnary that contains accuracy(on the form: {'train': [acc_epoch1, acc_epoch2, ...], 'test': [acc_epoch1, acc_epoch2, ...]})
    """

    x = [i for i in range(len(loss['train']))]

    plt.clf()

    ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)

    #Loss
    ax1.plot(x, loss['train'], c='b')
    ax1.plot(x, loss['test'], c='r')

    #Accuracy
    ax2.plot(x, accuracy['train'], c='b')
    ax2.plot(x, accuracy['test'], c='r')

    plt.show()
