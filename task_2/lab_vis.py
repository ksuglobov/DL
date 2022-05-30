import numpy as np

import matplotlib.pyplot as plt


def plot_loss(plot_name, n_epochs, train_loss, val_loss):
    fig = plt.figure(figsize=(10, 5))

    x = np.array(range(n_epochs)) + 1
    plt.plot(x, train_loss, label='train')
    plt.plot(x, val_loss, label='validation')

    plt.xlabel('# epochs')
    plt.ylabel('CrossEntropy')
    
    plt.xticks(x)

    plt.title(plot_name)
    plt.legend()
    plt.grid(linestyle=':')
    plt.show()
    
def plot_cv_losses(plot_name, xlabel,
                   train_losses, val_losses,
                   x_arr, graps_labels_arr):
    fig = plt.figure(figsize=(10, 5))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i, (label, color) in enumerate(zip(graps_labels_arr, colors)):
        train_loss, val_loss = train_losses[:,i], val_losses[:,i]
        plt.plot(x_arr, train_loss, color=color, linestyle='dashed')
        plt.plot(x_arr, val_loss, color=color, linestyle='solid', label=label)

    plt.xlabel(xlabel)
    plt.ylabel('CrossEntropy')
    
    plt.xticks(x_arr)

    plt.suptitle(plot_name)
    plt.title('(train - dashed, validation - solid)')
    plt.legend()
    plt.grid(linestyle=':')
    plt.show()
