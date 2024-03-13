import numpy as np
from functools import partial
import logging
import matplotlib.pyplot as plt

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, loss, data, label, nbr_epochs, batch_size, lr):
    """ Basic training routine with logging and error graphing.

    :param model: Model to be trained
    :param loss: Loss
    :param data: Training data
    :param label: Training labels
    :param nbr_epochs: Nbr of epochs
    :param batch_size: Batch size
    :param lr: Learning rate
    :return:
    """

    # Initialization
    model.initialize_parameter()
    logging.info("Initialization done")

    # Use standard gradient descent for updating the parameters
    up_fun = partial(vanilla_grad_desc, lr=lr)
    nbr_data, _ = data.shape

    epoch_losses = []  # To store mean loss of each epoch
    # Run training epochs
    for e in range(nbr_epochs):
        logging.info('-------- Running epoch %d --------' % e)

        indices = np.random.permutation(np.arange(nbr_data))
        epoch_loss = 0

        for batch_num in range(int(np.ceil(nbr_data / batch_size))):
            batch_indices = indices[batch_num * batch_size: min((batch_num + 1) * batch_size, nbr_data)]
            batch_data = data[batch_indices, :]
            batch_label = label[batch_indices]

            model_out = model.fprop(batch_data)
            loss.set_targets(batch_label)
            batch_losses = loss.fprop(model_out)
            epoch_loss += np.mean(batch_losses)

            z = loss.bprop(np.ones_like(batch_label) / len(batch_indices))
            model.bprop(z)
            model.update_parameters(up_fun)

        epoch_loss /= int(np.ceil(nbr_data / batch_size))
        epoch_losses.append(epoch_loss)
        

        logging.info("Epoch %d: Loss = %f", e, epoch_loss)
        
    # After training, plot the epoch losses.
    epochs_list = list(range(nbr_epochs))  # Generate a list of epoch numbers.
    plt.plot(epochs_list, epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.show()
    return epochs_list, epoch_losses


def vanilla_grad_desc(para, grad_para, lr):
    """ Update function for the vanilla gradient descent.

    :param para: Parameter to be updated
    :param grad_para: Gradient at the parameter
    :param lr: Learning rate
    :return:
    """
    return para - lr * grad_para
