import abc
import numpy as np


class NNModule:
    """ Class defining abstract interface every module has to implement

    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fprop(self, input):
        """ Forwardpropagate the input through the module

        :param input: Input tensor for the module
        :return Output tensor after module application
        """
        return

    @abc.abstractmethod
    def bprop(self, grad_out):
        """ Backpropagate the gradient the output to the input

        :param grad_out: Gradients at the output of the module
        :return: Gradient wrt. input
        """
        return

    @abc.abstractmethod
    def get_grad_param(self, grad_out):
        """ Return gradients wrt. the parameters
        Calculate the gardients wrt. to the parameters of the module. Function already
        accumulates gradients over the batch -> Save memory and implementation issues using numpy avoid loops

        :param grad_out: Gradients at the output
        :return: Gradients wrt. the internal parameter accumulated over the batch
        """
        return

    @abc.abstractmethod
    def apply_parameter_update(self, acc_grad_para, up_fun):
        """ Apply the update function to the internal parameters.

        :param acc_grad_para: Accumulated gradients over the batch
        :param up_fun: Update function used
        :return:
        """
        return

    # If we would like to support different initialization techniques, we could
    # use an Initializer class
    # For simplicity use a fixed initialize for each module
    @abc.abstractmethod
    def initialize_parameter(self):
        """ Initialize the internal parameter

        :return:
        """


class NNModuleParaFree(NNModule):
    """Specialization of the NNModule for modules which do not have any internal parameters

    """
    __metaclass__ = abc.ABCMeta

    def initialize_parameter(self):
        # No initialization necessary
        return

    def get_grad_param(self, grad_out):
        # No parameter gradients
        return None

    def apply_parameter_update(self, acc_grad_para, up_fun):
        # No parameters to update
        return


class LossModule(NNModule):
    """Specialization of NNModule for losses which need target values

    """
    __metaclass__ = abc.ABCMeta

    def set_targets(self, t):
        """Saves expected targets.
        Does not copy the input.

        :param t: Expected target values.
        :return:
        """
        self.t = t

    def initialize_parameter(self):
        # No internal parameters
        return

    def get_grad_param(self, grad_out):
        # No gradient for internal parameter
        return None

    def apply_parameter_update(self, acc_grad_para, up_fun):
        # No update needed
        return


# Task 2 a)
class Linear(NNModule):
    """Module which implements a linear layer"""

    #####Start Subtask 2a#####
    def __init__(self, n_in, n_out):
        # Nbr input neurons
        self.n_in = n_in
        # Nbr output neurons
        self.n_out = n_out
        # Weights
        self.W = None
        # Biases
        self.b = None
        # Input cache for bprop
        self.cache_input = None

    def initialize_parameter(self):
        # Glorot intialization
        sigma = np.sqrt(2.0 / (self.n_in + self.n_out))
        self.W = np.random.normal(0, sigma, (self.n_in, self.n_out))
        self.b = np.zeros((1, self.n_out))
        # self.W = np.zeros((self.n_in, self.n_out))

    def fprop(self, input):
        # A copy of the input for deriving parameter update
        self.cache_input = np.array(input)
        return np.matmul(input, self.W) + self.b

    def bprop(self, grad_out):
        return np.matmul(grad_out, self.W.transpose())

    def get_grad_param(self, grad_out):
        grad_w = np.matmul(self.cache_input.transpose(), grad_out)
        # Distinguish batch mode
        grad_b = np.sum(grad_out, 0) if grad_out.ndim > 1 else grad_out
        return grad_w, grad_b

    def apply_parameter_update(self, acc_grad_para, up_fun):
        self.W = up_fun(self.W, acc_grad_para[0])
        self.b = up_fun(self.b, acc_grad_para[1])

    #####End Subtask#####


# Task 2 b)
class Softmax(NNModuleParaFree):
    """Softmax layer"""

    #####Start Subtask 2b#####
    def __init__(self):
        # Cache output for bprob
        self.cache_out = None

    def fprop(self, input):
        # See 4a for stability reasons
        inp_max = np.max(input, 1)
        # Transpose -> Numpy subtracts from each batch is inp_max using numpy's broadcasting
        exponentials = np.exp((input.transpose() - inp_max).transpose())
        normalization = np.sum(exponentials, 1)

        # Transpose -> numpy broadcast -> see above
        output = (exponentials.transpose() / normalization).transpose()
        self.cache_out = np.array(output)

        return output

    def bprop(self, grad_out):
        if grad_out.ndim == 2:
            sz_batch, n_out = grad_out.shape
        else:
            sz_batch = 1
            n_out = len(grad_out)

        # 1. term
        v_s = np.empty((sz_batch, 1))
        for i in range(sz_batch):
            v_s[i, :] = np.dot(grad_out[i, :], self.cache_out[i, :])
        # 2. term
        v_v_s = grad_out - np.broadcast_to(v_s, (sz_batch, n_out))
        z = np.multiply(self.cache_out, v_v_s)
        return z

    #####End Subtask#####


class CrossEntropyLoss(LossModule):
    """Cross-Entropy-Loss"""

    def __init__(self):
        self.cache_in = None  # To store the input to fprop for use in bprop
        self.sz_batch = self.n_in = None

    def fprop(self, input):
        self.cache_in = np.array(input)  # Cache the input for use in bprop
        self.sz_batch, self.n_in = input.shape
        loss = -1 * np.log(input[np.arange(self.sz_batch), self.t])
        return loss

    def bprop(self, grad_out):
        # One-hot encode the true labels
        true_labels_one_hot = np.zeros_like(self.cache_in)
        true_labels_one_hot[np.arange(self.sz_batch), self.t] = 1
        
        # The gradient of cross-entropy loss w.r.t. softmax input is the difference between predictions and true labels
        z = self.cache_in - true_labels_one_hot
        
        # Scale the gradients by grad_out if necessary. Usually, grad_out is an array of ones.
        if grad_out is not None:
            z *= grad_out[:, None]  # Ensures grad_out is correctly shaped for broadcasting
        
        return z


# Task 3 b)
class Tanh(NNModuleParaFree):
    """Module implementing a Tanh acitivation function"""

    def __init__(self):
        # Cache output for bprop
        self.cache_out = None

    def fprop(self, input):
        output = np.tanh(input)
        self.cache_out = np.array(output)
        return output

    def bprop(self, grad_out):
        return np.multiply(grad_out, 1 - self.cache_out ** 2)


# Task 4 e)
class LogCrossEntropyLoss(LossModule):
    """Log-Cross-Entropy-Loss"""
    def __init__(self):
        self.sz_batch = self.n_in = None

    def fprop(self, input):
        self.sz_batch, self.n_in = input.shape
        # Assumes `input` is the output of a LogSoftmax layer
        loss = -1 * input[np.arange(self.sz_batch), self.t]
        return loss

    def bprop(self, grad_out):
        # Initialize the gradient tensor with zeros
        z = np.zeros((self.sz_batch, self.n_in))
        
        # The gradient for log-cross-entropy with respect to the input log probabilities is straightforward:
        # For each example in the batch, we only need to adjust the gradient for the true class index to -1,
        # as the derivative of -log(p) is -1/p, but since we're in log space, it simplifies to just -1 at the true class.
        z[np.arange(self.sz_batch), self.t] = -1
        
        # If `grad_out` is provided and not just an array of ones, it should be used to scale the gradients.
        # This is often unnecessary for the final loss layer, but included here for completeness.
        if grad_out is not None:
            z *= grad_out[:, None]  # Ensure proper broadcasting
        
        return z


# Task 4 e)
class LogSoftmax(NNModuleParaFree):
    """Log-Softmax-Module"""

    def __init__(self):
        # Save output for bprop
        self.cache_out = None

    def fprop(self, input):
        # See 4a for stability reasons
        inp_max = np.max(input, 1)
        # Transpose for numpy broadcasting -> Subtract each batch max from the batch
        input = (input.T - inp_max).T
        exponentials = np.exp(input)
        log_normalization = np.log(np.sum(exponentials, 1))

        # Transpose -> Subtract log normalization for each batch and reshape to batch \times output
        output = (input.T - log_normalization).T
        self.cache_out = np.array(output)

        return output

    def bprop(self, grad_out):
        sz_batch, n_in = grad_out.shape
        sum_grad = np.sum(grad_out, 1).reshape((sz_batch, 1))
        sigma = np.exp(self.cache_out)
        z = grad_out - np.multiply(sum_grad, sigma)
        return z
