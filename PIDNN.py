import numpy as np
from numpy import array
import pdb


class PIDNN(object):
    '''A PID Neural Network Controller Class

    ...
    This PID neural net is a 2-Layer neural network.
    The first layer is not fully connected but the second is.
    This implementation does not include added bias.
    ...
    We initiliaze the class with the following parameters:
    - learning_rate: it sets the learning rate for the gradiant descent
    - state_vector_dim; the dimension of the state vector fed to the controller
    '''

    def __init__(self, learning_rate, state_vector_dim):
        self.learning_rate = learning_rate
        self.dim_state = state_vector_dim
        self.loss = 10
        self.W1 = {}
        self.W2 = np.zeros((self.dim_state, 1))
        self.I = {}
        self.D = {}
        self.cache = {}

    def init_net(self):
        """
        Initialization of the Neural Network

        ...
        We randomly initialize the weights

        """
        self.W2 = np.random.randn(self.dim_state, 3 * self.dim_state) * 0.01
        for i in range(self.dim_state):
            self.W1["W1" + str(i)] = np.random.randn(3, 2) * 0.01
            self.I['I' + str(i)] = 0
            self.D["D" + str(i)] = 0

    # Activation functions

    def p_neuron(self,v):
        """
        P Neuron Activation Function

        ...
        Return input for input between -1 and 1
        """
        if np.fabs(v) > 1:
            v = v / np.fabs(v)
        return v

    def i_neuron(self,v, I):
        """
        I Neuron Activation Function

        ...
        Return input + integration term for input between -1 and 1
        """
        if np.fabs(v) > 1:
            v = v / np.fasb(v)
        else:
            v = v + I
        return v

    def d_neuron(self,v, D):
        """
        D Neuron Activation Function

        ...
        Return input - derivative term for input between -1 and 1
        """
        if np.fabs(v) > 1:
            v = v / np.fasb(v)
        else:
            v = v - D
        return v

    def neuron_back(self,z):
        """
        Derivative function for backpropagation computation

        ...
        """
        x = (np.fabs(z) < 1) * np.sign(z)
        return x

    def pid_forward_pass(self, x, r):
        """
        First forward pass function (PID layer)

        ...
        Input:

        x: state vector from feedback observation (n vector)
        r: retpoint contol signal (n vector)

        Return:
        A1: output from PID
        Z1_cache: for backpropagation step
        """
        A1 = {}
        Z1_cache = []
        for i in range(self.dim_state):
            a1 = np.zeros((3, 1))
            I = self.I["I" + str(i)]
            D = self.D["D" + str(i)]
            X = np.concatenate((x[i], r[i]))
            X = X.reshape((2, -1))
            W = self.W1['W1' + str(i)]
            Z1 = np.dot(W, X)
            Z1_cache = np.append(Z1_cache, Z1)
            a1[0] = self.p_neuron(Z1[0][0])
            a1[1] = self.i_neuron(Z1[1][0], I)
            a1[2] = self.d_neuron(Z1[2][0], D)
            A1["a1" + str(i)] = a1
            # update pid integrator and derivator
            self.I["I" + str(i)] = a1[1].squeeze()
            self.D["D" + str(i)] = a1[2].squeeze()
        Z1_cache = Z1_cache.reshape((Z1_cache.shape[0], -1))
        self.cache["Z1"] = Z1_cache
        return A1

    def forward_output(self, A1):
        """
        Second forward pass function

        ...
        Input:

        A1: output from the PID layer

        Return:

        A2: Output of the NN
        """
        _A1 = []
        for i in range(self.dim_state):
            a1 = A1['a1' + str(i)]
            _A1 = np.append(_A1, a1)
        _A1 = _A1.reshape((self.dim_state * 3, -1))
        self.cache['A1'] = _A1
        Z2 = np.dot(self.W2, _A1)
        A2 = np.zeros(Z2.shape)
        # saturate the output;
        for i in range(len(A2)):
            A2[i][0] = self.p_neuron(A2[i][0])
        self.cache["A2"] = A2

    def compute_cost(self, r, y):
        """
        Compute the cost of each pass
        ...
        """
        self.loss = 0.5 * np.dot((r - y).T, r - y)

    def backward_pass_1(self, Y):
        """
        First backward pass function

        ...
        Input:

        A2: output from the feedback propagation
        Y: output signal from the plant

        Return:
        dW2: weight gradient
        dZ2: output gradient
        """
        # compute dW2
        A2 = self.cache["A2"]
        dZ2 = A2 - Y
        self.cache["dZ2"] = dZ2
        A1 = self.cache['A1']
        dW2 = np.dot(dZ2, A1.T)
        return dW2

    def backward_pass_2(self,x, y):
        """
        Second backward pass function

        ...
        Input:

        Z1_cache: output from the first layer
        x: feedback signal
        r: setpoint signal

        Return:
        dW1: weight gradient
        """
        # compute dW1
        dZ2 = self.cache["dZ2"]
        Z1 = self.cache["Z1"]
        dZ1 = np.dot(self.W2.T, dZ2) * self.neuron_back(Z1)
        dW1 = {}
        for i in range(self.dim_state):
            X = array([x[i], r[i]])
            X = X.reshape((2, -1))
            dW1["dW1" + str(i)] = np.dot(dZ1[i:i + 3], X.T)
        return dW1

    def update_weights(self,dW1, dW2):
        """
        Gradient descent update step

        ...
        Input:

        dW1,dW2: gradient weight
        """
        self.W2 += -1.0 * learning_rate * dW2
        for i in range(self.dim_state):
            self.W1["W1" + str(i)] += -1.0 * learning_rate * dW1["dW1" + str(i)]

    def feedforward(self, x, y):
        # FeedFoward
        A1 = self.pid_forward_pass(x, y)
        self.forward_output(A1)
        u = self.cache["A2"]
        return u

    def backprop(self, x, r, Y):
        # Backprop
        dW2 = self.backward_pass_1(Y)
        dW1 = self.backward_pass_2(x, y)
        # Gradient Descent
        self.update_weights(dW1, dW2)
