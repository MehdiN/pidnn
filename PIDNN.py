import numpy as np
from numpy import array


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

    def __init__(self, learning_rate, learning_rate_2, state_vector_dim):
        self.mu = learning_rate
        self.sigma = learning_rate_2
        self.dim_state = state_vector_dim
        self.loss = 0
        self.W1 = {}
        self.W2 = np.zeros((self.dim_state, 1))
        self.I = {}
        self.D = {}
        self.cache = {}
        self.output = None

    def init_net(self):
        """
        Initialization of the Neural Network

        ...
        We randomly initialize the weights

        """
        self.W2 = np.random.randn(self.dim_state, 3 * self.dim_state) * 0.1
        for i in range(self.dim_state):
            self.W1["W1" + str(i)] = np.random.randn(3, 2) * 0.1
            self.I['I' + str(i)] = 0
            self.D["D" + str(i)] = 0

    # Activation functions

    def p_neuron(self, v):
        """
        P Neuron Activation Function

        ...
        Return input for input between -1 and 1
        """
        if np.fabs(v) > 1:
            v = v / np.fabs(v)
        return v

    def i_neuron(self, v, I):
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

    def d_neuron(self, v, D):
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

    def neuron_back(self, z):
        """
        Derivative function for backpropagation computation

        ...
        """
        x = (np.fabs(z) < 1) * np.sign(z)
        return x

    def pid_activation_neuron(self, Z):
        A = np.zeros((3, 1))
        A[0] = self.p_neuron(Z[0][0])
        A[1] = self.i_neuron(Z[1][0], I)
        A[2] = self.d_neuron(Z[2][0], D)
        return A

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
        Z1_cache = {}
        X = np.concatenate((x.T, r.T), axis=0)
        for i in range(self.dim_state):
            I = self.I["I" + str(i)]
            D = self.D["D" + str(i)]
            X_temp = X[:, i]
            X_temp = np.reshape(X[:, 1], (X.shape[0], -1))
            W1 = self.W1['W1' + str(i)]
            Z1 = np.dot(W1, X_temp)
            Z1_cache["Z1" + str(i)] = Z1
            A1_temp = self.pid_activation_neuron(Z1)
            A1["A1" + str(i)] = A1_temp
            # update pid integrator and derivator
            self.I["I" + str(i)] = a1[1].squeeze()
            self.D["D" + str(i)] = a1[2].squeeze()
        Z1_cache = Z1_cache.reshape((Z1_cache.shape[0], -1))
        self.cache["Z1"] = Z1_cache
        return A1

    def full_connected_output(self, A1_in):
        """
        Second forward pass function

        ...
        Input:

        A1: output from the PID layer

        Return:

        A2: Output of the NN
        """
        temp = [A1_in["A1" + str(i)] for i in range(dim_state)]
        A1 = np.concatenate(temp, axis=1)
        self.cache['A1'] = A1
        A2 = np.dot(self.W2, A1)
        # A2 = np.zeros(Z2.shape)
        # saturate the output;
        # for i in range(len(A2)):
        #    A2[i][0] = self.p_neuron(A2[i][0])
        self.cache["A2"] = A2

    def compute_cost(self, r, y):
        """
        Compute the cost of each pass
        ...
        """
        # pdb.set_trace()
        self.loss = 0.5 * np.dot((r - y).T, r - y)
        return self.loss

    def backward_pass_1(self, Y):
        """
        First backward pass function

        ...
        Input:

        A2: output from the feedback propagation
        Y: output signal from the plant i.e x at t-1

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

    def update_weights(self, err, u, u_old, y, y_old, W2_old):
        # update W2
        q = [np.sign((y[i, :] - y_old[i, :]) / (u[i, :] - u_old[i, :]))
             for i in range(len(y))]
        S = np.sum([q[i] * err[i, :] for i in range(len(q))])
        tmp_W2 = self.W2
        old_W2 = self.cache["old_W2"]
        for i in range(W2.shape[0]):
            for j in range(W2.shape[2]):
                self.W2[i, j] = self.W2[i, j] + self.mu * err[i, :] * \
                    q[i] * u[i, :] - self.sigma * \
                    (self.W2[i, j] - old_W2[i, j])
        old_W2 = tmp_W2
        self.cache["old_W2"] = old_W2
        # update W1
        pass

    # def backward_pass_2(self,x, y):
    #     """
    #     Second backward pass function

    #     ...
    #     Input:

    #     Z1_cache: output from the first layer
    #     x: feedback signal at t
    #     r: setpoint signal

    #     Return:
    #     dW1: weight gradient
    #     """
    #     # compute dW1
    #     dZ2 = self.cache["dZ2"]
    #     Z1 = self.cache["Z1"]
    #     dZ1 = np.dot(self.W2.T, dZ2) * self.neuron_back(Z1)
    #     dW1 = {}
    #     for i in range(self.dim_state):
    #         X = array([x[i], r[i]])
    #         X = X.reshape((2, -1))
    #         dW1["dW1" + str(i)] = np.dot(dZ1[i:i + 3], X.T)
    #     return dW1

    # def update_weights(self,dW1, dW2):
    #     """
    #     Gradient descent update step

    #     ...
    #     Input:

    #     dW1,dW2: gradient weight
    #     """
    #     self.W2 += -1.0 * learning_rate * dW2
    #     for i in range(self.dim_state):
    #         self.W1["W1" + str(i)] += -1.0 * learning_rate * dW1["dW1" + str(i)]

    # def feedforward(self, x, y):
    #     # FeedFoward
    #     A1 = self.pid_forward_pass(x, y)
    #     self.full_connected_output(A1)
    #     self.output = self.cache["A2"]

    # def backprop(self, x, r):
    #     # Backprop
    #     dW2 = self.backward_pass_1(x)
    #     dW1 = self.backward_pass_2(x, y)
    #     # Gradient Descent
    #     self.update_weights(dW1, dW2)

    # def run_nn(self,x,r):
    #     self.feedforward(x,y)
    #     self.backprop(x,r,y)
